"""Score OptiSet candidate path sets with a frozen downstream LLM.

Adds answer-token NLL scores from either a Transformers or vLLM backend.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "retrieve"))

from src.dataset.retriever import RetrieverDataset  # noqa: E402
# from selector_inference import build_config  # noqa: E402
from selector_optiset_utils import (  # noqa: E402
    build_path_prompt,
    load_jsonl,
    mean,
    pearson,
    safe_tag,
    write_jsonl,
)

def build_config(args, checkpoint):
    """Reconstruct the retriever dataset config from checkpoint args and CLI overrides."""
    saved_args = checkpoint.get("args", {})
    dataset = args.dataset or saved_args.get("dataset")
    if dataset is None:
        raise ValueError("Checkpoint does not record a dataset; please pass --dataset.")
    return {
        "env": {
            "num_threads": args.num_threads or saved_args.get("num_threads", 1),
            "seed": args.seed if args.seed is not None else saved_args.get("seed", 42),
        },
        "dataset": {
            "name": dataset,
            "text_encoder_name": saved_args.get("text_encoder_name", "gte-large-en-v1.5"),
            "data_dir": args.data_dir,
        },
        "retriever": {
            "topic_pe": saved_args.get("topic_pe", True),
            "DDE_kwargs": {
                "num_rounds": saved_args.get("num_rounds", 2),
                "num_reverse_rounds": saved_args.get("num_reverse_rounds", 2),
            },
        },
    }


def output_name(args):
    candidate = safe_tag(Path(args.candidate_path).stem, max_len=180)
    llm = safe_tag(args.model_name.split("/")[-1])
    quant = "4bit" if args.load_in_4bit else "8bit" if args.load_in_8bit else "none"
    return (
        f"optiset_scored_{candidate}_backend-{args.backend}_llm-{llm}"
        f"_maxlen{args.max_length}_ctx{args.max_context_chars}_maxans{args.max_answers}_dtype{args.dtype}_q{quant}.jsonl"
    )


def load_transformers_llm(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = None
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device_map = {"": args.llm_device_map} if args.llm_device_map.startswith(("cuda", "cpu")) else args.llm_device_map
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return tokenizer, model


def score_answer_nll_transformers(tokenizer, model, prompts, answers, max_length):
    import torch
    import torch.nn.functional as F

    full_texts = [p + a for p, a in zip(prompts, answers)]
    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
    enc = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    labels = input_ids.clone()
    answer_token_counts = []
    for i in range(labels.size(0)):
        answer_start = int(prompt_enc["attention_mask"][i].sum().item())
        labels[i, :answer_start] = -100
        labels[i, attention_mask[i] == 0] = -100
        answer_token_counts.append(int((labels[i] != -100).sum().item()))
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shifted_logits = outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        losses = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shifted_labels.shape)
        denom = (shifted_labels != -100).sum(dim=1).clamp_min(1)
        return (losses.sum(dim=1) / denom).detach().float().cpu().tolist(), answer_token_counts


def load_vllm_llm(args):
    from vllm import LLM

    dtype = "bfloat16" if args.dtype == "bf16" else "float16"
    return LLM(
        model=args.model_name,
        dtype=dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_length,
    )


def _token_logprob(entry):
    if entry is None:
        return None
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    if isinstance(entry, dict) and "logprob" in entry:
        return float(entry["logprob"])
    return None


def score_answer_nll_vllm(llm, prompts, answers, max_length):
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    texts = [p + a for p, a in zip(prompts, answers)]
    prompt_lengths = [len(tokenizer.encode(p, add_special_tokens=True)) for p in prompts]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    outputs = llm.generate(texts, sampling_params, use_tqdm=False)
    nlls = []
    answer_token_counts = []
    for output, prompt_len in zip(outputs, prompt_lengths):
        token_ids = output.prompt_token_ids[:max_length]
        prompt_logprobs = output.prompt_logprobs[:max_length]
        losses = []
        for idx in range(max(prompt_len, 1), len(token_ids)):
            choices = prompt_logprobs[idx]
            if not choices:
                continue
            token_id = token_ids[idx]
            entry = choices.get(token_id) if isinstance(choices, dict) else None
            logprob = _token_logprob(entry)
            if logprob is not None:
                losses.append(-logprob)
        answer_token_counts.append(len(losses))
        nlls.append(float(sum(losses) / max(len(losses), 1)))
    return nlls, answer_token_counts


def answer_list(row, sample=None):
    answers = row.get("gold_answers")
    if answers is None and sample is not None:
        answers = sample.get("a_entity")
    if answers is None:
        target = str(row.get("target_answer_string") or "").strip()
        answers = [part.strip() for part in target.split("|") if part.strip()] if target else []
    return sorted({str(answer).strip() for answer in answers if str(answer).strip()})


def row_answer_count(row, sample=None):
    if row.get("num_gold_answers") is not None:
        return int(row["num_gold_answers"])
    if row.get("answer_count") is not None:
        return int(row["answer_count"])
    return len(answer_list(row, sample))


def load_llm(args):
    if args.backend == "transformers":
        return load_transformers_llm(args)
    return load_vllm_llm(args)


def score_batch(args, scorer, prompts, answers):
    if args.backend == "transformers":
        tokenizer, model = scorer
        return score_answer_nll_transformers(tokenizer, model, prompts, answers, args.max_length)
    return score_answer_nll_vllm(scorer, prompts, answers, args.max_length)


def sample_lookup(args, candidate_rows):
    dataset_name = args.dataset or candidate_rows[0].get("dataset")
    if dataset_name is None:
        raise ValueError("Pass --dataset or use candidate rows with dataset metadata.")
    pseudo_args = argparse.Namespace(dataset=dataset_name, data_dir=args.data_dir, num_threads=args.num_threads, seed=args.seed)
    config = build_config(pseudo_args, {"args": {"dataset": dataset_name}})
    config["dataset"]["emb_mode"] = args.emb_mode
    dataset = RetrieverDataset(config=config, split=args.split or candidate_rows[0].get("split", "train"), skip_no_path=True, emb_mode=args.emb_mode)
    return {sample["id"]: sample for sample in dataset.processed_dict_list}, dataset_name


def summarize(rows):
    sets = [s for row in rows for s in row.get("sets", []) if "llm_gain" in s]
    gains = [s["llm_gain"] for s in sets]
    answer_gains = [s["llm_gain"] for s in sets if s.get("answer_path_count", 0) > 0]
    no_answer_gains = [s["llm_gain"] for s in sets if s.get("answer_path_count", 0) == 0]
    spreads, det_best, pairs = [], [], []
    for row in rows:
        row_sets = [s for s in row.get("sets", []) if "llm_gain" in s]
        if row_sets:
            row_gains = [s["llm_gain"] for s in row_sets]
            spreads.append(max(row_gains) - min(row_gains))
            best = max(row_sets, key=lambda s: s["llm_gain"])
            det_best.append(1.0 if best.get("source") == "deterministic_beam" else 0.0)
            for s in row_sets:
                pairs.append((s.get("selector_set_score_at_sampling", 0.0), s["llm_gain"]))
    return {
        "num_rows": len(rows),
        "num_sets": len(sets),
        "mean_llm_gain": mean(gains),
        "mean_answer_reaching_set_gain": mean(answer_gains),
        "mean_non_answer_set_gain": mean(no_answer_gains),
        "mean_gain_spread": mean(spreads),
        "median_gain_spread": float(statistics.median(spreads)) if spreads else 0.0,
        "deterministic_is_best_rate": mean(det_best),
        "corr_selector_set_score_at_sampling_llm_gain": pearson([x for x, _ in pairs], [y for _, y in pairs]),
    }


def main(args):
    from tqdm import tqdm
    scorer = load_transformers_llm(args) if args.backend == "transformers" else load_vllm_llm(args)
    # scorer = load_llm(args)
    rows = load_jsonl(args.candidate_path)
    samples, dataset_name = sample_lookup(args, rows)
    
    scored_rows = []
    scoring_config = {
        "backend": args.backend,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "max_context_chars": args.max_context_chars,
        "max_answers": args.max_answers,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
    }
    for row in tqdm(rows, desc="score_optiset_candidates"):
        sample = samples.get(row["id"])
        if sample is None:
            continue
        answers_for_row = answer_list(row, sample)
        answer_count = row_answer_count(row, sample)
        if args.max_answers > 0 and answer_count > args.max_answers:
            continue
        answer = row.get("target_answer_string") or " | ".join(answers_for_row)
        if not answer:
            continue
        prompts = [build_path_prompt(sample, [], max_context_chars=args.max_context_chars)]
        prompts.extend(build_path_prompt(sample, s.get("paths", []), max_paths=args.max_paths_per_set, max_context_chars=args.max_context_chars) for s in row.get("sets", []))
        answers = [answer] * len(prompts)
        nlls = []
        answer_token_counts = []
        for start in range(0, len(prompts), args.batch_size):
            batch_nlls, batch_answer_token_counts = score_batch(
                args,
                scorer,
                prompts[start:start + args.batch_size],
                answers[start:start + args.batch_size],
            )
            nlls.extend(batch_nlls)
            answer_token_counts.extend(batch_answer_token_counts)
        empty_nll = nlls[0]
        out = dict(row)
        out["schema_version"] = "selector_optiset_scored.v1"
        out["dataset"] = dataset_name
        out["gold_answers"] = answers_for_row
        out["num_gold_answers"] = int(answer_count)
        out["answer_count"] = int(answer_count)
        out["empty_answer_nll"] = float(empty_nll)
        out["llm_backend"] = args.backend
        out["llm_model_name"] = args.model_name
        out["scoring_config"] = scoring_config
        for set_obj, nll, answer_tokens in zip(out.get("sets", []), nlls[1:], answer_token_counts[1:]):
            set_obj["answer_nll"] = float(nll)
            set_obj["empty_answer_nll"] = float(empty_nll)
            set_obj["llm_gain"] = float(empty_nll - nll)
            set_obj["num_answer_tokens"] = int(answer_tokens)
            set_obj["llm_backend"] = args.backend
            set_obj["llm_model_name"] = args.model_name
        scored_rows.append(out)
    output = Path(args.output) if args.output else Path(args.output_dir) / output_name(args)
    write_jsonl(output, scored_rows)
    summary = summarize(scored_rows)
    summary["output"] = str(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Score OptiSet candidates with a frozen downstream LLM")
    p.add_argument("--candidate_path", required=True)
    p.add_argument("--dataset", default=None, choices=["webqsp", "cwq"])
    p.add_argument("--split", default=None)
    p.add_argument("--data_dir", default="retrieve")
    p.add_argument(
        "--emb_mode",
        default="none",
        choices=["none", "eager", "lazy", "sharded_lazy"],
        help="Embedding loading mode for RetrieverDataset; use 'none' for text/path-only scoring.",
    )
    p.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    p.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--llm_device_map", default="auto")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--max_context_chars", type=int, default=6000)
    p.add_argument("--max_paths_per_set", type=int, default=None)
    p.add_argument("--max_answers", type=int, default=50, help="Skip samples whose gold answer list is longer than this; use <=0 to disable.")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_threads", type=int, default=4)
    p.add_argument("--output_dir", default="outputs/optiset/scored")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    # if args.backend == "tranformers":
    #     args.backend = "transformers"
    return args


if __name__ == "__main__":
    main(parse_args())
