"""Run KGQA reasoning over paths decoded by selector_inference.py.

This entrypoint mirrors the original SubgraphRAG reasoning flow in ``reason/`` but
uses selector beam-search paths instead of a scored triple set.  Each path is
rendered as a compact arrow sequence, e.g. ``head -> relation -> tail`` for a
forward traversal and ``tail <- relation <- head`` for a reverse traversal.
"""

import argparse
import json
import os
import sys

import swanlab
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
REASON_ROOT = REPO_ROOT / "reason"
sys.path.insert(0, str(REASON_ROOT))



PATH_SYS_PROMPT = (
    "Based on the paths retrieved from a knowledge graph, please answer the question. "
    'Please return formatted answers as a list, each prefixed with "ans:".'
)

PATH_COT_PROMPT = (
    "Let's think step by step. "
    'Return the most possible answers based on the given paths by listing each answer on a separate line, starting with the prefix "ans:". '
    'Otherwise, if there is no sufficient information to answer the question, return "ans: not available".'
)


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_checkpoint(file_path):
    if os.path.exists(file_path):
        print("*" * 50)
        print(f"Resuming from {file_path}")
        with open(file_path, "r") as f:
            ckpt = [json.loads(line) for line in f if line.strip()]
        if ckpt:
            print(f"Last processed item: {ckpt[-1]['id']}")
        print("*" * 50)
        return ckpt
    return []


def save_checkpoint(file_handle, data):
    file_handle.write(json.dumps(data, ensure_ascii=False) + "\n")
    file_handle.flush()


def get_default_qa_path(dataset_name, split):
    return (
        REPO_ROOT
        / "results"
        / "KGQA"
        / dataset_name
        / "RoG"
        / split
        / f"results_gen_rule_path_RoG-{dataset_name}_RoG_{split}_predictions_3_False_jsonl"
        / "predictions.jsonl"
    )


def triple_to_path_step(step):
    h, r, t = step["triple"]
    direction = int(step.get("direction", 0))
    if direction == 0:
        return f"{h} -> {r} -> {t}"
    return f"{t} <- {r} <- {h}"


def path_to_str(path):
    steps = path.get("path_triples", [])
    if not steps:
        return ""
    return " ; ".join(triple_to_path_step(step) for step in steps)


def build_path_prompt(selector_item, max_paths=None):
    paths = selector_item.get("paths", [])
    if max_paths is not None:
        paths = paths[:max_paths]
    rendered_paths = [path_to_str(path) for path in paths]
    rendered_paths = [path for path in rendered_paths if path]
    return "Paths:\n" + "\n".join(rendered_paths)


def parse_max_paths(prompt_mode):
    if prompt_mode in {"path", "path_all"}:
        return None
    if prompt_mode.startswith("path_"):
        value = prompt_mode.split("_", 1)[1]
        if value == "all":
            return None
        return int(value)
    raise ValueError(f"Invalid prompt_mode for selector paths: {prompt_mode}")



PATH_ICL_USER_PROMPT = """Paths:
Lou Seal -> sports.mascot.team -> San Francisco Giants
San Francisco Giants -> sports.sports_team.championships -> 2012 World Series
San Francisco Giants -> sports.sports_championship_event.champion -> 2014 World Series
San Francisco Giants -> time.participant.event -> 2010 World Series
San Francisco Giants -> sports.sports_team.championships -> 2014 World Series
San Francisco Giants -> sports.sports_team.championships -> 2010 World Series
San Francisco Giants -> time.participant.event -> 2012 World Series
Crazy Crab -> sports.mascot.team -> San Francisco Giants

Question:
What year did the team with mascot named Lou Seal win the World Series?"""

PATH_ICL_ASSISTANT_PROMPT = """To find the year the team with mascot named Lou Seal won the World Series, first identify the team connected to Lou Seal. Lou Seal is the mascot of the San Francisco Giants. The paths show that the San Francisco Giants won the 2010 World Series, 2012 World Series, and 2014 World Series.

ans: 2014
ans: 2012
ans: 2010"""


def add_icl_to_prompt(user_query):
    return "\n\n".join([
        "Here is an example of the expected answer format.",
        PATH_ICL_USER_PROMPT,
        PATH_ICL_ASSISTANT_PROMPT,
        "Now answer the following question using the same format.",
        user_query,
    ])


def remove_icl_from_llm_mode(llm_mode):
    mode = llm_mode.replace("icl", "").replace("__", "_").strip("_")
    if not mode:
        return "sys_dc"
    if "sys" not in mode:
        mode = f"sys_{mode}"
    return mode


def build_reasoning_data(selector_results, qa_records, prompt_mode, use_icl=False):
    qa_by_id = {item["id"]: item for item in qa_records}
    max_paths = parse_max_paths(prompt_mode)
    data = []
    for sample_id, selector_item in selector_results.items():
        qa_item = qa_by_id.get(sample_id, {})
        question = selector_item.get("question") or qa_item.get("question")
        if not question:
            raise ValueError(f"Question not found for sample {sample_id}")
        ground_truth = qa_item.get("ground_truth") or selector_item.get("a_entity", [])
        question_prompt = "Question:\n" + question
        if question_prompt[-1] != "?":
            question_prompt += "?"
        path_prompt = build_path_prompt(selector_item, max_paths=max_paths)
        user_query = "\n\n".join([path_prompt, question_prompt])
        if use_icl:
            user_query = add_icl_to_prompt(user_query)
        all_query = "\n\n".join([PATH_SYS_PROMPT, user_query])
        item = {
            "id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "q_entity": selector_item.get("q_entity", qa_item.get("q_entity", [])),
            "a_entity": selector_item.get("a_entity", ground_truth),
            "paths": selector_item.get("paths", []),
            "sys_query": PATH_SYS_PROMPT,
            "user_query": user_query,
            "all_query": all_query,
            "cot_query": PATH_COT_PROMPT,
        }
        data.append(item)
    return data


def eval_all(pred_file_path, run, subset, split=None, eval_hops=-1, previous_pred_file_path=None):
    from evaluate_results import eval_results

    print("=" * 50)
    print(f"Evaluating on subset: {subset}")
    metrics = eval_results(
        str(pred_file_path),
        subset=subset,
        split=split,
        eval_hops=eval_hops,
        previous_pred_file_path=previous_pred_file_path,
    )
    if subset:
        postfix = "_sub"
    else:
        postfix = ""
    if run:
        run.log({f"results{postfix}/{key}": value for key, value in metrics.items()})
    print(
        f"hit@1={metrics['hit@1']}, hit={metrics['hit']}, macro_f1={metrics['macro_f1']}, "
        f"macro_precision={metrics['macro_precision']}, macro_recall={metrics['macro_recall']}, "
        f"micro_f1={metrics['micro_f1']}, micro_precision={metrics['micro_precision']}, "
        f"micro_recall={metrics['micro_recall']}, total_cnt={metrics['total_cnt']}, "
        f"no_ans_cnt={metrics['no_ans_cnt']}, no_ans_ratio={metrics['no_ans_ratio']}"
    )
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Reason over selector_inference.py path results")
    parser.add_argument("-d", "--dataset_name", default="cwq", choices=["cwq", "webqsp"])
    parser.add_argument("-p", "--selector_result_path", required=True, help="Path to selector_inference.py retrieval_result.pth")
    parser.add_argument("--qa_path", default=None, help="JSONL containing ids and ground_truth; defaults to the RoG prediction path")
    parser.add_argument("--prompt_mode", default="path_all", help="path_all/path or path_<N> to keep the first N decoded paths")
    parser.add_argument("--llm_mode", default="sys_dc", help="Combine tokens such as sys, icl, and dc; select remains unsupported here")
    parser.add_argument("-m", "--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--backend", default="openai", choices=["openai", "vllm"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--frequency_penalty", type=float, default=0.16)
    parser.add_argument("--pred_path_postfix", default="")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--eval_split", default=None, help="Optional split string passed to answer extraction during eval")
    parser.add_argument("--eval_hops", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of concurrent OpenAI requests for LLM inference")

    args = parser.parse_args()

    if "select" in args.llm_mode:
        raise ValueError("Path reasoning main.py does not support select mode.")

    import torch
    from tqdm import tqdm
    from llm_utils import llm_init, llm_inf_all

    selector_results = torch.load(args.selector_result_path, map_location="cpu", weights_only=False)
    qa_path = Path(args.qa_path) if args.qa_path else get_default_qa_path(args.dataset_name, args.split)
    qa_records = load_jsonl(qa_path)
    use_icl = "icl" in args.llm_mode
    data = build_reasoning_data(selector_results, qa_records, args.prompt_mode, use_icl=use_icl)
    inference_llm_mode = remove_icl_from_llm_mode(args.llm_mode)

    output_root = Path(args.pred_path_postfix) / args.model_name.split("/")[-1]
    # if args.pred_path_postfix:
    #     output_root = output_root / args.pred_path_postfix
    output_root.mkdir(parents=True, exist_ok=True)
    raw_pred_file_path = output_root / f"{args.prompt_mode}-{args.llm_mode}-{args.frequency_penalty}-{args.split}-predictions-resume.jsonl"

    project_name = f"GraphSieve-{args.dataset_name}"
    run_name = f"{output_root.name.replace('/', '_')}-{args.model_name}-{args.prompt_mode}-{args.llm_mode}-{args.frequency_penalty}-{args.split}"
    run = swanlab.init(project=project_name, name=run_name, config=vars(args))

    llm = llm_init(
        args.model_name,
        args.tensor_parallel_size,
        args.max_seq_len_to_capture,
        args.max_tokens,
        args.seed,
        args.temperature,
        args.frequency_penalty,
        backend=args.backend,
    )

    start_idx = len(load_checkpoint(raw_pred_file_path))
    pending_data = data[start_idx:]
    if args.backend != "openai" and args.num_workers > 1:
        print("--num_workers is only applied to the openai backend; falling back to serial vllm inference.")
    with open(raw_pred_file_path, "a") as pred_file:
        if args.backend == "openai" and args.num_workers > 1:
            results = llm_inf_all(
                llm,
                pending_data,
                inference_llm_mode,
                args.model_name,
                backend=args.backend,
                num_workers=args.num_workers,
            )
            for each_qa, res in tqdm(
                zip(pending_data, results),
                initial=start_idx,
                total=len(data),
                desc="reason",
            ):
                output = dict(each_qa)
                output.pop("paths", None)
                output["prediction"] = res[0]
                save_checkpoint(pred_file, output)
        else:
            for each_qa in tqdm(pending_data, initial=start_idx, total=len(data), desc="reason"):
                res = llm_inf_all(llm, each_qa, inference_llm_mode, args.model_name, backend=args.backend)
                output = dict(each_qa)
                output.pop("paths", None)
                output["prediction"] = res[0]
                save_checkpoint(pred_file, output)

    final_pred_file_path = raw_pred_file_path.with_name(raw_pred_file_path.stem.replace("-resume", "") + raw_pred_file_path.suffix)
    os.replace(raw_pred_file_path, final_pred_file_path)
    if not args.skip_eval:
        eval_all(final_pred_file_path, run, subset=True, split=args.eval_split, eval_hops=args.eval_hops, previous_pred_file_path=qa_path)
        eval_all(final_pred_file_path, run, subset=False, split=args.eval_split, eval_hops=args.eval_hops, previous_pred_file_path=qa_path)


if __name__ == "__main__":
    main()
