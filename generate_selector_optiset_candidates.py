"""Generate OptiSet candidate path sets with deterministic and stochastic selector beam search."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "retrieve"))

from src.dataset.retriever import RetrieverDataset  # noqa: E402
from src.setup import set_seed  # noqa: E402
from selector_inference import build_config, build_model  # noqa: E402
from selector_optiset_utils import (  # noqa: E402
    answer_string,
    checkpoint_tag,
    deterministic_beam_search_set,
    make_path_set,
    path_set_key,
    safe_tag,
    stochastic_beam_search_set,
    write_jsonl,
)


def output_name(args):
    ckpt = checkpoint_tag(args.selector_checkpoint)
    return (
        f"optiset_candidates_{safe_tag(args.dataset)}_{safe_tag(args.split)}_"
        f"ckpt-{ckpt}_hop{args.max_hop}_beam{args.beam_size}_sets{args.num_stochastic_sets}_"
        f"topk{args.local_topk}_gumbel{args.gumbel_temperature}_det{int(args.include_deterministic_beam)}_"
        f"entrep{int(args.allow_entity_repeat)}_trirep{int(args.allow_triple_repeat)}_seed{args.seed}.jsonl"
    )


def make_sets(args, model, sample, device, generator):
    sets, seen = [], set()
    if args.include_deterministic_beam:
        paths = deterministic_beam_search_set(
            model,
            sample,
            device,
            max_hop=args.max_hop,
            beam_size=args.beam_size,
            local_topk=args.local_topk,
            length_penalty=args.length_penalty,
            allow_entity_repeat=args.allow_entity_repeat,
            allow_triple_repeat=args.allow_triple_repeat,
        )
        key = path_set_key(paths)
        if key:
            seen.add(key)
            sets.append(make_path_set(len(sets), "deterministic_beam", paths))

    attempts = 0
    max_attempts = max(args.num_stochastic_sets * args.max_sampling_attempt_multiplier, args.num_stochastic_sets)
    while len([s for s in sets if s["source"] == "stochastic_beam"]) < args.num_stochastic_sets and attempts < max_attempts:
        attempts += 1
        paths = stochastic_beam_search_set(
            model,
            sample,
            device,
            max_hop=args.max_hop,
            beam_size=args.beam_size,
            local_topk=args.local_topk,
            gumbel_temperature=args.gumbel_temperature,
            allow_entity_repeat=args.allow_entity_repeat,
            allow_triple_repeat=args.allow_triple_repeat,
            generator=generator,
        )
        key = path_set_key(paths)
        if not key or key in seen:
            continue
        seen.add(key)
        sets.append(make_path_set(len(sets), "stochastic_beam", paths))
    return sets


def main(args):
    checkpoint = torch.load(args.selector_checkpoint, map_location="cpu", weights_only=False)
    if args.dataset is None:
        args.dataset = checkpoint.get("args", {}).get("dataset")
    if args.dataset is None:
        raise ValueError("Dataset must be provided or recorded in checkpoint args.")
    set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    config = build_config(args, checkpoint)
    config["dataset"]["emb_mode"] = args.emb_mode
    config["dataset"]["emb_cache_size"] = args.emb_cache_size
    config["dataset"]["emb_dir"] = args.emb_dir
    dataset = RetrieverDataset(config=config, split=args.split, skip_no_path=args.skip_no_path, emb_mode=args.emb_mode, emb_dir=args.emb_dir,)
    emb_size = checkpoint.get("emb_size", dataset[0]["q_emb"].shape[-1])
    model = build_model(emb_size, checkpoint, config, device)
    model.eval()

    rows = []
    limit = len(dataset) if args.num_samples <= 0 else min(args.num_samples, len(dataset))
    dataset_lock = Lock()
    generation_config = {
        "max_hop": args.max_hop,
        "beam_size": args.beam_size,
        "num_stochastic_sets": args.num_stochastic_sets,
        "local_topk": args.local_topk,
        "length_penalty": args.length_penalty,
        "gumbel_temperature": args.gumbel_temperature,
        "include_deterministic_beam": args.include_deterministic_beam,
        "allow_entity_repeat": args.allow_entity_repeat,
        "allow_triple_repeat": args.allow_triple_repeat,
        "seed": args.seed,
        "sample_workers": args.sample_workers,
    }
    def build_row(i):
        # RetrieverDataset can maintain an embedding cache in lazy modes; guard
        # sample materialization while allowing the expensive path-set search to
        # run concurrently after the sample is loaded.
        with dataset_lock:
            sample = dataset[i]
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + i)
        with torch.inference_mode():
            sets = make_sets(args, model, sample, device, generator)
        gold_answers = sorted({str(x).strip() for x in sample.get("a_entity", []) if str(x).strip()})
        return {
            "schema_version": "selector_optiset_candidates.v1",
            "id": sample["id"],
            "question": sample["question"],
            "gold_answers": gold_answers,
            "num_gold_answers": len(gold_answers),
            "answer_count": len(gold_answers),
            "target_answer_string": answer_string(sample, separator=args.answer_separator),
            "dataset": args.dataset,
            "split": args.split,
            "selector_checkpoint": args.selector_checkpoint,
            "generation_config": generation_config,
            "sets": sets,
        }

    worker_count = max(1, args.sample_workers)
    if worker_count == 1:
        for i in tqdm(range(limit), desc="generate_optiset_candidates"):
            rows.append(build_row(i))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            rows.extend(tqdm(
                executor.map(build_row, range(limit)),
                total=limit,
                desc=f"generate_optiset_candidates[{worker_count} workers]",
            ))

    output = Path(args.output) if args.output else Path(args.output_dir) / output_name(args)
    write_jsonl(output, rows)
    print(json.dumps({"num_rows": len(rows), "num_sets": sum(len(r["sets"]) for r in rows), "output": str(output)}, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Generate OptiSet candidate path sets")
    p.add_argument("--selector_checkpoint", required=True)
    p.add_argument("--dataset", default=None, choices=["webqsp", "cwq"])
    p.add_argument("--split", default="train")
    p.add_argument("--data_dir", default="retrieve")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--emb_mode", default="lazy", choices=["eager", "lazy", "sharded_lazy"])
    p.add_argument("--emb_cache_size", type=int, default=0)
    p.add_argument("--emb_dir", default=None)
    p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--max_hop", type=int, default=3)
    p.add_argument("--beam_size", type=int, default=8)
    p.add_argument("--num_stochastic_sets", type=int, default=8)
    p.add_argument("--local_topk", type=int, default=64)
    p.add_argument("--length_penalty", type=float, default=0.0)
    p.add_argument("--gumbel_temperature", type=float, default=1.0)
    p.add_argument("--include_deterministic_beam", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow_entity_repeat", action="store_true")
    p.add_argument("--allow_triple_repeat", action="store_true")
    p.add_argument("--max_sampling_attempt_multiplier", type=int, default=4)
    p.add_argument("--answer_separator", default="\n")
    p.add_argument("--skip_no_path", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_threads", type=int, default=4, help="Number of intra-op PyTorch CPU threads.")
    p.add_argument("--sample_workers", type=int, default=1, help="Number of samples to generate path sets for concurrently. Use 1 for serial generation.")
    p.add_argument("--output_dir", default="outputs/optiset/candidates")
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
