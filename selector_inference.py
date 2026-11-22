"""Beam-search inference for a trained path-conditioned selector checkpoint.

This script decodes paths on a dataset split with ``PathConditionedCSMSelector``
and stores each sample's beam-search paths as the retrieval result. Unlike the
standalone triple selector inference script, it does not apply a max-K/top-K
triple cutoff; the retrieved triple set is the union of triples appearing in the
beam-search paths.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "retrieve"))

from src.dataset.retriever import RetrieverDataset  # noqa: E402
from src.setup import set_seed  # noqa: E402
from selector import PathConditionedCSMSelector  # noqa: E402
from path_decode import beam_search_sample  # noqa: E402


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


def build_model(emb_size, checkpoint, config, device):
    saved_args = checkpoint.get("args", {})
    model = PathConditionedCSMSelector(
        emb_size=emb_size,
        topic_pe=config["retriever"].get("topic_pe", True),
        DDE_kwargs=config["retriever"].get("DDE_kwargs", {"num_rounds": 2, "num_reverse_rounds": 2}),
        hidden_size=saved_args.get("hidden_size", 256),
        num_layers=saved_args.get("num_layers", 3),
        num_heads=saved_args.get("num_heads", 8),
        dropout=saved_args.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint.get("selector_state_dict", checkpoint))
    model.to(device)
    model.eval()
    return model


def triple_name(sample, triple_id):
    entity_list = sample["text_entity_list"] + sample["non_text_entity_list"]
    relation_list = sample["relation_list"]
    h = sample["h_id_list"][triple_id]
    r = sample["r_id_list"][triple_id]
    t = sample["t_id_list"][triple_id]
    return (entity_list[int(h)], relation_list[int(r)], entity_list[int(t)])


def path_record(sample, beam):
    triples = []
    for triple_id, direction in beam["path_triples"]:
        h, r, t = triple_name(sample, int(triple_id))
        triples.append({
            "triple_id": int(triple_id),
            "direction": int(direction),
            "triple": (h, r, t),
        })
    return {
        "topic_entity": int(beam["topic_entity"]),
        "current_entity": int(beam["current_entity"]),
        "score": float(beam["score"]),
        "stopped": bool(beam["stopped"]),
        "path_triples": triples,
    }


def base_prediction(sample):
    entity_list = sample["text_entity_list"] + sample["non_text_entity_list"]
    target_ids = torch.nonzero(sample["target_triple_probs"] > 0, as_tuple=False).view(-1).tolist()
    return {
        "question": sample["question"],
        "q_entity": sample["q_entity"],
        "q_entity_in_graph": [entity_list[int(e)] for e in sample["q_entity_id_list"]],
        "a_entity": sample["a_entity"],
        "a_entity_in_graph": [entity_list[int(e)] for e in sample.get("a_entity_id_list", [])],
        "max_path_length": sample["max_path_length"],
        "target_relevant_triples": [triple_name(sample, int(tid)) for tid in target_ids],
    }


@torch.no_grad()
def main(args):
    # print(args)
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    checkpoint = torch.load(args.path, map_location="cpu", weights_only=False)
    config = build_config(args, checkpoint)
    torch.set_num_threads(config["env"].get("num_threads", 1))
    set_seed(config["env"].get("seed", 42))

    dataset = RetrieverDataset(config=config, split=args.split, skip_no_path=args.skip_no_path)
    emb_size = checkpoint.get("emb_size", dataset[0]["q_emb"].shape[-1])
    # checkpoint['args']['hidden_size'] = 256
    model = build_model(emb_size, checkpoint, config, device)

    saved_args = checkpoint.get("args", {})
    max_hop = args.max_hop if args.max_hop is not None else saved_args.get("max_hop", 3)
    beam_size = args.beam_size if args.beam_size is not None else saved_args.get("beam_size", 8)
    local_topk = args.local_topk if args.local_topk is not None else saved_args.get("local_topk", 8)

    length_penalty = args.length_penalty if args.length_penalty is not None else saved_args.get("length_penalty", 0.0)

    pred_dict = {}
    for sample in tqdm(dataset, desc="beam_search"):
        beams = beam_search_sample(
            model,
            sample,
            device,
            max_hop=max_hop,
            beam_size=beam_size,
            local_topk=local_topk,  # no top-K candidate pruning; keep beam-size pruning only
            length_penalty=length_penalty,
            allow_entity_repeat=args.allow_entity_repeat,
            allow_triple_repeat=args.allow_triple_repeat,
            action_score_mode=args.action_score_mode,
            force_stop=args.force_stop,
        )
        pred = base_prediction(sample)
        pred["paths"] = [path_record(sample, beam) for beam in beams]
        pred_dict[sample["id"]] = pred

    output_dir = Path(args.output_dir) if args.output_dir else Path(os.path.dirname(args.path))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pred_dict, output_dir / args.output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beam-search inference for a path-conditioned selector checkpoint")
    parser.add_argument("-p", "--path", required=True, help="Path to selector checkpoint, e.g. selector_path_sft/cpt.pth")
    parser.add_argument("-d", "--dataset", default=None, choices=["webqsp", "cwq"], help="Dataset name; defaults to checkpoint args")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data_dir", default="retrieve")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--max_hop", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--local_topk", type=int, default=None)
    parser.add_argument("--length_penalty", type=float, default=None)
    parser.add_argument("--allow_entity_repeat", action="store_true")
    parser.add_argument("--allow_triple_repeat", action="store_true")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_name", default="retrieval_result.pth")
    parser.add_argument("--skip_no_path", action="store_true", default=False)
    parser.add_argument("--action_score_mode", default="sigmoid")
    parser.add_argument("--force_stop", action=argparse.BooleanOptionalAction, default=True)

    main(parser.parse_args())
