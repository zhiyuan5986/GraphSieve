"""Train PathConditionedCSMSelector with offline OptiSet-style path-set preferences."""

import argparse
import importlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "retrieve"))

from src.dataset.retriever import RetrieverDataset  # noqa: E402
from src.setup import set_seed  # noqa: E402
from selector_dataset import PathSFTDataset, collate_path_sft, get_adjacent_candidate_triples, triple_key  # noqa: E402
from selector import PathConditionedCSMSelector  # noqa: E402
from selector_inference import build_config, build_model  # noqa: E402
from train_selector_path_sft import compute_loss, state_forward  # noqa: E402
from path_decode import evaluate_beam_search  # noqa: E402
from path_sft_metrics import mean_metrics  # noqa: E402
from selector_optiset_utils import checkpoint_tag, load_jsonl, safe_tag  # noqa: E402


def resolve_swanlab():
    if importlib.util.find_spec("swanlab") is None:
        class SwanLabStub:
            def init(self, *args, **kwargs):
                print("# swanlab unavailable; logging disabled")

            def log(self, *args, **kwargs):
                pass

            def finish(self):
                pass

        return SwanLabStub()
    return importlib.import_module("swanlab")


def output_dir_name(args):
    ckpt = checkpoint_tag(args.selector_checkpoint) if args.selector_checkpoint else "scratch"
    scored = safe_tag(Path(args.scored_path).stem, max_len=120)
    return (
        f"selector_path_optiset_{safe_tag(args.dataset)}_{safe_tag(args.split)}_init-{ckpt}_"
        f"scored-{scored}_llist{args.lambda_list}_lsft{args.lambda_sft}_"
        f"tauG{args.reward_temperature}_tauP{args.policy_temperature}_lr{args.lr}_seed{args.seed}"
    )


class LazySampleIndex:
    """Look up dataset samples by ID without retaining their embeddings.

    Iterating over a lazy ``RetrieverDataset`` calls ``__getitem__`` and loads the
    embedding shard for every sample.  Keeping those returned samples in a dict
    therefore turns lazy loading back into eager loading.  Index only the
    embedding-free metadata and resolve the full sample on demand instead.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = {
            sample["id"]: idx
            for idx, sample in enumerate(dataset.processed_dict_list)
        }

    def get(self, sample_id, default=None):
        idx = self.indices.get(sample_id)
        return self.dataset[idx] if idx is not None else default


def sample_index(dataset):
    return LazySampleIndex(dataset)


def multi_positive_softmax_loss(cand_logits, stop_logit, positive_indices, stop_positive):
    all_logits = torch.cat([cand_logits, stop_logit.view(1)])
    positives = list(positive_indices)
    if stop_positive:
        positives.append(cand_logits.numel())
    if not positives:
        return all_logits.new_zeros(())
    pos = torch.tensor(sorted(set(positives)), dtype=torch.long, device=all_logits.device)
    return -torch.logsumexp(F.log_softmax(all_logits, dim=0)[pos], dim=0)


def _advance_state(sample, current_entity, path_triples, action, step, allow_entity_repeat=False, allow_triple_repeat=False, visited_entities=None):
    visited_entities = visited_entities or {int(current_entity)}
    candidates = get_adjacent_candidate_triples(current_entity, sample)
    if not allow_triple_repeat:
        used = {tid for tid, _ in path_triples}
        candidates = [c for c in candidates if c["triple_id"] not in used]
    if not allow_entity_repeat:
        candidates = [c for c in candidates if c["next_entity"] not in visited_entities]
    chosen_idx, chosen = None, None
    stop = action == "STOP"
    if not stop:
        wanted = triple_key(step["triple_id"], step["direction"])
        for idx, cand in enumerate(candidates):
            if triple_key(cand["triple_id"], cand["direction"]) == wanted:
                chosen_idx, chosen = idx, cand
                break
    return candidates, chosen_idx, chosen, stop


def replay_path_score(model, sample, path, device, args):
    current = int(path["topic_entity"])
    visited = {current}
    prefix = []
    logps = []
    triples = path.get("path_triples", [])
    actions = path.get("actions") or ["TRIPLE"] * len(triples)
    triple_pos = 0
    for action in actions:
        step = triples[triple_pos] if action != "STOP" and triple_pos < len(triples) else None
        candidates, chosen_idx, chosen, stop = _advance_state(sample, current, prefix, action, step, args.allow_entity_repeat, args.allow_triple_repeat, visited)
        state = {"path_triples": list(prefix), "candidate_triples": candidates}
        item = {"sample": sample, "state": state}
        cand_logits, stop_logit = state_forward(model, item, device)
        log_probs = F.log_softmax(torch.cat([cand_logits, stop_logit.view(1)]), dim=0)
        if stop:
            logps.append(log_probs[-1])
            break
        if chosen_idx is None:
            continue
        logps.append(log_probs[chosen_idx])
        prefix.append((int(chosen["triple_id"]), int(chosen["direction"])))
        current = int(chosen["next_entity"])
        visited.add(current)
        triple_pos += 1
    if not logps:
        return None
    return torch.stack(logps).mean()


def set_score(model, sample, set_obj, device, args):
    scores = [replay_path_score(model, sample, path, device, args) for path in set_obj.get("paths", [])]
    scores = [s for s in scores if s is not None]
    if not scores:
        return None
    return torch.stack(scores).mean()


def best_set_mpce_loss(model, sample, set_obj, device, args):
    grouped = {}
    for path in set_obj.get("paths", []):
        current = int(path["topic_entity"])
        visited = {current}
        prefix = []
        triples = path.get("path_triples", [])
        actions = path.get("actions") or ["TRIPLE"] * len(triples)
        triple_pos = 0
        for action in actions:
            step = triples[triple_pos] if action != "STOP" and triple_pos < len(triples) else None
            candidates, chosen_idx, chosen, stop = _advance_state(sample, current, prefix, action, step, args.allow_entity_repeat, args.allow_triple_repeat, visited)
            key = (current, tuple(prefix))
            entry = grouped.setdefault(key, {"path_triples": list(prefix), "candidate_triples": candidates, "positive": set(), "stop": False})
            if stop:
                entry["stop"] = True
                break
            if chosen_idx is None:
                continue
            entry["positive"].add(triple_key(chosen["triple_id"], chosen["direction"]))
            prefix.append((int(chosen["triple_id"]), int(chosen["direction"])))
            current = int(chosen["next_entity"])
            visited.add(current)
            triple_pos += 1
    losses = []
    for entry in grouped.values():
        cand_logits, stop_logit = state_forward(model, {"sample": sample, "state": entry}, device)
        pos_indices = [idx for idx, cand in enumerate(entry["candidate_triples"]) if triple_key(cand["triple_id"], cand["direction"]) in entry["positive"]]
        losses.append(multi_positive_softmax_loss(cand_logits, stop_logit, pos_indices, entry["stop"]))
    if not losses:
        return None
    return torch.stack(losses).mean()


def listwise_loss(set_scores, gains, reward_temperature=1.0, policy_temperature=1.0):
    scores = torch.stack(set_scores)
    gain_tensor = torch.tensor(gains, dtype=scores.dtype, device=scores.device)
    gain_std = gain_tensor.std(unbiased=False)
    gain_hat = (gain_tensor - gain_tensor.mean()) / (gain_std + 1e-6)
    q = F.softmax(gain_hat / reward_temperature, dim=0).detach()
    log_p = F.log_softmax(scores / policy_temperature, dim=0)
    return F.kl_div(log_p, q, reduction="sum"), q


def row_answer_count(row):
    if row.get("num_gold_answers") is not None:
        return int(row["num_gold_answers"])
    if row.get("answer_count") is not None:
        return int(row["answer_count"])
    answers = row.get("gold_answers")
    if answers is not None:
        return len({str(answer).strip() for answer in answers if str(answer).strip()})
    target = str(row.get("target_answer_string") or "").strip()
    return len([part.strip() for part in target.split("|") if part.strip()]) if target else 0


def load_scored(args):
    rows = load_jsonl(args.scored_path)
    if args.max_answers > 0:
        rows = [row for row in rows if row_answer_count(row) <= args.max_answers]
    if args.max_train_samples > 0:
        rows = rows[: args.max_train_samples]
    return rows


def optiset_step(args, model, sample, row, device):
    valid_sets, scores, gains = [], [], []
    for set_obj in row.get("sets", []):
        if "llm_gain" not in set_obj:
            continue
        score = set_score(model, sample, set_obj, device, args)
        if score is None:
            continue
        valid_sets.append(set_obj)
        scores.append(score)
        gains.append(float(set_obj["llm_gain"]))
    if not valid_sets:
        return None, {}
    l_list, q = listwise_loss(scores, gains, args.reward_temperature, args.policy_temperature)
    best_idx = max(range(len(gains)), key=lambda i: gains[i])
    l_best = best_set_mpce_loss(model, sample, valid_sets[best_idx], device, args)
    if l_best is None:
        l_best = l_list.new_zeros(())
    det_sets = [i for i, s in enumerate(valid_sets) if s.get("source") == "deterministic_beam"]
    det_gain = gains[det_sets[0]] if det_sets else None
    stats = {
        "optiset_list_loss": float(l_list.detach().item()),
        "optiset_best_loss": float(l_best.detach().item()),
        "optiset_best_gain": float(gains[best_idx]),
        "optiset_gain_spread": float(max(gains) - min(gains)),
        "target_q_entropy": float(-(q * (q + 1e-12).log()).sum().detach().item()),
        "deterministic_is_best": float(valid_sets[best_idx].get("source") == "deterministic_beam"),
    }
    if det_gain is not None:
        stats["best_gain_minus_deterministic_gain"] = float(gains[best_idx] - det_gain)
    return l_best + args.lambda_list * l_list, stats


def train_epoch(args, model, rows, sample_by_id, sft_ds, device, optimizer, on_step=None):
    model.train()
    order = list(range(len(rows)))
    random.shuffle(order)
    sums = defaultdict(float)
    n = 0
    for idx in tqdm(order, desc="optiset_train"):
        row = rows[idx]
        sample = sample_by_id.get(row["id"])
        if sample is None:
            continue
        loss, stats = optiset_step(args, model, sample, row, device)
        if loss is None:
            continue
        if args.lambda_sft > 0 and sft_ds is not None and len(sft_ds) > 0:
            sft_item = sft_ds[random.randrange(len(sft_ds))]
            sft_loss, _ = compute_loss(args, *state_forward(model, sft_item, device), sft_item["state"]["labels"], sft_item["state"]["stop_label"])
            loss = loss + args.lambda_sft * sft_loss
            stats["sft_loss"] = float(sft_loss.detach().item())
        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            stats["grad_norm"] = float(grad_norm.item())
        optimizer.step()
        stats["loss"] = float(loss.detach().item())
        for k, v in stats.items():
            sums[k] += float(v)
        n += 1
        if on_step is not None:
            on_step(n, {k: v / max(n, 1) for k, v in sums.items()})
        if args.max_train_steps and n >= args.max_train_steps:
            break
    return {k: v / max(n, 1) for k, v in sums.items()}


def build_fresh_model(emb_size, args, config, device):
    model = PathConditionedCSMSelector(
        emb_size=emb_size,
        topic_pe=config["retriever"].get("topic_pe", True),
        DDE_kwargs=config["retriever"].get("DDE_kwargs", {"num_rounds": 2, "num_reverse_rounds": 2}),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model.to(device)
    return model


def validate(model, val_base, device, args):
    if val_base is None:
        return {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        beam_rows, _ = evaluate_beam_search(
            model,
            val_base,
            device,
            max_hop=args.max_hop,
            beam_size=args.val_beam_size,
            local_topk=args.val_local_topk,
            length_penalty=args.length_penalty,
            allow_entity_repeat=args.allow_entity_repeat,
            allow_triple_repeat=args.allow_triple_repeat,
            action_score_mode=args.val_action_score_mode,
            force_stop=args.force_stop,
        )
    if was_training:
        model.train()
    return mean_metrics(beam_rows)


def checkpoint_payload(model, optimizer, args, epoch, emb_size, global_step, best_metric=None, val_metrics=None):
    payload = {
        "selector_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "global_step": global_step,
        "emb_size": emb_size,
    }
    if best_metric is not None:
        payload["best_metric"] = best_metric
    if val_metrics is not None:
        payload["val_metrics"] = val_metrics
    return payload


def main(args):
    checkpoint = None
    if args.selector_checkpoint:
        checkpoint = torch.load(args.selector_checkpoint, map_location="cpu", weights_only=False)
        if args.dataset is None:
            args.dataset = checkpoint.get("args", {}).get("dataset")
        if args.dataset is None:
            raise ValueError("Dataset must be provided or recorded in checkpoint args.")
        args.topic_pe = checkpoint.get("args", {}).get("topic_pe", True)
        args.num_rounds = checkpoint.get("args", {}).get("num_rounds", 2)
        args.num_reverse_rounds = checkpoint.get("args", {}).get("num_reverse_rounds", 2)
        args.hidden_size = checkpoint.get("args", {}).get("hidden_size", 256)
        args.num_layers = checkpoint.get("args", {}).get("num_layers", 3)
        args.num_heads = checkpoint.get("args", {}).get("num_heads", 8)
        args.dropout = checkpoint.get("args", {}).get("dropout", 0.0)
    elif args.dataset is None:
        raise ValueError("Dataset must be provided when training the selector from scratch without --selector_checkpoint.")
    set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    config = build_config(args, checkpoint or {"args": vars(args)})
    config["dataset"]["emb_mode"] = args.emb_mode
    config["dataset"]["emb_cache_size"] = args.emb_cache_size
    train_base = RetrieverDataset(config=config, split=args.split, skip_no_path=True, emb_mode=args.emb_mode, emb_dir=args.emb_dir)
    val_base = RetrieverDataset(config=config, split=args.val_split, skip_no_path=True, emb_mode=args.emb_mode, emb_dir=args.emb_dir) if args.val_split else None
    sample_by_id = sample_index(train_base)
    rows = load_scored(args)
    emb_size = checkpoint.get("emb_size", train_base[0]["q_emb"].shape[-1]) if checkpoint is not None else train_base[0]["q_emb"].shape[-1]
    model = build_model(emb_size, checkpoint, config, device) if checkpoint is not None else build_fresh_model(emb_size, args, config, device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sft_ds = PathSFTDataset(train_base, max_hop=args.max_hop) if args.lambda_sft > 0 else None
    if sft_ds is not None:
        DataLoader(sft_ds, batch_size=1, shuffle=True, collate_fn=collate_path_sft)
    out = Path(args.output_dir or f"selector_path_optiset_{args.dataset}_{time.strftime('%b%d-%H%M%S', time.gmtime())}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    swanlab = resolve_swanlab()
    swanlab.init(
        project=args.swanlab_project or f"selector-path-optiset-{args.dataset}",
        name=args.swanlab_name or out.name,
        config=vars(args),
        mode=args.swanlab_mode,
    )
    best = -1.0
    patience = 0
    global_step = 0
    for epoch in range(args.num_epochs):
        if args.patience and patience > args.patience:
            break

        def on_train_step(epoch_step, running_train_metrics):
            nonlocal best, patience, global_step
            global_step += 1
            if args.val_n_steps <= 0 or global_step % args.val_n_steps != 0:
                return
            val_metrics = validate(model, val_base, device, args)
            log = {f"train/{k}": v for k, v in running_train_metrics.items()}
            log.update({f"val/{k}": v for k, v in val_metrics.items()})
            log.update({"epoch": epoch, "epoch_step": epoch_step, "global_step": global_step})
            torch.save(
                checkpoint_payload(model, optimizer, args, epoch, emb_size, global_step, val_metrics=val_metrics),
                out / f"val_epoch_{epoch}_step_{epoch_step}_global_{global_step}.pth",
            )
            torch.save(checkpoint_payload(model, optimizer, args, epoch, emb_size, global_step, val_metrics=val_metrics), out / "last_val.pth")
            score = val_metrics.get(args.best_metric, running_train_metrics.get("optiset_best_gain", 0.0))
            if score >= best:
                best = score
                torch.save(
                    checkpoint_payload(model, optimizer, args, epoch, emb_size, global_step, best_metric=best, val_metrics=val_metrics),
                    out / "cpt.pth",
                )
                patience = 0
            log["best"] = best
            log["patience"] = patience
            print(json.dumps(log, sort_keys=True))
            swanlab.log(log)

        train_metrics = train_epoch(args, model, rows, sample_by_id, sft_ds, device, optimizer, on_step=on_train_step)
        val_metrics = validate(model, val_base, device, args)
        log = {f"train/{k}": v for k, v in train_metrics.items()}
        log.update({f"val/{k}": v for k, v in val_metrics.items()})
        log["epoch"] = epoch
        log["global_step"] = global_step
        torch.save(checkpoint_payload(model, optimizer, args, epoch, emb_size, global_step, val_metrics=val_metrics), out / "last.pth")
        torch.save(checkpoint_payload(model, optimizer, args, epoch, emb_size, global_step, val_metrics=val_metrics), out / f"epoch_{epoch}.pth")
        score = val_metrics.get(args.best_metric, train_metrics.get("optiset_best_gain", 0.0))
        if score >= best:
            best = score
            torch.save(checkpoint_payload(model, optimizer, args, epoch, emb_size, global_step, best_metric=best, val_metrics=val_metrics), out / "cpt.pth")
            patience = 0
        else:
            patience += 1
        log['patience'] = patience
        print(json.dumps(log, sort_keys=True))
        swanlab.log(log)
    swanlab.finish()


def parse_args():
    p = argparse.ArgumentParser(description="Train selector with OptiSet-style offline path-set preferences")
    p.add_argument("--scored_path", required=True)
    p.add_argument("--selector_checkpoint", default=None, help="Optional SFT selector checkpoint. Omit to initialize and train the selector from scratch.")
    p.add_argument("--dataset", default=None, choices=["webqsp", "cwq"])
    p.add_argument("--split", default="train")
    p.add_argument("--val_split", default="val")
    p.add_argument("--val_n_steps", type=int, default=0, help="Run validation every N optimizer steps within each epoch and save a checkpoint for each validation; use <=0 to disable step-level validation.")
    p.add_argument("--data_dir", default="retrieve")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--emb_mode", default="lazy", choices=["eager", "lazy", "sharded_lazy"])
    p.add_argument("--emb_dir", default=None, help="Directory to store embeddings. If None, embeddings will be computed on-the-fly.")
    p.add_argument("--topic_pe", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_rounds", type=int, default=2)
    p.add_argument("--num_reverse_rounds", type=int, default=2)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--text_encoder_name", default="gte-large-en-v1.5")
    p.add_argument("--emb_cache_size", type=int, default=0)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_train_steps", type=int, default=0)
    p.add_argument("--max_answers", type=int, default=50, help="Skip scored samples whose gold answer list is longer than this; use <=0 to disable.")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lambda_list", type=float, default=0.1)
    p.add_argument("--lambda_sft", type=float, default=0.1)
    p.add_argument("--reward_temperature", type=float, default=1.0)
    p.add_argument("--policy_temperature", type=float, default=1.0)
    p.add_argument("--max_hop", type=int, default=3)
    p.add_argument("--allow_entity_repeat", action="store_true")
    p.add_argument("--allow_triple_repeat", action="store_true")
    p.add_argument("--length_penalty", type=float, default=0.0)
    p.add_argument("--val_beam_size", type=int, default=8)
    p.add_argument("--val_local_topk", type=int, default=64)
    p.add_argument("--val_action_score_mode", default="softmax")
    p.add_argument("--force_stop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--best_metric", default="answer_entity_coverage")
    p.add_argument("--bce_loss_weight", type=float, default=1.0)
    p.add_argument("--mpce_loss_weight", type=float, default=0.3)
    p.add_argument("--stop_loss_weight", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_threads", type=int, default=4)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--swanlab_project", default=None)
    p.add_argument("--swanlab_name", default=None)
    p.add_argument("--swanlab_mode", default="disabled", choices=["cloud", "local", "disabled"])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
