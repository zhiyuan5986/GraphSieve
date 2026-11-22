"""Shared utilities for OptiSet-style path-set generation, scoring, and training."""

import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from path_decode import _score_state, beam_search_sample
from selector_dataset import get_adjacent_candidate_triples


PATH_SYS_PROMPT = (
    "Based on the paths retrieved from a knowledge graph, please answer the question. "
    'Please return the final answer directly.'
)


def safe_tag(value, max_len=80):
    text = str(value).strip().replace("/", "-")
    text = re.sub(r"[^A-Za-z0-9_.=+-]+", "-", text).strip("-")
    return text[:max_len] or "none"


def checkpoint_tag(path):
    p = Path(path)
    parent = safe_tag(p.parent.name, max_len=40)
    stem = safe_tag(p.stem, max_len=30)
    return f"{parent}-{stem}" if parent else stem


def answer_string(sample, separator=" | "):
    answers = sorted({str(x).strip() for x in sample.get("a_entity", []) if str(x).strip()})
    return separator.join(answers)


def entity_names(sample):
    return sample["text_entity_list"] + sample["non_text_entity_list"]


def triple_text(sample, triple_id, direction):
    entities = entity_names(sample)
    h = int(sample["h_id_list"][triple_id])
    r = int(sample["r_id_list"][triple_id])
    t = int(sample["t_id_list"][triple_id])
    rel = sample["relation_list"][r]
    if int(direction) == 0:
        return f"{entities[h]} -> {rel} -> {entities[t]}"
    return f"{entities[t]} <- {rel} <- {entities[h]}"


def render_path_set(sample, paths, max_paths=None, max_context_chars=6000):
    selected = paths if max_paths is None else paths[:max_paths]
    seen_triples = set()
    lines = []
    for path in selected:
        steps = []
        for step in path.get("path_triples", []):
            key = (int(step["triple_id"]), int(step["direction"]))
            if key in seen_triples:
                continue
            seen_triples.add(key)
            steps.append(triple_text(sample, key[0], key[1]))
        if steps:
            lines.append(f"Path {len(lines) + 1}: " + " ; ".join(steps))
    text = "\n".join(lines)
    if max_context_chars and max_context_chars > 0 and len(text) > max_context_chars:
        text = text[:max_context_chars]
    return text


def build_path_prompt(sample, paths, max_paths=None, max_context_chars=6000):
    path_block = render_path_set(sample, paths, max_paths=max_paths, max_context_chars=max_context_chars)
    return "\n\n".join([
        PATH_SYS_PROMPT,
        f"Question:\n{sample['question']}",
        f"Knowledge graph paths:\n{path_block}",
        "Answer:\n",
    ])


def path_set_key(paths):
    return tuple(sorted(tuple((int(s["triple_id"]), int(s["direction"])) for s in p.get("path_triples", [])) for p in paths))


def unique_triple_ids(paths):
    return sorted({int(s["triple_id"]) for p in paths for s in p.get("path_triples", [])})


def _gumbel_like(x, generator=None):
    uniform = torch.rand(x.shape, device=x.device, generator=generator).clamp_(1e-6, 1 - 1e-6)
    return -torch.log(-torch.log(uniform))


def _candidate_filter(candidates, beam, allow_entity_repeat=False, allow_triple_repeat=False):
    if not allow_triple_repeat:
        used = {int(step["triple_id"]) for step in beam.get("path_triples", [])}
        candidates = [c for c in candidates if int(c["triple_id"]) not in used]
    if not allow_entity_repeat:
        candidates = [c for c in candidates if int(c["next_entity"]) not in beam.get("visited_entities", set())]
    return candidates


@torch.no_grad()
def stochastic_beam_search_set(
    model,
    sample,
    device,
    max_hop=3,
    beam_size=8,
    local_topk=64,
    gumbel_temperature=1.0,
    allow_entity_repeat=False,
    allow_triple_repeat=False,
    generator=None,
):
    model.eval()
    beams = []
    for topic in sample.get("q_entity_id_list", []):
        topic = int(topic)
        beams.append({
            "topic_entity": topic,
            "current_entity": topic,
            "path_triples": [],
            "step_logprobs": [],
            "actions": [],
            "score": 0.0,
            "rank_score": 0.0,
            "stopped": False,
            "visited_entities": {topic},
        })
    if not beams:
        return []

    for _ in range(max_hop):
        expanded = []
        for beam in beams:
            if beam["stopped"]:
                expanded.append(beam)
                continue
            candidates = get_adjacent_candidate_triples(beam["current_entity"], sample)
            candidates = _candidate_filter(candidates, beam, allow_entity_repeat, allow_triple_repeat)
            state = {
                "path_triples": [(s["triple_id"], s["direction"]) for s in beam["path_triples"]],
                "candidate_triples": candidates,
            }
            cand_logits, stop_logit = _score_state(model, sample, state, device)
            all_logits = torch.cat([cand_logits, stop_logit.view(1)])
            clean_log_probs = F.log_softmax(all_logits, dim=0)
            actions = [("STOP", None, clean_log_probs[-1], clean_log_probs[-1])]
            if candidates:
                k = min(local_topk, len(candidates))
                noisy = clean_log_probs[:-1] + gumbel_temperature * _gumbel_like(clean_log_probs[:-1], generator=generator)
                vals, idxs = torch.topk(noisy, k=k)
                for noisy_val, idx in zip(vals, idxs.tolist()):
                    actions.append(("TRIPLE", candidates[idx], clean_log_probs[idx], noisy_val))
            for kind, cand, clean_delta, noisy_delta in actions:
                nb = dict(beam)
                nb["path_triples"] = list(beam["path_triples"])
                nb["step_logprobs"] = list(beam["step_logprobs"])
                nb["actions"] = list(beam["actions"])
                nb["visited_entities"] = set(beam["visited_entities"])
                nb["score"] = float(beam["score"] + float(clean_delta.detach().item()))
                nb["rank_score"] = float(beam["rank_score"] + float(noisy_delta.detach().item()))
                if kind == "STOP":
                    nb["stopped"] = True
                    nb["actions"].append("STOP")
                    nb["step_logprobs"].append(float(clean_delta.detach().item()))
                else:
                    nb["actions"].append("TRIPLE")
                    nb["path_triples"].append({"triple_id": int(cand["triple_id"]), "direction": int(cand["direction"])})
                    nb["step_logprobs"].append(float(clean_delta.detach().item()))
                    nb["current_entity"] = int(cand["next_entity"])
                    nb["visited_entities"].add(int(cand["next_entity"]))
                expanded.append(nb)
        beams = sorted(expanded, key=lambda b: b["rank_score"], reverse=True)[:beam_size]
        if all(b["stopped"] for b in beams):
            break
    return finalize_paths(sample, beams)


@torch.no_grad()
def deterministic_beam_search_set(model, sample, device, **kwargs):
    beams = beam_search_sample(model, sample, device, action_score_mode="softmax", **kwargs)
    converted = []
    for beam in beams:
        converted.append({
            "topic_entity": int(beam["topic_entity"]),
            "current_entity": int(beam["current_entity"]),
            "path_triples": [{"triple_id": int(tid), "direction": int(direction)} for tid, direction in beam.get("path_triples", [])],
            "step_logprobs": [],
            "actions": ["TRIPLE"] * len(beam.get("path_triples", [])) + (["STOP"] if beam.get("stopped") else []),
            "score": float(beam.get("score", 0.0)),
            "stopped": bool(beam.get("stopped", False)),
        })
    return finalize_paths(sample, converted)


def finalize_paths(sample, beams):
    answer_ids = set(map(int, sample.get("a_entity_id_list", [])))
    output = []
    for beam in beams:
        step_logprobs = [float(x) for x in beam.get("step_logprobs", [])]
        score = float(sum(step_logprobs) / max(len(step_logprobs), 1)) if step_logprobs else float(beam.get("score", 0.0))
        output.append({
            "topic_entity": int(beam["topic_entity"]),
            "final_entity": int(beam.get("current_entity", beam.get("final_entity", beam["topic_entity"]))),
            "actions": list(beam.get("actions", [])),
            "path_triples": list(beam.get("path_triples", [])),
            "step_logprobs": step_logprobs,
            "selector_path_score_at_sampling": score,
            "stopped": bool(beam.get("stopped", False)),
            "reaches_answer": int(int(beam.get("current_entity", beam.get("final_entity", -1))) in answer_ids),
            "path_length": len(beam.get("path_triples", [])),
        })
    return output


def make_path_set(set_id, source, paths):
    return {
        "set_id": int(set_id),
        "source": source,
        "paths": paths,
        "unique_triple_ids": unique_triple_ids(paths),
        "selector_set_score_at_sampling": float(sum(p.get("selector_path_score_at_sampling", 0.0) for p in paths) / max(len(paths), 1)),
        "answer_path_count": int(sum(int(p.get("reaches_answer", 0)) for p in paths)),
    }


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values):
    return float(sum(values) / max(len(values), 1))


def pearson(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))
