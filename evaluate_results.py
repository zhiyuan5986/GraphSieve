"""Answer-level KGQA evaluation utilities for the root path reasoning entrypoint.

This module intentionally reuses the answer extraction and metric primitives from
``reason/metrics`` so root ``main.py`` can evaluate path-based predictions without
constructing triple-centric artifacts.  It reports only the answer-level metrics
needed by this project: macro/micro precision, recall, F1, Hit@1, and Hit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from reason.metrics.evaluate_results import eval_hit as eval_hit_original
from reason.metrics.evaluate_results_corrected import (
    eval_f1,
    eval_hal_score,
    eval_hit as eval_hit_at_1,
    eval_precision,
    eval_recall,
    get_pred,
    remove_duplicates,
)

METRIC_KEYS = (
    "hit@1",
    "hit",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "micro_f1",
    "micro_precision",
    "micro_recall",
    "total_cnt",
    "no_ans_cnt",
    "no_ans_ratio",
    "hal_score",
    "hal_stats",
)



def _load_previous_qa_by_id(previous_pred_file_path: Optional[str]) -> Optional[Dict[Any, Dict[str, Any]]]:
    """Load canonical QA records from a previous RoG prediction JSONL file.

    Root path predictions can be generated from selector outputs, so their embedded
    QA fields may not exactly match the original RoG evaluation set.  When a RoG
    prediction file is supplied, use its question/ground-truth records as the
    source of truth and only evaluate ids present in that file.
    """
    if previous_pred_file_path is None:
        return None

    qa_by_id: Dict[Any, Dict[str, Any]] = {}
    with open(previous_pred_file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except Exception:
                print(line)
                continue
            qa_by_id[data["id"]] = data
    return qa_by_id

def _infer_dataset_name(predict_file: str) -> str:

    for sp in predict_file.split('/')[::-1]:
        lowered = sp.lower()
    # lowered = predict_file.split('/')[-1].lower()
        if "webqsp" in lowered:
            return "webqsp"
        if "cwq" in lowered:
            return "cwq"
    raise NotImplementedError("Cannot infer dataset name from prediction path; expected 'webqsp' or 'cwq'.")


def _samples_to_eval_path(dataset_name: str) -> Path:
    if dataset_name == "webqsp":
        return Path("./scored_triples/webqsp_240912_unidir_test.pth")
    if dataset_name == "cwq":
        return Path("./scored_triples/cwq_240907_unidir_test.pth")
    raise NotImplementedError(f"Unsupported dataset: {dataset_name}")


def _extract_paths_block(record: Dict[str, Any]) -> str:
    prompt = record.get("user_query") or record.get("all_query") or ""
    marker = "Paths:\n"
    if marker not in prompt:
        return ""
    paths_block = prompt.split(marker, 1)[1]
    if "\n\nQuestion:" in paths_block:
        paths_block = paths_block.split("\n\nQuestion:", 1)[0]
    return paths_block.strip()


def _entities_from_path_step(step: str) -> list[str]:
    if " -> " in step:
        parts = step.split(" -> ")
    elif " <- " in step:
        parts = step.split(" <- ")
    else:
        return []
    if len(parts) < 3:
        return []
    return [parts[0].strip(), parts[-1].strip()]


def _retrieved_entities_for_hal(record: Dict[str, Any]) -> list[str]:
    entities: list[str] = []
    seen = set()
    for path_line in _extract_paths_block(record).splitlines():
        for step in path_line.split(" ; "):
            for entity in _entities_from_path_step(step.strip()):
                if entity and entity not in seen:
                    entities.append(entity)
                    seen.add(entity)
    return entities


def _passes_sample_filters(sample_meta: Dict[str, Any], subset: bool, bad_samples: bool, eval_hops: int) -> bool:
    if eval_hops > 0:
        max_path_length = sample_meta["max_path_length"]
        if eval_hops == 3:
            if max_path_length is None or max_path_length < 3:
                return False
        elif max_path_length != eval_hops:
            return False

    if subset and not sample_meta["a_entity_in_graph"]:
        return False
    if bad_samples and sample_meta["a_entity_in_graph"]:
        return False
    return True


def _normalize_answers_for_question(answers: Iterable[str], question: str) -> list[str]:
    answer = sorted(remove_duplicates(list(answers)), key=len, reverse=True)
    if "when" in question.lower() or "what year" in question.lower():
        for idx in range(len(answer)):
            if "-" in answer[idx] and answer[idx].split("-")[0].isdigit():
                answer[idx] = answer[idx].split("-")[0]
    return answer


def _should_double_check(question: str) -> bool:
    # Keep this condition textually aligned with reason/metrics/evaluate_results.py
    # and reason/metrics/evaluate_results_corrected.py. Those modules define it
    # inline at the call site rather than exposing a reusable symbol.
    return any([keyword in question.lower() for keyword in [
        "when",
        "what year",
        "which year",
        "where",
        "sport",
        "what countr",
        "language",
        "nba finals",
        "world series",
    ]])


def _has_no_answer(raw_prediction: str, prediction: list[str], split: Optional[str]) -> bool:
    if split == "\n":
        return len(prediction) == 0
    lowered = raw_prediction.lower() if raw_prediction else ""
    return (
        len(prediction) == 0
        or "ans:" not in raw_prediction
        or "ans: not available" in lowered
        or "ans: no information available" in lowered
    )


def _safe_micro_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def eval_results(
    predict_file: str,
    *,
    split: Optional[str] = None,
    subset: bool = False,
    bad_samples: bool = False,
    eval_hops: int = -1,
    detailed_eval_name: Optional[str] = None,
    previous_pred_file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate root-path predictions using ``reason/metrics`` semantics.

    ``hit@1`` is computed with ``reason.metrics.evaluate_results_corrected.eval_hit``;
    ``hit`` is computed with ``reason.metrics.evaluate_results.eval_hit``.  Answer
    extraction uses ``reason.metrics.evaluate_results_corrected.get_pred``.
    """
    assert not (subset and bad_samples)

    dataset_name = _infer_dataset_name(predict_file)
    samples_to_eval_path = _samples_to_eval_path(dataset_name)
    samples_to_eval = torch.load(samples_to_eval_path, weights_only=False)
    qa_by_id = _load_previous_qa_by_id(previous_pred_file_path)

    if detailed_eval_name is None:
        if subset:
            detailed_eval_name = f"sub_answer_metrics_hop{eval_hops}.jsonl"
        elif bad_samples:
            detailed_eval_name = f"bad_answer_metrics_hop{eval_hops}.jsonl"
        else:
            detailed_eval_name = f"full_answer_metrics_hop{eval_hops}.jsonl"
    detailed_eval_file = predict_file.replace("predictions.jsonl", detailed_eval_name)

    hit_at_1_list = []
    hit_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    total_pred = 0
    total_answer = 0
    total_match = 0
    total_cnt = 0
    no_ans_cnt = 0
    hal_score_list = []
    hal_stats = {
        "g_no_ans": 0,
        "g_c": 0,
        "g_w": 0,
        "b_no_ans": 0,
        "b_in_graph": 0,
        "b_out_graph_c": 0,
        "b_out_graph_w": 0,
        "total_ans": 0,
        "total_g_samples": 0,
        "total_b_samples": 0,
        "total_samples": 0,
        "total_g_ans": 0,
        "total_b_ans": 0,
        "g_c_out_graph": 0,
        "g_w_out_graph": 0,
        "g_c_in_graph": 0,
        "g_w_in_graph": 0,
    }

    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except Exception:
                print(line)
                continue

            sample_id = data["id"]
            if qa_by_id is not None and sample_id not in qa_by_id:
                continue
            if not _passes_sample_filters(samples_to_eval[sample_id], subset, bad_samples, eval_hops):
                continue

            qa_data = qa_by_id[sample_id] if qa_by_id is not None else data
            raw_prediction = data["prediction"]
            question = qa_data["question"]
            answer = _normalize_answers_for_question(qa_data["ground_truth"], question)
            double_check = _should_double_check(question)

            prediction = get_pred(raw_prediction, split)
            total_cnt += 1
            no_ans_flag = _has_no_answer(raw_prediction, prediction, split)
            if no_ans_flag:
                no_ans_cnt += 1

            precision_score, matched_1, num_pred = eval_precision(prediction, answer, double_check)
            recall_score, matched_2, num_answer = eval_recall(prediction, answer, double_check)
            assert matched_1 == matched_2
            f1_score = eval_f1(precision_score, recall_score)
            hit_at_1 = eval_hit_at_1(prediction, answer, double_check)
            hit = eval_hit_original(raw_prediction, answer, double_check)
            subgraph_ent = _retrieved_entities_for_hal(data)
            hal_score, hal_stats = eval_hal_score(
                prediction,
                answer,
                double_check,
                samples_to_eval[sample_id]["a_entity_in_graph"],
                no_ans_flag,
                subgraph_ent,
                hal_stats,
            )

            total_pred += num_pred
            total_answer += num_answer
            total_match += matched_1
            precision_list.append(precision_score)
            recall_list.append(recall_score)
            f1_list.append(f1_score)
            hit_at_1_list.append(hit_at_1)
            hit_list.append(hit)
            hal_score_list.append(hal_score)

            f2.write(json.dumps({
                "id": sample_id,
                "prediction": prediction,
                "ground_truth": answer,
                "hit@1": hit_at_1,
                "hit": hit,
                "f1": f1_score,
                "precision": precision_score,
                "recall": recall_score,
                "hal_score": hal_score,
            }, ensure_ascii=False) + "\n")

    if not hit_at_1_list:
        return {key: 0 for key in METRIC_KEYS}

    micro_precision = total_match / total_pred if total_pred else 0.0
    micro_recall = total_match / total_answer if total_answer else 0.0
    avg_hal_score = sum(hal_score_list) / len(hal_score_list)
    avg_hal_score = (avg_hal_score + 1.5) / (1 + 1.5) * 100
    metrics = {
        "hit@1": sum(hit_at_1_list) * 100 / len(hit_at_1_list),
        "hit": sum(hit_list) * 100 / len(hit_list),
        "macro_f1": sum(f1_list) * 100 / len(f1_list),
        "macro_precision": sum(precision_list) * 100 / len(precision_list),
        "macro_recall": sum(recall_list) * 100 / len(recall_list),
        "micro_f1": _safe_micro_f1(micro_precision, micro_recall),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "total_cnt": total_cnt,
        "no_ans_cnt": no_ans_cnt,
        "no_ans_ratio": no_ans_cnt / total_cnt if total_cnt else 0.0,
        "hal_score": avg_hal_score,
        "hal_stats": hal_stats,
    }

    result_str = (
        f"Hit@1: {metrics['hit@1']}, Hit: {metrics['hit']}, "
        f"Macro F1: {metrics['macro_f1']}, Macro Precision: {metrics['macro_precision']}, "
        f"Macro Recall: {metrics['macro_recall']}, Hal Score: {metrics['hal_score']}"
    )
    print(result_str)
    print(
        f"Micro F1: {metrics['micro_f1']}, Micro precision: {metrics['micro_precision']}, "
        f"Micro Recall: {metrics['micro_recall']}"
    )
    print(f"Total number of samples: {total_cnt}, no answer samples: {no_ans_cnt}, ratio: {metrics['no_ans_ratio']}")

    eval_result_path = predict_file.replace("predictions.jsonl", "answer_eval_result.txt")
    with open(eval_result_path, "w") as f:
        f.write(result_str)
    return metrics
