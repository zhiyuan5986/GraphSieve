import os
import re
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from .prepare_prompts import unique_preserve_order

DATA_PATH = "Your Path"

def get_subgraphs(dataset_name, split):
    if DATA_PATH is None:
        input_file = os.path.join("rmanluo", f"RoG-{dataset_name}")
        dataset = load_dataset(input_file, split=split)
    else:
        local_dir = os.path.join(DATA_PATH, f"RoG-{dataset_name}", "data")
        dataset = load_dataset("parquet", data_dir=local_dir)[split]
    return dataset


def extract_reasoning_paths(text):
    pattern = r"Reasoning Paths:(.*?)\n\nQuestion:"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        reasoning_paths = match.group(1).strip()
        return reasoning_paths
    else:
        return None


def add_good_triplets_from_rog(data):
    print("Adding good triplets from ROG...")
    total_good_triplets = 0
    total_good_triplets_in_graph = 0
    total_good_triplets_not_in_graph = 0
    for idx, each_qa in enumerate(tqdm(data)):
        all_paths = extract_reasoning_paths(each_qa["input"]).split("\n")
        data[idx]["good_paths_rog"] = all_paths
        all_good_triplets = []
        for each_path in all_paths:
            each_path = each_path.split(" -> ")
            good_triplets = []
            i = 0
            while i < len(each_path):
                if i + 2 < len(each_path):
                    triplet = (each_path[i], each_path[i + 1], each_path[i + 2])
                    temp_triplet = (each_path[i + 2], each_path[i + 1], each_path[i])
                    total_good_triplets += 1
                    # if triplet in each_qa["graph"] or temp_triplet in each_qa["graph"]:
                    #     total_good_triplets_in_graph += 1
                    # else:
                    #     total_good_triplets_not_in_graph += 1
                    good_triplets.append(triplet)
                i += 2
            all_good_triplets.extend(good_triplets)
        data[idx]["good_triplets_rog"] = unique_preserve_order(all_good_triplets)
    return data


def _normalize_triplet(triplet):
    """Return the (head, relation, tail) text fields from selector/RAG triples."""
    if isinstance(triplet, dict):
        triplet = triplet.get("triple") or triplet.get("triplet")
    if triplet is None or len(triplet) < 3:
        raise ValueError(f"Invalid triplet: {triplet}")
    return tuple(triplet[:3])


def _triplets_from_selector_record(record, seed=None):
    """Recover de-duplicated triples, optionally shuffled with a fixed seed."""
    triples = []
    seen = set()
    for path in record.get("paths", []):
        for path_triple in path.get("path_triples", []):
            triplet = _normalize_triplet(path_triple)
            if triplet not in seen:
                triples.append(triplet)
                seen.add(triplet)
    if seed is not None:
        random.Random(seed).shuffle(triples)
    return triples


def load_selector_path_results(selector_path, seed=None):
    """Load selector path output from .jsonl/.json/.pth and map question id to triples."""
    pred_dict = torch.load(selector_path, map_location="cpu", weights_only=False)

    selector_triples = {}
    for qid, record in pred_dict.items():
        if qid is None:
            qid = record.get("id")
        if qid is None:
            raise ValueError(f"Selector record without id in {selector_path}")
        selector_triples[str(qid)] = _triplets_from_selector_record(record, seed=seed)
    return selector_triples


def add_selector_path_triplets(data, selector_result_path, seed=None):
    print(f"Adding selector path triplets from {selector_result_path}...")
    selector_triples = load_selector_path_results(selector_result_path, seed=seed)
    new_data = []
    missing = 0
    empty = 0
    for each_qa in tqdm(data):
        triples = selector_triples.get(str(each_qa["id"]))
        if triples is None:
            missing += 1
            continue
        if not triples:
            empty += 1
        each_qa["scored_triplets"] = triples
        new_data.append(each_qa)
    print(f"Selector path triplets not found for {missing} questions; empty for {empty} questions")
    return new_data


def add_gt_if_not_present(triple_score_dict):
    st = [','.join(list(each)[:3]) for each in triple_score_dict['scored_triples']]
    tt = [','.join(list(each)[:3]) for each in triple_score_dict['target_relevant_triples']]
    for each in tt:
        if each in st:
            continue
        else:
            # put at the beginning
            triple_score_dict["scored_triples"].insert(0, tuple(each.split(',')))
    return triple_score_dict["scored_triples"]


def add_scored_triplets(data, score_dict_path, prompt_mode):
    print("Adding scored triplets...")
    new_data = []
    cnt = 0
    triple_score_dict = torch.load(score_dict_path, weights_only=False)

    running_baselines = False
    if 'triples' in triple_score_dict[next(iter(triple_score_dict))]:
        running_baselines = True
        for k, v in tqdm(triple_score_dict.items()):
            triple_score_dict[k]['scored_triples'] = v['triples']

    for each_qa in tqdm(data):
        if each_qa["id"] in triple_score_dict:
            if 'gt' in prompt_mode:
                scored_triples = add_gt_if_not_present(triple_score_dict[each_qa["id"]])
            else:
                scored_triples = triple_score_dict[each_qa["id"]]["scored_triples"]
            each_qa['scored_triplets'] = scored_triples
            new_data.append(each_qa)
        else:
            print(f"Triplets not found for {each_qa['id']}")
            if running_baselines:
                each_qa['scored_triplets'] = [('', '', '')]
                new_data.append(each_qa)
            elif 'gt' not in prompt_mode:
                raise ValueError
            else:
                cnt += 1
    print(f"Triplets not found for {cnt} questions")
    return new_data


def sample_random_triplets(data, num_triplets, seed=0):
    print(f"Sampling {num_triplets} random triplets...")
    np.random.seed(seed)
    for idx, each_qa in enumerate(tqdm(data)):
        all_triplets = np.array(each_qa["graph"])
        sampled_triplets = np.random.permutation(all_triplets)[:num_triplets]
        data[idx][f"sampled_triplets_{num_triplets}"] = sampled_triplets.tolist()
    return data


def get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode, seed=None, triplets_to_sample=[50, 100, 200, 300], selector_result_path=None):
    with open(pred_file_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    print("Loading subgraphs...")
    subgraphs = get_subgraphs(dataset_name, split)

    print("Adding subgraphs to data...")
    data = []
    for i, each_qa in enumerate(tqdm(raw_data)):
        assert each_qa["id"] == subgraphs[i]["id"]
        each_qa["graph"] = [tuple(each) for each in subgraphs[i]["graph"]]
        each_qa['a_entity'] = subgraphs[i]['a_entity']
        data.append(each_qa)
    # data = raw_data

    data = add_good_triplets_from_rog(data)
    if selector_result_path is not None:
        data = add_selector_path_triplets(data, selector_result_path, seed=seed)
    else:
        data = add_scored_triplets(data, score_dict_path, prompt_mode)
    # for num_triplets in triplets_to_sample:
    #     data = sample_random_triplets(data, num_triplets, seed)

    return data
