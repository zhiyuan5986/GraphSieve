"""Path-level SFT data construction for a path-conditioned CSM selector."""

from collections import OrderedDict, defaultdict
import os
import statistics
from urllib.parse import quote

import networkx as nx
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


def triple_key(triple_id, direction):
    return (int(triple_id), int(direction))


def get_adjacent_candidate_triples(current_entity, sample, allow_triple_ids=None):
    """Return current-entity adjacent triples with direction and next entity.

    direction: 0 means traversing stored h->t, 1 means traversing incoming t->h.
    """
    triples = []
    allow = set(allow_triple_ids) if allow_triple_ids is not None else None
    for tid, (h, r, t) in enumerate(zip(sample["h_id_list"], sample["r_id_list"], sample["t_id_list"])):
        if allow is not None and tid not in allow:
            continue
        if int(h) == int(current_entity):
            triples.append({"triple_id": tid, "h": int(h), "r": int(r), "t": int(t), "direction": 0, "next_entity": int(t)})
        if int(t) == int(current_entity):
            triples.append({"triple_id": tid, "h": int(h), "r": int(r), "t": int(t), "direction": 1, "next_entity": int(h)})
    return triples


def _all_shortest_entity_paths(sample, max_hop=None):
    graph = nx.DiGraph()
    edge_to_tid = {}
    for tid, (h, r, t) in enumerate(zip(sample["h_id_list"], sample["r_id_list"], sample["t_id_list"])):
        h, t = int(h), int(t)
        graph.add_edge(h, t)
        edge_to_tid[(h, t)] = tid

    paths = []
    for q in sample.get("q_entity_id_list", []):
        for a in sample.get("a_entity_id_list", []):
            q, a = int(q), int(a)
            oriented_entity_paths = []
            try:
                oriented_entity_paths.extend(list(nx.all_shortest_paths(graph, q, a)))
            except Exception:
                pass
            try:
                # Existing RetrieverDataset also treats answer->topic shortest paths as supervision.
                # Reverse them so path-conditioned states always start from the topic entity.
                oriented_entity_paths.extend([list(reversed(p)) for p in nx.all_shortest_paths(graph, a, q)])
            except Exception:
                pass
            if not oriented_entity_paths:
                continue
            min_len = min(len(p) for p in oriented_entity_paths)
            for ent_path in oriented_entity_paths:
                if len(ent_path) != min_len:
                    continue
                if max_hop is not None and len(ent_path) - 1 > max_hop:
                    continue
                triple_path = []
                ok = True
                for u, v in zip(ent_path[:-1], ent_path[1:]):
                    if (u, v) in edge_to_tid:
                        triple_path.append({"triple_id": edge_to_tid[(u, v)], "direction": 0, "next_entity": int(v)})
                    elif (v, u) in edge_to_tid:
                        triple_path.append({"triple_id": edge_to_tid[(v, u)], "direction": 1, "next_entity": int(v)})
                    else:
                        ok = False
                        break
                if ok:
                    paths.append({"topic_entity": q, "answer_entity": a, "entity_path": [int(x) for x in ent_path], "triple_path": triple_path})
    # de-duplicate
    seen, uniq = set(), []
    for p in paths:
        key = (p["topic_entity"], tuple((x["triple_id"], x["direction"], x["next_entity"]) for x in p["triple_path"]))
        if key not in seen:
            seen.add(key); uniq.append(p)
    return uniq


def build_gold_path_trie(sample, max_hop=None):
    trie = defaultdict(lambda: {"next": set(), "terminal": False, "current_entity": None, "topic_entity": None, "prefix": None})
    for path in _all_shortest_entity_paths(sample, max_hop=max_hop):
        prefix = []
        topic = path["topic_entity"]
        key = (topic, tuple(prefix))
        trie[key]["current_entity"] = topic
        trie[key]["topic_entity"] = topic
        trie[key]["prefix"] = tuple(prefix)
        for step in path["triple_path"]:
            key = (topic, tuple(prefix))
            trie[key]["next"].add(triple_key(step["triple_id"], step["direction"]))
            prefix.append(triple_key(step["triple_id"], step["direction"]))
            key = (topic, tuple(prefix))
            trie[key]["current_entity"] = step["next_entity"]
            trie[key]["topic_entity"] = topic
            trie[key]["prefix"] = tuple(prefix)
        trie[(topic, tuple(prefix))]["terminal"] = True
    return trie


def build_path_sft_states(sample, max_hop=None):
    trie = build_gold_path_trie(sample, max_hop=max_hop)
    answer_ids = set(map(int, sample.get("a_entity_id_list", [])))
    states = []
    for _, node in trie.items():
        prefix = node.get("prefix") or tuple()
        current = node["current_entity"]
        if current is None:
            continue
        candidates = get_adjacent_candidate_triples(current, sample)
        gold_next = set(node["next"])
        labels = [1.0 if triple_key(c["triple_id"], c["direction"]) in gold_next else 0.0 for c in candidates]
        stop_label = 1.0 if node["terminal"] or int(current) in answer_ids else 0.0
        if candidates or stop_label > 0 or gold_next:
            states.append({
                "question_id": sample.get("id"),
                "question_text": sample.get("question", sample.get("question_text", "")),
                "topic_entity": node["topic_entity"],
                "current_entity": current,
                "path_triples": list(prefix),
                "candidate_triples": candidates,
                "labels": labels,
                "stop_label": stop_label,
            })
    return states


class PathSFTDataset(Dataset):
    """Flattens RetrieverDataset questions into gold-prefix path SFT states.

    Embeddings are normally read through ``retriever_dataset`` exactly as before.
    Set ``lazy_emb=True`` only when the provided retriever dataset was built
    without lazy embeddings but selector training should load per-sample
    embedding shards on demand.  When the base retriever dataset already uses
    ``emb_mode="lazy"``/``"sharded_lazy"``, this class simply delegates to
    ``base[sample_idx]`` and inherits the retriever-side cache.
    """
    def __init__(
        self,
        retriever_dataset,
        max_hop=3,
        lazy_emb=False,
        emb_dir=None,
        emb_cache_size=0,
    ):
        self.base = retriever_dataset
        self.max_hop = max_hop
        self.lazy_emb = bool(lazy_emb)
        self.emb_dir = emb_dir
        self.emb_cache_size = int(emb_cache_size or 0)
        self._emb_cache = OrderedDict()
        if self.lazy_emb:
            self.emb_dir = self._resolve_emb_dir(emb_dir)
        self.items = []
        per_sample_counts = []
        for sample_idx, sample in tqdm(enumerate(self.base.processed_dict_list)):
            states = build_path_sft_states(sample, max_hop=max_hop)
            per_sample_counts.append(len(states))
            for state in states:
                self.items.append((sample_idx, state))
        self.summary = self._summary(per_sample_counts)

    def _summary(self, per_sample_counts):
        cand_counts = [len(s["candidate_triples"]) for _, s in self.items]
        action_counts = [len(s["candidate_triples"]) + 1 for _, s in self.items]
        pos_actions = [sum(s["labels"]) + s["stop_label"] for _, s in self.items]
        return {
            "num_questions": len(self.base),
            "num_states": len(self.items),
            "avg_states_per_question": float(statistics.mean(per_sample_counts)) if per_sample_counts else 0.0,
            "avg_candidates": float(statistics.mean(cand_counts)) if cand_counts else 0.0,
            "positive_action_ratio": float(sum(pos_actions) / max(sum(action_counts), 1)),
            "stop_action_ratio": float(sum(s["stop_label"] for _, s in self.items) / max(len(self.items), 1)),
        }

    def _resolve_emb_dir(self, emb_dir):
        if emb_dir is not None:
            return emb_dir
        if getattr(self.base, 'emb_dir', None) is not None:
            return self.base.emb_dir

        config = getattr(self.base, 'config', None)
        dataset_name = getattr(self.base, 'dataset_name', None)
        split = getattr(self.base, 'split', None)
        if config is None or dataset_name is None or split is None:
            raise ValueError(
                'lazy_emb=True requires emb_dir, or a retriever dataset with '
                'config/dataset_name/split metadata.'
            )

        text_encoder_name = config['dataset']['text_encoder_name']
        base_dir = os.path.join(
            config['dataset'].get('data_dir', ''),
            f'data_files/{dataset_name}/emb/{text_encoder_name}',
        )
        candidates = [
            os.path.join(base_dir, split),
            os.path.join(base_dir, f'{split}_sharded'),
            os.path.join(base_dir, 'sharded', split),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError(
            'Lazy selector embedding mode requires sharded per-sample embedding files. '
            f'Checked: {candidates}'
        )

    def _get_emb_file_path(self, sample_id):
        sample_id_str = str(sample_id)
        candidates = [
            os.path.join(self.emb_dir, f'{sample_id_str}.pth'),
            os.path.join(self.emb_dir, f'{sample_id_str}.pt'),
            os.path.join(self.emb_dir, quote(sample_id_str, safe='') + '.pth'),
            os.path.join(self.emb_dir, quote(sample_id_str, safe='') + '.pt'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"Could not find lazy selector embedding file for sample id '{sample_id}' in {self.emb_dir}."
        )

    def _load_emb_for_sample(self, sample_id):
        if self.emb_cache_size > 0 and sample_id in self._emb_cache:
            payload = self._emb_cache.pop(sample_id)
            self._emb_cache[sample_id] = payload
            return payload

        payload = torch.load(self._get_emb_file_path(sample_id), map_location='cpu')
        if self.emb_cache_size > 0:
            self._emb_cache[sample_id] = payload
            while len(self._emb_cache) > self.emb_cache_size:
                self._emb_cache.popitem(last=False)
        return payload

    def _sample_without_eager_emb(self, sample_idx):
        sample = self.base.processed_dict_list[sample_idx].copy()
        sample_id = sample.pop('_emb_id', sample.get('id'))
        for key in ('q_emb', 'entity_embs', 'relation_embs'):
            sample.pop(key, None)
        sample.update(self._load_emb_for_sample(sample_id))
        return sample

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample_idx, state = self.items[idx]
        if self.lazy_emb and getattr(self.base, 'emb_mode', 'eager') == 'eager':
            sample = self._sample_without_eager_emb(sample_idx)
        else:
            sample = self.base[sample_idx]
        return {"sample": sample, "state": state}


def collate_path_sft(batch):
    return batch[0]
