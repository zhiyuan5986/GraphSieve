import os
import pickle
from collections import OrderedDict, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.model.retriever import Retriever
from src.setup import prepare_sample
from src.dataset.retriever import collate_retriever


class BeamPathScoredRetrieverDataset:
    """Retriever-compatible dataset with continuous shortest+beam path scores.

    This class intentionally leaves ``RetrieverDataset`` unchanged. It reuses the
    same sample fields/collate contract, but replaces hard shortest-path labels
    with soft contribution scores computed from all shortest paths plus retriever
    guided beam-search paths that reach answer entities.
    """

    def __init__(
        self,
        config,
        split,
        retriever_path,
        device="cuda:0",
        skip_no_path=True,
        emb_mode=None,
        emb_cache_size=None,
        emb_dir=None,
        beam_width=8,
        beam_depth=4,
        beam_topk_edges=16,
        max_paths_per_pair=16,
        shortest_weight=1.0,
        beam_weight=0.8,
        overlap_weight=1.2,
        length_beta=0.7,
        pair_bonus=0.2,
        position_bonus=0.1,
        cache_tag=None,
        refresh_cache=False,
        path_num_workers=None,
    ):
        if retriever_path is None:
            raise ValueError("BeamPathScoredRetrieverDataset requires --retriever_path")

        dataset_name = config["dataset"]["name"]
        self.config = config
        self.dataset_name = dataset_name
        self.split = split
        self.emb_mode = emb_mode or config["dataset"].get("emb_mode", "eager")
        if self.emb_mode != "eager":
            raise ValueError("BeamPathScoredRetrieverDataset currently requires eager embeddings for retriever scoring.")
        self.emb_cache_size = int(emb_cache_size if emb_cache_size is not None else config["dataset"].get("emb_cache_size", 0) or 0)
        self._emb_cache = OrderedDict()
        self.emb_dir = emb_dir or config["dataset"].get("emb_dir")
        self.beam_width = int(beam_width)
        self.beam_depth = int(beam_depth)
        self.beam_topk_edges = int(beam_topk_edges)
        self.max_paths_per_pair = int(max_paths_per_pair)
        self.shortest_weight = float(shortest_weight)
        self.beam_weight = float(beam_weight)
        self.overlap_weight = float(overlap_weight)
        self.length_beta = float(length_beta)
        self.pair_bonus = float(pair_bonus)
        self.position_bonus = float(position_bonus)
        self.refresh_cache = bool(refresh_cache)
        self.path_num_workers = int(
            path_num_workers
            if path_num_workers is not None
            else config["dataset"].get("beam_path_num_workers", config["dataset"].get("path_num_workers", 0))
            or 0
        )

        self.processed_source_list = self._load_processed(config, dataset_name, split)
        emb_dict = self._load_emb(config, dataset_name, split)
        self.emb_size = next(iter(emb_dict.values()))["q_emb"].shape[-1] if emb_dict else None

        cache_tag = cache_tag or (
            f"bw{self.beam_width}_bd{self.beam_depth}_ek{self.beam_topk_edges}_mp{self.max_paths_per_pair}"
        )
        cache_tag = f"directed_{cache_tag}"
        retriever = self._load_retriever(retriever_path, device)
        try:
            triple_score_dict = self._get_triple_scores(dataset_name, split, emb_dict, retriever, device, cache_tag)
        finally:
            # The retriever is only needed to construct labels. Release it before
            # selector training so it does not occupy GPU memory.
            del retriever
            if torch.cuda.is_available() and "cuda" in str(device):
                torch.cuda.empty_cache()

        self._assembly(self.processed_source_list, triple_score_dict, emb_dict, skip_no_path)

    def _load_processed(self, config, dataset_name, split):
        processed_file = os.path.join(
            config["dataset"].get("data_dir", ""),
            f"data_files/{dataset_name}/processed/{split}.pkl",
        )
        with open(processed_file, "rb") as f:
            return pickle.load(f)

    def _load_emb(self, config, dataset_name, split):
        text_encoder_name = config["dataset"]["text_encoder_name"]
        file_path = os.path.join(
            config["dataset"].get("data_dir", ""),
            f"data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth",
        )
        return torch.load(file_path, map_location="cpu")

    def _load_retriever(self, retriever_path, device):
        cpt = torch.load(retriever_path, map_location="cpu")
        retriever_config = cpt.get("config", self.config).get("retriever", self.config["retriever"])
        model = Retriever(self.emb_size, **retriever_config).to(device)
        model.load_state_dict(cpt["model_state_dict"])
        model.eval()
        return model

    def _get_triple_scores(self, dataset_name, split, emb_dict, retriever, device, cache_tag):
        save_dir = os.path.join("data_files", dataset_name, "triple_scores_beam_csm")
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f"{split}_{cache_tag}.pth")
        if os.path.exists(save_file) and not self.refresh_cache:
            return torch.load(save_file, map_location="cpu")

        if self.path_num_workers > 1:
            triple_score_dict = self._get_triple_scores_threaded(split, emb_dict, retriever, device)
        else:
            triple_score_dict = {}
            for sample in tqdm(self.processed_source_list, desc=f"score {split}"):
                sample_id = sample["id"]
                if sample_id not in emb_dict:
                    print(f"[BeamPathScoredRetrieverDataset] skip scoring sample without embeddings: {sample_id}")
                    continue
                sample_with_emb = sample.copy()
                sample_with_emb.update(emb_dict[sample_id])
                retriever_scores = self._score_with_retriever(sample_with_emb, retriever, device)
                triple_score_dict[sample_id] = self._score_paths_for_sample(sample, retriever_scores)

        torch.save(triple_score_dict, save_file)
        return triple_score_dict

    def _get_triple_scores_threaded(self, split, emb_dict, retriever, device):
        """Score retriever logits serially and extract beam paths concurrently.

        Retriever inference can use the configured GPU, so it remains in the main
        thread. The NetworkX shortest-path and beam-search label construction is
        CPU-heavy per sample, and is dispatched to a worker pool while retriever
        inference continues for later samples.
        """
        triple_score_dict = {}
        pending = {}
        max_pending = max(self.path_num_workers * 4, self.path_num_workers)

        def drain(done):
            for future in done:
                sample_id = pending.pop(future)
                triple_score_dict[sample_id] = future.result()

        with ThreadPoolExecutor(max_workers=self.path_num_workers) as executor:
            for sample in tqdm(self.processed_source_list, desc=f"score {split}"):
                sample_id = sample["id"]
                if sample_id not in emb_dict:
                    print(f"[BeamPathScoredRetrieverDataset] skip scoring sample without embeddings: {sample_id}")
                    continue
                sample_with_emb = sample.copy()
                sample_with_emb.update(emb_dict[sample_id])
                retriever_scores = self._score_with_retriever(sample_with_emb, retriever, device)
                future = executor.submit(self._score_paths_for_sample, sample, retriever_scores)
                pending[future] = sample_id
                if len(pending) >= max_pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    drain(done)

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                drain(done)

        return triple_score_dict

    def _score_paths_for_sample(self, sample, retriever_scores):
        triple_scores, max_path_length, positive_paths = self._extract_paths_and_score(sample, retriever_scores)
        return {
            "triple_scores": triple_scores.cpu(),
            "max_path_length": max_path_length,
            "positive_paths": positive_paths,
        }

    @torch.no_grad()
    def _score_with_retriever(self, sample, retriever, device):
        num_entities = len(sample["text_entity_list"]) + len(sample["non_text_entity_list"])
        topic_entity_mask = torch.zeros(num_entities)
        topic_entity_mask[sample["q_entity_id_list"]] = 1.0
        sample["topic_entity_one_hot"] = F.one_hot(topic_entity_mask.long(), num_classes=2).float()
        sample["target_triple_probs"] = torch.zeros(len(sample["h_id_list"]))
        batch = collate_retriever([sample])
        prepared = prepare_sample(device, batch)
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, num_non_text_entities, relation_embs, topic_entity_one_hot, _, _ = prepared
        if len(h_id_tensor) == 0:
            return torch.empty(0)
        logits = retriever(
            h_id_tensor,
            r_id_tensor,
            t_id_tensor,
            q_emb,
            entity_embs,
            num_non_text_entities,
            relation_embs,
            topic_entity_one_hot,
        )
        return torch.sigmoid(logits).reshape(-1).detach().cpu()

    def _extract_paths_and_score(self, sample, retriever_scores):
        nx_g = self._get_nx_g(sample["h_id_list"], sample["r_id_list"], sample["t_id_list"])
        paths_by_key = {}
        for q_entity_id in sample["q_entity_id_list"]:
            for a_entity_id in sample["a_entity_id_list"]:
                pair_key = (int(q_entity_id), int(a_entity_id))
                for path in self._shortest_path(nx_g, q_entity_id, a_entity_id):
                    triple_path = self._entity_path_to_triple_path(nx_g, path)
                    if triple_path:
                        key = tuple(triple_path)
                        rec = paths_by_key.setdefault(key, {"triples": triple_path, "sources": set(), "pairs": set(), "ret": 1.0})
                        rec["sources"].add("shortest")
                        rec["pairs"].add(pair_key)

                for start_id, target_id in ((q_entity_id, a_entity_id), (a_entity_id, q_entity_id)):
                    for triple_path, path_ret_score in self._beam_paths(sample, start_id, target_id, retriever_scores):
                        if triple_path:
                            key = tuple(triple_path)
                            rec = paths_by_key.setdefault(key, {"triples": triple_path, "sources": set(), "pairs": set(), "ret": path_ret_score})
                            rec["sources"].add("beam")
                            rec["pairs"].add(pair_key)
                            rec["ret"] = max(float(rec["ret"]), float(path_ret_score))

        path_records = list(paths_by_key.values())
        max_path_length = max((len(rec["triples"]) for rec in path_records), default=None)
        triple_scores = self._score_triples(path_records, len(sample["h_id_list"]))
        positive_paths = [rec["triples"] for rec in path_records]
        return triple_scores, max_path_length, positive_paths

    def _get_nx_g(self, h_id_list, r_id_list, t_id_list):
        nx_g = nx.DiGraph()
        for i, (h_i, r_i, t_i) in enumerate(zip(h_id_list, r_id_list, t_id_list)):
            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)
        return nx_g

    def _shortest_path(self, nx_g, q_entity_id, a_entity_id):
        try:
            forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
        except Exception:
            forward_paths = []
        try:
            backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
        except Exception:
            backward_paths = []
        full_paths = forward_paths + backward_paths
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths
        min_path_len = min(len(path) for path in full_paths)
        return [path for path in full_paths if len(path) == min_path_len]

    def _entity_path_to_triple_path(self, nx_g, entity_path):
        triple_path = []
        for h_id, t_id in zip(entity_path[:-1], entity_path[1:]):
            edge_data = nx_g.get_edge_data(h_id, t_id)
            if edge_data is None:
                return []
            triple_path.append(int(edge_data["triple_id"]))
        return triple_path

    def _build_adjacency(self, sample, retriever_scores):
        adjacency = defaultdict(list)
        for triple_id, (h_id, t_id) in enumerate(zip(sample["h_id_list"], sample["t_id_list"])):
            score = float(retriever_scores[triple_id].item()) if len(retriever_scores) > triple_id else 0.0
            adjacency[int(h_id)].append((int(t_id), int(triple_id), score))
        for entity_id in adjacency:
            adjacency[entity_id].sort(key=lambda item: item[2], reverse=True)
            if self.beam_topk_edges > 0:
                adjacency[entity_id] = adjacency[entity_id][: self.beam_topk_edges]
        return adjacency

    def _beam_paths(self, sample, q_entity_id, a_entity_id, retriever_scores):
        if self.beam_width <= 0 or self.beam_depth <= 0:
            return []
        adjacency = self._build_adjacency(sample, retriever_scores)
        answer_id = int(a_entity_id)
        start_id = int(q_entity_id)
        beams = [(start_id, [], {start_id}, 0.0)]
        completed = []
        for _ in range(self.beam_depth):
            candidates = []
            for current_entity, triple_path, visited_entities, log_score_sum in beams:
                for next_entity, triple_id, edge_score in adjacency.get(current_entity, []):
                    if next_entity in visited_entities or triple_id in triple_path:
                        continue
                    new_path = triple_path + [triple_id]
                    new_visited = set(visited_entities)
                    new_visited.add(next_entity)
                    new_log_sum = log_score_sum + float(np.log(max(edge_score, 1e-8)))
                    avg_log = new_log_sum / len(new_path)
                    state = (next_entity, new_path, new_visited, new_log_sum)
                    if next_entity == answer_id:
                        completed.append((new_path, float(np.exp(avg_log))))
                    candidates.append((avg_log, state))
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0], reverse=True)
            beams = [state for _, state in candidates[: self.beam_width]]

        completed.sort(key=lambda item: item[1], reverse=True)
        unique = []
        seen = set()
        for path, score in completed:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append((path, score))
            if len(unique) >= self.max_paths_per_pair:
                break
        return unique

    def _score_triples(self, path_records, num_triples):
        scores = torch.zeros(num_triples, dtype=torch.float)
        pair_coverage = defaultdict(set)
        raw_scores = defaultdict(float)
        for rec in path_records:
            triple_path = rec["triples"]
            path_len = max(len(triple_path), 1)
            sources = rec["sources"]
            if "shortest" in sources and "beam" in sources:
                source_weight = self.overlap_weight
            elif "shortest" in sources:
                source_weight = self.shortest_weight
            else:
                source_weight = self.beam_weight
            path_weight = source_weight * (path_len ** (-self.length_beta)) * float(rec.get("ret", 1.0))
            for pos, triple_id in enumerate(triple_path):
                position_weight = 1.0 + self.position_bonus * ((path_len - pos) / path_len)
                raw_scores[triple_id] += path_weight * position_weight
                pair_coverage[triple_id].update(rec.get("pairs", set()))

        for triple_id, raw_score in raw_scores.items():
            coverage_bonus = 1.0 + self.pair_bonus * np.log1p(len(pair_coverage[triple_id]))
            scores[triple_id] = float(raw_score * coverage_bonus)
        max_score = float(scores.max().item()) if scores.numel() > 0 else 0.0
        if max_score > 0:
            scores = scores / max_score
        return scores

    def _assembly(self, processed_dict_list, triple_score_dict, emb_dict, skip_no_path):
        self.processed_dict_list = []
        num_relevant_triples = []
        num_skipped = 0
        print(f"# samples: {len(processed_dict_list)}")
        print(f"# embeddings: {len(emb_dict)}")
        for sample in tqdm(processed_dict_list, desc=f"assemble {self.split}"):
            sample_i = sample.copy()
            sample_id = sample_i["id"]
            if sample_id not in emb_dict:
                print(f"[BeamPathScoredRetrieverDataset] skip assembling sample without embeddings: {sample_id}")
                num_skipped += 1
                continue
            if sample_id not in triple_score_dict:
                print(f"[BeamPathScoredRetrieverDataset] skip assembling sample without triple scores: {sample_id}")
                num_skipped += 1
                continue
            payload = triple_score_dict[sample_id]
            triple_scores = payload["triple_scores"]
            max_path_length = payload["max_path_length"]
            num_relevant_triples.append(int((triple_scores > 0).sum().item()))
            sample_i["target_triple_probs"] = triple_scores
            sample_i["max_path_length"] = max_path_length
            sample_i["positive_paths"] = payload.get("positive_paths", [])
            if skip_no_path and (max_path_length in [None, 0]):
                num_skipped += 1
                continue
            sample_i.update(emb_dict[sample_id])
            sample_i["a_entity"] = list(set(sample_i["a_entity"]))
            sample_i["a_entity_id_list"] = list(set(sample_i["a_entity_id_list"]))
            num_entities = len(sample_i["text_entity_list"]) + len(sample_i["non_text_entity_list"])
            topic_entity_mask = torch.zeros(num_entities)
            topic_entity_mask[sample_i["q_entity_id_list"]] = 1.0
            sample_i["topic_entity_one_hot"] = F.one_hot(topic_entity_mask.long(), num_classes=2).float()
            self.processed_dict_list.append(sample_i)

        if num_relevant_triples:
            print(f"# skipped samples: {num_skipped}")
            print(
                "# relevant triples | median: "
                f"{int(np.median(num_relevant_triples))} | mean: {int(np.mean(num_relevant_triples))} | "
                f"max: {int(np.max(num_relevant_triples))}"
            )

    def __len__(self):
        return len(self.processed_dict_list)

    def __getitem__(self, i):
        return self.processed_dict_list[i]
