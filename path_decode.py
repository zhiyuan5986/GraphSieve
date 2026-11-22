"""Beam search decoding for PathConditionedCSMSelector."""
import torch
import torch.nn.functional as F

from selector_dataset import get_adjacent_candidate_triples


def _prepared_tensors(sample, device):
    return (
        torch.tensor(sample['h_id_list'], dtype=torch.long, device=device),
        torch.tensor(sample['r_id_list'], dtype=torch.long, device=device),
        torch.tensor(sample['t_id_list'], dtype=torch.long, device=device),
        sample['q_emb'].to(device), sample['entity_embs'].to(device),
        len(sample['non_text_entity_list']), sample['relation_embs'].to(device), sample['topic_entity_one_hot'].to(device),
    )


def _score_state(model, sample, state, device):
    h, r, t, q, ent, n_non, rel, topic_pe = _prepared_tensors(sample, device)
    cand_ids = [c['triple_id'] for c in state['candidate_triples']]
    cand_dirs = [c['direction'] for c in state['candidate_triples']]
    path_ids = [x[0] for x in state['path_triples']]
    path_dirs = [x[1] for x in state['path_triples']]
    return model(q, path_ids, path_dirs, cand_ids, cand_dirs, h, r, t, ent, n_non, rel, topic_pe)


@torch.no_grad()
def beam_search_sample(model, sample, device, max_hop=3, beam_size=8, local_topk=64, length_penalty=0.0,
                       allow_entity_repeat=False, allow_triple_repeat=False, action_score_mode='sigmoid',
                       force_stop=True):
    model.eval()
    beams = []
    for topic in sample.get('q_entity_id_list', []):
        topic = int(topic)
        beams.append({'topic_entity': topic, 'current_entity': topic, 'path_triples': [], 'score': 0.0,
                      'stopped': False, 'visited_entities': {topic}})
    if not beams:
        return []
    stop_actions = 0
    for _ in range(max_hop):
        expanded = []
        for beam in beams:
            if beam['stopped']:
                expanded.append(beam); continue
            candidates = get_adjacent_candidate_triples(beam['current_entity'], sample)
            if not allow_triple_repeat:
                used = {tid for tid, _ in beam['path_triples']}
                candidates = [c for c in candidates if c['triple_id'] not in used]
            if not allow_entity_repeat:
                candidates = [c for c in candidates if c['next_entity'] not in beam['visited_entities']]
            state = dict(beam, candidate_triples=candidates)
            cand_logits, stop_logit = _score_state(model, sample, state, device)
            if action_score_mode == 'softmax':
                all_logits = torch.cat([cand_logits, stop_logit.view(1)])
                log_probs = F.log_softmax(all_logits, dim=0)
                if force_stop:
                    actions = [('STOP', None, float(log_probs[-1].item()))]
                else:
                    actions = []
                if len(candidates) > 0:
                    k = min(local_topk, len(candidates))
                    vals, idxs = torch.topk(log_probs[:-1], k=k)
                    for val, idx in zip(vals.tolist(), idxs.tolist()):
                        actions.append(('TRIPLE', candidates[idx], float(val)))
            elif action_score_mode == 'sigmoid':
                if force_stop:
                    actions = [('STOP', None, float(F.logsigmoid(stop_logit).item()))]
                else:
                    actions = []
                if len(candidates) > 0:
                    k = min(local_topk, len(candidates))
                    vals, idxs = torch.topk(cand_logits, k=k)
                    for val, idx in zip(vals, idxs.tolist()):
                        actions.append(('TRIPLE', candidates[idx], float(F.logsigmoid(val).item())))
            else:
                raise ValueError(f"Unknown action_score_mode: {action_score_mode!r}")
            for kind, cand, delta in actions:
                nb = dict(beam)
                nb['visited_entities'] = set(beam['visited_entities'])
                nb['score'] = beam['score'] + delta
                if kind == 'STOP':
                    nb['stopped'] = True; stop_actions += 1
                else:
                    nb['path_triples'] = beam['path_triples'] + [(cand['triple_id'], cand['direction'])]
                    nb['current_entity'] = cand['next_entity']
                    nb['visited_entities'].add(cand['next_entity'])
                expanded.append(nb)
        def rank_score(b):
            denom = max(len(b['path_triples']), 1) ** length_penalty if length_penalty else 1.0
            return b['score'] / denom
        beams = sorted(expanded, key=rank_score, reverse=True)[:beam_size]
        if all(b['stopped'] for b in beams):
            break
    for b in beams:
        b['covered_answer_entities'] = {b['current_entity']} & set(map(int, sample.get('a_entity_id_list', [])))
    return beams


def evaluate_beam_search(model, dataset, device, **kwargs):
    was_training = model.training
    model.eval()
    try:
        rows, predictions = [], []
        for sample in dataset:
            beams = beam_search_sample(model, sample, device, **kwargs)
            answers = set(map(int, sample.get('a_entity_id_list', [])))
            generated_triples = {tid for b in beams for tid, _ in b['path_triples']}
            gold_triples = set(torch.nonzero(sample['target_triple_probs'] > 0, as_tuple=False).view(-1).tolist())
            covered = set().union(*(b['covered_answer_entities'] for b in beams)) if beams else set()
            num_paths = len(beams)
            reaches = sum(1 for b in beams if b['current_entity'] in answers)
            lengths = [len(b['path_triples']) for b in beams]
            rows.append({
                'avg_num_generated_paths': num_paths,
                'avg_num_generated_triples': len(generated_triples),
                'answer_entity_coverage': len(covered) / max(len(answers), 1),
                'path_reaches_answer_ratio': reaches / max(num_paths, 1),
                'shortest_path_triple_recall': len(generated_triples & gold_triples) / max(len(gold_triples), 1),
                'gold_answer_hit': 1.0 if covered else 0.0,
                'avg_path_length': sum(lengths) / max(len(lengths), 1),
                'stop_action_rate': sum(1 for b in beams if b['stopped']) / max(num_paths, 1),
            })
            predictions.append({'id': sample.get('id'), 'beams': [{k: v for k, v in b.items() if k != 'visited_entities'} for b in beams]})
        return rows, predictions
    finally:
        if was_training:
            model.train()
