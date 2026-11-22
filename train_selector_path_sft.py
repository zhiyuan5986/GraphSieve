"""Train a path-conditioned CSM selector with gold-prefix SFT."""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

try:
    import swanlab
except ModuleNotFoundError:
    class _SwanLabStub:
        def init(self, *args, **kwargs): print('# swanlab unavailable; logging disabled')
        def log(self, *args, **kwargs): pass
        def finish(self): pass
    swanlab = _SwanLabStub()
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'retrieve'))

from src.dataset.retriever import RetrieverDataset  # noqa: E402
from src.setup import set_seed  # noqa: E402
from selector import PathConditionedCSMSelector  # noqa: E402
from selector_dataset import PathSFTDataset, collate_path_sft  # noqa: E402
from path_decode import evaluate_beam_search  # noqa: E402
from path_sft_metrics import mean_metrics  # noqa: E402


def build_config(args):
    return {'env': {'num_threads': args.num_threads, 'seed': args.seed},
            'dataset': {'name': args.dataset, 'text_encoder_name': args.text_encoder_name, 'data_dir': args.data_dir},
            'retriever': {'topic_pe': args.topic_pe, 'DDE_kwargs': {'num_rounds': args.num_rounds, 'num_reverse_rounds': args.num_reverse_rounds}}}


def tensors(sample, device):
    return (torch.tensor(sample['h_id_list'], dtype=torch.long, device=device),
            torch.tensor(sample['r_id_list'], dtype=torch.long, device=device),
            torch.tensor(sample['t_id_list'], dtype=torch.long, device=device),
            sample['q_emb'].to(device), sample['entity_embs'].to(device),
            len(sample['non_text_entity_list']), sample['relation_embs'].to(device), sample['topic_entity_one_hot'].to(device))


def state_forward(model, item, device):
    sample, state = item['sample'], item['state']
    h, r, t, q, ent, n_non, rel, topic_pe = tensors(sample, device)
    path_ids = [x[0] for x in state['path_triples']]
    path_dirs = [x[1] for x in state['path_triples']]
    cand_ids = [c['triple_id'] for c in state['candidate_triples']]
    cand_dirs = [c['direction'] for c in state['candidate_triples']]
    return model(q, path_ids, path_dirs, cand_ids, cand_dirs, h, r, t, ent, n_non, rel, topic_pe)


def compute_loss(args, cand_logits, stop_logit, labels, stop_label):
    device = stop_logit.device
    labels = torch.tensor(labels, dtype=torch.float, device=device)
    if cand_logits.numel() > 0:
        cand_bce = F.binary_cross_entropy_with_logits(cand_logits, labels)
    else:
        cand_bce = stop_logit.new_zeros(())
    stop_target = torch.tensor(float(stop_label), dtype=torch.float, device=device)
    stop_bce = F.binary_cross_entropy_with_logits(stop_logit.view(()), stop_target)
    all_logits = torch.cat([cand_logits, stop_logit.view(1)])
    all_labels = torch.cat([labels, stop_target.view(1)])
    pos = torch.nonzero(all_labels > 0, as_tuple=False).view(-1)
    mpce = -torch.logsumexp(F.log_softmax(all_logits, dim=0)[pos], dim=0) if pos.numel() else stop_logit.new_zeros(())
    loss = args.bce_loss_weight * cand_bce + args.stop_loss_weight * stop_bce + args.mpce_loss_weight * mpce
    return loss, {'path_sft_loss': loss.item(), 'candidate_bce_loss': cand_bce.item(), 'stop_bce_loss': stop_bce.item(), 'mpce_loss': mpce.item()}


def _compute_item_loss(args, model, item, device):
    cand_logits, stop_logit = state_forward(model, item, device)
    return compute_loss(args, cand_logits, stop_logit, item['state']['labels'], item['state']['stop_label'])


def _sample_index_groups(dataset):
    groups = defaultdict(list)
    for item_idx, (sample_idx, _) in enumerate(dataset.items):
        groups[sample_idx].append(item_idx)
    return list(groups.values())


def run_per_state_loss_epoch(args, loader, model, device, optimizer=None, desc='eval_loss'):
    train = optimizer is not None
    model.train(train)
    sums = defaultdict(float); n = 0
    for item in tqdm(loader, desc=desc):
        loss, stats = _compute_item_loss(args, model, item, device)
        if train:
            optimizer.zero_grad(); loss.backward()
            if args.max_grad_norm > 0: grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm); stats['grad_norm'] = grad_norm.item()
            optimizer.step()
        for k, v in stats.items(): sums[k] += float(v)
        n += 1
        if args.max_train_steps and train and n >= args.max_train_steps: break
    return {k: v / max(n, 1) for k, v in sums.items()}


def run_per_sample_loss_epoch(args, loader, model, device, optimizer=None, desc='eval_loss'):
    train = optimizer is not None
    model.train(train)
    groups = _sample_index_groups(loader.dataset)
    if train:
        order = torch.randperm(len(groups)).tolist()
        groups = [groups[i] for i in order]
    sums = defaultdict(float); n = 0
    for indices in tqdm(groups, desc=desc):
        losses = []
        sample_sums = defaultdict(float)
        for idx in indices:
            loss, stats = _compute_item_loss(args, model, loader.dataset[idx], device)
            losses.append(loss)
            for k, v in stats.items(): sample_sums[k] += float(v)
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        stats = {k: v / len(losses) for k, v in sample_sums.items()}
        stats['path_sft_loss'] = float(loss.detach().item())
        if train:
            optimizer.zero_grad(); loss.backward()
            if args.max_grad_norm > 0: grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm); stats['grad_norm'] = grad_norm.item()
            optimizer.step()
        for k, v in stats.items(): sums[k] += float(v)
        n += 1
        if args.max_train_steps and train and n >= args.max_train_steps: break
    return {k: v / max(n, 1) for k, v in sums.items()}


def run_loss_epoch(args, loader, model, device, optimizer=None, desc='eval_loss'):
    if args.per_sample_loss:
        return run_per_sample_loss_epoch(args, loader, model, device, optimizer, desc)
    return run_per_state_loss_epoch(args, loader, model, device, optimizer, desc)


def print_summary(name, ds):
    print(f"# {name} path SFT summary: " + json.dumps(ds.summary, ensure_ascii=False, sort_keys=True))


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or 'cuda' not in args.device else 'cpu')
    torch.set_num_threads(args.num_threads); set_seed(args.seed)
    config = build_config(args)
    train_base = RetrieverDataset(config, args.split, skip_no_path=True, emb_mode=args.emb_mode, emb_dir=args.emb_dir, emb_cache_size=args.emb_cache_size)
    val_base = RetrieverDataset(config, args.val_split, skip_no_path=True, emb_mode=args.emb_mode, emb_dir=args.emb_dir, emb_cache_size=args.emb_cache_size)
    test_base = RetrieverDataset(config, args.test_split, skip_no_path=False) if args.test_split else None
    train_ds = PathSFTDataset(train_base, max_hop=args.max_hop, lazy_emb=args.selector_lazy_emb, emb_dir=args.emb_dir, emb_cache_size=args.emb_cache_size); val_ds = PathSFTDataset(val_base, max_hop=args.max_hop, lazy_emb=args.selector_lazy_emb, emb_dir=args.emb_dir, emb_cache_size=args.emb_cache_size)
    print_summary(args.split, train_ds); print_summary(args.val_split, val_ds)
    if test_base: print(f"# {args.test_split} questions: {len(test_base)}")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_path_sft)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_path_sft)
    emb_size = train_base[0]['q_emb'].shape[-1]
    model = PathConditionedCSMSelector(
        emb_size=emb_size,
        topic_pe=args.topic_pe,
        DDE_kwargs={"num_rounds": args.num_rounds, "num_reverse_rounds": args.num_reverse_rounds},
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.output_dir or f"selector_path_sft_{args.dataset}_{time.strftime('%b%d-%H%M%S', time.gmtime())}"); out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    swanlab.init(project=args.swanlab_project or f"selector-path-sft-{args.dataset}", name=args.swanlab_name or out.name,
                 config=vars(args), mode=args.swanlab_mode)
    best = -1.0
    patience = 0
    for epoch in range(args.num_epochs):
        if patience == args.patience: break
        train_m = run_loss_epoch(args, train_loader, model, device, opt, 'train')
        val_loss = run_loss_epoch(args, val_loader, model, device, None, 'val_loss')
        rows, preds = evaluate_beam_search(model, val_base, device, max_hop=args.max_hop, beam_size=args.beam_size,
                                           local_topk=args.local_topk, length_penalty=args.length_penalty,
                                           allow_entity_repeat=args.allow_entity_repeat, allow_triple_repeat=args.allow_triple_repeat,
                                           action_score_mode=args.val_action_score_mode,
                                           force_stop=args.force_stop,
                                           )
        val_beam = mean_metrics(rows); val_m = {**val_loss, **val_beam}
        # (out / 'val_predictions.json').write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding='utf-8')
        torch.save({'selector_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'args': vars(args), 'epoch': epoch, 'emb_size': emb_size}, out / 'last.pth')
        score = val_m.get(args.best_metric, 0.0)
        if score >= best:
            best = score; torch.save({'selector_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'args': vars(args), 'epoch': epoch, 'best_val_metric': best, 'emb_size': emb_size}, out / 'cpt.pth')
            patience = 0
        else:
            patience += 1
        log = {f'train/{k}': v for k, v in train_m.items()}; log.update({f'val/{k}': v for k, v in val_m.items()}); log['epoch'] = epoch; log['patience'] = patience
        print(json.dumps(log, sort_keys=True)); swanlab.log(log)
    swanlab.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='webqsp', choices=['webqsp','cwq']); p.add_argument('--split', default='train'); p.add_argument('--val_split', default='val'); p.add_argument('--test_split', default='test')
    p.add_argument('--data_dir', default='retrieve'); p.add_argument('--output_dir', default=None); p.add_argument('--device', default='cuda:0')
    p.add_argument('--num_epochs', type=int, default=10000); p.add_argument('--patience', type=int, default=10); p.add_argument('--batch_size', type=int, default=1); p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--max_hop', type=int, default=3); p.add_argument('--beam_size', type=int, default=8); p.add_argument('--local_topk', type=int, default=64); p.add_argument('--length_penalty', type=float, default=0.0)
    p.add_argument('--allow_entity_repeat', action='store_true'); p.add_argument('--allow_triple_repeat', action='store_true')
    p.add_argument('--val_action_score_mode', default='softmax', choices=['softmax', 'sigmoid'],
                   help='Scoring mode for validation beam search actions over candidate triples plus STOP.')
    p.add_argument('--force_stop', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--topic_pe', action=argparse.BooleanOptionalAction, default=True); p.add_argument('--num_rounds', type=int, default=2); p.add_argument('--num_reverse_rounds', type=int, default=2)
    p.add_argument('--hidden_size', type=int, default=192); p.add_argument('--num_layers', type=int, default=3); p.add_argument('--num_heads', type=int, default=8); p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--bce_loss_weight', type=float, default=1.0); p.add_argument('--mpce_loss_weight', type=float, default=0.3); p.add_argument('--stop_loss_weight', type=float, default=1.0)
    p.add_argument('--per_sample_loss', action='store_true', help='Average path-state losses within each question first, then optimize/evaluate the mean across questions instead of weighting every state equally.')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--num_threads', type=int, default=16); p.add_argument('--text_encoder_name', default='gte-large-en-v1.5')
    p.add_argument('--emb_mode', default='eager', choices=['eager','lazy','sharded_lazy'], help='Embedding loading mode for RetrieverDataset; use lazy/sharded_lazy for large CWQ shards.')
    p.add_argument('--emb_dir', default=None, help='Optional directory containing per-sample embedding shards for lazy modes.')
    p.add_argument('--emb_cache_size', type=int, default=0, help='Number of per-sample embedding payloads to keep in the lazy embedding LRU cache.')
    p.add_argument('--selector_lazy_emb', action='store_true', help='Let PathSFTDataset load per-sample embedding shards on demand when the base dataset is eager.')
    p.add_argument('--max_grad_norm', type=float, default=1.0); p.add_argument('--max_train_steps', type=int, default=0); p.add_argument('--best_metric', default='answer_entity_coverage')
    p.add_argument('--swanlab_project', default=None); p.add_argument('--swanlab_name', default=None); p.add_argument('--swanlab_mode', default='disabled', choices=['cloud','local','disabled'])
    return p.parse_args()

if __name__ == '__main__': main(parse_args())