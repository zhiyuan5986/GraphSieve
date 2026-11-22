import os
from urllib.parse import quote

import torch

from datasets import load_dataset
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset

def _embedding_shard_path(shard_dir, sample_id, quote_ids=False):
    filename_id = quote(str(sample_id), safe='') if quote_ids else str(sample_id)
    return os.path.join(shard_dir, f'{filename_id}.pth')


def get_emb(
    subset,
    text_encoder,
    save_file=None,
    shard_dir=None,
    save_monolithic=True,
    quote_shard_ids=False,
    overwrite_shards=False,
):
    if not save_monolithic and shard_dir is None:
        raise ValueError('Either save_monolithic must be True or shard_dir must be provided.')

    emb_dict = dict() if save_monolithic else None
    if shard_dir is not None:
        os.makedirs(shard_dir, exist_ok=True)

    for i in tqdm(range(len(subset))):
        id, q_text, text_entity_list, relation_list = subset[i]

        q_emb, entity_embs, relation_embs = text_encoder(
            q_text, text_entity_list, relation_list)
        emb_dict_i = {
            'q_emb': q_emb.cpu(),
            'entity_embs': entity_embs.cpu(),
            'relation_embs': relation_embs.cpu()
        }

        if shard_dir is not None:
            shard_file = _embedding_shard_path(shard_dir, id, quote_shard_ids)
            if overwrite_shards or not os.path.exists(shard_file):
                torch.save(emb_dict_i, shard_file)

        if save_monolithic:
            emb_dict[id] = emb_dict_i

    if save_monolithic:
        torch.save(emb_dict, save_file)

def main(args):
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/emb/gte-large-en-v1.5/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    torch.set_num_threads(config['env']['num_threads'])

    # if args.dataset == 'cwq':
    #     input_file = os.path.join('rmanluo', 'RoG-cwq') if args.dataset_path is None else os.path.join(args.dataset_path, 'RoG-cwq')
    # else:
    #     input_file = os.path.join('ml1996', 'webqsp') if args.dataset_path is None else os.path.join(args.dataset_path, 'webqsp')

    # train_set = load_dataset(input_file, split='train')
    # val_set = load_dataset(input_file, split='validation')
    # test_set = load_dataset(input_file, split='test')

    if args.dataset_path is None:
        if args.dataset == 'cwq':
            input_file = os.path.join('rmanluo', 'RoG-cwq') if args.dataset_path is None else os.path.join(args.dataset_path, 'RoG-cwq')
        else:
            input_file = os.path.join('ml1996', 'webqsp') if args.dataset_path is None else os.path.join(args.dataset_path, 'webqsp')

        train_set = load_dataset(input_file, split='train')
        val_set = load_dataset(input_file, split='validation')
        test_set = load_dataset(input_file, split='test')
    else:
        if args.dataset == 'cwq':
            local_dir = os.path.join(args.dataset_path, 'RoG-cwq', 'data')
        else:
            local_dir = os.path.join(args.dataset_path, 'webqsp', 'data')

        # data_files = {
        #     "train": os.path.join(local_dir, "train*.parquet"),
        #     "validation": os.path.join(local_dir, "validation*.parquet"),
        #     "test": os.path.join(local_dir, "test*.parquet"),
        # }

        dataset = load_dataset("parquet", data_dir=local_dir)

        train_set = dataset["train"]
        val_set = dataset["validation"]
        test_set = dataset["test"]
    
    entity_identifiers = []
    with open(config['entity_identifier_file'], 'r') as f:
        for line in f:
            entity_identifiers.append(line.strip())
    entity_identifiers = set(entity_identifiers)
    
    save_dir = args.save_dir_prefix + '/' + f'data_files/{args.dataset}/processed'if args.save_dir_prefix is not None else f'data_files/{args.dataset}/processed'
    os.makedirs(save_dir, exist_ok=True)

    train_set = EmbInferDataset(
        train_set,
        entity_identifiers,
        os.path.join(save_dir, 'train.pkl'))

    val_set = EmbInferDataset(
        val_set,
        entity_identifiers,
        os.path.join(save_dir, 'val.pkl'))

    test_set = EmbInferDataset(
        test_set,
        entity_identifiers,
        os.path.join(save_dir, 'test.pkl'),
        skip_no_topic=False,
        skip_no_ans=False)
    
    device = torch.device('cuda:0')
    
    text_encoder_name = config['text_encoder']['name']
    if text_encoder_name == 'gte-large-en-v1.5':
        from src.model.text_encoders import GTELargeEN
        text_encoder = GTELargeEN(args.model_path, device)
    else:
        raise NotImplementedError(text_encoder_name)
    
    emb_save_dir = args.save_dir_prefix + '/' + f'data_files/{args.dataset}/emb/{text_encoder_name}' if args.save_dir_prefix is not None else f'data_files/{args.dataset}/emb/{text_encoder_name}'
    os.makedirs(emb_save_dir, exist_ok=True)
    
    save_monolithic = not args.no_save_monolithic_emb
    save_sharded = args.save_sharded_emb or args.no_save_monolithic_emb

    get_emb(
        train_set,
        text_encoder,
        os.path.join(emb_save_dir, 'train.pth'),
        shard_dir=os.path.join(emb_save_dir, 'train') if save_sharded else None,
        save_monolithic=save_monolithic,
        quote_shard_ids=args.quote_shard_ids,
        overwrite_shards=args.overwrite_shards,
    )
    get_emb(
        val_set,
        text_encoder,
        os.path.join(emb_save_dir, 'val.pth'),
        shard_dir=os.path.join(emb_save_dir, 'val') if save_sharded else None,
        save_monolithic=save_monolithic,
        quote_shard_ids=args.quote_shard_ids,
        overwrite_shards=args.overwrite_shards,
    )
    get_emb(
        test_set,
        text_encoder,
        os.path.join(emb_save_dir, 'test.pth'),
        shard_dir=os.path.join(emb_save_dir, 'test') if save_sharded else None,
        save_monolithic=save_monolithic,
        quote_shard_ids=args.quote_shard_ids,
        overwrite_shards=args.overwrite_shards,
    )

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('--dataset-path', type=str, default=None, required=False, help='Path to the dataset files')
    parser.add_argument('--model-path', type=str, default=None, required=False, help='Path to the model')
    parser.add_argument('--save-dir-prefix', type=str, default=None, required=False, help='Prefix for the save directory')
    parser.add_argument(
        '--save-sharded-emb',
        action='store_true',
        help='Also save one .pth embedding payload per sample under <emb_save_dir>/<split>/.',
    )
    parser.add_argument(
        '--no-save-monolithic-emb',
        action='store_true',
        help='Do not keep the full split embedding dict in memory or save <split>.pth. Implies --save-sharded-emb.',
    )
    parser.add_argument(
        '--quote-shard-ids',
        action='store_true',
        help='URL-quote sample ids in sharded embedding filenames.',
    )
    parser.add_argument(
        '--overwrite-shards',
        action='store_true',
        help='Overwrite existing per-sample sharded embedding files.',
    )

    args = parser.parse_args()
    
    main(args)
