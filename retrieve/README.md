# Embedding

## Supported Datasets

We support two built-in multi-hop knowledge graph question answering (KGQA) datasets:

- `webqsp`
- `cwq`

## Entity and Relation Embedding Pre-Computation

We first pre-compute and cache entity and relation embeddings for all samples to save time for later training and inference of retrievers.

### Installation

We use `gte-large-en-v1.5` for text encoder, hence the environment name.

```bash
conda create -n gte_large_en_v1-5 python=3.10 -y
conda activate gte_large_en_v1-5
pip install -r requirements/gte_large_en_v1-5.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

### Inference (Embedding Computation)

```bash
python emb.py -d D
```
where `D` should be a dataset mentioned in ["Supported Datasets"](#supported-datasets).