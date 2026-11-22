# GraphSieve: AAAI Supplementary Code

This directory is the self-contained supplementary implementation for **GraphSieve**. It follows the paper's three-part pipeline: (1) encode the question and knowledge-graph (KG) text, (2) use a path-conditioned graph selector to construct a compact evidence subgraph, and (3) ask a frozen large language model (LLM) to answer from the selected paths. The optional **OptiSet** stage improves the selector with offline path-set preferences derived from downstream answer likelihood, without updating the LLM.

### Embedding Pre-Computation

Please refer to the `retrieve/README.md` for details on the Embedding Pre-Computation

### Installation

We now train a selector, employ it for selection (inference), and evaluate the selection results.

```bash
conda create -n retriever python=3.10 -y
conda activate retriever
pip install -r retrieve/requirements/retriever.txt
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install pyg_lib==0.3.1 torch_scatter==2.1.2 torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Stage-I Training

We train the selector with `train_selector_path_sft.py`.

### Stage-II Alighment

First, we generate candidate path sets with `generate_selector_optiset_candidates.py`.

Then, we score every path set with a frozen local LLM with `score_selector_optiset_candidates.py`.

Finally, we train the selector with `train_selector_path_optiset.py`.

### Final KGQA reasoning and evaluation

First, we employ the selector for selection (inference) with `selector_inference.py`.

Then, we evaluate the selection results with `selector_eval.py`.

Finally, we use the selected paths to answer the question with `main.py`.
