# BRAVE: Block-wise Structural Regularization via Controlled Evidence Feedback for Reliable Label Aggregation under Sparse Crowdsourcing

Official implementation of our paper accepted at **Transactions on Machine Learning Research (TMLR)**.

BRAVE aims to make crowdsourced labels more reliable when annotations are limited and noisy, with attention to both label quality and confidence.

BRAVE partitions workers into disjoint blocks of contiguous annotation-matrix
columns, preserving column order and letting the final block absorb the remainder.
The Python helper is `partition_annotations(labels, n_blocks=...)`.

## Quick start

Run these commands from the repository root:

```bash
python -m pip install -r requirements.txt

# Run on CrowdTruth RTE (downloads data on first use)
python main.py --dataset rte_ct

# Run on Bluebirds (downloads data on first use)
python main.py --dataset bluebirds

# View available options
python main.py --help
```
