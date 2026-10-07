# Dataset Setup

This directory contains dataset setup instructions. Loaders store downloaded
data in local caches, and Git ignores cached files under `data/`.
Run all commands below from the BRAVE repository root.

Use each dataset under its source license, terms of use, and citation
requirements. Label aggregation uses the crowd annotation and reference-label
files listed below.

## Synthetic Data

Generate synthetic data locally:

```bash
python main.py --dataset synthetic --seed 0
```

## Automatically Downloaded Data

These loaders download their inputs when the required cache files are absent.
The cache locations below are relative to the BRAVE repository unless noted.

| CLI dataset | Upstream source | Default cache |
| --- | --- | --- |
| `multipref_expert` | [AllenAI MultiPref](https://huggingface.co/datasets/allenai/multipref) | Hugging Face's configured cache |
| `bluebirds` | [CUBAM Bluebirds](https://github.com/welinder/cubam/tree/public/demo/bluebirds) | `data/bluebirds/` |
| `toloka_rel2` | [Crowd-Kit relevance-2 archive](https://tlk.s3.yandex.net/dataset/crowd-kit/relevance-2.zip) | `data/relevance-2/` |
| `toloka_rel5` | [Crowd-Kit relevance-5 archive](https://tlk.s3.yandex.net/dataset/crowd-kit/relevance-5.zip) | `data/relevance-5/` |
| `rte_ct` | [CrowdTruth RTE](https://github.com/CrowdTruth/CrowdTruth-core/blob/master/tutorial/data/rte.standardized.csv) | `data/rte_crowdtruth/` |
| `cifar10h` | [CIFAR-10H](https://github.com/jcpeterson/cifar-10h) | `data/cifar10h/` |
| `netease` | [Crowd-Kit datasets](https://github.com/Toloka/crowd-kit) (`netease_crowd`) | Crowd-Kit's default data cache |
| `weather` | [WeatherSentiment AMT](https://eprints.soton.ac.uk/376543/1/WeatherSentiment_amt.csv) | `data/weather_sentiment_amt/` |
| `mre_treat`, `mre_cause` | [CrowdTruth Medical Relation Extraction](https://github.com/CrowdTruth/Medical-Relation-Extraction) | `data/crowdtruth_mre/` |
| `nist_trec` | [Crowd-Kit relevance archive](https://tlk.s3.yandex.net/dataset/crowd-kit/relevance.zip) | `data/nist_trec_relevance/` |
| `music_genre` | [Music genre benchmark archive](https://fprodrigues.com/mturk-datasets.tar.gz) | `data/music_genre/` |

For example:

```bash
python main.py --dataset bluebirds --seed 0
python main.py --dataset rte_ct --seed 0
```

For offline use, place files from the listed dataset release in the loader's
expected cache.

The Music-Genre loader expects `mturk-datasets.tar.gz` in `data/music_genre/`
and extracts the four `music_genre_classification/` members:
`mturk_answers.csv`, `music_genre_gold.csv`, `music_genre_test.csv`, and
`music_genre_mturk.csv`. For an offline run, keep the archive as well as any
already-extracted files in that cache directory.

## WebCrowd25K: Manual Download

Download the dataset archive from the
[WebCrowd25K project page](https://ir.ischool.utexas.edu/webcrowd25k/).
Extract the relevant files and place them at these exact paths:

```text
data/webcrowd25k/
  crowd_judgements.csv
  gold_judgements.txt
```

The loader also accepts the release's alternate spelling `gold_judjements.txt`.
The crowd CSV must contain `tid`, `did`, `wid`, and `label`; the gold file has
four whitespace-separated columns: topic ID, unused field, document ID, and
gold relevance label. The CLI requires both crowd judgments and official gold.
It evaluates items with available gold, using the mapping
`{-2, 0} -> 0`, `1 -> 1`, `2 -> 2`, `{3, 4} -> 3`.

```bash
python main.py --dataset webcrowd25k --seed 0
```

## CrowdTruth ODRE: Manual Clone

Clone the upstream dataset into the directory expected by the loader:

```bash
mkdir -p data/crowdtruth_odre
git clone https://github.com/CrowdTruth/Open-Domain-Relation-Extraction.git data/crowdtruth_odre/Open-Domain-Relation-Extraction
python main.py --dataset odre --seed 0
```

The required structure is:

```text
data/crowdtruth_odre/Open-Domain-Relation-Extraction/
  data/input/AMT/*.csv
  data/output/aggregated_sentences.csv
```

The AMT files must contain `WorkerId`, `Input.sent_id`, and `Answer.Q1`.
The aggregate file must contain `input.sent_id` and `max_rel`. For a worker
response containing multiple relations separated by `|`, the loader uses the
first selected relation.

## Run Configuration

The CLI applies dataset-specific subsampling and minimum-annotation filters.
Use `--data-seed` for data sampling, `--seed` for single-run model initialization,
and `--model-seeds` for validation-selected test runs. Record the dataset release
and preprocessing settings when comparing runs. `predictions.npz` stores the
reference labels and evaluation mask used for scoring.
