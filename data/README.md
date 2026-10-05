# Dataset Setup

Raw benchmark data and third-party dataset repositories are not distributed with
this code. Downloaded files under `data/` are ignored by Git; only this README is
tracked. Run all commands below from the BRAVE repository root.

Datasets retain their upstream licenses, terms of use, and citation requirements.
Please consult the linked sources before using or redistributing them. The MIT
license for BRAVE does not apply to third-party data. No ClueWeb document corpus
is needed for the label-aggregation runs.

## Synthetic Data

Synthetic data are generated locally and require no downloads:

```bash
python main.py --dataset synthetic --seed 0
```

## Automatically Downloaded Data

These loaders download their inputs when the required cache files are absent.
Internet access and continued availability of the upstream source are required.
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

If an upstream download is unavailable, obtain the same dataset release from its
maintainers and populate the loader's expected cache. Do not replace the data
with a different benchmark or derive evaluation gold from worker votes to bypass
a missing gold file. Download failures are not model-training failures.

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
It evaluates only items with available gold, using the existing mapping
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
The aggregate file must contain `input.sent_id` and `max_rel`. To preserve the
existing single-label protocol, the loader uses the first selected relation
when a worker response contains multiple relations separated by `|`.

## Reproduction Notes

The CLI preserves the existing per-dataset preprocessing policies, including
subsampling and minimum-annotation filters. `--seed` controls the implemented
sampling and model initialization; it does not fix upstream dataset revisions.
For a historical result comparison, keep the same raw files and preprocessing
settings. Some datasets have evaluation labels for only a subset of items.
`predictions.npz` records the truth and evaluation mask used by each run.
