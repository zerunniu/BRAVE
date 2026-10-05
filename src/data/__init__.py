"""
Data-loading utilities.

Supported datasets include:
- MultiPref: A preference dataset hosted on Hugging Face.

Adding a dataset:
    1. Create a loader module in this directory.
    2. Inherit BaseDataLoader or implement the prepare_data interface directly.
    3. Import the loader in this __init__.py.
"""

from .base_loader import BaseDataLoader
from .multipref import MultiPrefLoader, quick_load_multipref
from .webcrowd25k import WebCrowd25KLoader, quick_load_webcrowd25k
from .bluebirds import BluebirdsLoader, quick_load_bluebirds
from .toloka_relevance import TolokaRelevanceLoader, quick_load_toloka_relevance
from .synthetic_rte import SyntheticRTELoader, SyntheticRTEConfig, quick_load_synthetic_rte
from .rte_crowdtruth import CrowdTruthRTELoader, quick_load_rte_crowdtruth
from .netease_crowd import NetEaseCrowdLoader, NetEaseCrowdConfig, quick_load_netease_crowd
from .cifar10h import CIFAR10HLoader, quick_load_cifar10h
from .crowdtruth_mre import CrowdTruthMRELoader, quick_load_crowdtruth_mre
from .crowdgleason import CrowdGleasonLoader, quick_load_crowdgleason
from .weather_sentiment_amt import WeatherSentimentAMTLoader, quick_load_weather_sentiment_amt
from .crowdtruth_odre import CrowdTruthODRELoader, quick_load_crowdtruth_odre
from .nist_trec_relevance import NistTrecRelevanceLoader, quick_load_nist_trec_relevance
from .music_genre import MusicGenreLoader, quick_load_music_genre
from .synthetic_aligned_error import SyntheticAlignedErrorConfig, generate_synthetic_aligned_error

__all__ = [
    "BaseDataLoader",
    "MultiPrefLoader",
    "quick_load_multipref",
    "WebCrowd25KLoader",
    "quick_load_webcrowd25k",
    "BluebirdsLoader",
    "quick_load_bluebirds",
    "TolokaRelevanceLoader",
    "quick_load_toloka_relevance",
    "SyntheticRTELoader",
    "SyntheticRTEConfig",
    "quick_load_synthetic_rte",
    "CrowdTruthRTELoader",
    "quick_load_rte_crowdtruth",
    "NetEaseCrowdLoader",
    "NetEaseCrowdConfig",
    "quick_load_netease_crowd",
    "CIFAR10HLoader",
    "quick_load_cifar10h",
    "CrowdTruthMRELoader",
    "quick_load_crowdtruth_mre",
    "CrowdGleasonLoader",
    "quick_load_crowdgleason",
    "WeatherSentimentAMTLoader",
    "quick_load_weather_sentiment_amt",
    "CrowdTruthODRELoader",
    "quick_load_crowdtruth_odre",
    "NistTrecRelevanceLoader",
    "quick_load_nist_trec_relevance",
    "MusicGenreLoader",
    "quick_load_music_genre",
    "SyntheticAlignedErrorConfig",
    "generate_synthetic_aligned_error",
]



