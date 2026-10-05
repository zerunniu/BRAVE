"""
MultiPref dataset loader.
Load and prepare the MultiPref dataset hosted on Hugging Face.

Dataset source: https://huggingface.co/datasets/allenai/multipref
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from datasets import load_dataset as hf_load_dataset

from .base_loader import BaseDataLoader


class MultiPrefLoader(BaseDataLoader):
    """
    MultiPref dataset loader.
    
    MultiPref is a preference dataset. Each instance contains:
    - A prompt.
    - Two model responses (completion_a, completion_b).
    - Four preference annotations (two regular workers and two experts).
    
    Preferences use a five-point Likert scale:
    - A-is-clearly-better (0)
    - A-is-slightly-better (1)
    - Tie (2)
    - B-is-slightly-better (3)
    - B-is-clearly-better (4)
    """
    
    # Preference label mapping.
    PREF_MAPPING = {
        'A-is-clearly-better': 0,
        'A-is-slightly-better': 1,
        'Tie': 2,
        'B-is-slightly-better': 3,
        'B-is-clearly-better': 4
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the MultiPref loader.
        
        Args:
            cache_dir: Dataset cache directory.
        """
        super().__init__()
        self.cache_dir = cache_dir
        self.n_classes = 5  # Five-point Likert scale.
    
    def load_dataset(self, subset: str = "default") -> None:
        """
        Load the MultiPref dataset from Hugging Face.
        
        Args:
            subset: Dataset subset.
                - "default": Full annotation data.
                - "human_overall_binarized": Binarized human annotations.
                - "gpt4_overall_binarized": Binarized GPT-4 annotations.
        """
        print(f"Loading MultiPref dataset (subset={subset})...")
        try:
            self.dataset = hf_load_dataset(
                "allenai/multipref",
                subset,
                split="train",
                cache_dir=self.cache_dir
            )
            print(f"✓ Loaded {len(self.dataset)} instances")
        except Exception as e:
            print(f"✗ Loading failed: {e}")
            raise
    
    def prepare_data(
        self,
        preference_type: str = "overall",
        include_experts: bool = True,
        include_normals: bool = True,
        min_annotations: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict]:
        """
        Convert MultiPref to the standard format for crowdsourcing aggregation.
        
        Args:
            preference_type: Preference dimension.
                - "overall": Overall preference.
                - "helpful": Helpfulness preference.
                - "truthful": Truthfulness preference.
                - "harmless": Harmlessness preference.
            include_experts: Whether to include expert annotations.
            include_normals: Whether to include regular-worker annotations.
            min_annotations: Minimum annotations per instance for filtering.
            
        Returns:
            labels: Worker annotation matrix (n_instances, n_workers).
            truth: Majority-vote reference labels (n_instances,).
            n_classes: Number of label classes (fixed at five).
            metadata: Dataset metadata.
        """
        if self.dataset is None:
            raise ValueError("Call load_dataset() before preparing data")
        
        if not include_experts and not include_normals:
            raise ValueError("At least one annotator type must be included")
        
        print(f"\nPreparing data...")
        print(f"  Preference dimension: {preference_type}")
        print(f"  Include experts: {include_experts}")
        print(f"  Include regular workers: {include_normals}")
        
        # Collect all annotations.
        all_annotations = []
        worker_ids = set()
        
        for idx, instance in enumerate(self.dataset):
            annotations = []
            
            # Collect regular-worker annotations.
            if include_normals:
                for ann in instance['normal_worker_annotations']:
                    pref_key = f"{preference_type}_pref"
                    if pref_key in ann and ann[pref_key] in self.PREF_MAPPING:
                        worker_id = f"normal_{ann['evaluator']}"
                        worker_ids.add(worker_id)
                        annotations.append({
                            'worker_id': worker_id,
                            'label': self.PREF_MAPPING[ann[pref_key]]
                        })
            
            # Collect expert annotations.
            if include_experts:
                for ann in instance['expert_worker_annotations']:
                    pref_key = f"{preference_type}_pref"
                    if pref_key in ann and ann[pref_key] in self.PREF_MAPPING:
                        worker_id = f"expert_{ann['evaluator']}"
                        worker_ids.add(worker_id)
                        annotations.append({
                            'worker_id': worker_id,
                            'label': self.PREF_MAPPING[ann[pref_key]]
                        })
            
            # Filter out instances with too few annotations.
            if len(annotations) >= min_annotations:
                all_annotations.append({
                    'instance_idx': idx,
                    'annotations': annotations,
                    'comparison_id': instance['comparison_id'],
                    'prompt_id': instance['prompt_id']
                })
        
        n_instances = len(all_annotations)
        worker_id_to_idx = {wid: idx for idx, wid in enumerate(sorted(worker_ids))}
        n_workers = len(worker_id_to_idx)
        
        print(f"\nDataset statistics:")
        print(f"  Instances: {n_instances}")
        print(f"  Workers: {n_workers}")
        print(f"  Classes: {self.n_classes}")
        
        # Build the annotation matrix.
        labels = np.full((n_instances, n_workers), -1, dtype=int)
        
        for i, inst in enumerate(all_annotations):
            for ann in inst['annotations']:
                worker_idx = worker_id_to_idx[ann['worker_id']]
                labels[i, worker_idx] = ann['label']
        
        # Compute statistics.
        coverage = (labels != -1).sum() / (n_instances * n_workers)
        avg_annotations = (labels != -1).sum(axis=1).mean()
        print(f"  Annotation coverage: {coverage:.2%}")
        print(f"  Mean annotations per instance: {avg_annotations:.1f}")
        
        # Use majority vote as the reference labels.
        truth = np.zeros(n_instances, dtype=int)
        for i in range(n_instances):
            instance_labels = labels[i][labels[i] != -1]
            if len(instance_labels) > 0:
                vote_counts = np.bincount(instance_labels, minlength=self.n_classes)
                truth[i] = np.argmax(vote_counts)
            else:
                truth[i] = 2  # Default to Tie.
        
        # Metadata.
        metadata = {
            'preference_type': preference_type,
            'n_instances': n_instances,
            'n_workers': n_workers,
            'n_classes': self.n_classes,
            'worker_id_to_idx': worker_id_to_idx,
            'label_mapping': self.PREF_MAPPING,
            'inverse_mapping': {v: k for k, v in self.PREF_MAPPING.items()},
            'coverage': coverage,
            'avg_annotations_per_instance': avg_annotations,
        }
        
        return labels, truth, self.n_classes, metadata
    
    def get_statistics(self, labels: np.ndarray, truth: np.ndarray, metadata: Dict) -> pd.DataFrame:
        """Get dataset statistics."""
        stats = []
        
        stats.append({'Metric': 'Total instances', 'Value': metadata['n_instances']})
        stats.append({'Metric': 'Total workers', 'Value': metadata['n_workers']})
        stats.append({'Metric': 'Number of classes', 'Value': self.n_classes})
        stats.append({'Metric': 'Annotation coverage', 'Value': f"{metadata['coverage']:.2%}"})
        stats.append({'Metric': 'Mean annotations per instance', 'Value': f"{metadata['avg_annotations_per_instance']:.2f}"})
        
        # Label distribution.
        truth_counts = np.bincount(truth, minlength=self.n_classes)
        for label_idx, count in enumerate(truth_counts):
            label_name = metadata['inverse_mapping'][label_idx]
            n_instances = metadata['n_instances']
            stats.append({
                'Metric': f'Reference label - {label_name}',
                'Value': f"{count} ({count/n_instances*100:.1f}%)"
            })
        
        return pd.DataFrame(stats)


def quick_load_multipref(
    preference_type: str = "overall",
    include_experts: bool = True,
    include_normals: bool = True
) -> Tuple[np.ndarray, np.ndarray, int, Dict]:
    """
    Convenience function for loading MultiPref quickly.
    
    Args:
        preference_type: Preference dimension.
        include_experts: Whether to include experts.
        include_normals: Whether to include regular workers.
        
    Returns:
        labels, truth, n_classes, metadata
    """
    loader = MultiPrefLoader()
    loader.load_dataset()
    return loader.prepare_data(
        preference_type=preference_type,
        include_experts=include_experts,
        include_normals=include_normals
    )



