"""
Base data loader.
Abstract base class for all dataset loaders.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any


class BaseDataLoader(ABC):
    """
    Base class for crowdsourcing dataset loaders.
    
    Concrete loaders should inherit this class and implement prepare_data.
    
    Extension guide:
        1. Inherit this class.
        2. Implement load_dataset() to load raw data.
        3. Implement prepare_data() to convert data to the standard format.
        4. Optionally implement get_statistics() to return dataset statistics.
    
    Standard output format:
        labels: np.ndarray, shape (n_instances, n_workers)
            Worker annotation matrix; -1 indicates a missing annotation.
        truth: np.ndarray, shape (n_instances,)
            Ground-truth labels, if available, or reference labels.
        n_classes: int
            Number of label classes.
        metadata: Dict
            Additional metadata.
    """
    
    def __init__(self):
        """Initialize the loader."""
        self.dataset = None
        self.n_classes = None
    
    @abstractmethod
    def load_dataset(self, **kwargs) -> None:
        """
        Load the raw dataset.
        
        Args:
            **kwargs: Dataset-specific parameters.
        """
        pass
    
    @abstractmethod
    def prepare_data(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        """
        Convert the dataset to the standard format.
        
        Returns:
            labels: Worker annotation matrix (n_instances, n_workers).
            truth: Ground-truth or reference labels (n_instances,).
            n_classes: Number of label classes.
            metadata: Metadata dictionary.
        """
        pass
    
    def get_statistics(self, labels: np.ndarray, truth: np.ndarray, metadata: Dict) -> Dict:
        """
        Get dataset statistics.
        
        Args:
            labels: Worker annotation matrix.
            truth: Ground-truth or reference labels.
            metadata: Dataset metadata.
            
        Returns:
            stats: Dictionary of dataset statistics.
        """
        n_instances, n_workers = labels.shape
        
        # Compute basic statistics.
        coverage = (labels != -1).sum() / (n_instances * n_workers)
        avg_annotations = (labels != -1).sum(axis=1).mean()
        
        # Label distribution.
        label_counts = np.bincount(truth, minlength=self.n_classes)
        
        return {
            'n_instances': n_instances,
            'n_workers': n_workers,
            'n_classes': self.n_classes,
            'coverage': coverage,
            'avg_annotations_per_instance': avg_annotations,
            'label_distribution': label_counts / n_instances,
        }



