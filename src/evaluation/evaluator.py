"""
Evaluation utilities.
Evaluate the performance of crowdsourcing label aggregation algorithms.

Supported metrics:
1. Standard classification metrics: Accuracy, Precision, Recall, F1.
2. Probability calibration metrics: ECE, Brier Score, NLL.
3. Additional classification metrics: Cohen's Kappa, MCC, Weighted F1.
4. Worker quality metrics: Worker Accuracy Correlation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    brier_score_loss
)
import pandas as pd


class Evaluator:
    """
    Evaluator for crowdsourcing label aggregation algorithms.
    
    Supported features:
    - Evaluate predictions from a single algorithm.
    - Compare multiple algorithms.
    - Evaluate probability calibration.
    - Evaluate worker quality estimates.
    - Export evaluation results.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = []
    
    def evaluate(
        self, 
        predictions: np.ndarray, 
        ground_truth: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        algorithm_name: str = "Unknown",
        dataset_name: str = "Unknown",
        training_time: Optional[float] = None,
        additional_metrics: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate predictions.
        
        Args:
            predictions: Predicted labels (n_instances,).
            ground_truth: Ground-truth labels (n_instances,).
            probabilities: Optional predicted probabilities (n_instances, n_classes).
            algorithm_name: Algorithm name.
            dataset_name: Dataset name.
            training_time: Training time in seconds.
            additional_metrics: Additional evaluation metrics.
            
        Returns:
            metrics: Dictionary of evaluation metrics.
        """
        # Exclude instances whose ground-truth label is -1.
        valid_mask = ground_truth != -1
        predictions = predictions[valid_mask]
        ground_truth = ground_truth[valid_mask]
        if probabilities is not None:
            probabilities = probabilities[valid_mask]
        
        if len(predictions) == 0:
            raise ValueError("No valid instances available for evaluation")
        
        # Use a stable class count for probability metrics.
        # Some datasets may have non-contiguous class ids in the evaluated subset
        # (e.g. labels {0,1,3,5,...}), where len(unique) is smaller than max(label)+1.
        # In that case, log_loss(labels=range(n_classes)) would fail.
        if probabilities is not None and hasattr(probabilities, "shape") and len(probabilities.shape) == 2:
            n_classes = int(probabilities.shape[1])
        else:
            n_classes = int(np.max(ground_truth)) + 1
        
        # ==================== Standard classification metrics ====================
        accuracy = accuracy_score(ground_truth, predictions)
        
        # Macro average.
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='macro', zero_division=0
        )
        
        # Weighted average to account for class imbalance.
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix.
        conf_matrix = confusion_matrix(ground_truth, predictions)
        
        # ==================== Additional classification metrics ====================
        # Cohen's Kappa accounts for chance agreement.
        kappa = cohen_kappa_score(ground_truth, predictions)
        
        # Matthews Correlation Coefficient is suitable for imbalanced data.
        mcc = matthews_corrcoef(ground_truth, predictions)
        
        metrics = {
            'algorithm': algorithm_name,
            'dataset': dataset_name,
            'n_instances': len(predictions),
            'n_classes': n_classes,
            # Standard metrics.
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            # Additional metrics.
            'cohen_kappa': kappa,
            'mcc': mcc,
            # Confusion matrix.
            'confusion_matrix': conf_matrix
        }
        
        # ==================== Probability calibration metrics ====================
        if probabilities is not None:
            prob_metrics = self._compute_probability_metrics(
                probabilities, ground_truth, n_classes
            )
            metrics.update(prob_metrics)
        
        if training_time is not None:
            metrics['training_time'] = training_time
        
        if additional_metrics is not None:
            metrics.update(additional_metrics)
        
        self.results.append(metrics)
        return metrics
    
    def _compute_probability_metrics(
        self, 
        probabilities: np.ndarray, 
        ground_truth: np.ndarray,
        n_classes: int
    ) -> Dict:
        """
        Compute probability calibration metrics.
        
        Args:
            probabilities: Predicted probabilities (n_instances, n_classes).
            ground_truth: Ground-truth labels (n_instances,).
            n_classes: Number of label classes.
            
        Returns:
            prob_metrics: Dictionary of probability metrics.
        """
        metrics = {}
        
        # 1. Negative Log-Likelihood (NLL)
        # Measure how well predicted probabilities match the ground-truth labels.
        try:
            nll = log_loss(ground_truth, probabilities, labels=range(n_classes))
            metrics['nll'] = nll
        except Exception:
            metrics['nll'] = np.nan
        
        # 2. Brier Score (multiclass).
        # Measure the accuracy of probability predictions.
        try:
            # Convert ground-truth labels to one-hot vectors.
            y_true_onehot = np.zeros((len(ground_truth), n_classes))
            y_true_onehot[np.arange(len(ground_truth)), ground_truth] = 1
            
            # Sum squared errors across classes, then average across instances.
            brier = np.mean(np.sum((probabilities - y_true_onehot) ** 2, axis=1))
            metrics['brier_score'] = brier
        except Exception:
            metrics['brier_score'] = np.nan
        
        # 3. Expected Calibration Error (ECE)
        # Measure the gap between predicted confidence and observed accuracy.
        try:
            ece = self._compute_ece(probabilities, ground_truth, n_bins=10)
            metrics['ece'] = ece
        except Exception:
            metrics['ece'] = np.nan
        
        # 4. Mean predictive entropy.
        # Measure prediction uncertainty.
        try:
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
            metrics['mean_entropy'] = np.mean(entropy)
            metrics['max_entropy'] = np.log(n_classes)  # Reference: entropy of the uniform distribution.
        except Exception:
            metrics['mean_entropy'] = np.nan
        
        # 5. Mean maximum probability (confidence).
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        wrong_mask = predictions != ground_truth
        metrics['mean_confidence'] = np.mean(confidences)
        metrics['wrong_confidence'] = float(np.mean(confidences[wrong_mask])) if np.any(wrong_mask) else 0.0
        
        return metrics
    
    def _compute_ece(
        self, 
        probabilities: np.ndarray, 
        ground_truth: np.ndarray, 
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures the gap between predicted confidence and observed accuracy.
        ECE = 0 indicates perfect calibration.
        
        Args:
            probabilities: Predicted probabilities.
            ground_truth: Ground-truth labels.
            n_bins: Number of bins.
            
        Returns:
            ece: Expected Calibration Error
        """
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        accuracies = (predictions == ground_truth).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(accuracies[in_bin])
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
        
        return ece
    
    def evaluate_worker_quality(
        self,
        estimated_quality: np.ndarray,
        labels: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict:
        """
        Evaluate the accuracy of worker quality estimates.
        
        Args:
            estimated_quality: Estimated worker quality (n_workers,).
            labels: Worker annotation matrix (n_instances, n_workers).
            ground_truth: Ground-truth labels (n_instances,).
            
        Returns:
            worker_metrics: Worker quality evaluation metrics.
        """
        n_instances, n_workers = labels.shape
        
        # Compute each worker's observed accuracy.
        true_accuracy = np.zeros(n_workers)
        annotation_counts = np.zeros(n_workers)
        
        for j in range(n_workers):
            mask = labels[:, j] != -1
            if mask.sum() > 0:
                worker_labels = labels[mask, j]
                true_labels = ground_truth[mask]
                true_accuracy[j] = np.mean(worker_labels == true_labels)
                annotation_counts[j] = mask.sum()
        
        # Consider only workers with annotations.
        active_workers = annotation_counts > 0
        if active_workers.sum() == 0:
            return {'worker_correlation': np.nan}
        
        est_quality = estimated_quality[active_workers]
        true_acc = true_accuracy[active_workers]
        
        # 1. Pearson correlation coefficient.
        if len(est_quality) > 1 and np.std(est_quality) > 0 and np.std(true_acc) > 0:
            correlation = np.corrcoef(est_quality, true_acc)[0, 1]
        else:
            correlation = np.nan
        
        # 2. Spearman rank correlation.
        try:
            from scipy.stats import spearmanr
            spearman_corr, _ = spearmanr(est_quality, true_acc)
        except Exception:
            spearman_corr = np.nan
        
        # 3. Mean squared error.
        mse = np.mean((est_quality - true_acc) ** 2)
        
        # 4. Mean absolute error.
        mae = np.mean(np.abs(est_quality - true_acc))
        
        return {
            'worker_pearson_correlation': correlation,
            'worker_spearman_correlation': spearman_corr,
            'worker_quality_mse': mse,
            'worker_quality_mae': mae,
            'n_active_workers': int(active_workers.sum()),
            'mean_true_accuracy': np.mean(true_acc),
            'mean_estimated_quality': np.mean(est_quality)
        }
    
    def compare_algorithms(
        self,
        algorithms: List,
        labels: np.ndarray,
        ground_truth: np.ndarray,
        dataset_name: str = "Unknown",
        verbose: bool = True,
        evaluate_probabilities: bool = True,
        evaluate_workers: bool = True
    ) -> pd.DataFrame:
        """
        Compare the performance of multiple algorithms.
        
        Args:
            algorithms: List of initialized aggregation algorithm objects.
            labels: Worker annotation matrix.
            ground_truth: Ground-truth labels.
            dataset_name: Dataset name.
            verbose: Whether to print detailed output.
            evaluate_probabilities: Whether to evaluate probability calibration.
            evaluate_workers: Whether to evaluate worker quality estimates.
            
        Returns:
            comparison_df: DataFrame containing comparison results.
        """
        comparison_results = []
        
        for algo in algorithms:
            algo_name = algo.__class__.__name__
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"Evaluating algorithm: {algo_name}")
                print(f"{'='*70}")
            
            # Train and predict.
            start_time = time.time()
            algo.fit(labels)
            predictions = algo.predict()
            training_time = time.time() - start_time
            
            # Retrieve probability outputs, if available.
            probabilities = None
            if evaluate_probabilities and hasattr(algo, 'instance_label_probs'):
                probs = algo.instance_label_probs
                if hasattr(probs, 'cpu'):  # PyTorch tensor
                    probabilities = probs.cpu().detach().numpy()
                else:
                    probabilities = probs
            
            # Evaluate predictions.
            metrics = self.evaluate(
                predictions=predictions,
                ground_truth=ground_truth,
                probabilities=probabilities,
                algorithm_name=algo_name,
                dataset_name=dataset_name,
                training_time=training_time
            )
            
            # Evaluate worker quality estimates.
            worker_metrics = {}
            if evaluate_workers and hasattr(algo, 'get_worker_quality'):
                try:
                    worker_quality = algo.get_worker_quality()
                    worker_metrics = self.evaluate_worker_quality(
                        worker_quality, labels, ground_truth
                    )
                    metrics.update(worker_metrics)
                except (NotImplementedError, Exception):
                    pass
            
            if verbose:
                self._print_metrics(metrics, worker_metrics)
            
            # Collect comparison results.
            result_row = {
                'Algorithm': algo_name,
                'Accuracy': metrics['accuracy'],
                'F1 (macro)': metrics['f1_macro'],
                'F1 (weighted)': metrics['f1_weighted'],
                'Cohen Kappa': metrics['cohen_kappa'],
                'MCC': metrics['mcc'],
                'Time (s)': metrics.get('training_time', np.nan)
            }
            
            # Add probability metrics.
            if probabilities is not None:
                result_row['NLL'] = metrics.get('nll', np.nan)
                result_row['Brier'] = metrics.get('brier_score', np.nan)
                result_row['ECE'] = metrics.get('ece', np.nan)
            
            # Add worker quality metrics.
            if worker_metrics:
                result_row['Worker Corr'] = worker_metrics.get('worker_pearson_correlation', np.nan)
            
            comparison_results.append(result_row)
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Algorithm comparison results - {dataset_name}")
            print(f"{'='*70}")
            print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def _print_metrics(self, metrics: Dict, worker_metrics: Dict = None):
        """Print evaluation metrics."""
        print("\n📊 Classification metrics:")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        print(f"  MCC:           {metrics['mcc']:.4f}")
        
        if 'nll' in metrics and not np.isnan(metrics.get('nll', np.nan)):
            print("\n📈 Probability calibration metrics:")
            print(f"  NLL:           {metrics['nll']:.4f}")
            print(f"  Brier Score:   {metrics['brier_score']:.4f}")
            print(f"  ECE:           {metrics['ece']:.4f}")
            print(f"  Mean Entropy:  {metrics['mean_entropy']:.4f} / {metrics['max_entropy']:.4f}")
            print(f"  Mean Confidence: {metrics['mean_confidence']:.4f}")
        
        if worker_metrics and not np.isnan(worker_metrics.get('worker_pearson_correlation', np.nan)):
            print("\n👷 Worker quality evaluation:")
            print(f"  Pearson Corr:  {worker_metrics['worker_pearson_correlation']:.4f}")
            print(f"  Spearman Corr: {worker_metrics['worker_spearman_correlation']:.4f}")
            print(f"  Quality MSE:   {worker_metrics['worker_quality_mse']:.4f}")
            print(f"  Active Workers: {worker_metrics['n_active_workers']}")
        
        if 'training_time' in metrics:
            print(f"\n⏱️  Training time: {metrics['training_time']:.4f}s")
    
    def get_summary(self) -> pd.DataFrame:
        """Get a summary of all evaluation results."""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.results:
            row = {
                'Algorithm': result['algorithm'],
                'Dataset': result['dataset'],
                'Accuracy': result['accuracy'],
                'F1 (macro)': result['f1_macro'],
                'F1 (weighted)': result['f1_weighted'],
                'Cohen Kappa': result['cohen_kappa'],
                'MCC': result['mcc'],
            }
            
            # Optional metrics.
            for key in ['nll', 'brier_score', 'ece', 'worker_pearson_correlation', 'training_time']:
                if key in result:
                    row[key] = result[key]
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def export_results(self, filename: str = "evaluation_results.csv"):
        """Export evaluation results to a CSV file."""
        summary_df = self.get_summary()
        if not summary_df.empty:
            summary_df.to_csv(filename, index=False)
            print(f"Evaluation results exported to: {filename}")
        else:
            print("No evaluation results to export")
    
    def clear_results(self):
        """Clear evaluation results."""
        self.results = []


def quick_evaluate(
    algorithm,
    labels: np.ndarray,
    ground_truth: np.ndarray,
    algorithm_name: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Convenience function for evaluating a single algorithm.
    
    Args:
        algorithm: Aggregation algorithm object.
        labels: Worker annotation matrix.
        ground_truth: Ground-truth labels.
        algorithm_name: Optional algorithm name.
        verbose: Whether to print results.
        
    Returns:
        metrics: Dictionary of evaluation metrics.
    """
    if algorithm_name is None:
        algorithm_name = algorithm.__class__.__name__
    
    evaluator = Evaluator()
    
    start_time = time.time()
    algorithm.fit(labels)
    predictions = algorithm.predict()
    training_time = time.time() - start_time
    
    # Retrieve predicted probabilities.
    probabilities = None
    if hasattr(algorithm, 'instance_label_probs'):
        probs = algorithm.instance_label_probs
        if hasattr(probs, 'cpu'):
            probabilities = probs.cpu().detach().numpy()
        else:
            probabilities = probs
    
    metrics = evaluator.evaluate(
        predictions=predictions,
        ground_truth=ground_truth,
        probabilities=probabilities,
        algorithm_name=algorithm_name,
        training_time=training_time
    )
    
    # Evaluate worker quality estimates.
    if hasattr(algorithm, 'get_worker_quality'):
        try:
            worker_quality = algorithm.get_worker_quality()
            worker_metrics = evaluator.evaluate_worker_quality(
                worker_quality, labels, ground_truth
            )
            metrics.update(worker_metrics)
        except (NotImplementedError, Exception):
            pass
    
    if verbose:
        evaluator._print_metrics(metrics, metrics)
    
    return metrics
