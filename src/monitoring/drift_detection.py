"""
Model Drift Detection

This module provides drift detection capabilities for monitoring
deployed ML models and triggering alerts when drift is detected.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from google.cloud import aiplatform
from google.cloud import monitoring_v3
from google.cloud import pubsub_v1

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift detection."""
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    CONCEPT_DRIFT = "concept_drift"
    DATA_QUALITY_DRIFT = "data_quality_drift"


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection."""
    project_id: str
    region: str = "us-central1"
    drift_threshold: float = 0.1
    statistical_test: str = "ks_test"  # ks_test, chi2_test, psi
    monitoring_frequency: str = "daily"
    baseline_window_days: int = 30
    detection_window_days: int = 7
    min_sample_size: int = 100
    enable_alerts: bool = True
    alert_topic: Optional[str] = None


class DriftDetector:
    """Drift detection for ML models."""
    
    def __init__(self, config: DriftDetectionConfig):
        """
        Initialize drift detector.
        
        Args:
            config: Drift detection configuration
        """
        self.config = config
        
        # Initialize clients
        aiplatform.init(project=config.project_id, location=config.region)
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        
        if config.enable_alerts and config.alert_topic:
            self.publisher = pubsub_v1.PublisherClient()
            self.topic_path = self.publisher.topic_path(
                config.project_id, config.alert_topic
            )
        else:
            self.publisher = None
            self.topic_path = None
        
        logger.info(f"Initialized drift detector for project {config.project_id}")
    
    def detect_feature_drift(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect feature drift between baseline and current data.
        
        Args:
            baseline_data: Baseline dataset
            current_data: Current dataset to compare
            feature_columns: Columns to analyze (if None, uses all numeric columns)
            
        Returns:
            Drift detection results
        """
        logger.info("Detecting feature drift")
        
        if feature_columns is None:
            feature_columns = baseline_data.select_dtypes(include=[np.number]).columns.tolist()
        
        drift_results = {
            "drift_detected": False,
            "overall_drift_score": 0.0,
            "feature_drift_scores": {},
            "drifted_features": [],
            "drift_type": DriftType.FEATURE_DRIFT.value,
            "detection_method": self.config.statistical_test,
            "threshold": self.config.drift_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
        total_drift_score = 0.0
        drifted_features = []
        
        for feature in feature_columns:
            if feature not in baseline_data.columns or feature not in current_data.columns:
                logger.warning(f"Feature {feature} not found in data")
                continue
            
            # Calculate drift score for this feature
            drift_score = self._calculate_feature_drift_score(
                baseline_data[feature].dropna(),
                current_data[feature].dropna(),
                feature
            )
            
            drift_results["feature_drift_scores"][feature] = drift_score
            total_drift_score += drift_score
            
            if drift_score > self.config.drift_threshold:
                drifted_features.append(feature)
        
        # Calculate overall drift score
        if len(feature_columns) > 0:
            overall_drift_score = total_drift_score / len(feature_columns)
            drift_results["overall_drift_score"] = overall_drift_score
            drift_results["drifted_features"] = drifted_features
            drift_results["drift_detected"] = overall_drift_score > self.config.drift_threshold
        
        logger.info(f"Feature drift detection completed. Drift detected: {drift_results['drift_detected']}")
        
        # Send alert if drift detected
        if drift_results["drift_detected"] and self.config.enable_alerts:
            self._send_drift_alert(drift_results)
        
        return drift_results
    
    def detect_prediction_drift(
        self,
        baseline_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect prediction drift between baseline and current predictions.
        
        Args:
            baseline_predictions: Baseline predictions
            current_predictions: Current predictions to compare
            
        Returns:
            Drift detection results
        """
        logger.info("Detecting prediction drift")
        
        # Calculate drift score
        drift_score = self._calculate_prediction_drift_score(
            baseline_predictions, current_predictions
        )
        
        drift_results = {
            "drift_detected": drift_score > self.config.drift_threshold,
            "drift_score": drift_score,
            "drift_type": DriftType.PREDICTION_DRIFT.value,
            "detection_method": self.config.statistical_test,
            "threshold": self.config.drift_threshold,
            "baseline_stats": self._calculate_distribution_stats(baseline_predictions),
            "current_stats": self._calculate_distribution_stats(current_predictions),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prediction drift detection completed. Drift detected: {drift_results['drift_detected']}")
        
        # Send alert if drift detected
        if drift_results["drift_detected"] and self.config.enable_alerts:
            self._send_drift_alert(drift_results)
        
        return drift_results
    
    def detect_data_quality_drift(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect data quality drift between baseline and current data.
        
        Args:
            baseline_data: Baseline dataset
            current_data: Current dataset to compare
            
        Returns:
            Data quality drift results
        """
        logger.info("Detecting data quality drift")
        
        # Calculate data quality metrics
        baseline_quality = self._calculate_data_quality_metrics(baseline_data)
        current_quality = self._calculate_data_quality_metrics(current_data)
        
        # Compare quality metrics
        quality_drift_score = self._calculate_quality_drift_score(
            baseline_quality, current_quality
        )
        
        drift_results = {
            "drift_detected": quality_drift_score > self.config.drift_threshold,
            "quality_drift_score": quality_drift_score,
            "drift_type": DriftType.DATA_QUALITY_DRIFT.value,
            "threshold": self.config.drift_threshold,
            "baseline_quality": baseline_quality,
            "current_quality": current_quality,
            "quality_changes": self._calculate_quality_changes(baseline_quality, current_quality),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Data quality drift detection completed. Drift detected: {drift_results['drift_detected']}")
        
        # Send alert if drift detected
        if drift_results["drift_detected"] and self.config.enable_alerts:
            self._send_drift_alert(drift_results)
        
        return drift_results
    
    def _calculate_feature_drift_score(
        self,
        baseline_values: pd.Series,
        current_values: pd.Series,
        feature_name: str
    ) -> float:
        """Calculate drift score for a single feature."""
        if len(baseline_values) < self.config.min_sample_size or len(current_values) < self.config.min_sample_size:
            logger.warning(f"Insufficient samples for feature {feature_name}")
            return 0.0
        
        if self.config.statistical_test == "ks_test":
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(baseline_values, current_values)
            return statistic
        
        elif self.config.statistical_test == "chi2_test":
            # Chi-square test for categorical data
            try:
                # Bin continuous data
                bins = np.histogram_bin_edges(
                    np.concatenate([baseline_values, current_values]), bins=10
                )
                baseline_hist, _ = np.histogram(baseline_values, bins=bins)
                current_hist, _ = np.histogram(current_values, bins=bins)
                
                # Avoid zero frequencies
                baseline_hist = baseline_hist + 1
                current_hist = current_hist + 1
                
                statistic, p_value = stats.chisquare(current_hist, baseline_hist)
                return min(statistic / 100.0, 1.0)  # Normalize
            except Exception as e:
                logger.warning(f"Chi-square test failed for {feature_name}: {e}")
                return 0.0
        
        elif self.config.statistical_test == "psi":
            # Population Stability Index
            return self._calculate_psi(baseline_values, current_values)
        
        else:
            logger.warning(f"Unknown statistical test: {self.config.statistical_test}")
            return 0.0
    
    def _calculate_prediction_drift_score(
        self,
        baseline_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> float:
        """Calculate drift score for predictions."""
        if len(baseline_predictions) < self.config.min_sample_size or len(current_predictions) < self.config.min_sample_size:
            logger.warning("Insufficient samples for prediction drift detection")
            return 0.0
        
        # Use KS test for prediction drift
        statistic, p_value = stats.ks_2samp(baseline_predictions, current_predictions)
        return statistic
    
    def _calculate_psi(self, baseline: pd.Series, current: pd.Series) -> float:
        """Calculate Population Stability Index."""
        try:
            # Create bins based on baseline data
            bins = np.percentile(baseline, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            bins = np.unique(bins)  # Remove duplicates
            
            if len(bins) < 2:
                return 0.0
            
            # Calculate frequencies
            baseline_freq = np.histogram(baseline, bins=bins)[0]
            current_freq = np.histogram(current, bins=bins)[0]
            
            # Convert to proportions
            baseline_prop = baseline_freq / len(baseline)
            current_prop = current_freq / len(current)
            
            # Avoid division by zero
            baseline_prop = np.where(baseline_prop == 0, 0.0001, baseline_prop)
            current_prop = np.where(current_prop == 0, 0.0001, current_prop)
            
            # Calculate PSI
            psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))
            return abs(psi)
        
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0
    
    def _calculate_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        return {
            "missing_value_rate": data.isnull().sum().sum() / (len(data) * len(data.columns)),
            "duplicate_rate": data.duplicated().sum() / len(data),
            "numeric_columns_count": len(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns_count": len(data.select_dtypes(include=['object']).columns),
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2
        }
    
    def _calculate_quality_drift_score(
        self,
        baseline_quality: Dict[str, Any],
        current_quality: Dict[str, Any]
    ) -> float:
        """Calculate overall data quality drift score."""
        # Compare key quality metrics
        missing_value_change = abs(
            current_quality["missing_value_rate"] - baseline_quality["missing_value_rate"]
        )
        duplicate_change = abs(
            current_quality["duplicate_rate"] - baseline_quality["duplicate_rate"]
        )
        
        # Weighted average of quality changes
        quality_drift = (missing_value_change * 0.6) + (duplicate_change * 0.4)
        return quality_drift
    
    def _calculate_quality_changes(
        self,
        baseline_quality: Dict[str, Any],
        current_quality: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate changes in quality metrics."""
        return {
            "missing_value_rate_change": current_quality["missing_value_rate"] - baseline_quality["missing_value_rate"],
            "duplicate_rate_change": current_quality["duplicate_rate"] - baseline_quality["duplicate_rate"],
            "row_count_change": current_quality["total_rows"] - baseline_quality["total_rows"],
            "column_count_change": current_quality["total_columns"] - baseline_quality["total_columns"]
        }
    
    def _calculate_distribution_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate distribution statistics."""
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75))
        }
    
    def _send_drift_alert(self, drift_results: Dict[str, Any]):
        """Send drift alert via Pub/Sub."""
        if not self.publisher or not self.topic_path:
            logger.warning("Pub/Sub not configured for alerts")
            return
        
        try:
            alert_message = {
                "alert_type": "model_drift",
                "drift_type": drift_results["drift_type"],
                "drift_detected": drift_results["drift_detected"],
                "drift_score": drift_results.get("overall_drift_score", drift_results.get("drift_score", 0.0)),
                "threshold": drift_results["threshold"],
                "timestamp": drift_results["timestamp"],
                "severity": self._determine_alert_severity(drift_results),
                "details": drift_results
            }
            
            message_data = json.dumps(alert_message).encode('utf-8')
            future = self.publisher.publish(self.topic_path, message_data)
            message_id = future.result()
            
            logger.info(f"Drift alert sent: {message_id}")
            
        except Exception as e:
            logger.error(f"Failed to send drift alert: {e}")
    
    def _determine_alert_severity(self, drift_results: Dict[str, Any]) -> str:
        """Determine alert severity based on drift score."""
        drift_score = drift_results.get("overall_drift_score", drift_results.get("drift_score", 0.0))
        threshold = drift_results["threshold"]
        
        if drift_score > threshold * 2:
            return "high"
        elif drift_score > threshold * 1.5:
            return "medium"
        else:
            return "low"
    
    def monitor_endpoint_drift(
        self,
        endpoint_id: str,
        baseline_data_path: str,
        monitoring_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Monitor an endpoint for drift over a specified time window.
        
        Args:
            endpoint_id: Vertex AI endpoint ID
            baseline_data_path: Path to baseline data
            monitoring_window_hours: Hours to look back for current data
            
        Returns:
            Comprehensive drift monitoring results
        """
        logger.info(f"Monitoring endpoint {endpoint_id} for drift")
        
        try:
            # Load baseline data
            baseline_data = pd.read_csv(baseline_data_path)
            
            # Get current prediction data (simplified - would integrate with actual logging)
            current_data = self._get_recent_prediction_data(endpoint_id, monitoring_window_hours)
            
            if current_data is None or len(current_data) < self.config.min_sample_size:
                return {
                    "status": "insufficient_data",
                    "message": f"Insufficient data for drift detection (minimum {self.config.min_sample_size} samples required)",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Perform drift detection
            results = {
                "endpoint_id": endpoint_id,
                "monitoring_window_hours": monitoring_window_hours,
                "baseline_samples": len(baseline_data),
                "current_samples": len(current_data),
                "timestamp": datetime.now().isoformat()
            }
            
            # Feature drift detection
            if len(baseline_data.columns) > 1:  # Has features beyond predictions
                feature_drift = self.detect_feature_drift(baseline_data, current_data)
                results["feature_drift"] = feature_drift
            
            # Prediction drift detection (if prediction column exists)
            if "prediction" in baseline_data.columns and "prediction" in current_data.columns:
                prediction_drift = self.detect_prediction_drift(
                    baseline_data["prediction"].values,
                    current_data["prediction"].values
                )
                results["prediction_drift"] = prediction_drift
            
            # Data quality drift
            quality_drift = self.detect_data_quality_drift(baseline_data, current_data)
            results["data_quality_drift"] = quality_drift
            
            # Overall drift assessment
            results["overall_drift_detected"] = any([
                results.get("feature_drift", {}).get("drift_detected", False),
                results.get("prediction_drift", {}).get("drift_detected", False),
                results.get("data_quality_drift", {}).get("drift_detected", False)
            ])
            
            logger.info(f"Drift monitoring completed for endpoint {endpoint_id}")
            return results
            
        except Exception as e:
            logger.error(f"Drift monitoring failed for endpoint {endpoint_id}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_recent_prediction_data(
        self,
        endpoint_id: str,
        hours: int
    ) -> Optional[pd.DataFrame]:
        """
        Get recent prediction data for an endpoint.
        
        This is a simplified implementation. In practice, this would
        integrate with Vertex AI prediction logging or custom logging.
        """
        # Placeholder implementation
        # In a real system, this would query prediction logs
        logger.info(f"Fetching recent prediction data for endpoint {endpoint_id}")
        
        # Return None to indicate no data available
        # Real implementation would fetch from BigQuery, Cloud Logging, etc.
        return None
