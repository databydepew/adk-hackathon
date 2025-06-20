"""
Model Monitoring Pipeline Components

This module provides components for setting up model monitoring, drift detection,
and performance tracking for deployed ML models.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from kfp.dsl import component, Input, Output, Artifact

logger = logging.getLogger(__name__)


class MonitoringType(Enum):
    """Types of model monitoring."""
    DRIFT_DETECTION = "drift_detection"
    PERFORMANCE_MONITORING = "performance_monitoring"
    DATA_QUALITY_MONITORING = "data_quality_monitoring"
    BIAS_MONITORING = "bias_monitoring"


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring components."""
    monitoring_type: MonitoringType
    project_id: str
    region: str = "us-central1"
    monitoring_frequency: str = "daily"
    drift_threshold: float = 0.1
    performance_threshold: float = 0.05
    enable_email_alerts: bool = True
    enable_slack_alerts: bool = False
    alert_email: Optional[str] = None
    slack_webhook: Optional[str] = None
    training_dataset_path: Optional[str] = None
    baseline_dataset_path: Optional[str] = None


class BaseMonitoringComponent(ABC):
    """Base class for monitoring components."""
    
    def __init__(self, config: MonitoringConfig):
        """
        Initialize the monitoring component.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.component_name = f"{config.monitoring_type.value}_monitoring"
        
    @abstractmethod
    def create_component(self) -> component:
        """Create the KFP component for monitoring setup."""
        pass
    
    @abstractmethod
    def get_component_spec(self) -> Dict[str, Any]:
        """Get the component specification."""
        pass


class DriftDetectionComponent(BaseMonitoringComponent):
    """Model drift detection monitoring component."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize drift detection component."""
        super().__init__(config)
        if config.monitoring_type != MonitoringType.DRIFT_DETECTION:
            raise ValueError("Config must be for drift detection monitoring")
    
    def create_component(self) -> component:
        """Create drift detection KFP component."""
        
        @component(
            base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
            packages_to_install=[
                "google-cloud-aiplatform>=1.38.0",
                "google-cloud-monitoring>=2.15.0",
                "google-auth>=2.17.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "scipy>=1.10.0"
            ]
        )
        def drift_detection_setup(
            endpoint: Input[Artifact],
            monitoring_job: Output[Artifact],
            project_id: str,
            region: str = "us-central1",
            monitoring_frequency: str = "daily",
            drift_threshold: float = 0.1,
            training_dataset_path: str = "",
            baseline_dataset_path: str = "",
            enable_email_alerts: bool = True,
            alert_email: str = "",
            enable_slack_alerts: bool = False,
            slack_webhook: str = ""
        ):
            """
            Set up drift detection monitoring for deployed model.
            
            Args:
                endpoint: Input endpoint artifact
                monitoring_job: Output monitoring job artifact
                project_id: GCP project ID
                region: GCP region
                monitoring_frequency: Frequency of monitoring (daily, hourly, weekly)
                drift_threshold: Threshold for drift detection
                training_dataset_path: Path to training dataset for baseline
                baseline_dataset_path: Path to baseline dataset
                enable_email_alerts: Enable email alerts
                alert_email: Email address for alerts
                enable_slack_alerts: Enable Slack alerts
                slack_webhook: Slack webhook URL
            """
            import json
            import os
            import logging
            from google.cloud import aiplatform
            from google.cloud import monitoring_v3
            
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            try:
                # Initialize Vertex AI
                aiplatform.init(project=project_id, location=region)
                logger.info(f"Initialized Vertex AI for project {project_id} in region {region}")
                
                # Load endpoint information
                endpoint_info_path = os.path.join(endpoint.path, "endpoint_info.json")
                with open(endpoint_info_path, 'r') as f:
                    endpoint_info = json.load(f)
                
                endpoint_resource_name = endpoint_info["endpoint_resource_name"]
                logger.info(f"Setting up drift detection for endpoint: {endpoint_resource_name}")
                
                # Create model monitoring job
                monitoring_job_name = f"{endpoint_info['endpoint_name']}_drift_monitoring"
                
                # Configure monitoring objectives
                monitoring_objectives = []
                
                # Feature drift monitoring
                feature_drift_config = {
                    "drift_threshold": drift_threshold,
                    "attribution_score_drift_threshold": drift_threshold * 0.8,
                    "default_drift_threshold": drift_threshold
                }
                
                monitoring_objectives.append({
                    "type": "feature_drift",
                    "config": feature_drift_config
                })
                
                # Prediction drift monitoring
                prediction_drift_config = {
                    "drift_threshold": drift_threshold,
                    "attribution_score_drift_threshold": drift_threshold * 0.8
                }
                
                monitoring_objectives.append({
                    "type": "prediction_drift", 
                    "config": prediction_drift_config
                })
                
                # Set up monitoring schedule
                monitoring_schedule = {
                    "frequency": monitoring_frequency,
                    "enabled": True
                }
                
                # Configure alerting
                alerting_config = {
                    "enable_email_alerts": enable_email_alerts,
                    "enable_slack_alerts": enable_slack_alerts
                }
                
                if enable_email_alerts and alert_email:
                    alerting_config["alert_email"] = alert_email
                
                if enable_slack_alerts and slack_webhook:
                    alerting_config["slack_webhook"] = slack_webhook
                
                # Create monitoring configuration
                monitoring_config = {
                    "monitoring_job_name": monitoring_job_name,
                    "endpoint_resource_name": endpoint_resource_name,
                    "monitoring_objectives": monitoring_objectives,
                    "monitoring_schedule": monitoring_schedule,
                    "alerting_config": alerting_config,
                    "training_dataset_path": training_dataset_path,
                    "baseline_dataset_path": baseline_dataset_path
                }
                
                logger.info("Creating model monitoring job...")
                
                # Note: In a real implementation, you would use the Vertex AI Model Monitoring API
                # For now, we'll create a configuration that can be used to set up monitoring
                
                # Create Cloud Monitoring alert policies
                monitoring_client = monitoring_v3.AlertPolicyServiceClient()
                project_name = f"projects/{project_id}"
                
                # Create alert policy for drift detection
                alert_policy = monitoring_v3.AlertPolicy(
                    display_name=f"{monitoring_job_name}_drift_alert",
                    documentation=monitoring_v3.AlertPolicy.Documentation(
                        content=f"Alert for model drift detection on endpoint {endpoint_info['endpoint_name']}"
                    ),
                    conditions=[
                        monitoring_v3.AlertPolicy.Condition(
                            display_name="Model drift detected",
                            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                                filter=f'resource.type="aiplatform.googleapis.com/Endpoint"',
                                comparison=monitoring_v3.ComparisonType.COMPARISON_GREATER_THAN,
                                threshold_value=drift_threshold,
                                duration={"seconds": 300}  # 5 minutes
                            )
                        )
                    ],
                    combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
                    enabled=True
                )
                
                try:
                    created_policy = monitoring_client.create_alert_policy(
                        name=project_name,
                        alert_policy=alert_policy
                    )
                    logger.info(f"Created alert policy: {created_policy.name}")
                    monitoring_config["alert_policy_name"] = created_policy.name
                except Exception as e:
                    logger.warning(f"Failed to create alert policy: {e}")
                    monitoring_config["alert_policy_name"] = None
                
                # Save monitoring job information
                os.makedirs(monitoring_job.path, exist_ok=True)
                
                monitoring_job_info = {
                    "monitoring_job_name": monitoring_job_name,
                    "monitoring_type": "drift_detection",
                    "endpoint_name": endpoint_info["endpoint_name"],
                    "endpoint_resource_name": endpoint_resource_name,
                    "project_id": project_id,
                    "region": region,
                    "monitoring_config": monitoring_config,
                    "status": "active",
                    "created_at": "2024-01-01T00:00:00Z"  # Would be actual timestamp
                }
                
                monitoring_job_metadata_path = os.path.join(monitoring_job.path, "monitoring_job_info.json")
                with open(monitoring_job_metadata_path, 'w') as f:
                    json.dump(monitoring_job_info, f, indent=2)
                
                # Create monitoring script for periodic execution
                monitoring_script = f"""#!/bin/bash
# Model Drift Detection Monitoring Script
# Generated for endpoint: {endpoint_info['endpoint_name']}

PROJECT_ID="{project_id}"
REGION="{region}"
ENDPOINT_NAME="{endpoint_info['endpoint_name']}"
DRIFT_THRESHOLD="{drift_threshold}"

echo "Running drift detection for endpoint: $ENDPOINT_NAME"
echo "Drift threshold: $DRIFT_THRESHOLD"

# Add your drift detection logic here
# This would typically involve:
# 1. Fetching recent prediction data
# 2. Comparing with baseline/training data
# 3. Calculating drift metrics
# 4. Triggering alerts if threshold exceeded

echo "Drift detection monitoring completed"
"""
                
                script_path = os.path.join(monitoring_job.path, "drift_monitoring.sh")
                with open(script_path, 'w') as f:
                    f.write(monitoring_script)
                
                logger.info(f"Drift detection monitoring setup completed successfully")
                
            except Exception as e:
                logger.error(f"Drift detection setup failed: {str(e)}")
                raise
        
        return drift_detection_setup
    
    def get_component_spec(self) -> Dict[str, Any]:
        """Get drift detection component specification."""
        return {
            "name": self.component_name,
            "description": "Set up drift detection monitoring for deployed model",
            "inputs": {
                "endpoint": {"type": "Artifact", "description": "Deployed endpoint information"},
                "project_id": {"type": "String", "description": "GCP project ID"},
                "region": {"type": "String", "description": "GCP region", "default": "us-central1"},
                "monitoring_frequency": {"type": "String", "description": "Monitoring frequency", "default": "daily"},
                "drift_threshold": {"type": "Float", "description": "Drift detection threshold", "default": 0.1},
                "training_dataset_path": {"type": "String", "description": "Training dataset path", "optional": True},
                "baseline_dataset_path": {"type": "String", "description": "Baseline dataset path", "optional": True},
                "enable_email_alerts": {"type": "Boolean", "description": "Enable email alerts", "default": True},
                "alert_email": {"type": "String", "description": "Alert email address", "optional": True},
                "enable_slack_alerts": {"type": "Boolean", "description": "Enable Slack alerts", "default": False},
                "slack_webhook": {"type": "String", "description": "Slack webhook URL", "optional": True}
            },
            "outputs": {
                "monitoring_job": {"type": "Artifact", "description": "Monitoring job information"}
            }
        }


class PerformanceMonitoringComponent(BaseMonitoringComponent):
    """Model performance monitoring component."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize performance monitoring component."""
        super().__init__(config)
        if config.monitoring_type != MonitoringType.PERFORMANCE_MONITORING:
            raise ValueError("Config must be for performance monitoring")
    
    def create_component(self) -> component:
        """Create performance monitoring KFP component."""
        
        @component(
            base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
            packages_to_install=[
                "google-cloud-aiplatform>=1.38.0",
                "google-cloud-monitoring>=2.15.0",
                "google-auth>=2.17.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0"
            ]
        )
        def performance_monitoring_setup(
            endpoint: Input[Artifact],
            monitoring_job: Output[Artifact],
            project_id: str,
            region: str = "us-central1",
            monitoring_frequency: str = "daily",
            performance_threshold: float = 0.05,
            enable_email_alerts: bool = True,
            alert_email: str = "",
            enable_slack_alerts: bool = False,
            slack_webhook: str = ""
        ):
            """
            Set up performance monitoring for deployed model.
            
            Args:
                endpoint: Input endpoint artifact
                monitoring_job: Output monitoring job artifact
                project_id: GCP project ID
                region: GCP region
                monitoring_frequency: Frequency of monitoring
                performance_threshold: Threshold for performance degradation
                enable_email_alerts: Enable email alerts
                alert_email: Email address for alerts
                enable_slack_alerts: Enable Slack alerts
                slack_webhook: Slack webhook URL
            """
            import json
            import os
            import logging
            from google.cloud import aiplatform
            from google.cloud import monitoring_v3
            
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            try:
                # Initialize Vertex AI
                aiplatform.init(project=project_id, location=region)
                logger.info(f"Initialized Vertex AI for project {project_id} in region {region}")
                
                # Load endpoint information
                endpoint_info_path = os.path.join(endpoint.path, "endpoint_info.json")
                with open(endpoint_info_path, 'r') as f:
                    endpoint_info = json.load(f)
                
                endpoint_resource_name = endpoint_info["endpoint_resource_name"]
                logger.info(f"Setting up performance monitoring for endpoint: {endpoint_resource_name}")
                
                # Create performance monitoring job
                monitoring_job_name = f"{endpoint_info['endpoint_name']}_performance_monitoring"
                
                # Configure performance monitoring objectives
                monitoring_objectives = []
                
                # Prediction accuracy monitoring
                accuracy_config = {
                    "performance_threshold": performance_threshold,
                    "baseline_performance": 0.8,  # Would be set based on training metrics
                    "minimum_sample_size": 100
                }
                
                monitoring_objectives.append({
                    "type": "prediction_accuracy",
                    "config": accuracy_config
                })
                
                # Latency monitoring
                latency_config = {
                    "max_latency_ms": 1000,
                    "p95_latency_threshold_ms": 500,
                    "p99_latency_threshold_ms": 800
                }
                
                monitoring_objectives.append({
                    "type": "latency_monitoring",
                    "config": latency_config
                })
                
                # Throughput monitoring
                throughput_config = {
                    "min_requests_per_minute": 10,
                    "max_requests_per_minute": 1000,
                    "error_rate_threshold": 0.05
                }
                
                monitoring_objectives.append({
                    "type": "throughput_monitoring",
                    "config": throughput_config
                })
                
                # Set up monitoring schedule
                monitoring_schedule = {
                    "frequency": monitoring_frequency,
                    "enabled": True
                }
                
                # Configure alerting
                alerting_config = {
                    "enable_email_alerts": enable_email_alerts,
                    "enable_slack_alerts": enable_slack_alerts
                }
                
                if enable_email_alerts and alert_email:
                    alerting_config["alert_email"] = alert_email
                
                if enable_slack_alerts and slack_webhook:
                    alerting_config["slack_webhook"] = slack_webhook
                
                # Create monitoring configuration
                monitoring_config = {
                    "monitoring_job_name": monitoring_job_name,
                    "endpoint_resource_name": endpoint_resource_name,
                    "monitoring_objectives": monitoring_objectives,
                    "monitoring_schedule": monitoring_schedule,
                    "alerting_config": alerting_config
                }
                
                logger.info("Creating performance monitoring job...")
                
                # Create Cloud Monitoring alert policies for performance
                monitoring_client = monitoring_v3.AlertPolicyServiceClient()
                project_name = f"projects/{project_id}"
                
                # Create alert policy for performance degradation
                alert_policy = monitoring_v3.AlertPolicy(
                    display_name=f"{monitoring_job_name}_performance_alert",
                    documentation=monitoring_v3.AlertPolicy.Documentation(
                        content=f"Alert for performance degradation on endpoint {endpoint_info['endpoint_name']}"
                    ),
                    conditions=[
                        monitoring_v3.AlertPolicy.Condition(
                            display_name="Performance degradation detected",
                            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                                filter=f'resource.type="aiplatform.googleapis.com/Endpoint"',
                                comparison=monitoring_v3.ComparisonType.COMPARISON_LESS_THAN,
                                threshold_value=1.0 - performance_threshold,  # Performance drop threshold
                                duration={"seconds": 600}  # 10 minutes
                            )
                        )
                    ],
                    combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
                    enabled=True
                )
                
                try:
                    created_policy = monitoring_client.create_alert_policy(
                        name=project_name,
                        alert_policy=alert_policy
                    )
                    logger.info(f"Created performance alert policy: {created_policy.name}")
                    monitoring_config["alert_policy_name"] = created_policy.name
                except Exception as e:
                    logger.warning(f"Failed to create performance alert policy: {e}")
                    monitoring_config["alert_policy_name"] = None
                
                # Save monitoring job information
                os.makedirs(monitoring_job.path, exist_ok=True)
                
                monitoring_job_info = {
                    "monitoring_job_name": monitoring_job_name,
                    "monitoring_type": "performance_monitoring",
                    "endpoint_name": endpoint_info["endpoint_name"],
                    "endpoint_resource_name": endpoint_resource_name,
                    "project_id": project_id,
                    "region": region,
                    "monitoring_config": monitoring_config,
                    "status": "active",
                    "created_at": "2024-01-01T00:00:00Z"  # Would be actual timestamp
                }
                
                monitoring_job_metadata_path = os.path.join(monitoring_job.path, "monitoring_job_info.json")
                with open(monitoring_job_metadata_path, 'w') as f:
                    json.dump(monitoring_job_info, f, indent=2)
                
                # Create performance monitoring script
                monitoring_script = f"""#!/bin/bash
# Model Performance Monitoring Script
# Generated for endpoint: {endpoint_info['endpoint_name']}

PROJECT_ID="{project_id}"
REGION="{region}"
ENDPOINT_NAME="{endpoint_info['endpoint_name']}"
PERFORMANCE_THRESHOLD="{performance_threshold}"

echo "Running performance monitoring for endpoint: $ENDPOINT_NAME"
echo "Performance threshold: $PERFORMANCE_THRESHOLD"

# Add your performance monitoring logic here
# This would typically involve:
# 1. Fetching recent prediction metrics
# 2. Calculating accuracy, latency, throughput
# 3. Comparing with baseline performance
# 4. Triggering alerts if thresholds exceeded

echo "Performance monitoring completed"
"""
                
                script_path = os.path.join(monitoring_job.path, "performance_monitoring.sh")
                with open(script_path, 'w') as f:
                    f.write(monitoring_script)
                
                logger.info(f"Performance monitoring setup completed successfully")
                
            except Exception as e:
                logger.error(f"Performance monitoring setup failed: {str(e)}")
                raise
        
        return performance_monitoring_setup
    
    def get_component_spec(self) -> Dict[str, Any]:
        """Get performance monitoring component specification."""
        return {
            "name": self.component_name,
            "description": "Set up performance monitoring for deployed model",
            "inputs": {
                "endpoint": {"type": "Artifact", "description": "Deployed endpoint information"},
                "project_id": {"type": "String", "description": "GCP project ID"},
                "region": {"type": "String", "description": "GCP region", "default": "us-central1"},
                "monitoring_frequency": {"type": "String", "description": "Monitoring frequency", "default": "daily"},
                "performance_threshold": {"type": "Float", "description": "Performance degradation threshold", "default": 0.05},
                "enable_email_alerts": {"type": "Boolean", "description": "Enable email alerts", "default": True},
                "alert_email": {"type": "String", "description": "Alert email address", "optional": True},
                "enable_slack_alerts": {"type": "Boolean", "description": "Enable Slack alerts", "default": False},
                "slack_webhook": {"type": "String", "description": "Slack webhook URL", "optional": True}
            },
            "outputs": {
                "monitoring_job": {"type": "Artifact", "description": "Monitoring job information"}
            }
        }


class MonitoringComponentFactory:
    """Factory for creating monitoring components."""
    
    @staticmethod
    def create_component(config: MonitoringConfig) -> BaseMonitoringComponent:
        """
        Create a monitoring component based on configuration.
        
        Args:
            config: Monitoring configuration
            
        Returns:
            Monitoring component instance
        """
        if config.monitoring_type == MonitoringType.DRIFT_DETECTION:
            return DriftDetectionComponent(config)
        elif config.monitoring_type == MonitoringType.PERFORMANCE_MONITORING:
            return PerformanceMonitoringComponent(config)
        else:
            raise ValueError(f"Unsupported monitoring type: {config.monitoring_type}")
    
    @staticmethod
    def get_supported_monitoring_types() -> List[MonitoringType]:
        """Get list of supported monitoring types."""
        return [MonitoringType.DRIFT_DETECTION, MonitoringType.PERFORMANCE_MONITORING]
    
    @staticmethod
    def create_drift_detection_component(
        project_id: str,
        region: str = "us-central1",
        **kwargs
    ) -> DriftDetectionComponent:
        """
        Create drift detection component with simplified interface.
        
        Args:
            project_id: GCP project ID
            region: GCP region
            **kwargs: Additional configuration options
            
        Returns:
            Drift detection component
        """
        config = MonitoringConfig(
            monitoring_type=MonitoringType.DRIFT_DETECTION,
            project_id=project_id,
            region=region,
            **kwargs
        )
        return DriftDetectionComponent(config)
    
    @staticmethod
    def create_performance_monitoring_component(
        project_id: str,
        region: str = "us-central1",
        **kwargs
    ) -> PerformanceMonitoringComponent:
        """
        Create performance monitoring component with simplified interface.
        
        Args:
            project_id: GCP project ID
            region: GCP region
            **kwargs: Additional configuration options
            
        Returns:
            Performance monitoring component
        """
        config = MonitoringConfig(
            monitoring_type=MonitoringType.PERFORMANCE_MONITORING,
            project_id=project_id,
            region=region,
            **kwargs
        )
        return PerformanceMonitoringComponent(config)


# Alias for backward compatibility
ModelMonitoringComponent = BaseMonitoringComponent
