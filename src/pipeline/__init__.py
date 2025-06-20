"""
Vertex AI Pipeline Agent - Pipeline Components Module

This module contains the pipeline generation components for creating
Vertex AI ML pipelines including data ingestion, training, deployment, and monitoring.
"""

from .data_ingestion import (
    BigQueryDataIngestionComponent,
    GCSDataIngestionComponent,
    DataIngestionComponentFactory
)
from .training import (
    ModelTrainingComponent,
    SklearnTrainingComponent,
    XGBoostTrainingComponent,
    TensorFlowTrainingComponent,
    TrainingComponentFactory
)
from .deployment import (
    ModelDeploymentComponent,
    EndpointDeploymentComponent,
    BatchDeploymentComponent,
    DeploymentComponentFactory
)
from .monitoring import (
    ModelMonitoringComponent,
    DriftDetectionComponent,
    PerformanceMonitoringComponent,
    MonitoringComponentFactory
)

__all__ = [
    # Data Ingestion
    "BigQueryDataIngestionComponent",
    "GCSDataIngestionComponent", 
    "DataIngestionComponentFactory",
    
    # Training
    "ModelTrainingComponent",
    "SklearnTrainingComponent",
    "XGBoostTrainingComponent",
    "TensorFlowTrainingComponent",
    "TrainingComponentFactory",
    
    # Deployment
    "ModelDeploymentComponent",
    "EndpointDeploymentComponent",
    "BatchDeploymentComponent",
    "DeploymentComponentFactory",
    
    # Monitoring
    "ModelMonitoringComponent",
    "DriftDetectionComponent",
    "PerformanceMonitoringComponent",
    "MonitoringComponentFactory",
]

__version__ = "1.0.0"
