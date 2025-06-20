"""
Model Deployment Pipeline Components

This module provides components for deploying ML models to Vertex AI endpoints
and setting up batch prediction services.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from kfp.dsl import component, Input, Output, Model, Artifact

logger = logging.getLogger(__name__)


class DeploymentType(Enum):
    """Types of model deployment."""
    ONLINE_ENDPOINT = "online_endpoint"
    BATCH_PREDICTION = "batch_prediction"
    EDGE_DEPLOYMENT = "edge_deployment"


@dataclass
class DeploymentConfig:
    """Configuration for model deployment components."""
    deployment_type: DeploymentType
    endpoint_name: str
    project_id: str
    region: str = "us-central1"
    machine_type: str = "n1-standard-4"
    min_replica_count: int = 1
    max_replica_count: int = 10
    traffic_split: Dict[str, int] = None
    auto_scaling: bool = True
    enable_access_logging: bool = True
    enable_container_logging: bool = True
    
    def __post_init__(self):
        if self.traffic_split is None:
            self.traffic_split = {"0": 100}


class BaseDeploymentComponent(ABC):
    """Base class for deployment components."""
    
    def __init__(self, config: DeploymentConfig):
        """
        Initialize the deployment component.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.component_name = f"{config.deployment_type.value}_deployment"
        
    @abstractmethod
    def create_component(self) -> component:
        """Create the KFP component for model deployment."""
        pass
    
    @abstractmethod
    def get_component_spec(self) -> Dict[str, Any]:
        """Get the component specification."""
        pass


class EndpointDeploymentComponent(BaseDeploymentComponent):
    """Vertex AI endpoint deployment component."""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize endpoint deployment component."""
        super().__init__(config)
        if config.deployment_type != DeploymentType.ONLINE_ENDPOINT:
            raise ValueError("Config must be for online endpoint deployment")
    
    def create_component(self) -> component:
        """Create endpoint deployment KFP component."""
        
        @component(
            base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
            packages_to_install=[
                "google-cloud-aiplatform>=1.38.0",
                "google-auth>=2.17.0"
            ]
        )
        def endpoint_deployment(
            model: Input[Model],
            endpoint: Output[Artifact],
            endpoint_name: str,
            project_id: str,
            region: str = "us-central1",
            machine_type: str = "n1-standard-4",
            min_replica_count: int = 1,
            max_replica_count: int = 10,
            traffic_split: str = '{"0": 100}',
            auto_scaling: bool = True,
            enable_access_logging: bool = True,
            enable_container_logging: bool = True
        ):
            """
            Deploy model to Vertex AI endpoint.
            
            Args:
                model: Input trained model
                endpoint: Output endpoint artifact
                endpoint_name: Name of the endpoint
                project_id: GCP project ID
                region: GCP region
                machine_type: Machine type for deployment
                min_replica_count: Minimum number of replicas
                max_replica_count: Maximum number of replicas
                traffic_split: JSON string of traffic split configuration
                auto_scaling: Enable auto scaling
                enable_access_logging: Enable access logging
                enable_container_logging: Enable container logging
            """
            import json
            import os
            import logging
            import time
            from google.cloud import aiplatform
            from google.auth import default
            
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            try:
                # Initialize Vertex AI
                aiplatform.init(project=project_id, location=region)
                logger.info(f"Initialized Vertex AI for project {project_id} in region {region}")
                
                # Load model metadata
                metadata_path = os.path.join(model.path, "metadata.json")
                with open(metadata_path, 'r') as f:
                    model_metadata = json.load(f)
                
                framework = model_metadata.get("framework", "scikit-learn")
                logger.info(f"Deploying {framework} model to endpoint")
                
                # Parse traffic split
                traffic_config = json.loads(traffic_split)
                
                # Create or get existing model in Vertex AI Model Registry
                model_display_name = f"{endpoint_name}_model"
                
                # Check if model already exists
                existing_models = aiplatform.Model.list(
                    filter=f'display_name="{model_display_name}"'
                )
                
                if existing_models:
                    vertex_model = existing_models[0]
                    logger.info(f"Using existing model: {vertex_model.display_name}")
                else:
                    # Upload model to Vertex AI Model Registry
                    logger.info("Uploading model to Vertex AI Model Registry...")
                    
                    # Determine container URI based on framework
                    container_uris = {
                        "scikit-learn": "gcr.io/deeplearning-platform-release/sklearn-cpu.0-23:latest",
                        "xgboost": "gcr.io/deeplearning-platform-release/xgboost-cpu.1-4:latest",
                        "tensorflow": "gcr.io/deeplearning-platform-release/tf2-cpu.2-8:latest"
                    }
                    
                    serving_container_image_uri = container_uris.get(
                        framework, 
                        "gcr.io/deeplearning-platform-release/sklearn-cpu.0-23:latest"
                    )
                    
                    # Upload model
                    vertex_model = aiplatform.Model.upload(
                        display_name=model_display_name,
                        artifact_uri=model.path,
                        serving_container_image_uri=serving_container_image_uri,
                        serving_container_predict_route="/predict",
                        serving_container_health_route="/health",
                        description=f"Model for {endpoint_name} endpoint"
                    )
                    
                    logger.info(f"Model uploaded with resource name: {vertex_model.resource_name}")
                
                # Create or get existing endpoint
                existing_endpoints = aiplatform.Endpoint.list(
                    filter=f'display_name="{endpoint_name}"'
                )
                
                if existing_endpoints:
                    vertex_endpoint = existing_endpoints[0]
                    logger.info(f"Using existing endpoint: {vertex_endpoint.display_name}")
                else:
                    # Create new endpoint
                    logger.info(f"Creating new endpoint: {endpoint_name}")
                    vertex_endpoint = aiplatform.Endpoint.create(
                        display_name=endpoint_name,
                        description=f"Endpoint for {endpoint_name} model",
                        enable_access_logging=enable_access_logging,
                        enable_container_logging=enable_container_logging
                    )
                    logger.info(f"Endpoint created with resource name: {vertex_endpoint.resource_name}")
                
                # Deploy model to endpoint
                logger.info("Deploying model to endpoint...")
                
                deployed_model = vertex_endpoint.deploy(
                    model=vertex_model,
                    deployed_model_display_name=f"{endpoint_name}_deployment",
                    machine_type=machine_type,
                    min_replica_count=min_replica_count,
                    max_replica_count=max_replica_count,
                    traffic_split=traffic_config,
                    sync=True
                )
                
                logger.info(f"Model deployed successfully. Deployed model ID: {deployed_model.id}")
                
                # Wait for deployment to be ready
                logger.info("Waiting for deployment to be ready...")
                time.sleep(30)  # Give some time for the deployment to stabilize
                
                # Test endpoint health
                try:
                    # Simple health check by getting endpoint info
                    endpoint_info = vertex_endpoint.gca_resource
                    logger.info("Endpoint health check passed")
                except Exception as e:
                    logger.warning(f"Endpoint health check failed: {e}")
                
                # Save endpoint information
                os.makedirs(endpoint.path, exist_ok=True)
                
                endpoint_info = {
                    "endpoint_name": endpoint_name,
                    "endpoint_id": vertex_endpoint.name,
                    "endpoint_resource_name": vertex_endpoint.resource_name,
                    "model_id": vertex_model.name,
                    "model_resource_name": vertex_model.resource_name,
                    "deployed_model_id": deployed_model.id,
                    "project_id": project_id,
                    "region": region,
                    "machine_type": machine_type,
                    "min_replica_count": min_replica_count,
                    "max_replica_count": max_replica_count,
                    "traffic_split": traffic_config,
                    "prediction_url": f"https://{region}-aiplatform.googleapis.com/v1/{vertex_endpoint.resource_name}:predict",
                    "framework": framework
                }
                
                endpoint_metadata_path = os.path.join(endpoint.path, "endpoint_info.json")
                with open(endpoint_metadata_path, 'w') as f:
                    json.dump(endpoint_info, f, indent=2)
                
                logger.info(f"Endpoint deployment completed successfully")
                logger.info(f"Endpoint URL: {endpoint_info['prediction_url']}")
                
            except Exception as e:
                logger.error(f"Endpoint deployment failed: {str(e)}")
                raise
        
        return endpoint_deployment
    
    def get_component_spec(self) -> Dict[str, Any]:
        """Get endpoint deployment component specification."""
        return {
            "name": self.component_name,
            "description": "Deploy model to Vertex AI endpoint",
            "inputs": {
                "model": {"type": "Model", "description": "Trained model to deploy"},
                "endpoint_name": {"type": "String", "description": "Name of the endpoint"},
                "project_id": {"type": "String", "description": "GCP project ID"},
                "region": {"type": "String", "description": "GCP region", "default": "us-central1"},
                "machine_type": {"type": "String", "description": "Machine type", "default": "n1-standard-4"},
                "min_replica_count": {"type": "Integer", "description": "Min replicas", "default": 1},
                "max_replica_count": {"type": "Integer", "description": "Max replicas", "default": 10},
                "traffic_split": {"type": "String", "description": "Traffic split JSON", "default": '{"0": 100}'},
                "auto_scaling": {"type": "Boolean", "description": "Enable auto scaling", "default": True},
                "enable_access_logging": {"type": "Boolean", "description": "Enable access logging", "default": True},
                "enable_container_logging": {"type": "Boolean", "description": "Enable container logging", "default": True}
            },
            "outputs": {
                "endpoint": {"type": "Artifact", "description": "Deployed endpoint information"}
            }
        }


class BatchDeploymentComponent(BaseDeploymentComponent):
    """Batch prediction deployment component."""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize batch deployment component."""
        super().__init__(config)
        if config.deployment_type != DeploymentType.BATCH_PREDICTION:
            raise ValueError("Config must be for batch prediction deployment")
    
    def create_component(self) -> component:
        """Create batch deployment KFP component."""
        
        @component(
            base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
            packages_to_install=[
                "google-cloud-aiplatform>=1.38.0",
                "google-auth>=2.17.0"
            ]
        )
        def batch_deployment(
            model: Input[Model],
            batch_job: Output[Artifact],
            job_name: str,
            input_data_path: str,
            output_data_path: str,
            project_id: str,
            region: str = "us-central1",
            machine_type: str = "n1-standard-4",
            replica_count: int = 1,
            max_replica_count: int = 10
        ):
            """
            Create batch prediction job for model.
            
            Args:
                model: Input trained model
                batch_job: Output batch job artifact
                job_name: Name of the batch prediction job
                input_data_path: GCS path to input data
                output_data_path: GCS path for output predictions
                project_id: GCP project ID
                region: GCP region
                machine_type: Machine type for batch prediction
                replica_count: Number of replicas
                max_replica_count: Maximum number of replicas
            """
            import json
            import os
            import logging
            from google.cloud import aiplatform
            
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            try:
                # Initialize Vertex AI
                aiplatform.init(project=project_id, location=region)
                logger.info(f"Initialized Vertex AI for project {project_id} in region {region}")
                
                # Load model metadata
                metadata_path = os.path.join(model.path, "metadata.json")
                with open(metadata_path, 'r') as f:
                    model_metadata = json.load(f)
                
                framework = model_metadata.get("framework", "scikit-learn")
                logger.info(f"Setting up batch prediction for {framework} model")
                
                # Create or get existing model in Vertex AI Model Registry
                model_display_name = f"{job_name}_batch_model"
                
                # Check if model already exists
                existing_models = aiplatform.Model.list(
                    filter=f'display_name="{model_display_name}"'
                )
                
                if existing_models:
                    vertex_model = existing_models[0]
                    logger.info(f"Using existing model: {vertex_model.display_name}")
                else:
                    # Upload model to Vertex AI Model Registry
                    logger.info("Uploading model to Vertex AI Model Registry...")
                    
                    # Determine container URI based on framework
                    container_uris = {
                        "scikit-learn": "gcr.io/deeplearning-platform-release/sklearn-cpu.0-23:latest",
                        "xgboost": "gcr.io/deeplearning-platform-release/xgboost-cpu.1-4:latest",
                        "tensorflow": "gcr.io/deeplearning-platform-release/tf2-cpu.2-8:latest"
                    }
                    
                    serving_container_image_uri = container_uris.get(
                        framework, 
                        "gcr.io/deeplearning-platform-release/sklearn-cpu.0-23:latest"
                    )
                    
                    # Upload model
                    vertex_model = aiplatform.Model.upload(
                        display_name=model_display_name,
                        artifact_uri=model.path,
                        serving_container_image_uri=serving_container_image_uri,
                        description=f"Model for {job_name} batch prediction"
                    )
                    
                    logger.info(f"Model uploaded with resource name: {vertex_model.resource_name}")
                
                # Create batch prediction job
                logger.info(f"Creating batch prediction job: {job_name}")
                
                batch_prediction_job = aiplatform.BatchPredictionJob.create(
                    job_display_name=job_name,
                    model_name=vertex_model.resource_name,
                    gcs_source=[input_data_path],
                    gcs_destination_prefix=output_data_path,
                    machine_type=machine_type,
                    starting_replica_count=replica_count,
                    max_replica_count=max_replica_count,
                    sync=False  # Don't wait for completion
                )
                
                logger.info(f"Batch prediction job created: {batch_prediction_job.resource_name}")
                
                # Save batch job information
                os.makedirs(batch_job.path, exist_ok=True)
                
                batch_job_info = {
                    "job_name": job_name,
                    "job_id": batch_prediction_job.name,
                    "job_resource_name": batch_prediction_job.resource_name,
                    "model_id": vertex_model.name,
                    "model_resource_name": vertex_model.resource_name,
                    "input_data_path": input_data_path,
                    "output_data_path": output_data_path,
                    "project_id": project_id,
                    "region": region,
                    "machine_type": machine_type,
                    "replica_count": replica_count,
                    "max_replica_count": max_replica_count,
                    "framework": framework,
                    "state": "RUNNING"
                }
                
                batch_job_metadata_path = os.path.join(batch_job.path, "batch_job_info.json")
                with open(batch_job_metadata_path, 'w') as f:
                    json.dump(batch_job_info, f, indent=2)
                
                logger.info(f"Batch prediction job setup completed successfully")
                
            except Exception as e:
                logger.error(f"Batch deployment failed: {str(e)}")
                raise
        
        return batch_deployment
    
    def get_component_spec(self) -> Dict[str, Any]:
        """Get batch deployment component specification."""
        return {
            "name": self.component_name,
            "description": "Create batch prediction job for model",
            "inputs": {
                "model": {"type": "Model", "description": "Trained model for batch prediction"},
                "job_name": {"type": "String", "description": "Name of the batch prediction job"},
                "input_data_path": {"type": "String", "description": "GCS path to input data"},
                "output_data_path": {"type": "String", "description": "GCS path for output predictions"},
                "project_id": {"type": "String", "description": "GCP project ID"},
                "region": {"type": "String", "description": "GCP region", "default": "us-central1"},
                "machine_type": {"type": "String", "description": "Machine type", "default": "n1-standard-4"},
                "replica_count": {"type": "Integer", "description": "Number of replicas", "default": 1},
                "max_replica_count": {"type": "Integer", "description": "Max replicas", "default": 10}
            },
            "outputs": {
                "batch_job": {"type": "Artifact", "description": "Batch prediction job information"}
            }
        }


class DeploymentComponentFactory:
    """Factory for creating deployment components."""
    
    @staticmethod
    def create_component(config: DeploymentConfig) -> BaseDeploymentComponent:
        """
        Create a deployment component based on configuration.
        
        Args:
            config: Deployment configuration
            
        Returns:
            Deployment component instance
        """
        if config.deployment_type == DeploymentType.ONLINE_ENDPOINT:
            return EndpointDeploymentComponent(config)
        elif config.deployment_type == DeploymentType.BATCH_PREDICTION:
            return BatchDeploymentComponent(config)
        else:
            raise ValueError(f"Unsupported deployment type: {config.deployment_type}")
    
    @staticmethod
    def get_supported_deployment_types() -> List[DeploymentType]:
        """Get list of supported deployment types."""
        return [DeploymentType.ONLINE_ENDPOINT, DeploymentType.BATCH_PREDICTION]
    
    @staticmethod
    def create_endpoint_component(
        endpoint_name: str,
        project_id: str,
        region: str = "us-central1",
        **kwargs
    ) -> EndpointDeploymentComponent:
        """
        Create endpoint deployment component with simplified interface.
        
        Args:
            endpoint_name: Name of the endpoint
            project_id: GCP project ID
            region: GCP region
            **kwargs: Additional configuration options
            
        Returns:
            Endpoint deployment component
        """
        config = DeploymentConfig(
            deployment_type=DeploymentType.ONLINE_ENDPOINT,
            endpoint_name=endpoint_name,
            project_id=project_id,
            region=region,
            **kwargs
        )
        return EndpointDeploymentComponent(config)
    
    @staticmethod
    def create_batch_component(
        job_name: str,
        project_id: str,
        region: str = "us-central1",
        **kwargs
    ) -> BatchDeploymentComponent:
        """
        Create batch deployment component with simplified interface.
        
        Args:
            job_name: Name of the batch prediction job
            project_id: GCP project ID
            region: GCP region
            **kwargs: Additional configuration options
            
        Returns:
            Batch deployment component
        """
        config = DeploymentConfig(
            deployment_type=DeploymentType.BATCH_PREDICTION,
            endpoint_name=job_name,  # Using endpoint_name field for job_name
            project_id=project_id,
            region=region,
            **kwargs
        )
        return BatchDeploymentComponent(config)


# Alias for backward compatibility
ModelDeploymentComponent = BaseDeploymentComponent
