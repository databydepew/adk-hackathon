"""
Vertex AI Integration for Pipeline Agent

This module provides integration with Google Cloud Vertex AI for model
registry, endpoints, and pipeline management.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic
import time

logger = logging.getLogger(__name__)


@dataclass
class VertexAIConfig:
    """Configuration for Vertex AI integration."""
    project_id: str
    region: str = "us-central1"
    staging_bucket: Optional[str] = None
    service_account: Optional[str] = None
    credentials_path: Optional[str] = None


class VertexAIClient:
    """Client for Vertex AI operations."""
    
    def __init__(self, config: VertexAIConfig):
        """
        Initialize Vertex AI client.
        
        Args:
            config: Vertex AI configuration
        """
        self.config = config
        
        # Initialize Vertex AI
        aiplatform.init(
            project=config.project_id,
            location=config.region,
            staging_bucket=config.staging_bucket,
            service_account=config.service_account
        )
        
        logger.info(f"Initialized Vertex AI client for project {config.project_id} in {config.region}")
    
    def list_models(self, filter_str: Optional[str] = None) -> List[aiplatform.Model]:
        """
        List models in Vertex AI Model Registry.
        
        Args:
            filter_str: Optional filter string
            
        Returns:
            List of Model objects
        """
        models = aiplatform.Model.list(filter=filter_str)
        logger.info(f"Found {len(models)} models")
        return models
    
    def get_model(self, model_id: str) -> aiplatform.Model:
        """
        Get a specific model from Model Registry.
        
        Args:
            model_id: Model ID or resource name
            
        Returns:
            Model object
        """
        model = aiplatform.Model(model_id)
        return model
    
    def upload_model(
        self,
        display_name: str,
        artifact_uri: str,
        serving_container_image_uri: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> aiplatform.Model:
        """
        Upload a model to Vertex AI Model Registry.
        
        Args:
            display_name: Display name for the model
            artifact_uri: GCS URI where model artifacts are stored
            serving_container_image_uri: Container image for serving
            description: Optional model description
            labels: Optional labels for the model
            
        Returns:
            Uploaded Model object
        """
        logger.info(f"Uploading model {display_name} from {artifact_uri}")
        
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
            description=description,
            labels=labels
        )
        
        logger.info(f"Model uploaded successfully: {model.resource_name}")
        return model
    
    def list_endpoints(self, filter_str: Optional[str] = None) -> List[aiplatform.Endpoint]:
        """
        List endpoints in Vertex AI.
        
        Args:
            filter_str: Optional filter string
            
        Returns:
            List of Endpoint objects
        """
        endpoints = aiplatform.Endpoint.list(filter=filter_str)
        logger.info(f"Found {len(endpoints)} endpoints")
        return endpoints
    
    def create_endpoint(
        self,
        display_name: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        enable_access_logging: bool = True
    ) -> aiplatform.Endpoint:
        """
        Create a new Vertex AI endpoint.
        
        Args:
            display_name: Display name for the endpoint
            description: Optional endpoint description
            labels: Optional labels for the endpoint
            enable_access_logging: Enable access logging
            
        Returns:
            Created Endpoint object
        """
        logger.info(f"Creating endpoint {display_name}")
        
        endpoint = aiplatform.Endpoint.create(
            display_name=display_name,
            description=description,
            labels=labels,
            enable_access_logging=enable_access_logging
        )
        
        logger.info(f"Endpoint created successfully: {endpoint.resource_name}")
        return endpoint
    
    def deploy_model_to_endpoint(
        self,
        model: aiplatform.Model,
        endpoint: aiplatform.Endpoint,
        deployed_model_display_name: str,
        machine_type: str = "n1-standard-4",
        min_replica_count: int = 1,
        max_replica_count: int = 10,
        traffic_percentage: int = 100
    ) -> aiplatform.Endpoint:
        """
        Deploy a model to an endpoint.
        
        Args:
            model: Model to deploy
            endpoint: Target endpoint
            deployed_model_display_name: Display name for deployed model
            machine_type: Machine type for deployment
            min_replica_count: Minimum number of replicas
            max_replica_count: Maximum number of replicas
            traffic_percentage: Percentage of traffic to route to this model
            
        Returns:
            Updated Endpoint object
        """
        logger.info(f"Deploying model {model.display_name} to endpoint {endpoint.display_name}")
        
        deployed_model = endpoint.deploy(
            model=model,
            deployed_model_display_name=deployed_model_display_name,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_percentage=traffic_percentage,
            sync=True
        )
        
        logger.info(f"Model deployed successfully: {deployed_model.id}")
        return endpoint
    
    def create_batch_prediction_job(
        self,
        job_display_name: str,
        model: aiplatform.Model,
        gcs_source: List[str],
        gcs_destination_prefix: str,
        machine_type: str = "n1-standard-4",
        starting_replica_count: int = 1,
        max_replica_count: int = 10
    ) -> aiplatform.BatchPredictionJob:
        """
        Create a batch prediction job.
        
        Args:
            job_display_name: Display name for the job
            model: Model to use for predictions
            gcs_source: List of GCS input paths
            gcs_destination_prefix: GCS output path prefix
            machine_type: Machine type for the job
            starting_replica_count: Starting number of replicas
            max_replica_count: Maximum number of replicas
            
        Returns:
            BatchPredictionJob object
        """
        logger.info(f"Creating batch prediction job {job_display_name}")
        
        job = aiplatform.BatchPredictionJob.create(
            job_display_name=job_display_name,
            model_name=model.resource_name,
            gcs_source=gcs_source,
            gcs_destination_prefix=gcs_destination_prefix,
            machine_type=machine_type,
            starting_replica_count=starting_replica_count,
            max_replica_count=max_replica_count,
            sync=False
        )
        
        logger.info(f"Batch prediction job created: {job.resource_name}")
        return job
    
    def get_pipeline_job(self, job_id: str) -> aiplatform.PipelineJob:
        """
        Get a pipeline job by ID.
        
        Args:
            job_id: Pipeline job ID
            
        Returns:
            PipelineJob object
        """
        job = aiplatform.PipelineJob.get(job_id)
        return job
    
    def list_pipeline_jobs(self, filter_str: Optional[str] = None) -> List[aiplatform.PipelineJob]:
        """
        List pipeline jobs.
        
        Args:
            filter_str: Optional filter string
            
        Returns:
            List of PipelineJob objects
        """
        jobs = aiplatform.PipelineJob.list(filter=filter_str)
        logger.info(f"Found {len(jobs)} pipeline jobs")
        return jobs


class VertexAIIntegration:
    """High-level Vertex AI integration for ML pipelines."""
    
    def __init__(self, config: VertexAIConfig):
        """
        Initialize Vertex AI integration.
        
        Args:
            config: Vertex AI configuration
        """
        self.config = config
        self.client = VertexAIClient(config)
        logger.info("Vertex AI integration initialized")
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information dictionary
        """
        model = self.client.get_model(model_id)
        
        return {
            "model_id": model.name,
            "display_name": model.display_name,
            "description": model.description,
            "labels": dict(model.labels) if model.labels else {},
            "create_time": model.create_time.isoformat() if model.create_time else None,
            "update_time": model.update_time.isoformat() if model.update_time else None,
            "artifact_uri": model.artifact_uri,
            "serving_container_image_uri": model.container_spec.image_uri if model.container_spec else None,
            "supported_deployment_resources_types": model.supported_deployment_resources_types,
            "supported_input_storage_formats": model.supported_input_storage_formats,
            "supported_output_storage_formats": model.supported_output_storage_formats
        }
    
    def get_endpoint_info(self, endpoint_id: str) -> Dict[str, Any]:
        """
        Get detailed information about an endpoint.
        
        Args:
            endpoint_id: Endpoint ID
            
        Returns:
            Endpoint information dictionary
        """
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        deployed_models = []
        for deployed_model in endpoint.list_models():
            deployed_models.append({
                "id": deployed_model.id,
                "display_name": deployed_model.display_name,
                "model_id": deployed_model.model,
                "create_time": deployed_model.create_time.isoformat() if deployed_model.create_time else None,
                "machine_type": deployed_model.machine_spec.machine_type if deployed_model.machine_spec else None,
                "min_replica_count": deployed_model.min_replica_count,
                "max_replica_count": deployed_model.max_replica_count,
                "traffic_percentage": deployed_model.traffic_percentage
            })
        
        return {
            "endpoint_id": endpoint.name,
            "display_name": endpoint.display_name,
            "description": endpoint.description,
            "labels": dict(endpoint.labels) if endpoint.labels else {},
            "create_time": endpoint.create_time.isoformat() if endpoint.create_time else None,
            "update_time": endpoint.update_time.isoformat() if endpoint.update_time else None,
            "deployed_models": deployed_models,
            "traffic_split": endpoint.traffic_split,
            "enable_access_logging": endpoint.enable_access_logging
        }
    
    def monitor_endpoint_health(self, endpoint_id: str) -> Dict[str, Any]:
        """
        Monitor endpoint health and performance.
        
        Args:
            endpoint_id: Endpoint ID
            
        Returns:
            Health monitoring information
        """
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        # Get basic endpoint status
        health_info = {
            "endpoint_id": endpoint_id,
            "status": "healthy",  # Would implement actual health checks
            "deployed_models_count": len(endpoint.list_models()),
            "traffic_split": endpoint.traffic_split,
            "last_checked": datetime.now().isoformat()
        }
        
        # Check each deployed model
        model_health = []
        for deployed_model in endpoint.list_models():
            model_info = {
                "model_id": deployed_model.id,
                "display_name": deployed_model.display_name,
                "status": "healthy",  # Would implement actual health checks
                "replica_count": deployed_model.min_replica_count,  # Would get actual count
                "traffic_percentage": deployed_model.traffic_percentage
            }
            model_health.append(model_info)
        
        health_info["model_health"] = model_health
        
        return health_info
    
    def get_pipeline_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a pipeline job.
        
        Args:
            job_id: Pipeline job ID
            
        Returns:
            Pipeline status information
        """
        job = self.client.get_pipeline_job(job_id)
        
        return {
            "job_id": job.name,
            "display_name": job.display_name,
            "state": job.state.name if job.state else "UNKNOWN",
            "create_time": job.create_time.isoformat() if job.create_time else None,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None,
            "pipeline_spec": job.pipeline_spec,
            "labels": dict(job.labels) if job.labels else {},
            "error": job.error.message if job.error else None
        }
    
    def cleanup_old_models(
        self, 
        days_old: int = 30, 
        keep_deployed: bool = True,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Clean up old models from Model Registry.
        
        Args:
            days_old: Delete models older than this many days
            keep_deployed: Keep models that are currently deployed
            dry_run: If True, only return what would be deleted
            
        Returns:
            Cleanup results
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        models = self.client.list_models()
        
        models_to_delete = []
        deployed_model_ids = set()
        
        # Get list of deployed models
        if keep_deployed:
            endpoints = self.client.list_endpoints()
            for endpoint in endpoints:
                for deployed_model in endpoint.list_models():
                    deployed_model_ids.add(deployed_model.model)
        
        # Find models to delete
        for model in models:
            if model.create_time and model.create_time.replace(tzinfo=None) < cutoff_date:
                if not keep_deployed or model.resource_name not in deployed_model_ids:
                    models_to_delete.append({
                        "model_id": model.name,
                        "display_name": model.display_name,
                        "create_time": model.create_time.isoformat(),
                        "is_deployed": model.resource_name in deployed_model_ids
                    })
        
        result = {
            "total_models": len(models),
            "models_to_delete": len(models_to_delete),
            "models": models_to_delete,
            "dry_run": dry_run
        }
        
        # Actually delete models if not dry run
        if not dry_run:
            deleted_count = 0
            for model_info in models_to_delete:
                try:
                    model = aiplatform.Model(model_info["model_id"])
                    model.delete()
                    deleted_count += 1
                    logger.info(f"Deleted model {model_info['display_name']}")
                except Exception as e:
                    logger.error(f"Failed to delete model {model_info['display_name']}: {e}")
            
            result["deleted_count"] = deleted_count
        
        return result
    
    def create_model_version(
        self,
        parent_model_id: str,
        artifact_uri: str,
        version_description: Optional[str] = None
    ) -> aiplatform.Model:
        """
        Create a new version of an existing model.
        
        Args:
            parent_model_id: ID of the parent model
            artifact_uri: GCS URI for the new model artifacts
            version_description: Optional description for the version
            
        Returns:
            New Model version object
        """
        parent_model = self.client.get_model(parent_model_id)
        
        # Create new version with same container spec as parent
        new_model = aiplatform.Model.upload(
            display_name=f"{parent_model.display_name}_v{int(time.time())}",
            artifact_uri=artifact_uri,
            serving_container_image_uri=parent_model.container_spec.image_uri,
            description=version_description or f"New version of {parent_model.display_name}",
            labels=parent_model.labels,
            parent_model=parent_model.resource_name
        )
        
        logger.info(f"Created new model version: {new_model.resource_name}")
        return new_model
