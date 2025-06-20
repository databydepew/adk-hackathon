"""
CI/CD Automation for ML Pipelines

This module provides CI/CD automation capabilities for ML pipeline
deployment, testing, and continuous integration workflows.
"""

import logging
import json
import yaml
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from google.cloud import aiplatform
from google.cloud import build_v1
from google.cloud import storage
from google.cloud import pubsub_v1

logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Deployment stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CICDConfig:
    """Configuration for CI/CD automation."""
    project_id: str
    region: str = "us-central1"
    
    # Repository settings
    source_repo_url: str = ""
    branch: str = "main"
    
    # Build settings
    enable_automated_builds: bool = True
    build_trigger_name: str = "ml-pipeline-build"
    build_config_path: str = "cloudbuild.yaml"
    
    # Testing settings
    enable_pipeline_testing: bool = True
    test_data_path: str = ""
    test_pipeline_template: str = ""
    
    # Deployment settings
    enable_automated_deployment: bool = True
    deployment_stages: List[str] = None
    approval_required_for_prod: bool = True
    
    # Monitoring settings
    enable_deployment_monitoring: bool = True
    rollback_on_failure: bool = True
    health_check_timeout_minutes: int = 30
    
    # Notification settings
    enable_notifications: bool = True
    notification_topic: Optional[str] = None
    
    def __post_init__(self):
        if self.deployment_stages is None:
            self.deployment_stages = ["development", "staging", "production"]


class CICDAutomation:
    """CI/CD automation system for ML pipelines."""
    
    def __init__(self, config: CICDConfig):
        """
        Initialize CI/CD automation system.
        
        Args:
            config: CI/CD configuration
        """
        self.config = config
        
        # Initialize clients
        aiplatform.init(project=config.project_id, location=config.region)
        self.build_client = build_v1.CloudBuildClient()
        self.storage_client = storage.Client(project=config.project_id)
        
        if config.enable_notifications and config.notification_topic:
            self.publisher = pubsub_v1.PublisherClient()
            self.topic_path = self.publisher.topic_path(
                config.project_id, config.notification_topic
            )
        else:
            self.publisher = None
            self.topic_path = None
        
        logger.info(f"Initialized CI/CD automation for project {config.project_id}")
    
    def create_build_trigger(self) -> Dict[str, Any]:
        """
        Create a Cloud Build trigger for automated pipeline builds.
        
        Returns:
            Build trigger information
        """
        logger.info("Creating Cloud Build trigger for ML pipelines")
        
        try:
            # Define build trigger
            trigger = build_v1.BuildTrigger(
                name=self.config.build_trigger_name,
                description="Automated build trigger for ML pipelines",
                github=build_v1.GitHubEventsConfig(
                    owner="egen",  # Would be configured
                    name="ml-pipelines",  # Would be configured
                    push=build_v1.PushFilter(branch=self.config.branch)
                ),
                filename=self.config.build_config_path,
                substitutions={
                    "_PROJECT_ID": self.config.project_id,
                    "_REGION": self.config.region
                }
            )
            
            # Create trigger
            parent = f"projects/{self.config.project_id}"
            created_trigger = self.build_client.create_build_trigger(
                parent=parent,
                build_trigger=trigger
            )
            
            trigger_info = {
                "trigger_id": created_trigger.id,
                "trigger_name": created_trigger.name,
                "description": created_trigger.description,
                "status": "created",
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"Build trigger created: {created_trigger.id}")
            return trigger_info
            
        except Exception as e:
            logger.error(f"Failed to create build trigger: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_cloudbuild_config(self) -> str:
        """
        Generate Cloud Build configuration for ML pipeline CI/CD.
        
        Returns:
            YAML configuration string
        """
        logger.info("Generating Cloud Build configuration")
        
        build_config = {
            "steps": [
                # Step 1: Install dependencies
                {
                    "name": "python:3.9",
                    "entrypoint": "pip",
                    "args": ["install", "-r", "requirements.txt"]
                },
                
                # Step 2: Run pipeline validation
                {
                    "name": "python:3.9",
                    "entrypoint": "python",
                    "args": ["-m", "pytest", "tests/", "-v"],
                    "env": [
                        "PROJECT_ID=${PROJECT_ID}",
                        "REGION=${_REGION}"
                    ]
                },
                
                # Step 3: Compile pipeline
                {
                    "name": "python:3.9",
                    "entrypoint": "python",
                    "args": ["scripts/compile_pipeline.py"],
                    "env": [
                        "PROJECT_ID=${PROJECT_ID}",
                        "REGION=${_REGION}"
                    ]
                },
                
                # Step 4: Deploy to development
                {
                    "name": "python:3.9",
                    "entrypoint": "python",
                    "args": ["scripts/deploy_pipeline.py", "--stage=development"],
                    "env": [
                        "PROJECT_ID=${PROJECT_ID}",
                        "REGION=${_REGION}"
                    ]
                },
                
                # Step 5: Run integration tests
                {
                    "name": "python:3.9",
                    "entrypoint": "python",
                    "args": ["scripts/run_integration_tests.py"],
                    "env": [
                        "PROJECT_ID=${PROJECT_ID}",
                        "REGION=${_REGION}",
                        "STAGE=development"
                    ]
                }
            ],
            
            "substitutions": {
                "_REGION": self.config.region
            },
            
            "options": {
                "logging": "CLOUD_LOGGING_ONLY",
                "machineType": "N1_HIGHCPU_8"
            },
            
            "timeout": "1800s"
        }
        
        # Add conditional production deployment
        if not self.config.approval_required_for_prod:
            build_config["steps"].extend([
                {
                    "name": "python:3.9",
                    "entrypoint": "python",
                    "args": ["scripts/deploy_pipeline.py", "--stage=production"],
                    "env": [
                        "PROJECT_ID=${PROJECT_ID}",
                        "REGION=${_REGION}"
                    ]
                }
            ])
        
        return yaml.dump(build_config, default_flow_style=False)
    
    def deploy_pipeline(
        self,
        pipeline_template_path: str,
        stage: str,
        pipeline_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Deploy a pipeline to a specific stage.
        
        Args:
            pipeline_template_path: Path to pipeline template
            stage: Deployment stage (development, staging, production)
            pipeline_parameters: Optional pipeline parameters
            
        Returns:
            Deployment information
        """
        logger.info(f"Deploying pipeline to {stage} stage")
        
        try:
            # Validate stage
            if stage not in self.config.deployment_stages:
                raise ValueError(f"Invalid deployment stage: {stage}")
            
            # Check approval for production
            if stage == "production" and self.config.approval_required_for_prod:
                approval_status = self._check_production_approval()
                if not approval_status["approved"]:
                    return {
                        "status": "pending_approval",
                        "message": "Production deployment requires approval",
                        "approval_required": True,
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Prepare pipeline parameters
            final_parameters = pipeline_parameters or {}
            final_parameters.update({
                "deployment_stage": stage,
                "project_id": self.config.project_id,
                "region": self.config.region
            })
            
            # Create pipeline job
            job_display_name = f"pipeline-{stage}-{int(datetime.now().timestamp())}"
            
            job = aiplatform.PipelineJob(
                display_name=job_display_name,
                template_path=pipeline_template_path,
                parameter_values=final_parameters,
                enable_caching=False
            )
            
            job.submit()
            
            deployment_info = {
                "status": "deployed",
                "job_id": job.name,
                "job_display_name": job_display_name,
                "stage": stage,
                "pipeline_template": pipeline_template_path,
                "parameters": final_parameters,
                "deployed_at": datetime.now().isoformat()
            }
            
            logger.info(f"Pipeline deployed to {stage}: {job.name}")
            
            # Send notification
            if self.config.enable_notifications:
                self._send_deployment_notification(deployment_info)
            
            # Start monitoring if enabled
            if self.config.enable_deployment_monitoring:
                self._start_deployment_monitoring(deployment_info)
            
            return deployment_info
            
        except Exception as e:
            logger.error(f"Failed to deploy pipeline to {stage}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "stage": stage,
                "timestamp": datetime.now().isoformat()
            }
    
    def run_pipeline_tests(
        self,
        test_pipeline_template: str,
        test_data_path: str
    ) -> Dict[str, Any]:
        """
        Run automated tests on a pipeline.
        
        Args:
            test_pipeline_template: Path to test pipeline template
            test_data_path: Path to test data
            
        Returns:
            Test results
        """
        logger.info("Running pipeline tests")
        
        try:
            # Prepare test parameters
            test_parameters = {
                "test_data_path": test_data_path,
                "project_id": self.config.project_id,
                "region": self.config.region,
                "test_mode": True
            }
            
            # Create test pipeline job
            job_display_name = f"pipeline-test-{int(datetime.now().timestamp())}"
            
            job = aiplatform.PipelineJob(
                display_name=job_display_name,
                template_path=test_pipeline_template,
                parameter_values=test_parameters,
                enable_caching=False
            )
            
            job.submit()
            job.wait()  # Wait for completion
            
            # Analyze test results
            test_results = self._analyze_test_results(job)
            
            logger.info(f"Pipeline tests completed. Status: {test_results['status']}")
            return test_results
            
        except Exception as e:
            logger.error(f"Pipeline tests failed: {e}")
            return {
                "status": "failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def rollback_deployment(
        self,
        stage: str,
        previous_version: str
    ) -> Dict[str, Any]:
        """
        Rollback a deployment to a previous version.
        
        Args:
            stage: Deployment stage to rollback
            previous_version: Previous version to rollback to
            
        Returns:
            Rollback information
        """
        logger.info(f"Rolling back {stage} deployment to version {previous_version}")
        
        try:
            # Implementation would restore previous pipeline version
            # This is a simplified placeholder
            
            rollback_info = {
                "status": "rolled_back",
                "stage": stage,
                "previous_version": previous_version,
                "rollback_reason": "automated_rollback_on_failure",
                "rolled_back_at": datetime.now().isoformat()
            }
            
            logger.info(f"Rollback completed for {stage}")
            
            # Send notification
            if self.config.enable_notifications:
                self._send_rollback_notification(rollback_info)
            
            return rollback_info
            
        except Exception as e:
            logger.error(f"Rollback failed for {stage}: {e}")
            return {
                "status": "rollback_failed",
                "message": str(e),
                "stage": stage,
                "timestamp": datetime.now().isoformat()
            }
    
    def monitor_deployment_health(
        self,
        deployment_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Monitor the health of a deployed pipeline.
        
        Args:
            deployment_info: Deployment information
            
        Returns:
            Health monitoring results
        """
        logger.info(f"Monitoring deployment health for {deployment_info['stage']}")
        
        try:
            job_id = deployment_info["job_id"]
            job = aiplatform.PipelineJob.get(job_id)
            
            # Check pipeline status
            pipeline_status = job.state.name if job.state else "UNKNOWN"
            
            # Perform health checks
            health_checks = {
                "pipeline_status": pipeline_status,
                "pipeline_healthy": pipeline_status == "PIPELINE_STATE_SUCCEEDED",
                "error_message": job.error.message if job.error else None,
                "start_time": job.start_time.isoformat() if job.start_time else None,
                "end_time": job.end_time.isoformat() if job.end_time else None
            }
            
            # Check if rollback is needed
            if (self.config.rollback_on_failure and 
                pipeline_status == "PIPELINE_STATE_FAILED"):
                
                logger.warning(f"Pipeline failed, initiating rollback for {deployment_info['stage']}")
                rollback_result = self.rollback_deployment(
                    deployment_info["stage"],
                    "previous_stable_version"
                )
                health_checks["rollback_initiated"] = True
                health_checks["rollback_result"] = rollback_result
            
            health_results = {
                "deployment_stage": deployment_info["stage"],
                "job_id": job_id,
                "health_checks": health_checks,
                "overall_health": "healthy" if health_checks["pipeline_healthy"] else "unhealthy",
                "monitored_at": datetime.now().isoformat()
            }
            
            return health_results
            
        except Exception as e:
            logger.error(f"Health monitoring failed: {e}")
            return {
                "status": "monitoring_error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _check_production_approval(self) -> Dict[str, Any]:
        """Check if production deployment is approved."""
        # Simplified implementation - would integrate with approval system
        return {
            "approved": False,
            "approval_required": True,
            "message": "Manual approval required for production deployment"
        }
    
    def _analyze_test_results(self, job: aiplatform.PipelineJob) -> Dict[str, Any]:
        """Analyze pipeline test results."""
        pipeline_status = job.state.name if job.state else "UNKNOWN"
        
        return {
            "status": "passed" if pipeline_status == "PIPELINE_STATE_SUCCEEDED" else "failed",
            "job_id": job.name,
            "pipeline_status": pipeline_status,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None,
            "error_message": job.error.message if job.error else None,
            "test_completed_at": datetime.now().isoformat()
        }
    
    def _start_deployment_monitoring(self, deployment_info: Dict[str, Any]):
        """Start monitoring a deployment."""
        logger.info(f"Starting deployment monitoring for {deployment_info['stage']}")
        
        # In a real implementation, this would set up monitoring jobs
        # For now, just log the monitoring start
        pass
    
    def _send_deployment_notification(self, deployment_info: Dict[str, Any]):
        """Send deployment notification."""
        if not self.publisher or not self.topic_path:
            return
        
        try:
            notification = {
                "notification_type": "deployment_completed",
                "deployment_info": deployment_info,
                "timestamp": datetime.now().isoformat()
            }
            
            message_data = json.dumps(notification).encode('utf-8')
            future = self.publisher.publish(self.topic_path, message_data)
            message_id = future.result()
            
            logger.info(f"Deployment notification sent: {message_id}")
            
        except Exception as e:
            logger.error(f"Failed to send deployment notification: {e}")
    
    def _send_rollback_notification(self, rollback_info: Dict[str, Any]):
        """Send rollback notification."""
        if not self.publisher or not self.topic_path:
            return
        
        try:
            notification = {
                "notification_type": "deployment_rolled_back",
                "rollback_info": rollback_info,
                "timestamp": datetime.now().isoformat()
            }
            
            message_data = json.dumps(notification).encode('utf-8')
            future = self.publisher.publish(self.topic_path, message_data)
            message_id = future.result()
            
            logger.info(f"Rollback notification sent: {message_id}")
            
        except Exception as e:
            logger.error(f"Failed to send rollback notification: {e}")
    
    def create_deployment_pipeline(self) -> str:
        """
        Create a comprehensive deployment pipeline configuration.
        
        Returns:
            Pipeline configuration as YAML string
        """
        logger.info("Creating deployment pipeline configuration")
        
        pipeline_config = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Workflow",
            "metadata": {
                "name": "ml-pipeline-deployment",
                "annotations": {
                    "description": "Automated ML pipeline deployment workflow"
                }
            },
            "spec": {
                "entrypoint": "deploy-pipeline",
                "arguments": {
                    "parameters": [
                        {"name": "pipeline-template", "value": ""},
                        {"name": "deployment-stage", "value": "development"},
                        {"name": "project-id", "value": self.config.project_id},
                        {"name": "region", "value": self.config.region}
                    ]
                },
                "templates": [
                    {
                        "name": "deploy-pipeline",
                        "dag": {
                            "tasks": [
                                {
                                    "name": "validate-pipeline",
                                    "template": "validate-template"
                                },
                                {
                                    "name": "run-tests",
                                    "template": "run-pipeline-tests",
                                    "dependencies": ["validate-pipeline"]
                                },
                                {
                                    "name": "deploy-to-stage",
                                    "template": "deploy-template",
                                    "dependencies": ["run-tests"]
                                },
                                {
                                    "name": "monitor-deployment",
                                    "template": "monitor-template",
                                    "dependencies": ["deploy-to-stage"]
                                }
                            ]
                        }
                    },
                    {
                        "name": "validate-template",
                        "container": {
                            "image": "python:3.9",
                            "command": ["python"],
                            "args": ["scripts/validate_pipeline.py"]
                        }
                    },
                    {
                        "name": "run-pipeline-tests",
                        "container": {
                            "image": "python:3.9",
                            "command": ["python"],
                            "args": ["scripts/run_tests.py"]
                        }
                    },
                    {
                        "name": "deploy-template",
                        "container": {
                            "image": "python:3.9",
                            "command": ["python"],
                            "args": ["scripts/deploy.py"]
                        }
                    },
                    {
                        "name": "monitor-template",
                        "container": {
                            "image": "python:3.9",
                            "command": ["python"],
                            "args": ["scripts/monitor.py"]
                        }
                    }
                ]
            }
        }
        
        return yaml.dump(pipeline_config, default_flow_style=False)
