"""
Automated Retraining System

This module provides automated retraining triggers based on drift detection,
performance degradation, and scheduled retraining policies.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from google.cloud import aiplatform
from google.cloud import pubsub_v1
from google.cloud import scheduler_v1

logger = logging.getLogger(__name__)


class RetrainingTrigger(Enum):
    """Types of retraining triggers."""
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED = "scheduled"
    DATA_VOLUME_THRESHOLD = "data_volume_threshold"
    MANUAL = "manual"


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining."""
    project_id: str
    region: str = "us-central1"
    
    # Drift-based triggers
    enable_drift_retraining: bool = True
    drift_threshold: float = 0.1
    drift_retraining_delay_hours: int = 24
    
    # Performance-based triggers
    enable_performance_retraining: bool = True
    performance_threshold: float = 0.05
    performance_retraining_delay_hours: int = 12
    
    # Scheduled retraining
    enable_scheduled_retraining: bool = True
    retraining_schedule: str = "weekly"  # daily, weekly, monthly
    
    # Data volume triggers
    enable_data_volume_retraining: bool = False
    data_volume_threshold: int = 10000
    
    # Retraining pipeline configuration
    pipeline_template_path: str = ""
    pipeline_parameters: Dict[str, Any] = None
    
    # Notification settings
    enable_notifications: bool = True
    notification_topic: Optional[str] = None
    
    def __post_init__(self):
        if self.pipeline_parameters is None:
            self.pipeline_parameters = {}


class RetrainingTrigger:
    """Automated retraining trigger system."""
    
    def __init__(self, config: RetrainingConfig):
        """
        Initialize retraining trigger system.
        
        Args:
            config: Retraining configuration
        """
        self.config = config
        
        # Initialize clients
        aiplatform.init(project=config.project_id, location=config.region)
        
        if config.enable_notifications and config.notification_topic:
            self.publisher = pubsub_v1.PublisherClient()
            self.topic_path = self.publisher.topic_path(
                config.project_id, config.notification_topic
            )
        else:
            self.publisher = None
            self.topic_path = None
        
        # Initialize scheduler client for scheduled retraining
        if config.enable_scheduled_retraining:
            self.scheduler_client = scheduler_v1.CloudSchedulerClient()
        else:
            self.scheduler_client = None
        
        # Track retraining history
        self.retraining_history = []
        
        logger.info(f"Initialized retraining trigger system for project {config.project_id}")
    
    def evaluate_retraining_triggers(
        self,
        model_id: str,
        drift_results: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        data_volume: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all retraining triggers for a model.
        
        Args:
            model_id: Model ID to evaluate
            drift_results: Recent drift detection results
            performance_metrics: Recent performance metrics
            data_volume: Recent data volume
            
        Returns:
            Trigger evaluation results
        """
        logger.info(f"Evaluating retraining triggers for model {model_id}")
        
        evaluation_results = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "triggers_evaluated": [],
            "triggers_activated": [],
            "retraining_recommended": False,
            "retraining_urgency": "low",
            "next_evaluation": None
        }
        
        # Check drift-based triggers
        if self.config.enable_drift_retraining and drift_results:
            drift_trigger = self._evaluate_drift_trigger(model_id, drift_results)
            evaluation_results["triggers_evaluated"].append("drift_detected")
            if drift_trigger["triggered"]:
                evaluation_results["triggers_activated"].append("drift_detected")
                evaluation_results["drift_trigger"] = drift_trigger
        
        # Check performance-based triggers
        if self.config.enable_performance_retraining and performance_metrics:
            performance_trigger = self._evaluate_performance_trigger(model_id, performance_metrics)
            evaluation_results["triggers_evaluated"].append("performance_degradation")
            if performance_trigger["triggered"]:
                evaluation_results["triggers_activated"].append("performance_degradation")
                evaluation_results["performance_trigger"] = performance_trigger
        
        # Check data volume triggers
        if self.config.enable_data_volume_retraining and data_volume is not None:
            volume_trigger = self._evaluate_data_volume_trigger(model_id, data_volume)
            evaluation_results["triggers_evaluated"].append("data_volume_threshold")
            if volume_trigger["triggered"]:
                evaluation_results["triggers_activated"].append("data_volume_threshold")
                evaluation_results["volume_trigger"] = volume_trigger
        
        # Check scheduled triggers
        if self.config.enable_scheduled_retraining:
            scheduled_trigger = self._evaluate_scheduled_trigger(model_id)
            evaluation_results["triggers_evaluated"].append("scheduled")
            if scheduled_trigger["triggered"]:
                evaluation_results["triggers_activated"].append("scheduled")
                evaluation_results["scheduled_trigger"] = scheduled_trigger
        
        # Determine overall retraining recommendation
        if evaluation_results["triggers_activated"]:
            evaluation_results["retraining_recommended"] = True
            evaluation_results["retraining_urgency"] = self._determine_urgency(
                evaluation_results["triggers_activated"]
            )
        
        # Set next evaluation time
        evaluation_results["next_evaluation"] = self._calculate_next_evaluation_time()
        
        logger.info(f"Trigger evaluation completed. Retraining recommended: {evaluation_results['retraining_recommended']}")
        
        # Send notification if retraining is recommended
        if evaluation_results["retraining_recommended"] and self.config.enable_notifications:
            self._send_retraining_notification(evaluation_results)
        
        return evaluation_results
    
    def trigger_retraining(
        self,
        model_id: str,
        trigger_reason: str,
        trigger_data: Dict[str, Any],
        pipeline_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Trigger model retraining.
        
        Args:
            model_id: Model ID to retrain
            trigger_reason: Reason for retraining
            trigger_data: Data about the trigger
            pipeline_parameters: Optional pipeline parameters
            
        Returns:
            Retraining job information
        """
        logger.info(f"Triggering retraining for model {model_id}. Reason: {trigger_reason}")
        
        try:
            # Prepare pipeline parameters
            final_parameters = self.config.pipeline_parameters.copy()
            if pipeline_parameters:
                final_parameters.update(pipeline_parameters)
            
            # Add model-specific parameters
            final_parameters.update({
                "model_id": model_id,
                "trigger_reason": trigger_reason,
                "trigger_timestamp": datetime.now().isoformat()
            })
            
            # Create pipeline job
            job_display_name = f"retrain-{model_id}-{int(datetime.now().timestamp())}"
            
            if self.config.pipeline_template_path:
                # Submit pipeline job
                job = aiplatform.PipelineJob(
                    display_name=job_display_name,
                    template_path=self.config.pipeline_template_path,
                    parameter_values=final_parameters,
                    enable_caching=False
                )
                
                job.submit()
                
                retraining_info = {
                    "status": "submitted",
                    "job_id": job.name,
                    "job_display_name": job_display_name,
                    "pipeline_template": self.config.pipeline_template_path,
                    "parameters": final_parameters,
                    "trigger_reason": trigger_reason,
                    "trigger_data": trigger_data,
                    "submitted_at": datetime.now().isoformat()
                }
                
                logger.info(f"Retraining job submitted: {job.name}")
                
            else:
                # No pipeline template configured
                retraining_info = {
                    "status": "no_pipeline_configured",
                    "message": "Retraining triggered but no pipeline template configured",
                    "trigger_reason": trigger_reason,
                    "trigger_data": trigger_data,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.warning("Retraining triggered but no pipeline template configured")
            
            # Record retraining event
            self.retraining_history.append(retraining_info)
            
            return retraining_info
            
        except Exception as e:
            logger.error(f"Failed to trigger retraining for model {model_id}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "trigger_reason": trigger_reason,
                "timestamp": datetime.now().isoformat()
            }
    
    def _evaluate_drift_trigger(
        self,
        model_id: str,
        drift_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate drift-based retraining trigger."""
        drift_detected = drift_results.get("drift_detected", False)
        drift_score = drift_results.get("overall_drift_score", drift_results.get("drift_score", 0.0))
        
        # Check if drift exceeds threshold
        triggered = drift_detected and drift_score > self.config.drift_threshold
        
        # Check if enough time has passed since last drift-based retraining
        if triggered:
            last_drift_retraining = self._get_last_retraining_time(model_id, "drift_detected")
            if last_drift_retraining:
                time_since_last = datetime.now() - last_drift_retraining
                if time_since_last.total_seconds() < self.config.drift_retraining_delay_hours * 3600:
                    triggered = False
        
        return {
            "triggered": triggered,
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "threshold": self.config.drift_threshold,
            "delay_hours": self.config.drift_retraining_delay_hours,
            "evaluation_time": datetime.now().isoformat()
        }
    
    def _evaluate_performance_trigger(
        self,
        model_id: str,
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate performance-based retraining trigger."""
        # Extract performance metrics
        current_performance = performance_metrics.get("current_accuracy", 0.0)
        baseline_performance = performance_metrics.get("baseline_accuracy", 1.0)
        
        # Calculate performance degradation
        performance_drop = baseline_performance - current_performance
        triggered = performance_drop > self.config.performance_threshold
        
        # Check if enough time has passed since last performance-based retraining
        if triggered:
            last_performance_retraining = self._get_last_retraining_time(model_id, "performance_degradation")
            if last_performance_retraining:
                time_since_last = datetime.now() - last_performance_retraining
                if time_since_last.total_seconds() < self.config.performance_retraining_delay_hours * 3600:
                    triggered = False
        
        return {
            "triggered": triggered,
            "current_performance": current_performance,
            "baseline_performance": baseline_performance,
            "performance_drop": performance_drop,
            "threshold": self.config.performance_threshold,
            "delay_hours": self.config.performance_retraining_delay_hours,
            "evaluation_time": datetime.now().isoformat()
        }
    
    def _evaluate_data_volume_trigger(
        self,
        model_id: str,
        data_volume: int
    ) -> Dict[str, Any]:
        """Evaluate data volume-based retraining trigger."""
        triggered = data_volume >= self.config.data_volume_threshold
        
        return {
            "triggered": triggered,
            "current_volume": data_volume,
            "threshold": self.config.data_volume_threshold,
            "evaluation_time": datetime.now().isoformat()
        }
    
    def _evaluate_scheduled_trigger(self, model_id: str) -> Dict[str, Any]:
        """Evaluate scheduled retraining trigger."""
        last_scheduled_retraining = self._get_last_retraining_time(model_id, "scheduled")
        
        if not last_scheduled_retraining:
            # No previous scheduled retraining
            triggered = True
        else:
            # Check if enough time has passed based on schedule
            time_since_last = datetime.now() - last_scheduled_retraining
            
            if self.config.retraining_schedule == "daily":
                triggered = time_since_last.days >= 1
            elif self.config.retraining_schedule == "weekly":
                triggered = time_since_last.days >= 7
            elif self.config.retraining_schedule == "monthly":
                triggered = time_since_last.days >= 30
            else:
                triggered = False
        
        return {
            "triggered": triggered,
            "schedule": self.config.retraining_schedule,
            "last_scheduled_retraining": last_scheduled_retraining.isoformat() if last_scheduled_retraining else None,
            "evaluation_time": datetime.now().isoformat()
        }
    
    def _get_last_retraining_time(
        self,
        model_id: str,
        trigger_reason: str
    ) -> Optional[datetime]:
        """Get the last retraining time for a specific trigger reason."""
        # In a real implementation, this would query a database or storage
        # For now, check the in-memory history
        for event in reversed(self.retraining_history):
            if (event.get("model_id") == model_id and 
                event.get("trigger_reason") == trigger_reason and
                event.get("status") == "submitted"):
                try:
                    return datetime.fromisoformat(event["submitted_at"])
                except (KeyError, ValueError):
                    continue
        
        return None
    
    def _determine_urgency(self, activated_triggers: List[str]) -> str:
        """Determine retraining urgency based on activated triggers."""
        if "performance_degradation" in activated_triggers:
            return "high"
        elif "drift_detected" in activated_triggers:
            return "medium"
        elif "data_volume_threshold" in activated_triggers:
            return "medium"
        elif "scheduled" in activated_triggers:
            return "low"
        else:
            return "low"
    
    def _calculate_next_evaluation_time(self) -> str:
        """Calculate when the next trigger evaluation should occur."""
        # Default to daily evaluation
        next_evaluation = datetime.now() + timedelta(days=1)
        return next_evaluation.isoformat()
    
    def _send_retraining_notification(self, evaluation_results: Dict[str, Any]):
        """Send retraining notification via Pub/Sub."""
        if not self.publisher or not self.topic_path:
            logger.warning("Pub/Sub not configured for notifications")
            return
        
        try:
            notification_message = {
                "notification_type": "retraining_recommended",
                "model_id": evaluation_results["model_id"],
                "triggers_activated": evaluation_results["triggers_activated"],
                "urgency": evaluation_results["retraining_urgency"],
                "timestamp": evaluation_results["timestamp"],
                "details": evaluation_results
            }
            
            message_data = json.dumps(notification_message).encode('utf-8')
            future = self.publisher.publish(self.topic_path, message_data)
            message_id = future.result()
            
            logger.info(f"Retraining notification sent: {message_id}")
            
        except Exception as e:
            logger.error(f"Failed to send retraining notification: {e}")
    
    def setup_scheduled_retraining(
        self,
        model_id: str,
        schedule: str = "0 2 * * 0"  # Weekly at 2 AM on Sunday
    ) -> str:
        """
        Set up scheduled retraining using Cloud Scheduler.
        
        Args:
            model_id: Model ID to schedule retraining for
            schedule: Cron schedule expression
            
        Returns:
            Scheduler job name
        """
        if not self.scheduler_client:
            raise ValueError("Cloud Scheduler not initialized")
        
        logger.info(f"Setting up scheduled retraining for model {model_id}")
        
        # Create scheduler job
        parent = f"projects/{self.config.project_id}/locations/{self.config.region}"
        job_name = f"retrain-{model_id}-scheduled"
        
        # Pub/Sub target for triggering retraining
        pubsub_target = scheduler_v1.PubsubTarget(
            topic_name=self.topic_path,
            data=json.dumps({
                "trigger_type": "scheduled",
                "model_id": model_id,
                "schedule": schedule
            }).encode('utf-8')
        )
        
        job = scheduler_v1.Job(
            name=f"{parent}/jobs/{job_name}",
            description=f"Scheduled retraining for model {model_id}",
            schedule=schedule,
            time_zone="UTC",
            pubsub_target=pubsub_target
        )
        
        try:
            created_job = self.scheduler_client.create_job(
                parent=parent,
                job=job
            )
            logger.info(f"Scheduled retraining job created: {created_job.name}")
            return created_job.name
            
        except Exception as e:
            logger.error(f"Failed to create scheduled retraining job: {e}")
            raise
    
    def get_retraining_history(
        self,
        model_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get retraining history.
        
        Args:
            model_id: Optional model ID to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of retraining events
        """
        history = self.retraining_history
        
        if model_id:
            history = [event for event in history if event.get("model_id") == model_id]
        
        # Sort by timestamp (most recent first)
        history = sorted(
            history,
            key=lambda x: x.get("submitted_at", x.get("timestamp", "")),
            reverse=True
        )
        
        return history[:limit]
