"""
Pub/Sub Integration for Pipeline Agent

This module provides integration with Google Cloud Pub/Sub for
event-driven automation and pipeline triggers.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.types import PubsubMessage

logger = logging.getLogger(__name__)


@dataclass
class PubSubConfig:
    """Configuration for Pub/Sub integration."""
    project_id: str
    credentials_path: Optional[str] = None


class PubSubClient:
    """Client for Pub/Sub operations."""
    
    def __init__(self, config: PubSubConfig):
        """
        Initialize Pub/Sub client.
        
        Args:
            config: Pub/Sub configuration
        """
        self.config = config
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        logger.info(f"Initialized Pub/Sub client for project {config.project_id}")
    
    def create_topic(self, topic_name: str) -> str:
        """
        Create a Pub/Sub topic.
        
        Args:
            topic_name: Name of the topic
            
        Returns:
            Topic path
        """
        topic_path = self.publisher.topic_path(self.config.project_id, topic_name)
        
        try:
            topic = self.publisher.create_topic(request={"name": topic_path})
            logger.info(f"Created topic: {topic.name}")
            return topic.name
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Topic already exists: {topic_path}")
                return topic_path
            else:
                raise
    
    def delete_topic(self, topic_name: str):
        """
        Delete a Pub/Sub topic.
        
        Args:
            topic_name: Name of the topic
        """
        topic_path = self.publisher.topic_path(self.config.project_id, topic_name)
        self.publisher.delete_topic(request={"topic": topic_path})
        logger.info(f"Deleted topic: {topic_path}")
    
    def list_topics(self) -> List[str]:
        """
        List all topics in the project.
        
        Returns:
            List of topic paths
        """
        project_path = f"projects/{self.config.project_id}"
        topics = self.publisher.list_topics(request={"project": project_path})
        topic_names = [topic.name for topic in topics]
        logger.info(f"Found {len(topic_names)} topics")
        return topic_names
    
    def publish_message(
        self, 
        topic_name: str, 
        data: Union[str, Dict[str, Any]], 
        attributes: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Publish a message to a topic.
        
        Args:
            topic_name: Name of the topic
            data: Message data (string or dict)
            attributes: Optional message attributes
            
        Returns:
            Message ID
        """
        topic_path = self.publisher.topic_path(self.config.project_id, topic_name)
        
        # Convert data to bytes
        if isinstance(data, dict):
            message_data = json.dumps(data).encode('utf-8')
        else:
            message_data = str(data).encode('utf-8')
        
        # Publish message
        future = self.publisher.publish(
            topic_path, 
            message_data, 
            **(attributes or {})
        )
        
        message_id = future.result()
        logger.info(f"Published message {message_id} to {topic_name}")
        return message_id
    
    def create_subscription(
        self, 
        topic_name: str, 
        subscription_name: str,
        ack_deadline_seconds: int = 60
    ) -> str:
        """
        Create a subscription to a topic.
        
        Args:
            topic_name: Name of the topic
            subscription_name: Name of the subscription
            ack_deadline_seconds: Acknowledgment deadline
            
        Returns:
            Subscription path
        """
        topic_path = self.publisher.topic_path(self.config.project_id, topic_name)
        subscription_path = self.subscriber.subscription_path(
            self.config.project_id, subscription_name
        )
        
        try:
            subscription = self.subscriber.create_subscription(
                request={
                    "name": subscription_path,
                    "topic": topic_path,
                    "ack_deadline_seconds": ack_deadline_seconds
                }
            )
            logger.info(f"Created subscription: {subscription.name}")
            return subscription.name
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Subscription already exists: {subscription_path}")
                return subscription_path
            else:
                raise
    
    def delete_subscription(self, subscription_name: str):
        """
        Delete a subscription.
        
        Args:
            subscription_name: Name of the subscription
        """
        subscription_path = self.subscriber.subscription_path(
            self.config.project_id, subscription_name
        )
        self.subscriber.delete_subscription(request={"subscription": subscription_path})
        logger.info(f"Deleted subscription: {subscription_path}")
    
    def pull_messages(
        self, 
        subscription_name: str, 
        max_messages: int = 10,
        timeout: float = 10.0
    ) -> List[PubsubMessage]:
        """
        Pull messages from a subscription.
        
        Args:
            subscription_name: Name of the subscription
            max_messages: Maximum number of messages to pull
            timeout: Timeout in seconds
            
        Returns:
            List of received messages
        """
        subscription_path = self.subscriber.subscription_path(
            self.config.project_id, subscription_name
        )
        
        response = self.subscriber.pull(
            request={
                "subscription": subscription_path,
                "max_messages": max_messages
            },
            timeout=timeout
        )
        
        messages = response.received_messages
        logger.info(f"Pulled {len(messages)} messages from {subscription_name}")
        return messages
    
    def acknowledge_messages(
        self, 
        subscription_name: str, 
        ack_ids: List[str]
    ):
        """
        Acknowledge received messages.
        
        Args:
            subscription_name: Name of the subscription
            ack_ids: List of acknowledgment IDs
        """
        subscription_path = self.subscriber.subscription_path(
            self.config.project_id, subscription_name
        )
        
        self.subscriber.acknowledge(
            request={
                "subscription": subscription_path,
                "ack_ids": ack_ids
            }
        )
        logger.info(f"Acknowledged {len(ack_ids)} messages")


class PubSubIntegration:
    """High-level Pub/Sub integration for ML pipelines."""
    
    def __init__(self, config: PubSubConfig):
        """
        Initialize Pub/Sub integration.
        
        Args:
            config: Pub/Sub configuration
        """
        self.config = config
        self.client = PubSubClient(config)
        logger.info("Pub/Sub integration initialized")
    
    def setup_pipeline_triggers(self) -> Dict[str, str]:
        """
        Set up Pub/Sub topics for pipeline triggers.
        
        Returns:
            Dictionary mapping trigger types to topic names
        """
        logger.info("Setting up pipeline trigger topics")
        
        trigger_topics = {
            "model_drift_detected": "model-drift-alerts",
            "performance_degradation": "performance-alerts", 
            "data_quality_issues": "data-quality-alerts",
            "retraining_scheduled": "retraining-triggers",
            "deployment_requested": "deployment-triggers",
            "pipeline_completed": "pipeline-completion",
            "pipeline_failed": "pipeline-failures"
        }
        
        created_topics = {}
        for trigger_type, topic_name in trigger_topics.items():
            topic_path = self.client.create_topic(topic_name)
            created_topics[trigger_type] = topic_path
        
        logger.info(f"Created {len(created_topics)} trigger topics")
        return created_topics
    
    def publish_drift_alert(
        self, 
        endpoint_id: str, 
        drift_metrics: Dict[str, Any],
        severity: str = "medium"
    ) -> str:
        """
        Publish a model drift alert.
        
        Args:
            endpoint_id: Endpoint ID where drift was detected
            drift_metrics: Drift detection metrics
            severity: Alert severity (low, medium, high)
            
        Returns:
            Message ID
        """
        alert_data = {
            "alert_type": "model_drift",
            "endpoint_id": endpoint_id,
            "drift_metrics": drift_metrics,
            "severity": severity,
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "action_required": severity in ["medium", "high"]
        }
        
        attributes = {
            "alert_type": "model_drift",
            "severity": severity,
            "endpoint_id": endpoint_id
        }
        
        return self.client.publish_message(
            "model-drift-alerts", 
            alert_data, 
            attributes
        )
    
    def publish_performance_alert(
        self, 
        endpoint_id: str, 
        performance_metrics: Dict[str, Any],
        severity: str = "medium"
    ) -> str:
        """
        Publish a performance degradation alert.
        
        Args:
            endpoint_id: Endpoint ID with performance issues
            performance_metrics: Performance metrics
            severity: Alert severity
            
        Returns:
            Message ID
        """
        alert_data = {
            "alert_type": "performance_degradation",
            "endpoint_id": endpoint_id,
            "performance_metrics": performance_metrics,
            "severity": severity,
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "action_required": severity in ["medium", "high"]
        }
        
        attributes = {
            "alert_type": "performance_degradation",
            "severity": severity,
            "endpoint_id": endpoint_id
        }
        
        return self.client.publish_message(
            "performance-alerts", 
            alert_data, 
            attributes
        )
    
    def publish_retraining_trigger(
        self, 
        model_id: str, 
        trigger_reason: str,
        retraining_config: Dict[str, Any]
    ) -> str:
        """
        Publish a retraining trigger event.
        
        Args:
            model_id: Model ID to retrain
            trigger_reason: Reason for retraining
            retraining_config: Retraining configuration
            
        Returns:
            Message ID
        """
        trigger_data = {
            "event_type": "retraining_trigger",
            "model_id": model_id,
            "trigger_reason": trigger_reason,
            "retraining_config": retraining_config,
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "priority": "normal"
        }
        
        attributes = {
            "event_type": "retraining_trigger",
            "model_id": model_id,
            "trigger_reason": trigger_reason
        }
        
        return self.client.publish_message(
            "retraining-triggers", 
            trigger_data, 
            attributes
        )
    
    def publish_pipeline_completion(
        self, 
        pipeline_job_id: str, 
        pipeline_name: str,
        status: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Publish a pipeline completion event.
        
        Args:
            pipeline_job_id: Pipeline job ID
            pipeline_name: Pipeline name
            status: Completion status (success, failed)
            metrics: Optional pipeline metrics
            
        Returns:
            Message ID
        """
        completion_data = {
            "event_type": "pipeline_completion",
            "pipeline_job_id": pipeline_job_id,
            "pipeline_name": pipeline_name,
            "status": status,
            "metrics": metrics or {},
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
        }
        
        attributes = {
            "event_type": "pipeline_completion",
            "pipeline_name": pipeline_name,
            "status": status
        }
        
        topic_name = "pipeline-completion" if status == "success" else "pipeline-failures"
        
        return self.client.publish_message(
            topic_name, 
            completion_data, 
            attributes
        )
    
    def setup_alert_subscriptions(
        self, 
        callback_functions: Dict[str, Callable]
    ) -> Dict[str, str]:
        """
        Set up subscriptions for alert handling.
        
        Args:
            callback_functions: Dictionary mapping alert types to callback functions
            
        Returns:
            Dictionary mapping alert types to subscription names
        """
        logger.info("Setting up alert subscriptions")
        
        subscriptions = {}
        
        for alert_type, callback in callback_functions.items():
            topic_mapping = {
                "drift_alerts": "model-drift-alerts",
                "performance_alerts": "performance-alerts",
                "retraining_triggers": "retraining-triggers",
                "pipeline_completions": "pipeline-completion",
                "pipeline_failures": "pipeline-failures"
            }
            
            if alert_type in topic_mapping:
                topic_name = topic_mapping[alert_type]
                subscription_name = f"{alert_type}_subscription"
                
                subscription_path = self.client.create_subscription(
                    topic_name, subscription_name
                )
                subscriptions[alert_type] = subscription_path
        
        logger.info(f"Created {len(subscriptions)} alert subscriptions")
        return subscriptions
    
    def start_message_listener(
        self, 
        subscription_name: str, 
        callback: Callable[[PubsubMessage], None],
        max_workers: int = 4
    ):
        """
        Start a message listener for a subscription.
        
        Args:
            subscription_name: Name of the subscription
            callback: Callback function to handle messages
            max_workers: Maximum number of worker threads
        """
        logger.info(f"Starting message listener for {subscription_name}")
        
        subscription_path = self.client.subscriber.subscription_path(
            self.config.project_id, subscription_name
        )
        
        # Configure flow control
        flow_control = pubsub_v1.types.FlowControl(max_messages=100)
        
        # Start listening
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            streaming_pull_future = self.client.subscriber.subscribe(
                subscription_path, 
                callback=callback,
                flow_control=flow_control
            )
            
            logger.info(f"Listening for messages on {subscription_path}")
            
            try:
                streaming_pull_future.result()
            except KeyboardInterrupt:
                streaming_pull_future.cancel()
                logger.info("Message listener stopped")
    
    def create_pipeline_event_handler(self) -> Callable[[PubsubMessage], None]:
        """
        Create a generic pipeline event handler.
        
        Returns:
            Message handler function
        """
        def handle_pipeline_event(message: PubsubMessage):
            """Handle pipeline events from Pub/Sub."""
            try:
                # Parse message data
                message_data = json.loads(message.data.decode('utf-8'))
                event_type = message_data.get('event_type')
                
                logger.info(f"Received pipeline event: {event_type}")
                
                # Handle different event types
                if event_type == "model_drift":
                    self._handle_drift_event(message_data)
                elif event_type == "performance_degradation":
                    self._handle_performance_event(message_data)
                elif event_type == "retraining_trigger":
                    self._handle_retraining_event(message_data)
                elif event_type == "pipeline_completion":
                    self._handle_completion_event(message_data)
                else:
                    logger.warning(f"Unknown event type: {event_type}")
                
                # Acknowledge message
                message.ack()
                
            except Exception as e:
                logger.error(f"Error handling pipeline event: {e}")
                message.nack()
        
        return handle_pipeline_event
    
    def _handle_drift_event(self, event_data: Dict[str, Any]):
        """Handle model drift events."""
        logger.info(f"Handling drift event for endpoint {event_data.get('endpoint_id')}")
        # Implementation would trigger appropriate actions
    
    def _handle_performance_event(self, event_data: Dict[str, Any]):
        """Handle performance degradation events."""
        logger.info(f"Handling performance event for endpoint {event_data.get('endpoint_id')}")
        # Implementation would trigger appropriate actions
    
    def _handle_retraining_event(self, event_data: Dict[str, Any]):
        """Handle retraining trigger events."""
        logger.info(f"Handling retraining event for model {event_data.get('model_id')}")
        # Implementation would trigger retraining pipeline
    
    def _handle_completion_event(self, event_data: Dict[str, Any]):
        """Handle pipeline completion events."""
        logger.info(f"Handling completion event for pipeline {event_data.get('pipeline_name')}")
        # Implementation would handle post-completion actions
