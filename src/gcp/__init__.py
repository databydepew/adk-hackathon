"""
Vertex AI Pipeline Agent - GCP Integration Module

This module contains integration components for various Google Cloud Platform services
including BigQuery, Vertex AI, Cloud Functions, Pub/Sub, and Artifact Registry.
"""

from .bigquery import BigQueryClient, BigQueryIntegration
from .vertex_ai import VertexAIClient, VertexAIIntegration
from .cloud_functions import CloudFunctionsClient, CloudFunctionsIntegration
from .pubsub import PubSubClient, PubSubIntegration

__all__ = [
    # BigQuery
    "BigQueryClient",
    "BigQueryIntegration",
    
    # Vertex AI
    "VertexAIClient", 
    "VertexAIIntegration",
    
    # Cloud Functions
    "CloudFunctionsClient",
    "CloudFunctionsIntegration",
    
    # Pub/Sub
    "PubSubClient",
    "PubSubIntegration",
]

__version__ = "1.0.0"
