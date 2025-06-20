"""
Vertex AI Pipeline Agent - Core Agent Module

This module contains the core agent implementation using Google Agent ADK
and Gemini Pro for natural language instruction parsing and ML pipeline generation.
"""

from .core import VertexAIAgent
from .config import AgentConfig

__all__ = [
    "VertexAIAgent",
    "AgentConfig",
]

__version__ = "1.0.0"
