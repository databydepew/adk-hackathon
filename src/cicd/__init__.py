"""
Vertex AI Pipeline Agent - CI/CD Module

This module contains CI/CD automation components for pipeline
deployment, testing, and continuous integration workflows.
"""

from .automation import CICDAutomation, CICDConfig

__all__ = [
    "CICDAutomation",
    "CICDConfig",
]

__version__ = "1.0.0"
