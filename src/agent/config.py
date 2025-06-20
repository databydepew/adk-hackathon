"""
Agent Configuration Management

Handles configuration loading and validation for the Vertex AI Pipeline Agent.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GeminiConfig:
    """Configuration for Gemini Pro integration."""
    model: str = "gemini-pro"
    temperature: float = 0.1
    max_tokens: int = 8192
    safety_settings: List[Dict[str, str]] = field(default_factory=lambda: [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ])


@dataclass
class GCPConfig:
    """Configuration for Google Cloud Platform services."""
    project_id: str
    region: str = "us-central1"
    zone: str = "us-central1-a"
    vertex_ai_location: str = "us-central1"
    staging_bucket: Optional[str] = None
    service_account: Optional[str] = None
    bigquery_location: str = "US"
    artifact_registry_repo: str = "ml-pipelines"
    artifact_registry_location: str = "us-central1"


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline generation."""
    default_machine_type: str = "n1-standard-4"
    default_disk_size_gb: int = 100
    python_version: str = "3.9"
    supported_frameworks: List[Dict[str, str]] = field(default_factory=lambda: [
        {"name": "scikit-learn", "container_uri": "gcr.io/deeplearning-platform-release/sklearn-cpu"},
        {"name": "xgboost", "container_uri": "gcr.io/deeplearning-platform-release/xgboost-cpu"},
        {"name": "tensorflow", "container_uri": "gcr.io/deeplearning-platform-release/tf2-cpu"},
        {"name": "pytorch", "container_uri": "gcr.io/deeplearning-platform-release/pytorch-cpu"},
    ])


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring and drift detection."""
    enable_drift_detection: bool = True
    drift_threshold: float = 0.1
    monitoring_frequency: str = "daily"
    enable_performance_tracking: bool = True
    performance_threshold: float = 0.05
    enable_data_validation: bool = True
    schema_validation: bool = True


@dataclass
class AgentBehaviorConfig:
    """Configuration for agent behavior and conversation handling."""
    max_conversation_turns: int = 10
    context_window: int = 4000
    auto_optimize_pipelines: bool = True
    include_monitoring: bool = True
    include_cicd: bool = True
    validate_sql_queries: bool = True
    validate_pipeline_configs: bool = True
    require_approval_for_deployment: bool = False


class AgentConfig:
    """Main configuration class for the Vertex AI Pipeline Agent."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize agent configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        self.config_path = config_path or "config.yaml"
        self._config_data = self._load_config()
        
        # Initialize configuration sections
        self.gemini = self._init_gemini_config()
        self.gcp = self._init_gcp_config()
        self.pipeline = self._init_pipeline_config()
        self.monitoring = self._init_monitoring_config()
        self.agent_behavior = self._init_agent_behavior_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable substitution."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(config_file, 'r') as f:
            config_content = f.read()
            
        # Substitute environment variables
        config_content = self._substitute_env_vars(config_content)
        
        try:
            return yaml.safe_load(config_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in configuration content."""
        import re
        
        def replace_env_var(match):
            var_expr = match.group(1)
            if ':-' in var_expr:
                # Handle default values: ${VAR:-default}
                var_name, default_value = var_expr.split(':-', 1)
                return os.getenv(var_name, default_value)
            else:
                # Simple variable: ${VAR}
                value = os.getenv(var_expr)
                if value is None:
                    raise ValueError(f"Required environment variable not set: {var_expr}")
                return value
        
        # Replace ${VAR} and ${VAR:-default} patterns
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_env_var, content)
    
    def _init_gemini_config(self) -> GeminiConfig:
        """Initialize Gemini configuration."""
        gemini_data = self._config_data.get('gemini', {})
        return GeminiConfig(
            model=gemini_data.get('model', 'gemini-pro'),
            temperature=gemini_data.get('temperature', 0.1),
            max_tokens=gemini_data.get('max_tokens', 8192),
            safety_settings=gemini_data.get('safety_settings', GeminiConfig().safety_settings)
        )
    
    def _init_gcp_config(self) -> GCPConfig:
        """Initialize GCP configuration."""
        gcp_data = self._config_data.get('gcp', {})
        
        project_id = gcp_data.get('project_id')
        if not project_id:
            raise ValueError("GCP project_id is required in configuration")
            
        return GCPConfig(
            project_id=project_id,
            region=gcp_data.get('region', 'us-central1'),
            zone=gcp_data.get('zone', 'us-central1-a'),
            vertex_ai_location=gcp_data.get('vertex_ai', {}).get('location', 'us-central1'),
            staging_bucket=gcp_data.get('vertex_ai', {}).get('staging_bucket'),
            service_account=gcp_data.get('vertex_ai', {}).get('service_account'),
            bigquery_location=gcp_data.get('bigquery', {}).get('dataset_location', 'US'),
            artifact_registry_repo=gcp_data.get('artifact_registry', {}).get('repository', 'ml-pipelines'),
            artifact_registry_location=gcp_data.get('artifact_registry', {}).get('location', 'us-central1')
        )
    
    def _init_pipeline_config(self) -> PipelineConfig:
        """Initialize pipeline configuration."""
        pipeline_data = self._config_data.get('pipeline', {})
        defaults = pipeline_data.get('defaults', {})
        
        return PipelineConfig(
            default_machine_type=defaults.get('machine_type', 'n1-standard-4'),
            default_disk_size_gb=defaults.get('disk_size_gb', 100),
            python_version=defaults.get('python_version', '3.9'),
            supported_frameworks=pipeline_data.get('frameworks', PipelineConfig().supported_frameworks)
        )
    
    def _init_monitoring_config(self) -> MonitoringConfig:
        """Initialize monitoring configuration."""
        monitoring_data = self._config_data.get('monitoring', {})
        model_monitoring = monitoring_data.get('model_monitoring', {})
        performance_monitoring = monitoring_data.get('performance_monitoring', {})
        data_quality = monitoring_data.get('data_quality', {})
        
        return MonitoringConfig(
            enable_drift_detection=model_monitoring.get('enable_drift_detection', True),
            drift_threshold=model_monitoring.get('drift_threshold', 0.1),
            monitoring_frequency=model_monitoring.get('monitoring_frequency', 'daily'),
            enable_performance_tracking=performance_monitoring.get('enable_performance_tracking', True),
            performance_threshold=performance_monitoring.get('performance_threshold', 0.05),
            enable_data_validation=data_quality.get('enable_data_validation', True),
            schema_validation=data_quality.get('schema_validation', True)
        )
    
    def _init_agent_behavior_config(self) -> AgentBehaviorConfig:
        """Initialize agent behavior configuration."""
        behavior_data = self._config_data.get('agent_behavior', {})
        conversation = behavior_data.get('conversation', {})
        pipeline_generation = behavior_data.get('pipeline_generation', {})
        validation = behavior_data.get('validation', {})
        
        return AgentBehaviorConfig(
            max_conversation_turns=conversation.get('max_turns', 10),
            context_window=conversation.get('context_window', 4000),
            auto_optimize_pipelines=pipeline_generation.get('auto_optimize', True),
            include_monitoring=pipeline_generation.get('include_monitoring', True),
            include_cicd=pipeline_generation.get('include_cicd', True),
            validate_sql_queries=validation.get('validate_sql_queries', True),
            validate_pipeline_configs=validation.get('validate_pipeline_configs', True),
            require_approval_for_deployment=validation.get('require_approval_for_deployment', False)
        )
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate required GCP settings
        if not self.gcp.project_id:
            raise ValueError("GCP project_id is required")
            
        # Validate Gemini settings
        if self.gemini.temperature < 0 or self.gemini.temperature > 1:
            raise ValueError("Gemini temperature must be between 0 and 1")
            
        if self.gemini.max_tokens <= 0:
            raise ValueError("Gemini max_tokens must be positive")
            
        # Validate pipeline settings
        if self.pipeline.default_disk_size_gb <= 0:
            raise ValueError("Pipeline disk size must be positive")
            
        # Validate monitoring settings
        if self.monitoring.drift_threshold < 0 or self.monitoring.drift_threshold > 1:
            raise ValueError("Monitoring drift threshold must be between 0 and 1")
            
        return True
    
    def get_framework_container(self, framework_name: str) -> Optional[str]:
        """
        Get container URI for a specific ML framework.
        
        Args:
            framework_name: Name of the ML framework
            
        Returns:
            Container URI if found, None otherwise
        """
        for framework in self.pipeline.supported_frameworks:
            if framework['name'].lower() == framework_name.lower():
                return framework['container_uri']
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'gemini': {
                'model': self.gemini.model,
                'temperature': self.gemini.temperature,
                'max_tokens': self.gemini.max_tokens,
                'safety_settings': self.gemini.safety_settings
            },
            'gcp': {
                'project_id': self.gcp.project_id,
                'region': self.gcp.region,
                'zone': self.gcp.zone,
                'vertex_ai_location': self.gcp.vertex_ai_location,
                'staging_bucket': self.gcp.staging_bucket,
                'service_account': self.gcp.service_account,
                'bigquery_location': self.gcp.bigquery_location,
                'artifact_registry_repo': self.gcp.artifact_registry_repo,
                'artifact_registry_location': self.gcp.artifact_registry_location
            },
            'pipeline': {
                'default_machine_type': self.pipeline.default_machine_type,
                'default_disk_size_gb': self.pipeline.default_disk_size_gb,
                'python_version': self.pipeline.python_version,
                'supported_frameworks': self.pipeline.supported_frameworks
            },
            'monitoring': {
                'enable_drift_detection': self.monitoring.enable_drift_detection,
                'drift_threshold': self.monitoring.drift_threshold,
                'monitoring_frequency': self.monitoring.monitoring_frequency,
                'enable_performance_tracking': self.monitoring.enable_performance_tracking,
                'performance_threshold': self.monitoring.performance_threshold,
                'enable_data_validation': self.monitoring.enable_data_validation,
                'schema_validation': self.monitoring.schema_validation
            },
            'agent_behavior': {
                'max_conversation_turns': self.agent_behavior.max_conversation_turns,
                'context_window': self.agent_behavior.context_window,
                'auto_optimize_pipelines': self.agent_behavior.auto_optimize_pipelines,
                'include_monitoring': self.agent_behavior.include_monitoring,
                'include_cicd': self.agent_behavior.include_cicd,
                'validate_sql_queries': self.agent_behavior.validate_sql_queries,
                'validate_pipeline_configs': self.agent_behavior.validate_pipeline_configs,
                'require_approval_for_deployment': self.agent_behavior.require_approval_for_deployment
            }
        }
