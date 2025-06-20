"""
Instruction Parser for Natural Language ML Pipeline Requirements

This module provides advanced parsing capabilities to extract structured
ML pipeline requirements from natural language instructions.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Supported ML task types."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    RECOMMENDATION = "recommendation"
    OTHER = "other"


class DataSourceType(Enum):
    """Supported data source types."""
    BIGQUERY = "bigquery"
    GCS = "gcs"
    CLOUD_SQL = "cloud_sql"
    FIRESTORE = "firestore"
    OTHER = "other"


class MLFramework(Enum):
    """Supported ML frameworks."""
    SCIKIT_LEARN = "scikit-learn"
    XGBOOST = "xgboost"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    AUTO_ML = "automl"


@dataclass
class DataSource:
    """Data source specification."""
    type: DataSourceType
    location: str
    description: str = ""
    schema_info: Optional[Dict[str, Any]] = None
    access_config: Optional[Dict[str, Any]] = None


@dataclass
class ModelConfig:
    """Model configuration specification."""
    framework: MLFramework
    algorithm: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: List[str] = field(default_factory=list)


@dataclass
class DeploymentSpec:
    """Deployment specification."""
    endpoint_name: str
    machine_type: str = "n1-standard-4"
    min_replica_count: int = 1
    max_replica_count: int = 10
    traffic_split: Dict[str, int] = field(default_factory=lambda: {"100": 100})
    auto_scaling: bool = True


@dataclass
class MonitoringSpec:
    """Monitoring specification."""
    enable_drift_detection: bool = True
    enable_performance_monitoring: bool = True
    enable_data_quality_monitoring: bool = True
    drift_threshold: float = 0.1
    performance_threshold: float = 0.05
    monitoring_frequency: str = "daily"
    alert_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedInstruction:
    """Parsed instruction result with structured ML pipeline requirements."""
    task_type: TaskType
    data_source: DataSource
    model_config: ModelConfig
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    deployment_spec: Optional[DeploymentSpec] = None
    monitoring_spec: Optional[MonitoringSpec] = None
    pipeline_name: Optional[str] = None
    description: str = ""
    confidence_score: float = 0.0
    raw_instruction: str = ""
    parsing_metadata: Dict[str, Any] = field(default_factory=dict)


class InstructionParser:
    """
    Advanced instruction parser for extracting ML pipeline requirements
    from natural language instructions.
    """
    
    def __init__(self):
        """Initialize the instruction parser."""
        self._init_patterns()
        self._init_keyword_mappings()
        logger.info("Instruction parser initialized")
    
    def _init_patterns(self):
        """Initialize regex patterns for parsing."""
        self.patterns = {
            # Task type patterns
            'task_types': {
                TaskType.BINARY_CLASSIFICATION: [
                    r'binary\s+classif\w*', r'fraud\s+detection', r'churn\s+prediction',
                    r'spam\s+detection', r'anomaly\s+classif\w*', r'yes/no\s+prediction',
                    r'true/false\s+prediction', r'0/1\s+prediction'
                ],
                TaskType.MULTICLASS_CLASSIFICATION: [
                    r'multi-?class\s+classif\w*', r'categoriz\w*', r'sentiment\s+analysis',
                    r'image\s+classif\w*', r'text\s+classif\w*', r'product\s+categoriz\w*'
                ],
                TaskType.REGRESSION: [
                    r'regression', r'predict\s+value', r'forecast\s+price',
                    r'estimate\s+cost', r'predict\s+revenue', r'sales\s+forecast'
                ],
                TaskType.CLUSTERING: [
                    r'cluster\w*', r'segment\w*', r'group\w*',
                    r'customer\s+segment\w*', r'market\s+segment\w*'
                ],
                TaskType.ANOMALY_DETECTION: [
                    r'anomaly\s+detect\w*', r'outlier\s+detect\w*', r'fraud\s+detect\w*',
                    r'unusual\s+pattern', r'abnormal\s+behavior'
                ],
                TaskType.TIME_SERIES_FORECASTING: [
                    r'time\s+series', r'forecast\w*', r'predict\s+future',
                    r'trend\s+analysis', r'seasonal\s+predict\w*'
                ],
                TaskType.RECOMMENDATION: [
                    r'recommend\w*', r'suggest\w*', r'collaborative\s+filter\w*',
                    r'content\s+filter\w*', r'personalization'
                ]
            },
            
            # Framework patterns
            'frameworks': {
                MLFramework.XGBOOST: [
                    r'xgboost', r'xgb', r'gradient\s+boost\w*', r'gbm'
                ],
                MLFramework.SCIKIT_LEARN: [
                    r'scikit-?learn', r'sklearn', r'random\s+forest', r'svm',
                    r'support\s+vector', r'logistic\s+regression', r'linear\s+regression'
                ],
                MLFramework.TENSORFLOW: [
                    r'tensorflow', r'tf', r'neural\s+network', r'deep\s+learning',
                    r'keras', r'dnn'
                ],
                MLFramework.PYTORCH: [
                    r'pytorch', r'torch'
                ],
                MLFramework.LIGHTGBM: [
                    r'lightgbm', r'lgb', r'light\s+gradient\s+boost\w*'
                ],
                MLFramework.CATBOOST: [
                    r'catboost', r'categorical\s+boost\w*'
                ],
                MLFramework.AUTO_ML: [
                    r'automl', r'auto\s+ml', r'automated\s+ml', r'auto-?ml'
                ]
            },
            
            # Data source patterns
            'data_sources': {
                DataSourceType.BIGQUERY: [
                    r'bigquery', r'bq', r'sql\s+table', r'dataset\s+table',
                    r'\w+\.\w+\.\w+'  # project.dataset.table pattern
                ],
                DataSourceType.GCS: [
                    r'gcs', r'cloud\s+storage', r'bucket', r'gs://',
                    r'csv\s+file', r'parquet\s+file', r'json\s+file'
                ],
                DataSourceType.CLOUD_SQL: [
                    r'cloud\s+sql', r'mysql', r'postgresql', r'postgres'
                ],
                DataSourceType.FIRESTORE: [
                    r'firestore', r'document\s+database', r'nosql'
                ]
            },
            
            # Column and feature patterns
            'columns': {
                'target': [
                    r'target\s+column', r'label\s+column', r'predict\s+(\w+)',
                    r'outcome\s+variable', r'dependent\s+variable'
                ],
                'features': [
                    r'feature\s+columns?', r'input\s+columns?', r'predictor\s+variables?',
                    r'independent\s+variables?', r'attributes'
                ]
            },
            
            # Deployment patterns
            'deployment': {
                'endpoint': [
                    r'deploy\s+to\s+endpoint', r'create\s+endpoint', r'serve\s+model',
                    r'online\s+prediction', r'real-?time\s+prediction'
                ],
                'batch': [
                    r'batch\s+prediction', r'offline\s+prediction', r'bulk\s+prediction'
                ]
            }
        }
    
    def _init_keyword_mappings(self):
        """Initialize keyword mappings for enhanced parsing."""
        self.keyword_mappings = {
            'algorithms': {
                'random forest': 'RandomForestClassifier',
                'logistic regression': 'LogisticRegression',
                'linear regression': 'LinearRegression',
                'svm': 'SVC',
                'support vector machine': 'SVC',
                'gradient boosting': 'GradientBoostingClassifier',
                'neural network': 'MLPClassifier',
                'k-means': 'KMeans',
                'dbscan': 'DBSCAN'
            },
            'metrics': {
                'classification': ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'roc_auc'],
                'regression': ['mse', 'rmse', 'mae', 'r2_score', 'mean_absolute_error'],
                'clustering': ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
            }
        }
    
    def parse(self, instruction: str) -> ParsedInstruction:
        """
        Parse a natural language instruction into structured ML pipeline requirements.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            ParsedInstruction object with extracted requirements
        """
        logger.info(f"Parsing instruction: {instruction[:100]}...")
        
        instruction_lower = instruction.lower()
        
        # Extract task type
        task_type = self._extract_task_type(instruction_lower)
        
        # Extract data source
        data_source = self._extract_data_source(instruction, instruction_lower)
        
        # Extract model configuration
        model_config = self._extract_model_config(instruction_lower, task_type)
        
        # Extract target and feature columns
        target_column, feature_columns = self._extract_columns(instruction, instruction_lower)
        
        # Extract deployment specification
        deployment_spec = self._extract_deployment_spec(instruction_lower)
        
        # Extract monitoring specification
        monitoring_spec = self._extract_monitoring_spec(instruction_lower)
        
        # Generate pipeline name
        pipeline_name = self._generate_pipeline_name(task_type, data_source)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            task_type, data_source, model_config, target_column
        )
        
        # Create parsed instruction
        parsed = ParsedInstruction(
            task_type=task_type,
            data_source=data_source,
            model_config=model_config,
            target_column=target_column,
            feature_columns=feature_columns,
            deployment_spec=deployment_spec,
            monitoring_spec=monitoring_spec,
            pipeline_name=pipeline_name,
            description=self._generate_description(instruction, task_type),
            confidence_score=confidence_score,
            raw_instruction=instruction,
            parsing_metadata={
                'parsing_method': 'pattern_based',
                'patterns_matched': self._get_matched_patterns(instruction_lower)
            }
        )
        
        logger.info(f"Parsed instruction with confidence {confidence_score:.2f}")
        return parsed
    
    def _extract_task_type(self, instruction_lower: str) -> TaskType:
        """Extract ML task type from instruction."""
        for task_type, patterns in self.patterns['task_types'].items():
            for pattern in patterns:
                if re.search(pattern, instruction_lower):
                    return task_type
        
        # Default fallback based on common keywords
        if any(word in instruction_lower for word in ['classify', 'classification', 'predict class']):
            return TaskType.BINARY_CLASSIFICATION
        elif any(word in instruction_lower for word in ['predict', 'forecast', 'estimate']):
            return TaskType.REGRESSION
        
        return TaskType.OTHER
    
    def _extract_data_source(self, instruction: str, instruction_lower: str) -> DataSource:
        """Extract data source information from instruction."""
        # Check for BigQuery table patterns
        bq_match = re.search(r'(\w+\.\w+\.\w+)', instruction)
        if bq_match or any(re.search(pattern, instruction_lower) 
                          for pattern in self.patterns['data_sources'][DataSourceType.BIGQUERY]):
            location = bq_match.group(1) if bq_match else ""
            return DataSource(
                type=DataSourceType.BIGQUERY,
                location=location,
                description="BigQuery table"
            )
        
        # Check for GCS patterns
        gcs_match = re.search(r'gs://[\w\-./]+', instruction)
        if gcs_match or any(re.search(pattern, instruction_lower) 
                           for pattern in self.patterns['data_sources'][DataSourceType.GCS]):
            location = gcs_match.group(0) if gcs_match else ""
            return DataSource(
                type=DataSourceType.GCS,
                location=location,
                description="Google Cloud Storage file"
            )
        
        # Check for other data sources
        for source_type, patterns in self.patterns['data_sources'].items():
            if source_type in [DataSourceType.BIGQUERY, DataSourceType.GCS]:
                continue
            for pattern in patterns:
                if re.search(pattern, instruction_lower):
                    return DataSource(
                        type=source_type,
                        location="",
                        description=f"{source_type.value} data source"
                    )
        
        return DataSource(
            type=DataSourceType.OTHER,
            location="",
            description="Unspecified data source"
        )
    
    def _extract_model_config(self, instruction_lower: str, task_type: TaskType) -> ModelConfig:
        """Extract model configuration from instruction."""
        # Detect framework
        framework = MLFramework.SCIKIT_LEARN  # default
        for fw, patterns in self.patterns['frameworks'].items():
            for pattern in patterns:
                if re.search(pattern, instruction_lower):
                    framework = fw
                    break
            if framework != MLFramework.SCIKIT_LEARN:
                break
        
        # Detect algorithm
        algorithm = None
        for algo_name, algo_class in self.keyword_mappings['algorithms'].items():
            if algo_name in instruction_lower:
                algorithm = algo_class
                break
        
        # Get default metrics for task type
        metrics = self._get_default_metrics(task_type)
        
        # Extract hyperparameters (basic extraction)
        hyperparameters = self._extract_hyperparameters(instruction_lower)
        
        return ModelConfig(
            framework=framework,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            evaluation_metrics=metrics
        )
    
    def _extract_columns(self, instruction: str, instruction_lower: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """Extract target and feature column information."""
        target_column = None
        feature_columns = None
        
        # Extract target column
        target_patterns = [
            r'target\s+(?:column\s+)?["\']?(\w+)["\']?',
            r'predict\s+["\']?(\w+)["\']?',
            r'label\s+(?:column\s+)?["\']?(\w+)["\']?'
        ]
        
        for pattern in target_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                target_column = match.group(1)
                break
        
        # Extract feature columns (basic extraction)
        feature_patterns = [
            r'features?\s+(?:columns?\s+)?["\']?([^"\']+)["\']?',
            r'using\s+(?:columns?\s+)?["\']?([^"\']+)["\']?'
        ]
        
        for pattern in feature_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                features_str = match.group(1)
                feature_columns = [f.strip() for f in features_str.split(',')]
                break
        
        return target_column, feature_columns
    
    def _extract_deployment_spec(self, instruction_lower: str) -> Optional[DeploymentSpec]:
        """Extract deployment specification from instruction."""
        # Check if deployment is mentioned
        deployment_mentioned = any(
            re.search(pattern, instruction_lower)
            for patterns in self.patterns['deployment'].values()
            for pattern in patterns
        )
        
        if not deployment_mentioned:
            return None
        
        # Extract endpoint name
        endpoint_match = re.search(r'endpoint\s+["\']?(\w+)["\']?', instruction_lower)
        endpoint_name = endpoint_match.group(1) if endpoint_match else "ml_model_endpoint"
        
        # Extract machine type
        machine_match = re.search(r'machine\s+type\s+["\']?([\w\-]+)["\']?', instruction_lower)
        machine_type = machine_match.group(1) if machine_match else "n1-standard-4"
        
        return DeploymentSpec(
            endpoint_name=endpoint_name,
            machine_type=machine_type
        )
    
    def _extract_monitoring_spec(self, instruction_lower: str) -> MonitoringSpec:
        """Extract monitoring specification from instruction."""
        # Check for monitoring keywords
        enable_monitoring = any(word in instruction_lower 
                              for word in ['monitor', 'drift', 'performance', 'alert'])
        
        # Extract drift threshold
        drift_match = re.search(r'drift\s+threshold\s+([\d.]+)', instruction_lower)
        drift_threshold = float(drift_match.group(1)) if drift_match else 0.1
        
        return MonitoringSpec(
            enable_drift_detection=enable_monitoring,
            enable_performance_monitoring=enable_monitoring,
            drift_threshold=drift_threshold
        )
    
    def _extract_hyperparameters(self, instruction_lower: str) -> Dict[str, Any]:
        """Extract hyperparameters from instruction."""
        hyperparameters = {}
        
        # Extract common hyperparameters
        param_patterns = {
            'n_estimators': r'(?:n_estimators|estimators|trees)\s*[=:]\s*(\d+)',
            'max_depth': r'max_depth\s*[=:]\s*(\d+)',
            'learning_rate': r'learning_rate\s*[=:]\s*([\d.]+)',
            'random_state': r'random_state\s*[=:]\s*(\d+)'
        }
        
        for param, pattern in param_patterns.items():
            match = re.search(pattern, instruction_lower)
            if match:
                value = match.group(1)
                hyperparameters[param] = int(value) if value.isdigit() else float(value)
        
        return hyperparameters
    
    def _get_default_metrics(self, task_type: TaskType) -> List[str]:
        """Get default evaluation metrics for task type."""
        if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            return self.keyword_mappings['metrics']['classification']
        elif task_type == TaskType.REGRESSION:
            return self.keyword_mappings['metrics']['regression']
        elif task_type == TaskType.CLUSTERING:
            return self.keyword_mappings['metrics']['clustering']
        else:
            return ['accuracy']
    
    def _generate_pipeline_name(self, task_type: TaskType, data_source: DataSource) -> str:
        """Generate a pipeline name based on task type and data source."""
        task_name = task_type.value.replace('_', '-')
        source_name = data_source.type.value
        return f"{task_name}-{source_name}-pipeline"
    
    def _generate_description(self, instruction: str, task_type: TaskType) -> str:
        """Generate a description for the pipeline."""
        return f"{task_type.value.replace('_', ' ').title()} pipeline based on: {instruction[:100]}..."
    
    def _calculate_confidence_score(
        self, 
        task_type: TaskType, 
        data_source: DataSource, 
        model_config: ModelConfig,
        target_column: Optional[str]
    ) -> float:
        """Calculate confidence score for the parsing result."""
        score = 0.0
        
        # Task type confidence
        if task_type != TaskType.OTHER:
            score += 0.3
        
        # Data source confidence
        if data_source.type != DataSourceType.OTHER and data_source.location:
            score += 0.3
        elif data_source.type != DataSourceType.OTHER:
            score += 0.2
        
        # Model configuration confidence
        if model_config.framework != MLFramework.SCIKIT_LEARN:  # Non-default framework
            score += 0.2
        if model_config.algorithm:
            score += 0.1
        
        # Target column confidence
        if target_column:
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_matched_patterns(self, instruction_lower: str) -> Dict[str, List[str]]:
        """Get patterns that matched during parsing."""
        matched = {}
        
        for category, subcategories in self.patterns.items():
            matched[category] = []
            for subcategory, patterns in subcategories.items():
                for pattern in patterns:
                    if re.search(pattern, instruction_lower):
                        matched[category].append(f"{subcategory}: {pattern}")
        
        return matched
    
    def validate_parsed_instruction(self, parsed: ParsedInstruction) -> Tuple[bool, List[str]]:
        """
        Validate a parsed instruction for completeness and consistency.
        
        Args:
            parsed: ParsedInstruction to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        if parsed.task_type == TaskType.OTHER:
            issues.append("Task type could not be determined")
        
        if parsed.data_source.type == DataSourceType.OTHER:
            issues.append("Data source type could not be determined")
        
        if not parsed.data_source.location and parsed.data_source.type in [
            DataSourceType.BIGQUERY, DataSourceType.GCS
        ]:
            issues.append(f"Data source location not specified for {parsed.data_source.type.value}")
        
        # Check task-specific requirements
        if parsed.task_type in [
            TaskType.BINARY_CLASSIFICATION, 
            TaskType.MULTICLASS_CLASSIFICATION, 
            TaskType.REGRESSION
        ] and not parsed.target_column:
            issues.append("Target column not specified for supervised learning task")
        
        # Check confidence score
        if parsed.confidence_score < 0.5:
            issues.append(f"Low confidence score: {parsed.confidence_score:.2f}")
        
        return len(issues) == 0, issues
