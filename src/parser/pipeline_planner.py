"""
Pipeline Planner for Vertex AI ML Pipelines

This module converts parsed instructions into structured Vertex AI pipeline plans
with components for data ingestion, training, deployment, and monitoring.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .instruction_parser import ParsedInstruction, TaskType, DataSourceType, MLFramework

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of pipeline components."""
    DATA_INGESTION = "data_ingestion"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_VALIDATION = "model_validation"
    MODEL_REGISTRATION = "model_registration"
    MODEL_DEPLOYMENT = "model_deployment"
    MONITORING_SETUP = "monitoring_setup"
    CICD_SETUP = "cicd_setup"


@dataclass
class PipelineComponent:
    """Individual pipeline component specification."""
    name: str
    type: ComponentType
    description: str
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    container_spec: Optional[Dict[str, Any]] = None
    resource_spec: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class PipelinePlan:
    """Complete ML pipeline plan for Vertex AI."""
    name: str
    description: str
    components: List[PipelineComponent]
    pipeline_parameters: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[str] = None
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelinePlanner:
    """
    Pipeline planner that converts parsed instructions into Vertex AI pipeline plans.
    """
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        """
        Initialize the pipeline planner.
        
        Args:
            project_id: GCP project ID
            region: GCP region for Vertex AI resources
        """
        self.project_id = project_id
        self.region = region
        self._init_component_templates()
        self._init_container_mappings()
        logger.info(f"Pipeline planner initialized for project {project_id}")
    
    def _init_component_templates(self):
        """Initialize component templates for different pipeline stages."""
        self.component_templates = {
            ComponentType.DATA_INGESTION: {
                "bigquery": {
                    "container_uri": "gcr.io/ml-pipeline/bigquery-component:latest",
                    "default_parameters": {
                        "query": "",
                        "destination_table": "",
                        "write_disposition": "WRITE_TRUNCATE"
                    }
                },
                "gcs": {
                    "container_uri": "gcr.io/ml-pipeline/gcs-component:latest",
                    "default_parameters": {
                        "source_path": "",
                        "destination_path": "",
                        "file_format": "csv"
                    }
                }
            },
            ComponentType.DATA_PREPROCESSING: {
                "default": {
                    "container_uri": "gcr.io/ml-pipeline/preprocessing-component:latest",
                    "default_parameters": {
                        "missing_value_strategy": "mean",
                        "scaling_method": "standard",
                        "encoding_method": "onehot"
                    }
                }
            },
            ComponentType.MODEL_TRAINING: {
                "scikit-learn": {
                    "container_uri": "gcr.io/deeplearning-platform-release/sklearn-cpu:latest",
                    "default_parameters": {
                        "algorithm": "RandomForestClassifier",
                        "hyperparameters": {},
                        "cross_validation_folds": 5
                    }
                },
                "xgboost": {
                    "container_uri": "gcr.io/deeplearning-platform-release/xgboost-cpu:latest",
                    "default_parameters": {
                        "objective": "binary:logistic",
                        "eval_metric": "auc",
                        "num_boost_round": 100
                    }
                },
                "tensorflow": {
                    "container_uri": "gcr.io/deeplearning-platform-release/tf2-cpu:latest",
                    "default_parameters": {
                        "model_type": "dnn",
                        "hidden_units": [128, 64, 32],
                        "epochs": 100
                    }
                }
            },
            ComponentType.MODEL_DEPLOYMENT: {
                "endpoint": {
                    "container_uri": "gcr.io/ml-pipeline/deployment-component:latest",
                    "default_parameters": {
                        "machine_type": "n1-standard-4",
                        "min_replica_count": 1,
                        "max_replica_count": 10
                    }
                }
            }
        }
    
    def _init_container_mappings(self):
        """Initialize container URI mappings for different frameworks."""
        self.container_mappings = {
            MLFramework.SCIKIT_LEARN: "gcr.io/deeplearning-platform-release/sklearn-cpu:latest",
            MLFramework.XGBOOST: "gcr.io/deeplearning-platform-release/xgboost-cpu:latest",
            MLFramework.TENSORFLOW: "gcr.io/deeplearning-platform-release/tf2-cpu:latest",
            MLFramework.PYTORCH: "gcr.io/deeplearning-platform-release/pytorch-cpu:latest",
            MLFramework.LIGHTGBM: "gcr.io/ml-pipeline/lightgbm:latest",
            MLFramework.CATBOOST: "gcr.io/ml-pipeline/catboost:latest",
            MLFramework.AUTO_ML: "gcr.io/cloud-aiplatform/training/automl-tabular:latest"
        }
    
    def create_pipeline_plan(self, parsed_instruction: ParsedInstruction) -> PipelinePlan:
        """
        Create a complete pipeline plan from parsed instruction.
        
        Args:
            parsed_instruction: Parsed instruction with ML requirements
            
        Returns:
            Complete pipeline plan for Vertex AI
        """
        logger.info(f"Creating pipeline plan for {parsed_instruction.task_type.value}")
        
        # Generate pipeline components
        components = []
        
        # 1. Data ingestion component
        data_component = self._create_data_ingestion_component(parsed_instruction)
        components.append(data_component)
        
        # 2. Data preprocessing component
        preprocessing_component = self._create_preprocessing_component(parsed_instruction)
        components.append(preprocessing_component)
        
        # 3. Feature engineering component (if needed)
        if self._needs_feature_engineering(parsed_instruction):
            feature_component = self._create_feature_engineering_component(parsed_instruction)
            components.append(feature_component)
        
        # 4. Model training component
        training_component = self._create_training_component(parsed_instruction)
        components.append(training_component)
        
        # 5. Model evaluation component
        evaluation_component = self._create_evaluation_component(parsed_instruction)
        components.append(evaluation_component)
        
        # 6. Model validation component
        validation_component = self._create_validation_component(parsed_instruction)
        components.append(validation_component)
        
        # 7. Model registration component
        registration_component = self._create_registration_component(parsed_instruction)
        components.append(registration_component)
        
        # 8. Model deployment component (if specified)
        if parsed_instruction.deployment_spec:
            deployment_component = self._create_deployment_component(parsed_instruction)
            components.append(deployment_component)
        
        # 9. Monitoring setup component
        monitoring_component = self._create_monitoring_component(parsed_instruction)
        components.append(monitoring_component)
        
        # 10. CI/CD setup component
        cicd_component = self._create_cicd_component(parsed_instruction)
        components.append(cicd_component)
        
        # Create pipeline plan
        pipeline_plan = PipelinePlan(
            name=parsed_instruction.pipeline_name or "ml-pipeline",
            description=parsed_instruction.description,
            components=components,
            pipeline_parameters=self._create_pipeline_parameters(parsed_instruction),
            monitoring_config=self._create_monitoring_config(parsed_instruction),
            deployment_config=self._create_deployment_config(parsed_instruction),
            metadata={
                "created_at": datetime.now().isoformat(),
                "task_type": parsed_instruction.task_type.value,
                "framework": parsed_instruction.model_config.framework.value,
                "data_source_type": parsed_instruction.data_source.type.value,
                "confidence_score": parsed_instruction.confidence_score
            }
        )
        
        logger.info(f"Created pipeline plan with {len(components)} components")
        return pipeline_plan
    
    def _create_data_ingestion_component(self, parsed: ParsedInstruction) -> PipelineComponent:
        """Create data ingestion component."""
        data_source = parsed.data_source
        
        if data_source.type == DataSourceType.BIGQUERY:
            template = self.component_templates[ComponentType.DATA_INGESTION]["bigquery"]
            parameters = template["default_parameters"].copy()
            parameters.update({
                "source_table": data_source.location,
                "project_id": self.project_id
            })
        elif data_source.type == DataSourceType.GCS:
            template = self.component_templates[ComponentType.DATA_INGESTION]["gcs"]
            parameters = template["default_parameters"].copy()
            parameters.update({
                "source_path": data_source.location,
                "project_id": self.project_id
            })
        else:
            # Default template
            template = {"container_uri": "gcr.io/ml-pipeline/data-ingestion:latest"}
            parameters = {"data_source": data_source.location}
        
        return PipelineComponent(
            name="data-ingestion",
            type=ComponentType.DATA_INGESTION,
            description=f"Ingest data from {data_source.type.value}",
            outputs={"dataset": "Dataset"},
            parameters=parameters,
            container_spec={
                "image_uri": template["container_uri"],
                "command": ["python", "ingest_data.py"],
                "args": ["--source-type", data_source.type.value]
            },
            resource_spec={
                "machine_type": "n1-standard-4",
                "disk_size_gb": 100
            }
        )
    
    def _create_preprocessing_component(self, parsed: ParsedInstruction) -> PipelineComponent:
        """Create data preprocessing component."""
        template = self.component_templates[ComponentType.DATA_PREPROCESSING]["default"]
        parameters = template["default_parameters"].copy()
        
        # Add task-specific preprocessing
        if parsed.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            parameters["target_encoding"] = "label"
        elif parsed.task_type == TaskType.REGRESSION:
            parameters["target_scaling"] = "standard"
        
        if parsed.target_column:
            parameters["target_column"] = parsed.target_column
        
        if parsed.feature_columns:
            parameters["feature_columns"] = parsed.feature_columns
        
        return PipelineComponent(
            name="data-preprocessing",
            type=ComponentType.DATA_PREPROCESSING,
            description="Preprocess and clean the dataset",
            inputs={"dataset": "Dataset"},
            outputs={"processed_dataset": "Dataset"},
            parameters=parameters,
            container_spec={
                "image_uri": template["container_uri"],
                "command": ["python", "preprocess_data.py"]
            },
            resource_spec={
                "machine_type": "n1-standard-4",
                "disk_size_gb": 50
            },
            dependencies=["data-ingestion"]
        )
    
    def _create_feature_engineering_component(self, parsed: ParsedInstruction) -> PipelineComponent:
        """Create feature engineering component."""
        return PipelineComponent(
            name="feature-engineering",
            type=ComponentType.FEATURE_ENGINEERING,
            description="Engineer features for the model",
            inputs={"processed_dataset": "Dataset"},
            outputs={"feature_dataset": "Dataset"},
            parameters={
                "feature_selection": True,
                "feature_importance_threshold": 0.01,
                "polynomial_features": False
            },
            container_spec={
                "image_uri": "gcr.io/ml-pipeline/feature-engineering:latest",
                "command": ["python", "engineer_features.py"]
            },
            resource_spec={
                "machine_type": "n1-standard-4",
                "disk_size_gb": 50
            },
            dependencies=["data-preprocessing"]
        )
    
    def _create_training_component(self, parsed: ParsedInstruction) -> PipelineComponent:
        """Create model training component."""
        framework = parsed.model_config.framework
        
        # Get framework-specific template
        framework_key = framework.value
        if framework_key in self.component_templates[ComponentType.MODEL_TRAINING]:
            template = self.component_templates[ComponentType.MODEL_TRAINING][framework_key]
        else:
            template = self.component_templates[ComponentType.MODEL_TRAINING]["scikit-learn"]
        
        parameters = template["default_parameters"].copy()
        
        # Add parsed hyperparameters
        if parsed.model_config.hyperparameters:
            parameters["hyperparameters"] = parsed.model_config.hyperparameters
        
        # Add algorithm if specified
        if parsed.model_config.algorithm:
            parameters["algorithm"] = parsed.model_config.algorithm
        
        # Add task-specific parameters
        if parsed.task_type == TaskType.BINARY_CLASSIFICATION:
            if framework == MLFramework.XGBOOST:
                parameters["objective"] = "binary:logistic"
                parameters["eval_metric"] = "auc"
        elif parsed.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            if framework == MLFramework.XGBOOST:
                parameters["objective"] = "multi:softprob"
                parameters["eval_metric"] = "mlogloss"
        elif parsed.task_type == TaskType.REGRESSION:
            if framework == MLFramework.XGBOOST:
                parameters["objective"] = "reg:squarederror"
                parameters["eval_metric"] = "rmse"
        
        input_key = "feature_dataset" if self._needs_feature_engineering(parsed) else "processed_dataset"
        
        return PipelineComponent(
            name="model-training",
            type=ComponentType.MODEL_TRAINING,
            description=f"Train {framework.value} model",
            inputs={input_key: "Dataset"},
            outputs={"model": "Model", "metrics": "Metrics"},
            parameters=parameters,
            container_spec={
                "image_uri": template["container_uri"],
                "command": ["python", "train_model.py"],
                "args": ["--framework", framework.value]
            },
            resource_spec={
                "machine_type": "n1-standard-8",
                "disk_size_gb": 100,
                "accelerator_type": "NVIDIA_TESLA_T4" if framework in [MLFramework.TENSORFLOW, MLFramework.PYTORCH] else None,
                "accelerator_count": 1 if framework in [MLFramework.TENSORFLOW, MLFramework.PYTORCH] else 0
            },
            dependencies=["feature-engineering"] if self._needs_feature_engineering(parsed) else ["data-preprocessing"]
        )
    
    def _create_evaluation_component(self, parsed: ParsedInstruction) -> PipelineComponent:
        """Create model evaluation component."""
        metrics = parsed.model_config.evaluation_metrics or ["accuracy"]
        
        return PipelineComponent(
            name="model-evaluation",
            type=ComponentType.MODEL_EVALUATION,
            description="Evaluate model performance",
            inputs={"model": "Model", "test_dataset": "Dataset"},
            outputs={"evaluation_metrics": "Metrics", "evaluation_report": "Report"},
            parameters={
                "metrics": metrics,
                "cross_validation": True,
                "test_size": 0.2
            },
            container_spec={
                "image_uri": "gcr.io/ml-pipeline/model-evaluation:latest",
                "command": ["python", "evaluate_model.py"]
            },
            resource_spec={
                "machine_type": "n1-standard-4",
                "disk_size_gb": 50
            },
            dependencies=["model-training"]
        )
    
    def _create_validation_component(self, parsed: ParsedInstruction) -> PipelineComponent:
        """Create model validation component."""
        return PipelineComponent(
            name="model-validation",
            type=ComponentType.MODEL_VALIDATION,
            description="Validate model meets quality thresholds",
            inputs={"evaluation_metrics": "Metrics"},
            outputs={"validation_result": "ValidationResult"},
            parameters={
                "min_accuracy": 0.8,
                "max_bias": 0.1,
                "performance_thresholds": {
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.7
                }
            },
            container_spec={
                "image_uri": "gcr.io/ml-pipeline/model-validation:latest",
                "command": ["python", "validate_model.py"]
            },
            resource_spec={
                "machine_type": "n1-standard-2",
                "disk_size_gb": 20
            },
            dependencies=["model-evaluation"]
        )
    
    def _create_registration_component(self, parsed: ParsedInstruction) -> PipelineComponent:
        """Create model registration component."""
        return PipelineComponent(
            name="model-registration",
            type=ComponentType.MODEL_REGISTRATION,
            description="Register model in Vertex AI Model Registry",
            inputs={"model": "Model", "validation_result": "ValidationResult"},
            outputs={"registered_model": "RegisteredModel"},
            parameters={
                "model_name": f"{parsed.pipeline_name}-model",
                "model_version": "1.0.0",
                "project_id": self.project_id,
                "region": self.region
            },
            container_spec={
                "image_uri": "gcr.io/ml-pipeline/model-registration:latest",
                "command": ["python", "register_model.py"]
            },
            resource_spec={
                "machine_type": "n1-standard-2",
                "disk_size_gb": 20
            },
            dependencies=["model-validation"]
        )
    
    def _create_deployment_component(self, parsed: ParsedInstruction) -> PipelineComponent:
        """Create model deployment component."""
        deployment_spec = parsed.deployment_spec
        
        return PipelineComponent(
            name="model-deployment",
            type=ComponentType.MODEL_DEPLOYMENT,
            description="Deploy model to Vertex AI Endpoint",
            inputs={"registered_model": "RegisteredModel"},
            outputs={"endpoint": "Endpoint"},
            parameters={
                "endpoint_name": deployment_spec.endpoint_name,
                "machine_type": deployment_spec.machine_type,
                "min_replica_count": deployment_spec.min_replica_count,
                "max_replica_count": deployment_spec.max_replica_count,
                "traffic_split": deployment_spec.traffic_split,
                "project_id": self.project_id,
                "region": self.region
            },
            container_spec={
                "image_uri": "gcr.io/ml-pipeline/model-deployment:latest",
                "command": ["python", "deploy_model.py"]
            },
            resource_spec={
                "machine_type": "n1-standard-2",
                "disk_size_gb": 20
            },
            dependencies=["model-registration"]
        )
    
    def _create_monitoring_component(self, parsed: ParsedInstruction) -> PipelineComponent:
        """Create monitoring setup component."""
        monitoring_spec = parsed.monitoring_spec
        
        return PipelineComponent(
            name="monitoring-setup",
            type=ComponentType.MONITORING_SETUP,
            description="Set up model monitoring and drift detection",
            inputs={"endpoint": "Endpoint"} if parsed.deployment_spec else {"registered_model": "RegisteredModel"},
            outputs={"monitoring_job": "MonitoringJob"},
            parameters={
                "enable_drift_detection": monitoring_spec.enable_drift_detection,
                "enable_performance_monitoring": monitoring_spec.enable_performance_monitoring,
                "drift_threshold": monitoring_spec.drift_threshold,
                "monitoring_frequency": monitoring_spec.monitoring_frequency,
                "project_id": self.project_id,
                "region": self.region
            },
            container_spec={
                "image_uri": "gcr.io/ml-pipeline/monitoring-setup:latest",
                "command": ["python", "setup_monitoring.py"]
            },
            resource_spec={
                "machine_type": "n1-standard-2",
                "disk_size_gb": 20
            },
            dependencies=["model-deployment"] if parsed.deployment_spec else ["model-registration"]
        )
    
    def _create_cicd_component(self, parsed: ParsedInstruction) -> PipelineComponent:
        """Create CI/CD setup component."""
        return PipelineComponent(
            name="cicd-setup",
            type=ComponentType.CICD_SETUP,
            description="Set up CI/CD triggers for retraining",
            inputs={"monitoring_job": "MonitoringJob"},
            outputs={"cicd_config": "CICDConfig"},
            parameters={
                "trigger_on_drift": True,
                "trigger_on_performance_degradation": True,
                "retraining_schedule": "weekly",
                "project_id": self.project_id,
                "region": self.region
            },
            container_spec={
                "image_uri": "gcr.io/ml-pipeline/cicd-setup:latest",
                "command": ["python", "setup_cicd.py"]
            },
            resource_spec={
                "machine_type": "n1-standard-2",
                "disk_size_gb": 20
            },
            dependencies=["monitoring-setup"]
        )
    
    def _needs_feature_engineering(self, parsed: ParsedInstruction) -> bool:
        """Determine if feature engineering component is needed."""
        # Add feature engineering for complex tasks or when explicitly mentioned
        return (
            parsed.task_type in [TaskType.TIME_SERIES_FORECASTING, TaskType.RECOMMENDATION] or
            (parsed.feature_columns and len(parsed.feature_columns) > 10) or
            "feature" in parsed.raw_instruction.lower()
        )
    
    def _create_pipeline_parameters(self, parsed: ParsedInstruction) -> Dict[str, Any]:
        """Create pipeline-level parameters."""
        return {
            "project_id": self.project_id,
            "region": self.region,
            "pipeline_name": parsed.pipeline_name,
            "task_type": parsed.task_type.value,
            "framework": parsed.model_config.framework.value,
            "data_source": parsed.data_source.location,
            "target_column": parsed.target_column,
            "feature_columns": parsed.feature_columns
        }
    
    def _create_monitoring_config(self, parsed: ParsedInstruction) -> Dict[str, Any]:
        """Create monitoring configuration."""
        monitoring_spec = parsed.monitoring_spec
        
        return {
            "drift_detection": {
                "enabled": monitoring_spec.enable_drift_detection,
                "threshold": monitoring_spec.drift_threshold,
                "frequency": monitoring_spec.monitoring_frequency
            },
            "performance_monitoring": {
                "enabled": monitoring_spec.enable_performance_monitoring,
                "threshold": monitoring_spec.performance_threshold
            },
            "data_quality": {
                "enabled": monitoring_spec.enable_data_quality_monitoring,
                "schema_validation": True
            },
            "alerting": {
                "email_notifications": True,
                "slack_notifications": False
            }
        }
    
    def _create_deployment_config(self, parsed: ParsedInstruction) -> Dict[str, Any]:
        """Create deployment configuration."""
        if not parsed.deployment_spec:
            return {}
        
        deployment_spec = parsed.deployment_spec
        
        return {
            "endpoint": {
                "name": deployment_spec.endpoint_name,
                "machine_type": deployment_spec.machine_type,
                "min_replicas": deployment_spec.min_replica_count,
                "max_replicas": deployment_spec.max_replica_count,
                "auto_scaling": deployment_spec.auto_scaling
            },
            "traffic_allocation": deployment_spec.traffic_split,
            "health_checks": {
                "enabled": True,
                "timeout": 30,
                "interval": 60
            }
        }
    
    def generate_kfp_pipeline(self, pipeline_plan: PipelinePlan) -> str:
        """
        Generate Kubeflow Pipelines (KFP) Python code for the pipeline plan.
        
        Args:
            pipeline_plan: Complete pipeline plan
            
        Returns:
            Python code string for KFP pipeline
        """
        logger.info(f"Generating KFP pipeline code for {pipeline_plan.name}")
        
        # Generate imports
        imports = [
            "from kfp import dsl",
            "from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics",
            "from typing import NamedTuple"
        ]
        
        # Generate component definitions
        component_defs = []
        for comp in pipeline_plan.components:
            comp_def = self._generate_component_definition(comp)
            component_defs.append(comp_def)
        
        # Generate pipeline definition
        pipeline_def = self._generate_pipeline_definition(pipeline_plan)
        
        # Combine all parts
        kfp_code = "\n\n".join([
            "\n".join(imports),
            "\n\n".join(component_defs),
            pipeline_def
        ])
        
        return kfp_code
    
    def _generate_component_definition(self, component: PipelineComponent) -> str:
        """Generate KFP component definition."""
        # Simplified component generation - in practice, this would be more complex
        return f"""
@component(
    base_image="{component.container_spec['image_uri'] if component.container_spec else 'python:3.9'}",
    packages_to_install=[]
)
def {component.name.replace('-', '_')}({self._generate_component_signature(component)}) -> NamedTuple('Outputs', {self._generate_output_signature(component)}):
    '''
    {component.description}
    '''
    # Component implementation would go here
    pass
"""
    
    def _generate_component_signature(self, component: PipelineComponent) -> str:
        """Generate component function signature."""
        inputs = []
        for name, type_name in component.inputs.items():
            inputs.append(f"{name}: Input[{type_name}]")
        
        for name, value in component.parameters.items():
            if isinstance(value, str):
                inputs.append(f"{name}: str = '{value}'")
            elif isinstance(value, (int, float)):
                inputs.append(f"{name}: {type(value).__name__} = {value}")
            else:
                inputs.append(f"{name}: str = '{json.dumps(value)}'")
        
        return ", ".join(inputs)
    
    def _generate_output_signature(self, component: PipelineComponent) -> str:
        """Generate component output signature."""
        outputs = []
        for name, type_name in component.outputs.items():
            outputs.append(f"('{name}', {type_name})")
        
        return "[" + ", ".join(outputs) + "]"
    
    def _generate_pipeline_definition(self, pipeline_plan: PipelinePlan) -> str:
        """Generate KFP pipeline definition."""
        component_calls = []
        
        for component in pipeline_plan.components:
            call = f"    {component.name.replace('-', '_')}_task = {component.name.replace('-', '_')}("
            
            # Add input connections
            inputs = []
            for input_name in component.inputs.keys():
                # Find the component that produces this input
                for dep_comp in pipeline_plan.components:
                    if input_name in dep_comp.outputs:
                        inputs.append(f"{input_name}={dep_comp.name.replace('-', '_')}_task.outputs['{input_name}']")
                        break
            
            call += ", ".join(inputs) + ")"
            component_calls.append(call)
        
        return f"""
@pipeline(
    name="{pipeline_plan.name}",
    description="{pipeline_plan.description}"
)
def {pipeline_plan.name.replace('-', '_')}_pipeline():
    '''
    {pipeline_plan.description}
    '''
{chr(10).join(component_calls)}
"""
    
    def export_pipeline_yaml(self, pipeline_plan: PipelinePlan) -> str:
        """
        Export pipeline plan as YAML configuration.
        
        Args:
            pipeline_plan: Complete pipeline plan
            
        Returns:
            YAML string representation
        """
        import yaml
        
        pipeline_dict = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Workflow",
            "metadata": {
                "name": pipeline_plan.name,
                "annotations": {
                    "description": pipeline_plan.description
                }
            },
            "spec": {
                "entrypoint": "main",
                "templates": []
            }
        }
        
        # Add component templates
        for component in pipeline_plan.components:
            template = {
                "name": component.name,
                "container": {
                    "image": component.container_spec["image_uri"] if component.container_spec else "python:3.9",
                    "command": component.container_spec.get("command", ["python"]) if component.container_spec else ["python"],
                    "args": component.container_spec.get("args", []) if component.container_spec else [],
                    "env": [
                        {"name": k, "value": str(v)} for k, v in component.parameters.items()
                    ]
                },
                "inputs": {
                    "artifacts": [
                        {"name": name, "path": f"/tmp/{name}"} for name in component.inputs.keys()
                    ]
                },
                "outputs": {
                    "artifacts": [
                        {"name": name, "path": f"/tmp/{name}"} for name in component.outputs.keys()
                    ]
                }
            }
            
            if component.resource_spec:
                template["container"]["resources"] = {
                    "requests": {
                        "memory": "4Gi",
                        "cpu": "2"
                    }
                }
            
            pipeline_dict["spec"]["templates"].append(template)
        
        # Add main template with DAG
        main_template = {
            "name": "main",
            "dag": {
                "tasks": []
            }
        }
        
        for component in pipeline_plan.components:
            task = {
                "name": component.name,
                "template": component.name
            }
            
            if component.dependencies:
                task["dependencies"] = component.dependencies
            
            main_template["dag"]["tasks"].append(task)
        
        pipeline_dict["spec"]["templates"].append(main_template)
        
        return yaml.dump(pipeline_dict, default_flow_style=False)
