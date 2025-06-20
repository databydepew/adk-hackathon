"""
Model Training Pipeline Components

This module provides components for training ML models using different frameworks
including scikit-learn, XGBoost, TensorFlow, and PyTorch.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

logger = logging.getLogger(__name__)


class MLFramework(Enum):
    """Supported ML frameworks."""
    SCIKIT_LEARN = "scikit-learn"
    XGBOOST = "xgboost"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


class TaskType(Enum):
    """ML task types."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


@dataclass
class TrainingConfig:
    """Configuration for model training components."""
    framework: MLFramework
    task_type: TaskType
    algorithm: Optional[str] = None
    hyperparameters: Dict[str, Any] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    test_size: float = 0.2
    validation_size: float = 0.1
    cross_validation_folds: int = 5
    random_state: int = 42
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {}


class BaseModelTrainingComponent(ABC):
    """Base class for model training components."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the model training component.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.component_name = f"{config.framework.value}_training"
        
    @abstractmethod
    def create_component(self) -> component:
        """Create the KFP component for model training."""
        pass
    
    @abstractmethod
    def get_component_spec(self) -> Dict[str, Any]:
        """Get the component specification."""
        pass


class SklearnTrainingComponent(BaseModelTrainingComponent):
    """Scikit-learn model training component."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize scikit-learn training component."""
        super().__init__(config)
        if config.framework != MLFramework.SCIKIT_LEARN:
            raise ValueError("Config must be for scikit-learn framework")
    
    def create_component(self) -> component:
        """Create scikit-learn training KFP component."""
        
        @component(
            base_image="gcr.io/deeplearning-platform-release/sklearn-cpu:latest",
            packages_to_install=[
                "scikit-learn>=1.3.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "joblib>=1.3.0"
            ]
        )
        def sklearn_training(
            dataset: Input[Dataset],
            model: Output[Model],
            metrics: Output[Metrics],
            algorithm: str = "RandomForestClassifier",
            hyperparameters: str = "{}",
            target_column: str = "target",
            feature_columns: str = "[]",
            test_size: float = 0.2,
            validation_size: float = 0.1,
            cross_validation_folds: int = 5,
            random_state: int = 42,
            task_type: str = "binary_classification"
        ):
            """
            Train a scikit-learn model.
            
            Args:
                dataset: Input dataset
                model: Output trained model
                metrics: Output training metrics
                algorithm: Algorithm to use
                hyperparameters: JSON string of hyperparameters
                target_column: Target column name
                feature_columns: JSON string of feature column names
                test_size: Test set size ratio
                validation_size: Validation set size ratio
                cross_validation_folds: Number of CV folds
                random_state: Random state for reproducibility
                task_type: Type of ML task
            """
            import pandas as pd
            import numpy as np
            import json
            import joblib
            import os
            import logging
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                mean_squared_error, mean_absolute_error, r2_score,
                classification_report, confusion_matrix
            )
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import LogisticRegression, LinearRegression
            from sklearn.svm import SVC, SVR
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            from sklearn.naive_bayes import GaussianNB
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            try:
                # Load dataset
                data_path = os.path.join(dataset.path, "data.csv")
                if not os.path.exists(data_path):
                    data_path = os.path.join(dataset.path, "data.parquet")
                
                if data_path.endswith('.csv'):
                    df = pd.read_csv(data_path)
                else:
                    df = pd.read_parquet(data_path)
                
                logger.info(f"Loaded dataset with shape: {df.shape}")
                
                # Parse hyperparameters and feature columns
                hyperparams = json.loads(hyperparameters) if hyperparameters != "{}" else {}
                feature_cols = json.loads(feature_columns) if feature_columns != "[]" else None
                
                # Prepare features and target
                if feature_cols:
                    X = df[feature_cols]
                else:
                    X = df.drop(columns=[target_column])
                
                y = df[target_column]
                
                logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
                
                # Split data
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y if task_type != "regression" else None
                )
                
                val_size_adjusted = validation_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
                    stratify=y_temp if task_type != "regression" else None
                )
                
                logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
                
                # Initialize model
                model_classes = {
                    "RandomForestClassifier": RandomForestClassifier,
                    "RandomForestRegressor": RandomForestRegressor,
                    "LogisticRegression": LogisticRegression,
                    "LinearRegression": LinearRegression,
                    "SVC": SVC,
                    "SVR": SVR,
                    "DecisionTreeClassifier": DecisionTreeClassifier,
                    "DecisionTreeRegressor": DecisionTreeRegressor,
                    "GaussianNB": GaussianNB,
                    "KNeighborsClassifier": KNeighborsClassifier,
                    "KNeighborsRegressor": KNeighborsRegressor
                }
                
                if algorithm not in model_classes:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")
                
                # Set default hyperparameters
                default_params = {"random_state": random_state}
                if algorithm in ["RandomForestClassifier", "RandomForestRegressor"]:
                    default_params.update({"n_estimators": 100, "max_depth": None})
                
                # Merge with provided hyperparameters
                final_params = {**default_params, **hyperparams}
                
                # Remove random_state for algorithms that don't support it
                if algorithm in ["GaussianNB"]:
                    final_params.pop("random_state", None)
                
                model_instance = model_classes[algorithm](**final_params)
                logger.info(f"Initialized {algorithm} with parameters: {final_params}")
                
                # Train model
                logger.info("Training model...")
                model_instance.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model_instance, X_train, y_train, cv=cross_validation_folds,
                    scoring='accuracy' if task_type != "regression" else 'r2'
                )
                
                # Predictions
                train_pred = model_instance.predict(X_train)
                val_pred = model_instance.predict(X_val)
                test_pred = model_instance.predict(X_test)
                
                # Calculate metrics
                metrics_dict = {}
                
                if task_type == "regression":
                    # Regression metrics
                    metrics_dict.update({
                        "train_mse": float(mean_squared_error(y_train, train_pred)),
                        "train_mae": float(mean_absolute_error(y_train, train_pred)),
                        "train_r2": float(r2_score(y_train, train_pred)),
                        "val_mse": float(mean_squared_error(y_val, val_pred)),
                        "val_mae": float(mean_absolute_error(y_val, val_pred)),
                        "val_r2": float(r2_score(y_val, val_pred)),
                        "test_mse": float(mean_squared_error(y_test, test_pred)),
                        "test_mae": float(mean_absolute_error(y_test, test_pred)),
                        "test_r2": float(r2_score(y_test, test_pred)),
                        "cv_mean": float(cv_scores.mean()),
                        "cv_std": float(cv_scores.std())
                    })
                else:
                    # Classification metrics
                    average = 'binary' if task_type == "binary_classification" else 'weighted'
                    
                    metrics_dict.update({
                        "train_accuracy": float(accuracy_score(y_train, train_pred)),
                        "train_precision": float(precision_score(y_train, train_pred, average=average, zero_division=0)),
                        "train_recall": float(recall_score(y_train, train_pred, average=average, zero_division=0)),
                        "train_f1": float(f1_score(y_train, train_pred, average=average, zero_division=0)),
                        "val_accuracy": float(accuracy_score(y_val, val_pred)),
                        "val_precision": float(precision_score(y_val, val_pred, average=average, zero_division=0)),
                        "val_recall": float(recall_score(y_val, val_pred, average=average, zero_division=0)),
                        "val_f1": float(f1_score(y_val, val_pred, average=average, zero_division=0)),
                        "test_accuracy": float(accuracy_score(y_test, test_pred)),
                        "test_precision": float(precision_score(y_test, test_pred, average=average, zero_division=0)),
                        "test_recall": float(recall_score(y_test, test_pred, average=average, zero_division=0)),
                        "test_f1": float(f1_score(y_test, test_pred, average=average, zero_division=0)),
                        "cv_mean": float(cv_scores.mean()),
                        "cv_std": float(cv_scores.std())
                    })
                
                logger.info(f"Training completed. Test accuracy/R2: {metrics_dict.get('test_accuracy', metrics_dict.get('test_r2')):.4f}")
                
                # Save model
                os.makedirs(model.path, exist_ok=True)
                model_path = os.path.join(model.path, "model.joblib")
                joblib.dump(model_instance, model_path)
                
                # Save model metadata
                model_metadata = {
                    "algorithm": algorithm,
                    "framework": "scikit-learn",
                    "task_type": task_type,
                    "hyperparameters": final_params,
                    "feature_columns": feature_cols or list(X.columns),
                    "target_column": target_column,
                    "model_path": model_path
                }
                
                metadata_path = os.path.join(model.path, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata, f, indent=2)
                
                # Save metrics
                os.makedirs(metrics.path, exist_ok=True)
                metrics_path = os.path.join(metrics.path, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_dict, f, indent=2)
                
                logger.info("Model training completed successfully")
                
            except Exception as e:
                logger.error(f"Model training failed: {str(e)}")
                raise
        
        return sklearn_training
    
    def get_component_spec(self) -> Dict[str, Any]:
        """Get scikit-learn component specification."""
        return {
            "name": self.component_name,
            "description": "Train a scikit-learn model",
            "inputs": {
                "dataset": {"type": "Dataset", "description": "Input dataset"},
                "algorithm": {"type": "String", "description": "Algorithm to use", "default": "RandomForestClassifier"},
                "hyperparameters": {"type": "String", "description": "JSON hyperparameters", "default": "{}"},
                "target_column": {"type": "String", "description": "Target column name", "default": "target"},
                "feature_columns": {"type": "String", "description": "JSON feature columns", "default": "[]"},
                "test_size": {"type": "Float", "description": "Test set size", "default": 0.2},
                "validation_size": {"type": "Float", "description": "Validation set size", "default": 0.1},
                "cross_validation_folds": {"type": "Integer", "description": "CV folds", "default": 5},
                "random_state": {"type": "Integer", "description": "Random state", "default": 42},
                "task_type": {"type": "String", "description": "Task type", "default": "binary_classification"}
            },
            "outputs": {
                "model": {"type": "Model", "description": "Trained model"},
                "metrics": {"type": "Metrics", "description": "Training metrics"}
            }
        }


class XGBoostTrainingComponent(BaseModelTrainingComponent):
    """XGBoost model training component."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize XGBoost training component."""
        super().__init__(config)
        if config.framework != MLFramework.XGBOOST:
            raise ValueError("Config must be for XGBoost framework")
    
    def create_component(self) -> component:
        """Create XGBoost training KFP component."""
        
        @component(
            base_image="gcr.io/deeplearning-platform-release/xgboost-cpu:latest",
            packages_to_install=[
                "xgboost>=1.7.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "scikit-learn>=1.3.0"
            ]
        )
        def xgboost_training(
            dataset: Input[Dataset],
            model: Output[Model],
            metrics: Output[Metrics],
            objective: str = "binary:logistic",
            eval_metric: str = "auc",
            hyperparameters: str = "{}",
            target_column: str = "target",
            feature_columns: str = "[]",
            test_size: float = 0.2,
            validation_size: float = 0.1,
            num_boost_round: int = 100,
            early_stopping_rounds: int = 10,
            random_state: int = 42,
            task_type: str = "binary_classification"
        ):
            """
            Train an XGBoost model.
            
            Args:
                dataset: Input dataset
                model: Output trained model
                metrics: Output training metrics
                objective: XGBoost objective function
                eval_metric: Evaluation metric
                hyperparameters: JSON string of hyperparameters
                target_column: Target column name
                feature_columns: JSON string of feature column names
                test_size: Test set size ratio
                validation_size: Validation set size ratio
                num_boost_round: Number of boosting rounds
                early_stopping_rounds: Early stopping rounds
                random_state: Random state for reproducibility
                task_type: Type of ML task
            """
            import pandas as pd
            import numpy as np
            import json
            import os
            import logging
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                mean_squared_error, mean_absolute_error, r2_score
            )
            
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            try:
                # Load dataset
                data_path = os.path.join(dataset.path, "data.csv")
                if not os.path.exists(data_path):
                    data_path = os.path.join(dataset.path, "data.parquet")
                
                if data_path.endswith('.csv'):
                    df = pd.read_csv(data_path)
                else:
                    df = pd.read_parquet(data_path)
                
                logger.info(f"Loaded dataset with shape: {df.shape}")
                
                # Parse hyperparameters and feature columns
                hyperparams = json.loads(hyperparameters) if hyperparameters != "{}" else {}
                feature_cols = json.loads(feature_columns) if feature_columns != "[]" else None
                
                # Prepare features and target
                if feature_cols:
                    X = df[feature_cols]
                else:
                    X = df.drop(columns=[target_column])
                
                y = df[target_column]
                
                logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
                
                # Split data
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=y if task_type != "regression" else None
                )
                
                val_size_adjusted = validation_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
                    stratify=y_temp if task_type != "regression" else None
                )
                
                logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
                
                # Create DMatrix objects
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                dtest = xgb.DMatrix(X_test, label=y_test)
                
                # Set default parameters
                default_params = {
                    "objective": objective,
                    "eval_metric": eval_metric,
                    "random_state": random_state,
                    "verbosity": 1
                }
                
                # Merge with provided hyperparameters
                final_params = {**default_params, **hyperparams}
                
                logger.info(f"Training XGBoost with parameters: {final_params}")
                
                # Train model
                evallist = [(dtrain, 'train'), (dval, 'val')]
                model_instance = xgb.train(
                    final_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=evallist,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False
                )
                
                # Predictions
                train_pred = model_instance.predict(dtrain)
                val_pred = model_instance.predict(dval)
                test_pred = model_instance.predict(dtest)
                
                # Convert probabilities to classes for classification
                if task_type != "regression":
                    if task_type == "binary_classification":
                        train_pred_class = (train_pred > 0.5).astype(int)
                        val_pred_class = (val_pred > 0.5).astype(int)
                        test_pred_class = (test_pred > 0.5).astype(int)
                    else:
                        train_pred_class = np.argmax(train_pred, axis=1)
                        val_pred_class = np.argmax(val_pred, axis=1)
                        test_pred_class = np.argmax(test_pred, axis=1)
                
                # Calculate metrics
                metrics_dict = {}
                
                if task_type == "regression":
                    # Regression metrics
                    metrics_dict.update({
                        "train_mse": float(mean_squared_error(y_train, train_pred)),
                        "train_mae": float(mean_absolute_error(y_train, train_pred)),
                        "train_r2": float(r2_score(y_train, train_pred)),
                        "val_mse": float(mean_squared_error(y_val, val_pred)),
                        "val_mae": float(mean_absolute_error(y_val, val_pred)),
                        "val_r2": float(r2_score(y_val, val_pred)),
                        "test_mse": float(mean_squared_error(y_test, test_pred)),
                        "test_mae": float(mean_absolute_error(y_test, test_pred)),
                        "test_r2": float(r2_score(y_test, test_pred))
                    })
                else:
                    # Classification metrics
                    average = 'binary' if task_type == "binary_classification" else 'weighted'
                    
                    metrics_dict.update({
                        "train_accuracy": float(accuracy_score(y_train, train_pred_class)),
                        "train_precision": float(precision_score(y_train, train_pred_class, average=average, zero_division=0)),
                        "train_recall": float(recall_score(y_train, train_pred_class, average=average, zero_division=0)),
                        "train_f1": float(f1_score(y_train, train_pred_class, average=average, zero_division=0)),
                        "val_accuracy": float(accuracy_score(y_val, val_pred_class)),
                        "val_precision": float(precision_score(y_val, val_pred_class, average=average, zero_division=0)),
                        "val_recall": float(recall_score(y_val, val_pred_class, average=average, zero_division=0)),
                        "val_f1": float(f1_score(y_val, val_pred_class, average=average, zero_division=0)),
                        "test_accuracy": float(accuracy_score(y_test, test_pred_class)),
                        "test_precision": float(precision_score(y_test, test_pred_class, average=average, zero_division=0)),
                        "test_recall": float(recall_score(y_test, test_pred_class, average=average, zero_division=0)),
                        "test_f1": float(f1_score(y_test, test_pred_class, average=average, zero_division=0))
                    })
                
                logger.info(f"Training completed. Test accuracy/R2: {metrics_dict.get('test_accuracy', metrics_dict.get('test_r2')):.4f}")
                
                # Save model
                os.makedirs(model.path, exist_ok=True)
                model_path = os.path.join(model.path, "model.xgb")
                model_instance.save_model(model_path)
                
                # Save model metadata
                model_metadata = {
                    "framework": "xgboost",
                    "task_type": task_type,
                    "objective": objective,
                    "eval_metric": eval_metric,
                    "hyperparameters": final_params,
                    "feature_columns": feature_cols or list(X.columns),
                    "target_column": target_column,
                    "model_path": model_path,
                    "num_boost_round": model_instance.num_boosted_rounds()
                }
                
                metadata_path = os.path.join(model.path, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata, f, indent=2)
                
                # Save metrics
                os.makedirs(metrics.path, exist_ok=True)
                metrics_path = os.path.join(metrics.path, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_dict, f, indent=2)
                
                logger.info("XGBoost training completed successfully")
                
            except Exception as e:
                logger.error(f"XGBoost training failed: {str(e)}")
                raise
        
        return xgboost_training
    
    def get_component_spec(self) -> Dict[str, Any]:
        """Get XGBoost component specification."""
        return {
            "name": self.component_name,
            "description": "Train an XGBoost model",
            "inputs": {
                "dataset": {"type": "Dataset", "description": "Input dataset"},
                "objective": {"type": "String", "description": "XGBoost objective", "default": "binary:logistic"},
                "eval_metric": {"type": "String", "description": "Evaluation metric", "default": "auc"},
                "hyperparameters": {"type": "String", "description": "JSON hyperparameters", "default": "{}"},
                "target_column": {"type": "String", "description": "Target column name", "default": "target"},
                "feature_columns": {"type": "String", "description": "JSON feature columns", "default": "[]"},
                "test_size": {"type": "Float", "description": "Test set size", "default": 0.2},
                "validation_size": {"type": "Float", "description": "Validation set size", "default": 0.1},
                "num_boost_round": {"type": "Integer", "description": "Boosting rounds", "default": 100},
                "early_stopping_rounds": {"type": "Integer", "description": "Early stopping", "default": 10},
                "random_state": {"type": "Integer", "description": "Random state", "default": 42},
                "task_type": {"type": "String", "description": "Task type", "default": "binary_classification"}
            },
            "outputs": {
                "model": {"type": "Model", "description": "Trained model"},
                "metrics": {"type": "Metrics", "description": "Training metrics"}
            }
        }


class TensorFlowTrainingComponent(BaseModelTrainingComponent):
    """TensorFlow model training component."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize TensorFlow training component."""
        super().__init__(config)
        if config.framework != MLFramework.TENSORFLOW:
            raise ValueError("Config must be for TensorFlow framework")
    
    def create_component(self) -> component:
        """Create TensorFlow training KFP component."""
        
        @component(
            base_image="gcr.io/deeplearning-platform-release/tf2-cpu:latest",
            packages_to_install=[
                "tensorflow>=2.13.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "scikit-learn>=1.3.0"
            ]
        )
        def tensorflow_training(
            dataset: Input[Dataset],
            model: Output[Model],
            metrics: Output[Metrics],
            model_type: str = "dnn",
            hidden_units: str = "[128, 64, 32]",
            hyperparameters: str = "{}",
            target_column: str = "target",
            feature_columns: str = "[]",
            test_size: float = 0.2,
            validation_size: float = 0.1,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            random_state: int = 42,
            task_type: str = "binary_classification"
        ):
            """
            Train a TensorFlow model.
            
            Args:
                dataset: Input dataset
                model: Output trained model
                metrics: Output training metrics
                model_type: Type of model (dnn, cnn, rnn)
                hidden_units: JSON list of hidden units
                hyperparameters: JSON string of hyperparameters
                target_column: Target column name
                feature_columns: JSON string of feature column names
                test_size: Test set size ratio
                validation_size: Validation set size ratio
                epochs: Number of training epochs
                batch_size: Batch size for training
                learning_rate: Learning rate
                random_state: Random state for reproducibility
                task_type: Type of ML task
            """
            import pandas as pd
            import numpy as np
            import json
            import os
            import logging
            import tensorflow as tf
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                mean_squared_error, mean_absolute_error, r2_score
            )
            
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            try:
                # Set random seeds
                tf.random.set_seed(random_state)
                np.random.seed(random_state)
                
                # Load dataset
                data_path = os.path.join(dataset.path, "data.csv")
                if not os.path.exists(data_path):
                    data_path = os.path.join(dataset.path, "data.parquet")
                
                if data_path.endswith('.csv'):
                    df = pd.read_csv(data_path)
                else:
                    df = pd.read_parquet(data_path)
                
                logger.info(f"Loaded dataset with shape: {df.shape}")
                
                # Parse hyperparameters and feature columns
                hyperparams = json.loads(hyperparameters) if hyperparameters != "{}" else {}
                feature_cols = json.loads(feature_columns) if feature_columns != "[]" else None
                hidden_units_list = json.loads(hidden_units)
                
                # Prepare features and target
                if feature_cols:
                    X = df[feature_cols]
                else:
                    X = df.drop(columns=[target_column])
                
                y = df[target_column]
                
                logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
                
                # Preprocessing
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Encode target for classification
                if task_type != "regression":
                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y)
                    num_classes = len(label_encoder.classes_)
                else:
                    y_encoded = y.values
                    num_classes = 1
                
                # Split data
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X_scaled, y_encoded, test_size=test_size, random_state=random_state,
                    stratify=y_encoded if task_type != "regression" else None
                )
                
                val_size_adjusted = validation_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
                    stratify=y_temp if task_type != "regression" else None
                )
                
                logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
                
                # Build model
                model_instance = tf.keras.Sequential()
                
                # Input layer
                model_instance.add(tf.keras.layers.Dense(
                    hidden_units_list[0], 
                    activation='relu', 
                    input_shape=(X_train.shape[1],)
                ))
                model_instance.add(tf.keras.layers.Dropout(0.2))
                
                # Hidden layers
                for units in hidden_units_list[1:]:
                    model_instance.add(tf.keras.layers.Dense(units, activation='relu'))
                    model_instance.add(tf.keras.layers.Dropout(0.2))
                
                # Output layer
                if task_type == "regression":
                    model_instance.add(tf.keras.layers.Dense(1))
                    loss = 'mse'
                    metrics_list = ['mae']
                elif task_type == "binary_classification":
                    model_instance.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                    loss = 'binary_crossentropy'
                    metrics_list = ['accuracy']
                else:  # multiclass
                    model_instance.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
                    loss = 'sparse_categorical_crossentropy'
                    metrics_list = ['accuracy']
                
                # Compile model
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                model_instance.compile(optimizer=optimizer, loss=loss, metrics=metrics_list)
                
                logger.info(f"Built model with {model_instance.count_params()} parameters")
                
                # Callbacks
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=10, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
                    )
                ]
                
                # Train model
                logger.info("Training model...")
                history = model_instance.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Predictions
                train_pred = model_instance.predict(X_train, verbose=0)
                val_pred = model_instance.predict(X_val, verbose=0)
                test_pred = model_instance.predict(X_test, verbose=0)
                
                # Convert predictions for classification
                if task_type != "regression":
                    if task_type == "binary_classification":
                        train_pred_class = (train_pred > 0.5).astype(int).flatten()
                        val_pred_class = (val_pred > 0.5).astype(int).flatten()
                        test_pred_class = (test_pred > 0.5).astype(int).flatten()
                    else:
                        train_pred_class = np.argmax(train_pred, axis=1)
                        val_pred_class = np.argmax(val_pred, axis=1)
                        test_pred_class = np.argmax(test_pred, axis=1)
                else:
                    train_pred = train_pred.flatten()
                    val_pred = val_pred.flatten()
                    test_pred = test_pred.flatten()
                
                # Calculate metrics
                metrics_dict = {}
                
                if task_type == "regression":
                    # Regression metrics
                    metrics_dict.update({
                        "train_mse": float(mean_squared_error(y_train, train_pred)),
                        "train_mae": float(mean_absolute_error(y_train, train_pred)),
                        "train_r2": float(r2_score(y_train, train_pred)),
                        "val_mse": float(mean_squared_error(y_val, val_pred)),
                        "val_mae": float(mean_absolute_error(y_val, val_pred)),
                        "val_r2": float(r2_score(y_val, val_pred)),
                        "test_mse": float(mean_squared_error(y_test, test_pred)),
                        "test_mae": float(mean_absolute_error(y_test, test_pred)),
                        "test_r2": float(r2_score(y_test, test_pred))
                    })
                else:
                    # Classification metrics
                    average = 'binary' if task_type == "binary_classification" else 'weighted'
                    
                    metrics_dict.update({
                        "train_accuracy": float(accuracy_score(y_train, train_pred_class)),
                        "train_precision": float(precision_score(y_train, train_pred_class, average=average, zero_division=0)),
                        "train_recall": float(recall_score(y_train, train_pred_class, average=average, zero_division=0)),
                        "train_f1": float(f1_score(y_train, train_pred_class, average=average, zero_division=0)),
                        "val_accuracy": float(accuracy_score(y_val, val_pred_class)),
                        "val_precision": float(precision_score(y_val, val_pred_class, average=average, zero_division=0)),
                        "val_recall": float(recall_score(y_val, val_pred_class, average=average, zero_division=0)),
                        "val_f1": float(f1_score(y_val, val_pred_class, average=average, zero_division=0)),
                        "test_accuracy": float(accuracy_score(y_test, test_pred_class)),
                        "test_precision": float(precision_score(y_test, test_pred_class, average=average, zero_division=0)),
                        "test_recall": float(recall_score(y_test, test_pred_class, average=average, zero_division=0)),
                        "test_f1": float(f1_score(y_test, test_pred_class, average=average, zero_division=0))
                    })
                
                # Add training history
                metrics_dict["training_history"] = {
                    "loss": [float(x) for x in history.history['loss']],
                    "val_loss": [float(x) for x in history.history['val_loss']]
                }
                
                logger.info(f"Training completed. Test accuracy/R2: {metrics_dict.get('test_accuracy', metrics_dict.get('test_r2')):.4f}")
                
                # Save model
                os.makedirs(model.path, exist_ok=True)
                model_path = os.path.join(model.path, "model")
                model_instance.save(model_path)
                
                # Save preprocessing objects
                import joblib
                scaler_path = os.path.join(model.path, "scaler.joblib")
                joblib.dump(scaler, scaler_path)
                
                if task_type != "regression":
                    encoder_path = os.path.join(model.path, "label_encoder.joblib")
                    joblib.dump(label_encoder, encoder_path)
                
                # Save model metadata
                model_metadata = {
                    "framework": "tensorflow",
                    "task_type": task_type,
                    "model_type": model_type,
                    "hidden_units": hidden_units_list,
                    "hyperparameters": hyperparams,
                    "feature_columns": feature_cols or list(X.columns),
                    "target_column": target_column,
                    "model_path": model_path,
                    "scaler_path": scaler_path,
                    "num_classes": num_classes if task_type != "regression" else None,
                    "epochs_trained": len(history.history['loss'])
                }
                
                if task_type != "regression":
                    model_metadata["label_encoder_path"] = encoder_path
                    model_metadata["classes"] = label_encoder.classes_.tolist()
                
                metadata_path = os.path.join(model.path, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata, f, indent=2)
                
                # Save metrics
                os.makedirs(metrics.path, exist_ok=True)
                metrics_path = os.path.join(metrics.path, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_dict, f, indent=2)
                
                logger.info("TensorFlow training completed successfully")
                
            except Exception as e:
                logger.error(f"TensorFlow training failed: {str(e)}")
                raise
        
        return tensorflow_training
    
    def get_component_spec(self) -> Dict[str, Any]:
        """Get TensorFlow component specification."""
        return {
            "name": self.component_name,
            "description": "Train a TensorFlow model",
            "inputs": {
                "dataset": {"type": "Dataset", "description": "Input dataset"},
                "model_type": {"type": "String", "description": "Model type", "default": "dnn"},
                "hidden_units": {"type": "String", "description": "JSON hidden units", "default": "[128, 64, 32]"},
                "hyperparameters": {"type": "String", "description": "JSON hyperparameters", "default": "{}"},
                "target_column": {"type": "String", "description": "Target column name", "default": "target"},
                "feature_columns": {"type": "String", "description": "JSON feature columns", "default": "[]"},
                "test_size": {"type": "Float", "description": "Test set size", "default": 0.2},
                "validation_size": {"type": "Float", "description": "Validation set size", "default": 0.1},
                "epochs": {"type": "Integer", "description": "Training epochs", "default": 100},
                "batch_size": {"type": "Integer", "description": "Batch size", "default": 32},
                "learning_rate": {"type": "Float", "description": "Learning rate", "default": 0.001},
                "random_state": {"type": "Integer", "description": "Random state", "default": 42},
                "task_type": {"type": "String", "description": "Task type", "default": "binary_classification"}
            },
            "outputs": {
                "model": {"type": "Model", "description": "Trained model"},
                "metrics": {"type": "Metrics", "description": "Training metrics"}
            }
        }


class TrainingComponentFactory:
    """Factory for creating training components."""
    
    @staticmethod
    def create_component(config: TrainingConfig) -> BaseModelTrainingComponent:
        """
        Create a training component based on configuration.
        
        Args:
            config: Training configuration
            
        Returns:
            Training component instance
        """
        if config.framework == MLFramework.SCIKIT_LEARN:
            return SklearnTrainingComponent(config)
        elif config.framework == MLFramework.XGBOOST:
            return XGBoostTrainingComponent(config)
        elif config.framework == MLFramework.TENSORFLOW:
            return TensorFlowTrainingComponent(config)
        else:
            raise ValueError(f"Unsupported framework: {config.framework}")
    
    @staticmethod
    def get_supported_frameworks() -> List[MLFramework]:
        """Get list of supported ML frameworks."""
        return [MLFramework.SCIKIT_LEARN, MLFramework.XGBOOST, MLFramework.TENSORFLOW]


# Alias for backward compatibility
ModelTrainingComponent = BaseModelTrainingComponent
