"""
BigQuery Integration for Vertex AI Pipeline Agent

This module provides integration with Google BigQuery for data access,
analysis, and pipeline data source management.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BigQueryConfig:
    """Configuration for BigQuery integration."""
    project_id: str
    dataset_location: str = "US"
    default_dataset: Optional[str] = None
    credentials_path: Optional[str] = None


class BigQueryClient:
    """Client for BigQuery operations."""
    
    def __init__(self, config: BigQueryConfig):
        """
        Initialize BigQuery client.
        
        Args:
            config: BigQuery configuration
        """
        self.config = config
        self.client = bigquery.Client(
            project=config.project_id,
            location=config.dataset_location
        )
        logger.info(f"Initialized BigQuery client for project {config.project_id}")
    
    def execute_query(
        self, 
        query: str, 
        job_config: Optional[bigquery.QueryJobConfig] = None
    ) -> bigquery.QueryJob:
        """
        Execute a BigQuery SQL query.
        
        Args:
            query: SQL query to execute
            job_config: Optional query job configuration
            
        Returns:
            Query job object
        """
        logger.info(f"Executing query: {query[:100]}...")
        
        if job_config is None:
            job_config = bigquery.QueryJobConfig()
        
        query_job = self.client.query(query, job_config=job_config)
        return query_job
    
    def query_to_dataframe(
        self, 
        query: str, 
        max_results: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Execute query and return results as pandas DataFrame.
        
        Args:
            query: SQL query to execute
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with query results
        """
        query_job = self.execute_query(query)
        df = query_job.to_dataframe(max_results=max_results)
        logger.info(f"Query returned {len(df)} rows")
        return df
    
    def get_table_schema(self, table_id: str) -> List[bigquery.SchemaField]:
        """
        Get schema for a BigQuery table.
        
        Args:
            table_id: Table ID in format 'project.dataset.table'
            
        Returns:
            List of schema fields
        """
        table_ref = self.client.get_table(table_id)
        return table_ref.schema
    
    def get_table_info(self, table_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a BigQuery table.
        
        Args:
            table_id: Table ID in format 'project.dataset.table'
            
        Returns:
            Dictionary with table information
        """
        try:
            table = self.client.get_table(table_id)
            
            return {
                "table_id": table_id,
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "created": table.created.isoformat() if table.created else None,
                "modified": table.modified.isoformat() if table.modified else None,
                "schema": [
                    {
                        "name": field.name,
                        "field_type": field.field_type,
                        "mode": field.mode,
                        "description": field.description
                    }
                    for field in table.schema
                ],
                "description": table.description,
                "location": table.location
            }
        except NotFound:
            raise ValueError(f"Table {table_id} not found")
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a SQL query without executing it.
        
        Args:
            query: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            self.client.query(query, job_config=job_config)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def estimate_query_cost(self, query: str) -> Dict[str, Any]:
        """
        Estimate the cost of running a query.
        
        Args:
            query: SQL query to estimate
            
        Returns:
            Dictionary with cost estimation
        """
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = self.client.query(query, job_config=job_config)
        
        # Estimate cost based on bytes processed
        bytes_processed = query_job.total_bytes_processed
        cost_per_tb = 5.0  # $5 per TB as of 2024
        estimated_cost = (bytes_processed / (1024**4)) * cost_per_tb
        
        return {
            "bytes_processed": bytes_processed,
            "estimated_cost_usd": estimated_cost,
            "cost_per_tb": cost_per_tb
        }
    
    def create_dataset(
        self, 
        dataset_id: str, 
        description: Optional[str] = None,
        location: Optional[str] = None
    ) -> bigquery.Dataset:
        """
        Create a new BigQuery dataset.
        
        Args:
            dataset_id: Dataset ID
            description: Optional dataset description
            location: Optional dataset location
            
        Returns:
            Created dataset object
        """
        dataset_ref = bigquery.DatasetReference(self.config.project_id, dataset_id)
        dataset = bigquery.Dataset(dataset_ref)
        
        if description:
            dataset.description = description
        
        if location:
            dataset.location = location
        else:
            dataset.location = self.config.dataset_location
        
        dataset = self.client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Created dataset {dataset_id}")
        return dataset
    
    def list_tables(self, dataset_id: str) -> List[str]:
        """
        List all tables in a dataset.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            List of table IDs
        """
        dataset_ref = bigquery.DatasetReference(self.config.project_id, dataset_id)
        tables = self.client.list_tables(dataset_ref)
        return [table.table_id for table in tables]


class BigQueryIntegration:
    """High-level BigQuery integration for ML pipelines."""
    
    def __init__(self, config: BigQueryConfig):
        """
        Initialize BigQuery integration.
        
        Args:
            config: BigQuery configuration
        """
        self.config = config
        self.client = BigQueryClient(config)
        logger.info("BigQuery integration initialized")
    
    def analyze_table_for_ml(self, table_id: str) -> Dict[str, Any]:
        """
        Analyze a BigQuery table for ML suitability.
        
        Args:
            table_id: Table ID to analyze
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing table {table_id} for ML suitability")
        
        # Get table info
        table_info = self.client.get_table_info(table_id)
        
        # Analyze schema
        schema_analysis = self._analyze_schema(table_info["schema"])
        
        # Get data sample for analysis
        sample_query = f"""
        SELECT *
        FROM `{table_id}`
        LIMIT 1000
        """
        
        try:
            sample_df = self.client.query_to_dataframe(sample_query)
            data_analysis = self._analyze_data_sample(sample_df)
        except Exception as e:
            logger.warning(f"Could not analyze data sample: {e}")
            data_analysis = {}
        
        return {
            "table_info": table_info,
            "schema_analysis": schema_analysis,
            "data_analysis": data_analysis,
            "ml_recommendations": self._generate_ml_recommendations(
                schema_analysis, data_analysis
            )
        }
    
    def _analyze_schema(self, schema: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze table schema for ML features."""
        numeric_columns = []
        categorical_columns = []
        text_columns = []
        datetime_columns = []
        
        for field in schema:
            field_type = field["field_type"]
            field_name = field["name"]
            
            if field_type in ["INTEGER", "FLOAT", "NUMERIC"]:
                numeric_columns.append(field_name)
            elif field_type in ["STRING"]:
                # Could be categorical or text
                if len(field_name) < 50:  # Heuristic for categorical
                    categorical_columns.append(field_name)
                else:
                    text_columns.append(field_name)
            elif field_type in ["TIMESTAMP", "DATE", "DATETIME"]:
                datetime_columns.append(field_name)
        
        return {
            "total_columns": len(schema),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "text_columns": text_columns,
            "datetime_columns": datetime_columns,
            "column_types": {field["name"]: field["field_type"] for field in schema}
        }
    
    def _analyze_data_sample(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data sample for ML insights."""
        analysis = {
            "sample_size": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            analysis["numeric_stats"] = df[numeric_cols].describe().to_dict()
        
        # Analyze categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            analysis["categorical_stats"] = {}
            for col in categorical_cols:
                analysis["categorical_stats"][col] = {
                    "unique_values": df[col].nunique(),
                    "top_values": df[col].value_counts().head(5).to_dict()
                }
        
        return analysis
    
    def _generate_ml_recommendations(
        self, 
        schema_analysis: Dict[str, Any], 
        data_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate ML recommendations based on analysis."""
        recommendations = {
            "suggested_target_columns": [],
            "suggested_feature_columns": [],
            "preprocessing_recommendations": [],
            "task_type_suggestions": []
        }
        
        # Suggest target columns (typically numeric or binary categorical)
        numeric_cols = schema_analysis["numeric_columns"]
        categorical_cols = schema_analysis["categorical_columns"]
        
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in 
                   ["target", "label", "outcome", "result", "score", "rating"]):
                recommendations["suggested_target_columns"].append(col)
        
        for col in categorical_cols:
            if any(keyword in col.lower() for keyword in 
                   ["class", "category", "type", "status", "flag"]):
                recommendations["suggested_target_columns"].append(col)
        
        # Suggest feature columns (exclude likely IDs and timestamps)
        feature_cols = []
        for col in numeric_cols + categorical_cols:
            if not any(keyword in col.lower() for keyword in 
                      ["id", "key", "timestamp", "created", "updated"]):
                feature_cols.append(col)
        
        recommendations["suggested_feature_columns"] = feature_cols
        
        # Preprocessing recommendations
        if data_analysis.get("missing_values"):
            missing_cols = [col for col, count in data_analysis["missing_values"].items() if count > 0]
            if missing_cols:
                recommendations["preprocessing_recommendations"].append(
                    f"Handle missing values in columns: {', '.join(missing_cols)}"
                )
        
        if data_analysis.get("duplicate_rows", 0) > 0:
            recommendations["preprocessing_recommendations"].append(
                "Remove duplicate rows"
            )
        
        # Task type suggestions
        if len(recommendations["suggested_target_columns"]) > 0:
            target_col = recommendations["suggested_target_columns"][0]
            if target_col in numeric_cols:
                recommendations["task_type_suggestions"].append("regression")
            else:
                recommendations["task_type_suggestions"].append("classification")
        
        return recommendations
    
    def generate_training_query(
        self, 
        table_id: str, 
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        limit: Optional[int] = None
    ) -> str:
        """
        Generate a SQL query for ML training data.
        
        Args:
            table_id: Source table ID
            target_column: Target column name
            feature_columns: List of feature columns (if None, uses all except target)
            where_clause: Optional WHERE clause
            limit: Optional LIMIT clause
            
        Returns:
            SQL query string
        """
        if feature_columns is None:
            # Get all columns except target
            table_info = self.client.get_table_info(table_id)
            all_columns = [field["name"] for field in table_info["schema"]]
            feature_columns = [col for col in all_columns if col != target_column]
        
        # Build SELECT clause
        select_columns = feature_columns + [target_column]
        select_clause = ", ".join(select_columns)
        
        # Build query
        query = f"SELECT {select_clause} FROM `{table_id}`"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        # Add basic data quality filters
        query += f" WHERE {target_column} IS NOT NULL"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return query
    
    def create_ml_dataset(
        self, 
        source_table_id: str,
        target_column: str,
        dataset_name: str,
        feature_columns: Optional[List[str]] = None,
        train_split: float = 0.8,
        validation_split: float = 0.1,
        test_split: float = 0.1
    ) -> Dict[str, str]:
        """
        Create train/validation/test splits for ML.
        
        Args:
            source_table_id: Source table ID
            target_column: Target column name
            dataset_name: Name for the ML dataset
            feature_columns: List of feature columns
            train_split: Training set proportion
            validation_split: Validation set proportion
            test_split: Test set proportion
            
        Returns:
            Dictionary with table IDs for each split
        """
        if abs(train_split + validation_split + test_split - 1.0) > 0.001:
            raise ValueError("Split proportions must sum to 1.0")
        
        # Generate training query
        base_query = self.generate_training_query(
            source_table_id, target_column, feature_columns
        )
        
        # Create dataset if it doesn't exist
        dataset_id = f"{dataset_name}_ml"
        self.client.create_dataset(dataset_id, f"ML dataset for {dataset_name}")
        
        # Create split tables
        splits = {
            "train": train_split,
            "validation": validation_split,
            "test": test_split
        }
        
        table_ids = {}
        cumulative_split = 0.0
        
        for split_name, split_ratio in splits.items():
            if split_ratio <= 0:
                continue
                
            table_id = f"{self.config.project_id}.{dataset_id}.{dataset_name}_{split_name}"
            
            # Create split query using FARM_FINGERPRINT for deterministic splits
            split_query = f"""
            CREATE OR REPLACE TABLE `{table_id}` AS
            WITH split_data AS (
                {base_query}
            )
            SELECT *
            FROM split_data
            WHERE MOD(ABS(FARM_FINGERPRINT(CAST({target_column} AS STRING))), 1000) 
                  BETWEEN {int(cumulative_split * 1000)} 
                  AND {int((cumulative_split + split_ratio) * 1000) - 1}
            """
            
            self.client.execute_query(split_query)
            table_ids[split_name] = table_id
            cumulative_split += split_ratio
            
            logger.info(f"Created {split_name} split table: {table_id}")
        
        return table_ids
