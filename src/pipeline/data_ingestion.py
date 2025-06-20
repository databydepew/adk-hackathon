"""
Data Ingestion Pipeline Components

This module provides components for ingesting data from various sources
including BigQuery and Google Cloud Storage for ML pipelines.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from kfp.dsl import component, Input, Output, Dataset
from google.cloud import bigquery, storage

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Supported data source types."""
    BIGQUERY = "bigquery"
    GCS = "gcs"
    CLOUD_SQL = "cloud_sql"
    FIRESTORE = "firestore"


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion components."""
    source_type: DataSourceType
    source_location: str
    destination_path: str
    project_id: str
    region: str = "us-central1"
    format: str = "csv"
    compression: Optional[str] = None
    schema_validation: bool = True
    data_quality_checks: bool = True


class BaseDataIngestionComponent(ABC):
    """Base class for data ingestion components."""
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the data ingestion component.
        
        Args:
            config: Data ingestion configuration
        """
        self.config = config
        self.component_name = f"{config.source_type.value}_data_ingestion"
        
    @abstractmethod
    def create_component(self) -> component:
        """Create the KFP component for data ingestion."""
        pass
    
    @abstractmethod
    def get_component_spec(self) -> Dict[str, Any]:
        """Get the component specification."""
        pass


class BigQueryDataIngestionComponent(BaseDataIngestionComponent):
    """BigQuery data ingestion component for ML pipelines."""
    
    def __init__(self, config: DataIngestionConfig):
        """Initialize BigQuery data ingestion component."""
        super().__init__(config)
        if config.source_type != DataSourceType.BIGQUERY:
            raise ValueError("Config must be for BigQuery data source")
    
    def create_component(self) -> component:
        """Create BigQuery data ingestion KFP component."""
        
        @component(
            base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
            packages_to_install=[
                "google-cloud-bigquery>=3.11.0",
                "pandas>=2.0.0",
                "pyarrow>=12.0.0"
            ]
        )
        def bigquery_data_ingestion(
            query: str,
            project_id: str,
            destination_path: Output[Dataset],
            table_id: str = "",
            dataset_location: str = "US",
            format: str = "csv",
            schema_validation: bool = True,
            data_quality_checks: bool = True,
            max_results: int = 1000000
        ):
            """
            Ingest data from BigQuery for ML pipeline.
            
            Args:
                query: SQL query to extract data
                project_id: GCP project ID
                destination_path: Output dataset path
                table_id: Optional table ID if not using query
                dataset_location: BigQuery dataset location
                format: Output format (csv, parquet, json)
                schema_validation: Enable schema validation
                data_quality_checks: Enable data quality checks
                max_results: Maximum number of results to fetch
            """
            import pandas as pd
            from google.cloud import bigquery
            import os
            import json
            import logging
            
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            try:
                # Initialize BigQuery client
                client = bigquery.Client(project=project_id, location=dataset_location)
                logger.info(f"Initialized BigQuery client for project {project_id}")
                
                # Execute query or read from table
                if query:
                    logger.info(f"Executing query: {query[:100]}...")
                    query_job = client.query(query)
                    df = query_job.to_dataframe(max_results=max_results)
                elif table_id:
                    logger.info(f"Reading from table: {table_id}")
                    table_ref = client.get_table(table_id)
                    df = client.list_rows(table_ref, max_results=max_results).to_dataframe()
                else:
                    raise ValueError("Either query or table_id must be provided")
                
                logger.info(f"Retrieved {len(df)} rows with {len(df.columns)} columns")
                
                # Data quality checks
                if data_quality_checks:
                    logger.info("Performing data quality checks...")
                    
                    # Check for empty dataset
                    if len(df) == 0:
                        raise ValueError("Dataset is empty")
                    
                    # Check for duplicate rows
                    duplicates = df.duplicated().sum()
                    if duplicates > 0:
                        logger.warning(f"Found {duplicates} duplicate rows")
                    
                    # Check for missing values
                    missing_values = df.isnull().sum().sum()
                    if missing_values > 0:
                        logger.warning(f"Found {missing_values} missing values")
                    
                    # Log basic statistics
                    logger.info(f"Dataset shape: {df.shape}")
                    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                # Schema validation
                if schema_validation:
                    logger.info("Performing schema validation...")
                    schema_info = {
                        "columns": list(df.columns),
                        "dtypes": df.dtypes.astype(str).to_dict(),
                        "shape": df.shape,
                        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
                    }
                    
                    # Save schema information
                    schema_path = os.path.join(destination_path.path, "schema.json")
                    os.makedirs(os.path.dirname(schema_path), exist_ok=True)
                    with open(schema_path, 'w') as f:
                        json.dump(schema_info, f, indent=2)
                    logger.info(f"Schema saved to {schema_path}")
                
                # Save dataset
                os.makedirs(destination_path.path, exist_ok=True)
                
                if format.lower() == "csv":
                    output_file = os.path.join(destination_path.path, "data.csv")
                    df.to_csv(output_file, index=False)
                elif format.lower() == "parquet":
                    output_file = os.path.join(destination_path.path, "data.parquet")
                    df.to_parquet(output_file, index=False)
                elif format.lower() == "json":
                    output_file = os.path.join(destination_path.path, "data.json")
                    df.to_json(output_file, orient="records", lines=True)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                logger.info(f"Dataset saved to {output_file}")
                
                # Save metadata
                metadata = {
                    "source_type": "bigquery",
                    "source_query": query,
                    "source_table": table_id,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "format": format,
                    "file_path": output_file,
                    "data_quality_checks": data_quality_checks,
                    "schema_validation": schema_validation
                }
                
                metadata_path = os.path.join(destination_path.path, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info("Data ingestion completed successfully")
                
            except Exception as e:
                logger.error(f"Data ingestion failed: {str(e)}")
                raise
        
        return bigquery_data_ingestion
    
    def get_component_spec(self) -> Dict[str, Any]:
        """Get BigQuery component specification."""
        return {
            "name": self.component_name,
            "description": "Ingest data from BigQuery for ML pipeline",
            "inputs": {
                "query": {"type": "String", "description": "SQL query to extract data"},
                "project_id": {"type": "String", "description": "GCP project ID"},
                "table_id": {"type": "String", "description": "Optional table ID", "optional": True},
                "dataset_location": {"type": "String", "description": "BigQuery dataset location", "default": "US"},
                "format": {"type": "String", "description": "Output format", "default": "csv"},
                "schema_validation": {"type": "Boolean", "description": "Enable schema validation", "default": True},
                "data_quality_checks": {"type": "Boolean", "description": "Enable data quality checks", "default": True},
                "max_results": {"type": "Integer", "description": "Maximum results", "default": 1000000}
            },
            "outputs": {
                "destination_path": {"type": "Dataset", "description": "Output dataset"}
            },
            "implementation": {
                "container": {
                    "image": "gcr.io/deeplearning-platform-release/base-cpu:latest",
                    "command": ["python"],
                    "args": ["-c", "# Component implementation"]
                }
            }
        }


class GCSDataIngestionComponent(BaseDataIngestionComponent):
    """Google Cloud Storage data ingestion component for ML pipelines."""
    
    def __init__(self, config: DataIngestionConfig):
        """Initialize GCS data ingestion component."""
        super().__init__(config)
        if config.source_type != DataSourceType.GCS:
            raise ValueError("Config must be for GCS data source")
    
    def create_component(self) -> component:
        """Create GCS data ingestion KFP component."""
        
        @component(
            base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
            packages_to_install=[
                "google-cloud-storage>=2.10.0",
                "pandas>=2.0.0",
                "pyarrow>=12.0.0"
            ]
        )
        def gcs_data_ingestion(
            source_path: str,
            project_id: str,
            destination_path: Output[Dataset],
            format: str = "csv",
            compression: str = "",
            schema_validation: bool = True,
            data_quality_checks: bool = True,
            file_pattern: str = "*"
        ):
            """
            Ingest data from Google Cloud Storage for ML pipeline.
            
            Args:
                source_path: GCS path (gs://bucket/path)
                project_id: GCP project ID
                destination_path: Output dataset path
                format: File format (csv, parquet, json)
                compression: Compression type (gzip, bz2, etc.)
                schema_validation: Enable schema validation
                data_quality_checks: Enable data quality checks
                file_pattern: File pattern to match
            """
            import pandas as pd
            from google.cloud import storage
            import os
            import json
            import logging
            import glob
            from urllib.parse import urlparse
            
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            try:
                # Parse GCS path
                parsed_url = urlparse(source_path)
                bucket_name = parsed_url.netloc
                blob_prefix = parsed_url.path.lstrip('/')
                
                logger.info(f"Reading from GCS: bucket={bucket_name}, prefix={blob_prefix}")
                
                # Initialize GCS client
                client = storage.Client(project=project_id)
                bucket = client.bucket(bucket_name)
                
                # List blobs matching pattern
                blobs = list(bucket.list_blobs(prefix=blob_prefix))
                if file_pattern != "*":
                    blobs = [blob for blob in blobs if file_pattern in blob.name]
                
                if not blobs:
                    raise ValueError(f"No files found at {source_path}")
                
                logger.info(f"Found {len(blobs)} files to process")
                
                # Download and process files
                dataframes = []
                for blob in blobs:
                    logger.info(f"Processing file: {blob.name}")
                    
                    # Download blob content
                    content = blob.download_as_bytes()
                    
                    # Read based on format
                    if format.lower() == "csv":
                        df = pd.read_csv(
                            content, 
                            compression=compression if compression else None
                        )
                    elif format.lower() == "parquet":
                        df = pd.read_parquet(content)
                    elif format.lower() == "json":
                        df = pd.read_json(content, lines=True)
                    else:
                        raise ValueError(f"Unsupported format: {format}")
                    
                    dataframes.append(df)
                
                # Combine all dataframes
                if len(dataframes) == 1:
                    df = dataframes[0]
                else:
                    df = pd.concat(dataframes, ignore_index=True)
                
                logger.info(f"Combined dataset: {len(df)} rows, {len(df.columns)} columns")
                
                # Data quality checks
                if data_quality_checks:
                    logger.info("Performing data quality checks...")
                    
                    # Check for empty dataset
                    if len(df) == 0:
                        raise ValueError("Dataset is empty")
                    
                    # Check for duplicate rows
                    duplicates = df.duplicated().sum()
                    if duplicates > 0:
                        logger.warning(f"Found {duplicates} duplicate rows")
                    
                    # Check for missing values
                    missing_values = df.isnull().sum().sum()
                    if missing_values > 0:
                        logger.warning(f"Found {missing_values} missing values")
                    
                    # Log basic statistics
                    logger.info(f"Dataset shape: {df.shape}")
                    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                # Schema validation
                if schema_validation:
                    logger.info("Performing schema validation...")
                    schema_info = {
                        "columns": list(df.columns),
                        "dtypes": df.dtypes.astype(str).to_dict(),
                        "shape": df.shape,
                        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
                    }
                    
                    # Save schema information
                    schema_path = os.path.join(destination_path.path, "schema.json")
                    os.makedirs(os.path.dirname(schema_path), exist_ok=True)
                    with open(schema_path, 'w') as f:
                        json.dump(schema_info, f, indent=2)
                    logger.info(f"Schema saved to {schema_path}")
                
                # Save dataset
                os.makedirs(destination_path.path, exist_ok=True)
                
                if format.lower() == "csv":
                    output_file = os.path.join(destination_path.path, "data.csv")
                    df.to_csv(output_file, index=False)
                elif format.lower() == "parquet":
                    output_file = os.path.join(destination_path.path, "data.parquet")
                    df.to_parquet(output_file, index=False)
                elif format.lower() == "json":
                    output_file = os.path.join(destination_path.path, "data.json")
                    df.to_json(output_file, orient="records", lines=True)
                
                logger.info(f"Dataset saved to {output_file}")
                
                # Save metadata
                metadata = {
                    "source_type": "gcs",
                    "source_path": source_path,
                    "files_processed": len(blobs),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "format": format,
                    "file_path": output_file,
                    "data_quality_checks": data_quality_checks,
                    "schema_validation": schema_validation
                }
                
                metadata_path = os.path.join(destination_path.path, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info("Data ingestion completed successfully")
                
            except Exception as e:
                logger.error(f"Data ingestion failed: {str(e)}")
                raise
        
        return gcs_data_ingestion
    
    def get_component_spec(self) -> Dict[str, Any]:
        """Get GCS component specification."""
        return {
            "name": self.component_name,
            "description": "Ingest data from Google Cloud Storage for ML pipeline",
            "inputs": {
                "source_path": {"type": "String", "description": "GCS path (gs://bucket/path)"},
                "project_id": {"type": "String", "description": "GCP project ID"},
                "format": {"type": "String", "description": "File format", "default": "csv"},
                "compression": {"type": "String", "description": "Compression type", "optional": True},
                "schema_validation": {"type": "Boolean", "description": "Enable schema validation", "default": True},
                "data_quality_checks": {"type": "Boolean", "description": "Enable data quality checks", "default": True},
                "file_pattern": {"type": "String", "description": "File pattern to match", "default": "*"}
            },
            "outputs": {
                "destination_path": {"type": "Dataset", "description": "Output dataset"}
            },
            "implementation": {
                "container": {
                    "image": "gcr.io/deeplearning-platform-release/base-cpu:latest",
                    "command": ["python"],
                    "args": ["-c", "# Component implementation"]
                }
            }
        }


class DataIngestionComponentFactory:
    """Factory for creating data ingestion components."""
    
    @staticmethod
    def create_component(config: DataIngestionConfig) -> BaseDataIngestionComponent:
        """
        Create a data ingestion component based on configuration.
        
        Args:
            config: Data ingestion configuration
            
        Returns:
            Data ingestion component instance
        """
        if config.source_type == DataSourceType.BIGQUERY:
            return BigQueryDataIngestionComponent(config)
        elif config.source_type == DataSourceType.GCS:
            return GCSDataIngestionComponent(config)
        else:
            raise ValueError(f"Unsupported data source type: {config.source_type}")
    
    @staticmethod
    def get_supported_sources() -> List[DataSourceType]:
        """Get list of supported data source types."""
        return [DataSourceType.BIGQUERY, DataSourceType.GCS]
    
    @staticmethod
    def create_bigquery_component(
        query: str,
        project_id: str,
        destination_path: str,
        **kwargs
    ) -> BigQueryDataIngestionComponent:
        """
        Create BigQuery data ingestion component with simplified interface.
        
        Args:
            query: SQL query to extract data
            project_id: GCP project ID
            destination_path: Output dataset path
            **kwargs: Additional configuration options
            
        Returns:
            BigQuery data ingestion component
        """
        config = DataIngestionConfig(
            source_type=DataSourceType.BIGQUERY,
            source_location=query,
            destination_path=destination_path,
            project_id=project_id,
            **kwargs
        )
        return BigQueryDataIngestionComponent(config)
    
    @staticmethod
    def create_gcs_component(
        source_path: str,
        project_id: str,
        destination_path: str,
        **kwargs
    ) -> GCSDataIngestionComponent:
        """
        Create GCS data ingestion component with simplified interface.
        
        Args:
            source_path: GCS path (gs://bucket/path)
            project_id: GCP project ID
            destination_path: Output dataset path
            **kwargs: Additional configuration options
            
        Returns:
            GCS data ingestion component
        """
        config = DataIngestionConfig(
            source_type=DataSourceType.GCS,
            source_location=source_path,
            destination_path=destination_path,
            project_id=project_id,
            **kwargs
        )
        return GCSDataIngestionComponent(config)
