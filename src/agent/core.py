"""
Core Vertex AI Pipeline Agent Implementation

This module contains the main agent class that uses Google Agent ADK and Gemini Pro
to interpret natural language instructions and generate ML pipelines.
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

import google.generativeai as genai
from google.cloud import aiplatform
from google.api_core import exceptions as gcp_exceptions

from .config import AgentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InstructionParseResult:
    """Result of parsing a natural language instruction."""
    task_type: str  # 'classification', 'regression', 'clustering', etc.
    data_source: Dict[str, Any]  # BigQuery table, GCS path, etc.
    model_framework: str  # 'scikit-learn', 'xgboost', 'tensorflow', etc.
    target_column: Optional[str] = None
    features: Optional[List[str]] = None
    model_parameters: Optional[Dict[str, Any]] = None
    deployment_config: Optional[Dict[str, Any]] = None
    monitoring_config: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    raw_instruction: str = ""
    parsed_components: Dict[str, Any] = None


@dataclass
class ConversationContext:
    """Context for multi-turn conversations."""
    conversation_id: str
    turn_count: int = 0
    instruction_history: List[str] = None
    parsed_results: List[InstructionParseResult] = None
    current_pipeline_plan: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.instruction_history is None:
            self.instruction_history = []
        if self.parsed_results is None:
            self.parsed_results = []


class VertexAIAgent:
    """
    Main agent class for interpreting NLP instructions and generating ML pipelines.
    
    Uses Google Agent ADK and Gemini Pro for natural language understanding
    and Vertex AI for pipeline generation and execution.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the Vertex AI Pipeline Agent.
        
        Args:
            config: Agent configuration object
        """
        self.config = config
        self.config.validate()
        
        # Initialize Gemini Pro
        self._init_gemini()
        
        # Initialize Vertex AI
        self._init_vertex_ai()
        
        # Conversation contexts
        self.conversations: Dict[str, ConversationContext] = {}
        
        # Instruction parsing patterns
        self._init_parsing_patterns()
        
        logger.info("Vertex AI Pipeline Agent initialized successfully")
    
    def _init_gemini(self):
        """Initialize Gemini Pro for natural language processing."""
        try:
            # Configure Gemini API
            genai.configure(api_key=self._get_api_key())
            
            # Initialize the model
            self.gemini_model = genai.GenerativeModel(
                model_name=self.config.gemini.model,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.gemini.temperature,
                    max_output_tokens=self.config.gemini.max_tokens,
                ),
                safety_settings=self._convert_safety_settings()
            )
            
            logger.info(f"Gemini {self.config.gemini.model} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def _init_vertex_ai(self):
        """Initialize Vertex AI client."""
        try:
            aiplatform.init(
                project=self.config.gcp.project_id,
                location=self.config.gcp.vertex_ai_location,
                staging_bucket=self.config.gcp.staging_bucket,
                service_account=self.config.gcp.service_account
            )
            
            logger.info("Vertex AI initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    def _get_api_key(self) -> str:
        """Get Gemini API key from environment or configuration."""
        import os
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        return api_key
    
    def _convert_safety_settings(self) -> List[Dict[str, Any]]:
        """Convert safety settings to Gemini format."""
        safety_settings = []
        for setting in self.config.gemini.safety_settings:
            safety_settings.append({
                "category": getattr(genai.types.HarmCategory, setting["category"]),
                "threshold": getattr(genai.types.HarmBlockThreshold, setting["threshold"])
            })
        return safety_settings
    
    def _init_parsing_patterns(self):
        """Initialize regex patterns for instruction parsing."""
        self.patterns = {
            'task_types': {
                'classification': [
                    r'classif\w*', r'predict\s+class', r'binary\s+classification',
                    r'multi-?class', r'categoriz\w*', r'fraud\s+detection',
                    r'churn\s+prediction', r'sentiment\s+analysis'
                ],
                'regression': [
                    r'regression', r'predict\s+value', r'forecast\w*',
                    r'price\s+prediction', r'sales\s+forecast', r'demand\s+forecast'
                ],
                'clustering': [
                    r'cluster\w*', r'segment\w*', r'group\w*',
                    r'customer\s+segment', r'market\s+segment'
                ]
            },
            'frameworks': {
                'xgboost': [r'xgboost', r'xgb', r'gradient\s+boost'],
                'scikit-learn': [r'scikit-?learn', r'sklearn', r'random\s+forest', r'svm'],
                'tensorflow': [r'tensorflow', r'tf', r'neural\s+network', r'deep\s+learning'],
                'pytorch': [r'pytorch', r'torch']
            },
            'data_sources': {
                'bigquery': [r'bigquery', r'bq', r'sql\s+table', r'dataset'],
                'gcs': [r'gcs', r'cloud\s+storage', r'bucket', r'csv\s+file']
            }
        }
    
    async def process_instruction(
        self, 
        instruction: str, 
        conversation_id: Optional[str] = None
    ) -> InstructionParseResult:
        """
        Process a natural language instruction and extract ML pipeline requirements.
        
        Args:
            instruction: Natural language instruction
            conversation_id: Optional conversation ID for multi-turn conversations
            
        Returns:
            Parsed instruction result
        """
        try:
            # Get or create conversation context
            context = self._get_conversation_context(conversation_id, instruction)
            
            # Parse instruction using Gemini Pro
            parse_result = await self._parse_with_gemini(instruction, context)
            
            # Validate and enhance the parsed result
            validated_result = self._validate_parse_result(parse_result, instruction)
            
            # Update conversation context
            context.parsed_results.append(validated_result)
            context.turn_count += 1
            
            logger.info(f"Successfully parsed instruction: {instruction[:100]}...")
            return validated_result
            
        except Exception as e:
            logger.error(f"Failed to process instruction: {e}")
            raise
    
    def _get_conversation_context(
        self, 
        conversation_id: Optional[str], 
        instruction: str
    ) -> ConversationContext:
        """Get or create conversation context."""
        if conversation_id is None:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id
            )
        
        context = self.conversations[conversation_id]
        context.instruction_history.append(instruction)
        
        return context
    
    async def _parse_with_gemini(
        self, 
        instruction: str, 
        context: ConversationContext
    ) -> InstructionParseResult:
        """Parse instruction using Gemini Pro."""
        
        # Build the prompt for Gemini
        prompt = self._build_parsing_prompt(instruction, context)
        
        try:
            # Generate response from Gemini
            response = await asyncio.to_thread(
                self.gemini_model.generate_content, prompt
            )
            
            # Parse the JSON response
            response_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_text = response_text
            
            parsed_data = json.loads(json_text)
            
            # Create InstructionParseResult
            result = InstructionParseResult(
                task_type=parsed_data.get('task_type', 'unknown'),
                data_source=parsed_data.get('data_source', {}),
                model_framework=parsed_data.get('model_framework', 'scikit-learn'),
                target_column=parsed_data.get('target_column'),
                features=parsed_data.get('features'),
                model_parameters=parsed_data.get('model_parameters', {}),
                deployment_config=parsed_data.get('deployment_config', {}),
                monitoring_config=parsed_data.get('monitoring_config', {}),
                confidence_score=parsed_data.get('confidence_score', 0.0),
                raw_instruction=instruction,
                parsed_components=parsed_data
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            # Fallback to pattern-based parsing
            return self._fallback_pattern_parsing(instruction)
            
        except Exception as e:
            logger.error(f"Gemini parsing failed: {e}")
            # Fallback to pattern-based parsing
            return self._fallback_pattern_parsing(instruction)
    
    def _build_parsing_prompt(self, instruction: str, context: ConversationContext) -> str:
        """Build the prompt for Gemini Pro instruction parsing."""
        
        base_prompt = f"""
You are an expert ML engineer who specializes in parsing natural language instructions 
into structured ML pipeline requirements for Google Cloud Vertex AI.

Parse the following instruction and extract the ML pipeline components:

INSTRUCTION: "{instruction}"

CONTEXT:
- Previous instructions in this conversation: {len(context.instruction_history) - 1}
- Available ML frameworks: {[fw['name'] for fw in self.config.pipeline.supported_frameworks]}
- Target platform: Google Cloud Vertex AI
- Supported data sources: BigQuery, Google Cloud Storage

REQUIREMENTS:
Extract and return a JSON object with the following structure:

{{
    "task_type": "classification|regression|clustering|other",
    "data_source": {{
        "type": "bigquery|gcs|other",
        "location": "project.dataset.table or gs://bucket/path",
        "description": "description of the data"
    }},
    "model_framework": "scikit-learn|xgboost|tensorflow|pytorch",
    "target_column": "name of target/label column (if applicable)",
    "features": ["list", "of", "feature", "columns"] or null,
    "model_parameters": {{
        "hyperparameters": "any specific model parameters mentioned"
    }},
    "deployment_config": {{
        "endpoint_name": "suggested endpoint name",
        "machine_type": "suggested machine type",
        "min_replica_count": 1,
        "max_replica_count": 10
    }},
    "monitoring_config": {{
        "enable_drift_detection": true,
        "enable_performance_monitoring": true,
        "metrics": ["list of metrics to monitor"]
    }},
    "confidence_score": 0.0-1.0
}}

PARSING GUIDELINES:
1. Infer task type from keywords like "classify", "predict", "forecast", "cluster"
2. Extract data source information (BigQuery tables, GCS paths)
3. Identify preferred ML framework or suggest best fit
4. Extract target column and feature specifications
5. Infer deployment requirements (endpoint configuration)
6. Set up appropriate monitoring based on task type
7. Assign confidence score based on clarity of instruction

Return ONLY the JSON object, no additional text.
"""
        
        return base_prompt
    
    def _fallback_pattern_parsing(self, instruction: str) -> InstructionParseResult:
        """Fallback pattern-based parsing when Gemini fails."""
        logger.info("Using fallback pattern-based parsing")
        
        instruction_lower = instruction.lower()
        
        # Detect task type
        task_type = 'unknown'
        for task, patterns in self.patterns['task_types'].items():
            if any(re.search(pattern, instruction_lower) for pattern in patterns):
                task_type = task
                break
        
        # Detect framework
        framework = 'scikit-learn'  # default
        for fw, patterns in self.patterns['frameworks'].items():
            if any(re.search(pattern, instruction_lower) for pattern in patterns):
                framework = fw
                break
        
        # Detect data source
        data_source = {'type': 'unknown', 'location': '', 'description': ''}
        for source, patterns in self.patterns['data_sources'].items():
            if any(re.search(pattern, instruction_lower) for pattern in patterns):
                data_source['type'] = source
                break
        
        # Extract table/dataset names
        bigquery_match = re.search(r'(\w+\.\w+\.\w+)', instruction)
        if bigquery_match:
            data_source['location'] = bigquery_match.group(1)
            data_source['type'] = 'bigquery'
        
        gcs_match = re.search(r'gs://[\w\-./]+', instruction)
        if gcs_match:
            data_source['location'] = gcs_match.group(0)
            data_source['type'] = 'gcs'
        
        return InstructionParseResult(
            task_type=task_type,
            data_source=data_source,
            model_framework=framework,
            confidence_score=0.6,  # Lower confidence for pattern-based parsing
            raw_instruction=instruction,
            parsed_components={
                'parsing_method': 'pattern_based_fallback'
            }
        )
    
    def _validate_parse_result(
        self, 
        result: InstructionParseResult, 
        instruction: str
    ) -> InstructionParseResult:
        """Validate and enhance the parsed result."""
        
        # Validate task type
        valid_task_types = ['classification', 'regression', 'clustering', 'other']
        if result.task_type not in valid_task_types:
            result.task_type = 'other'
        
        # Validate framework
        supported_frameworks = [fw['name'] for fw in self.config.pipeline.supported_frameworks]
        if result.model_framework not in supported_frameworks:
            result.model_framework = 'scikit-learn'  # default fallback
        
        # Validate data source
        if not result.data_source or not result.data_source.get('type'):
            result.data_source = {
                'type': 'unknown',
                'location': '',
                'description': 'Data source not specified'
            }
        
        # Set default deployment config if missing
        if not result.deployment_config:
            result.deployment_config = {
                'endpoint_name': f"{result.task_type}_model_endpoint",
                'machine_type': self.config.pipeline.default_machine_type,
                'min_replica_count': 1,
                'max_replica_count': 3
            }
        
        # Set default monitoring config if missing
        if not result.monitoring_config:
            result.monitoring_config = {
                'enable_drift_detection': self.config.monitoring.enable_drift_detection,
                'enable_performance_monitoring': self.config.monitoring.enable_performance_tracking,
                'metrics': self._get_default_metrics(result.task_type)
            }
        
        return result
    
    def _get_default_metrics(self, task_type: str) -> List[str]:
        """Get default metrics for a task type."""
        metrics_map = {
            'classification': ['accuracy', 'precision', 'recall', 'f1_score', 'auc'],
            'regression': ['mse', 'rmse', 'mae', 'r2_score'],
            'clustering': ['silhouette_score', 'calinski_harabasz_score']
        }
        return metrics_map.get(task_type, ['accuracy'])
    
    async def refine_instruction(
        self, 
        conversation_id: str, 
        refinement: str
    ) -> InstructionParseResult:
        """
        Refine a previous instruction with additional details.
        
        Args:
            conversation_id: ID of the conversation to refine
            refinement: Additional refinement instruction
            
        Returns:
            Updated instruction parse result
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        context = self.conversations[conversation_id]
        
        if not context.parsed_results:
            raise ValueError("No previous instructions to refine")
        
        # Combine previous instruction with refinement
        last_instruction = context.instruction_history[-1]
        combined_instruction = f"{last_instruction}\n\nAdditional requirements: {refinement}"
        
        # Process the refined instruction
        return await self.process_instruction(combined_instruction, conversation_id)
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a summary of a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation summary
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        context = self.conversations[conversation_id]
        
        return {
            'conversation_id': conversation_id,
            'turn_count': context.turn_count,
            'instructions': context.instruction_history,
            'latest_parse_result': context.parsed_results[-1].to_dict() if context.parsed_results else None,
            'pipeline_plan': context.current_pipeline_plan
        }
    
    def clear_conversation(self, conversation_id: str):
        """Clear a conversation context."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")
    
    def list_conversations(self) -> List[str]:
        """List all active conversation IDs."""
        return list(self.conversations.keys())


# Add to_dict method to InstructionParseResult for serialization
def _instruction_parse_result_to_dict(self) -> Dict[str, Any]:
    """Convert InstructionParseResult to dictionary."""
    return {
        'task_type': self.task_type,
        'data_source': self.data_source,
        'model_framework': self.model_framework,
        'target_column': self.target_column,
        'features': self.features,
        'model_parameters': self.model_parameters,
        'deployment_config': self.deployment_config,
        'monitoring_config': self.monitoring_config,
        'confidence_score': self.confidence_score,
        'raw_instruction': self.raw_instruction,
        'parsed_components': self.parsed_components
    }

# Monkey patch the to_dict method
InstructionParseResult.to_dict = _instruction_parse_result_to_dict
