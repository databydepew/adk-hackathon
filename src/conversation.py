"""
Conversation Manager for Multi-turn Interactions

This module handles conversation state, context management, and multi-turn
interactions for the Vertex AI Pipeline Agent.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from src.agent import VertexAIAgent

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    turn_id: int
    timestamp: str
    user_input: str
    agent_response: Dict[str, Any]
    context: Dict[str, Any]
    success: bool
    error: Optional[str] = None


@dataclass
class ConversationContext:
    """Maintains conversation context across turns."""
    current_pipeline: Optional[Dict[str, Any]] = None
    data_sources: List[Dict[str, Any]] = None
    model_preferences: Dict[str, Any] = None
    deployment_preferences: Dict[str, Any] = None
    previous_instructions: List[str] = None
    clarifications_needed: List[str] = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = []
        if self.model_preferences is None:
            self.model_preferences = {}
        if self.deployment_preferences is None:
            self.deployment_preferences = {}
        if self.previous_instructions is None:
            self.previous_instructions = []
        if self.clarifications_needed is None:
            self.clarifications_needed = []


class ConversationManager:
    """Manages multi-turn conversations with context awareness."""
    
    def __init__(self, agent: VertexAIAgent):
        """
        Initialize conversation manager.
        
        Args:
            agent: Vertex AI agent instance
        """
        self.agent = agent
        self.turns: List[ConversationTurn] = []
        self.context = ConversationContext()
        self.conversation_id = f"conv_{int(datetime.now().timestamp())}"
        
        logger.info(f"Initialized conversation manager: {self.conversation_id}")
    
    def process_turn(self, user_input: str) -> Dict[str, Any]:
        """
        Process a single conversation turn.
        
        Args:
            user_input: User's natural language input
            
        Returns:
            Processing result with context
        """
        turn_id = len(self.turns) + 1
        timestamp = datetime.now().isoformat()
        
        logger.info(f"Processing turn {turn_id}: {user_input[:100]}...")
        
        try:
            # Enhance input with conversation context
            enhanced_input = self._enhance_input_with_context(user_input)
            
            # Check for conversation commands
            if self._is_conversation_command(user_input):
                result = self._handle_conversation_command(user_input)
            else:
                # Process with agent
                result = self.agent.process_instruction(enhanced_input)
                
                # Update context based on result
                self._update_context(user_input, result)
            
            # Create conversation turn
            turn = ConversationTurn(
                turn_id=turn_id,
                timestamp=timestamp,
                user_input=user_input,
                agent_response=result,
                context=asdict(self.context),
                success=result.get("success", False),
                error=result.get("error")
            )
            
            self.turns.append(turn)
            
            # Add conversation metadata to result
            result["conversation"] = {
                "turn_id": turn_id,
                "conversation_id": self.conversation_id,
                "context_updated": True
            }
            
            logger.info(f"Turn {turn_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing turn {turn_id}: {e}")
            
            # Create error turn
            error_result = {
                "success": False,
                "error": str(e),
                "timestamp": timestamp
            }
            
            turn = ConversationTurn(
                turn_id=turn_id,
                timestamp=timestamp,
                user_input=user_input,
                agent_response=error_result,
                context=asdict(self.context),
                success=False,
                error=str(e)
            )
            
            self.turns.append(turn)
            return error_result
    
    def _enhance_input_with_context(self, user_input: str) -> str:
        """Enhance user input with conversation context."""
        # Build context string
        context_parts = []
        
        # Add previous pipeline context
        if self.context.current_pipeline:
            pipeline_name = self.context.current_pipeline.get("name", "current pipeline")
            context_parts.append(f"Current pipeline: {pipeline_name}")
        
        # Add data source context
        if self.context.data_sources:
            data_sources = [ds.get("location", "unknown") for ds in self.context.data_sources]
            context_parts.append(f"Known data sources: {', '.join(data_sources)}")
        
        # Add model preferences
        if self.context.model_preferences:
            framework = self.context.model_preferences.get("framework")
            if framework:
                context_parts.append(f"Preferred framework: {framework}")
        
        # Add deployment preferences
        if self.context.deployment_preferences:
            endpoint_name = self.context.deployment_preferences.get("endpoint_name")
            if endpoint_name:
                context_parts.append(f"Deployment endpoint: {endpoint_name}")
        
        # Add recent instructions for context
        if self.context.previous_instructions:
            recent_instructions = self.context.previous_instructions[-2:]  # Last 2 instructions
            context_parts.append(f"Recent instructions: {'; '.join(recent_instructions)}")
        
        # Combine context with current input
        if context_parts:
            context_string = "Context: " + "; ".join(context_parts)
            enhanced_input = f"{context_string}\n\nCurrent instruction: {user_input}"
        else:
            enhanced_input = user_input
        
        return enhanced_input
    
    def _is_conversation_command(self, user_input: str) -> bool:
        """Check if input is a conversation management command."""
        commands = [
            "show pipeline", "current pipeline", "pipeline status",
            "show context", "what do you know", "context",
            "modify pipeline", "update pipeline", "change pipeline",
            "deploy pipeline", "deploy current", "deploy model",
            "pipeline history", "show history", "what have we done"
        ]
        
        user_lower = user_input.lower().strip()
        return any(cmd in user_lower for cmd in commands)
    
    def _handle_conversation_command(self, user_input: str) -> Dict[str, Any]:
        """Handle conversation management commands."""
        user_lower = user_input.lower().strip()
        
        if any(cmd in user_lower for cmd in ["show pipeline", "current pipeline", "pipeline status"]):
            return self._show_current_pipeline()
        
        elif any(cmd in user_lower for cmd in ["show context", "what do you know", "context"]):
            return self._show_context()
        
        elif any(cmd in user_lower for cmd in ["pipeline history", "show history", "what have we done"]):
            return self._show_history()
        
        elif any(cmd in user_lower for cmd in ["deploy pipeline", "deploy current", "deploy model"]):
            return self._deploy_current_pipeline()
        
        else:
            return {
                "success": False,
                "error": "Unknown conversation command",
                "suggestions": [
                    "show pipeline - Display current pipeline",
                    "show context - Show conversation context",
                    "show history - Display conversation history",
                    "deploy pipeline - Deploy current pipeline"
                ]
            }
    
    def _show_current_pipeline(self) -> Dict[str, Any]:
        """Show information about the current pipeline."""
        if not self.context.current_pipeline:
            return {
                "success": True,
                "message": "No current pipeline. Create one by describing your ML task.",
                "suggestions": [
                    "Create a fraud detection model using BigQuery data",
                    "Build a recommendation system with TensorFlow",
                    "Set up a time series forecasting pipeline"
                ]
            }
        
        pipeline = self.context.current_pipeline
        
        return {
            "success": True,
            "message": "Current Pipeline Information",
            "pipeline": {
                "name": pipeline.get("name", "Unnamed"),
                "description": pipeline.get("description", "No description"),
                "components": len(pipeline.get("components", [])),
                "task_type": pipeline.get("metadata", {}).get("task_type", "Unknown"),
                "framework": pipeline.get("metadata", {}).get("framework", "Unknown"),
                "status": "planned"
            }
        }
    
    def _show_context(self) -> Dict[str, Any]:
        """Show current conversation context."""
        context_info = {
            "conversation_id": self.conversation_id,
            "turns_completed": len(self.turns),
            "current_pipeline": bool(self.context.current_pipeline),
            "data_sources": len(self.context.data_sources),
            "model_preferences": self.context.model_preferences,
            "deployment_preferences": self.context.deployment_preferences,
            "recent_instructions": self.context.previous_instructions[-3:] if self.context.previous_instructions else []
        }
        
        return {
            "success": True,
            "message": "Conversation Context",
            "context": context_info
        }
    
    def _show_history(self) -> Dict[str, Any]:
        """Show conversation history."""
        history = []
        
        for turn in self.turns:
            history.append({
                "turn": turn.turn_id,
                "timestamp": turn.timestamp,
                "user_input": turn.user_input[:100] + "..." if len(turn.user_input) > 100 else turn.user_input,
                "success": turn.success,
                "error": turn.error
            })
        
        return {
            "success": True,
            "message": f"Conversation History ({len(history)} turns)",
            "history": history
        }
    
    def _deploy_current_pipeline(self) -> Dict[str, Any]:
        """Deploy the current pipeline."""
        if not self.context.current_pipeline:
            return {
                "success": False,
                "error": "No current pipeline to deploy",
                "suggestions": ["Create a pipeline first by describing your ML task"]
            }
        
        try:
            # Execute pipeline deployment
            result = self.agent.execute_pipeline(self.context.current_pipeline)
            
            # Update deployment preferences
            if result.get("success", False):
                self.context.deployment_preferences.update({
                    "last_deployment": datetime.now().isoformat(),
                    "deployment_status": "deployed"
                })
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Deployment failed: {str(e)}"
            }
    
    def _update_context(self, user_input: str, result: Dict[str, Any]) -> None:
        """Update conversation context based on processing result."""
        # Add to previous instructions
        self.context.previous_instructions.append(user_input)
        
        # Keep only last 10 instructions
        if len(self.context.previous_instructions) > 10:
            self.context.previous_instructions = self.context.previous_instructions[-10:]
        
        if not result.get("success", False):
            return
        
        # Update current pipeline
        if "pipeline_plan" in result:
            self.context.current_pipeline = result["pipeline_plan"]
        
        # Update data sources
        if "parsed_instruction" in result:
            parsed = result["parsed_instruction"]
            
            if "data_source" in parsed:
                data_source = parsed["data_source"]
                
                # Check if this data source is already known
                existing_sources = [ds.get("location") for ds in self.context.data_sources]
                if data_source.get("location") not in existing_sources:
                    self.context.data_sources.append(data_source)
            
            # Update model preferences
            if "model_config" in parsed:
                model_config = parsed["model_config"]
                self.context.model_preferences.update({
                    "framework": model_config.get("framework"),
                    "algorithm": model_config.get("algorithm")
                })
            
            # Update deployment preferences
            if "deployment_spec" in parsed and parsed["deployment_spec"]:
                deployment_spec = parsed["deployment_spec"]
                self.context.deployment_preferences.update({
                    "endpoint_name": deployment_spec.get("endpoint_name"),
                    "machine_type": deployment_spec.get("machine_type")
                })
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        successful_turns = sum(1 for turn in self.turns if turn.success)
        failed_turns = len(self.turns) - successful_turns
        
        return {
            "conversation_id": self.conversation_id,
            "total_turns": len(self.turns),
            "successful_turns": successful_turns,
            "failed_turns": failed_turns,
            "has_current_pipeline": bool(self.context.current_pipeline),
            "data_sources_count": len(self.context.data_sources),
            "start_time": self.turns[0].timestamp if self.turns else None,
            "last_activity": self.turns[-1].timestamp if self.turns else None
        }
    
    def save_conversation(self, filepath: str) -> None:
        """Save conversation to file."""
        conversation_data = {
            "conversation_id": self.conversation_id,
            "summary": self.get_conversation_summary(),
            "context": asdict(self.context),
            "turns": [asdict(turn) for turn in self.turns]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        logger.info(f"Conversation saved to {filepath}")
    
    def load_conversation(self, filepath: str) -> None:
        """Load conversation from file."""
        with open(filepath, 'r') as f:
            conversation_data = json.load(f)
        
        # Restore conversation state
        self.conversation_id = conversation_data["conversation_id"]
        
        # Restore context
        context_data = conversation_data["context"]
        self.context = ConversationContext(**context_data)
        
        # Restore turns
        self.turns = []
        for turn_data in conversation_data["turns"]:
            turn = ConversationTurn(**turn_data)
            self.turns.append(turn)
        
        logger.info(f"Conversation loaded from {filepath}")
    
    def clear_history(self) -> None:
        """Clear conversation history but keep context."""
        self.turns.clear()
        logger.info("Conversation history cleared")
    
    def reset_context(self) -> None:
        """Reset conversation context."""
        self.context = ConversationContext()
        logger.info("Conversation context reset")
    
    def print_history(self) -> None:
        """Print conversation history to console."""
        if not self.turns:
            print("No conversation history.")
            return
        
        print(f"\nConversation History ({len(self.turns)} turns):")
        print("-" * 60)
        
        for turn in self.turns:
            status = "✅" if turn.success else "❌"
            timestamp = turn.timestamp.split('T')[1][:8]  # Show only time
            
            print(f"{status} Turn {turn.turn_id} ({timestamp})")
            print(f"   User: {turn.user_input[:80]}{'...' if len(turn.user_input) > 80 else ''}")
            
            if turn.success:
                response = turn.agent_response
                if "parsed_instruction" in response:
                    task_type = response["parsed_instruction"].get("task_type", "Unknown")
                    print(f"   Agent: Created {task_type} pipeline")
                else:
                    print(f"   Agent: {response.get('message', 'Processed successfully')}")
            else:
                print(f"   Error: {turn.error}")
            
            print()
