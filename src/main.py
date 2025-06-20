"""
Vertex AI Pipeline Agent - Main Application

This is the main entry point for the Vertex AI Pipeline Agent that converts
natural language instructions into end-to-end ML pipelines.
"""

import argparse
import logging
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path

from src.agent import VertexAIAgent, AgentConfig
from src.conversation import ConversationManager
from src.parser import InstructionParser, PipelinePlanner


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline_agent.log')
        ]
    )


def load_config(config_path: str) -> AgentConfig:
    """Load agent configuration from file."""
    try:
        config = AgentConfig.load_from_file(config_path)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Vertex AI Pipeline Agent - Convert natural language to ML pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m src.main --interactive

  # Single instruction
  python -m src.main --instruction "Create a fraud detection model using BigQuery data"

  # Batch mode with file
  python -m src.main --batch instructions.txt

  # Custom configuration
  python -m src.main --config custom_config.yaml --interactive
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive conversation mode"
    )
    
    mode_group.add_argument(
        "--instruction",
        type=str,
        help="Single instruction to process"
    )
    
    mode_group.add_argument(
        "--batch",
        type=str,
        help="Process instructions from file (one per line)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files (default: output)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "yaml", "kfp"],
        default="json",
        help="Output format for pipeline definitions (default: json)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and plan only, don't execute pipelines"
    )
    
    parser.add_argument(
        "--save-conversation",
        type=str,
        help="Save conversation history to file"
    )
    
    return parser


def process_single_instruction(
    agent: VertexAIAgent,
    instruction: str,
    output_dir: str,
    output_format: str,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Process a single instruction and generate pipeline."""
    logging.info(f"Processing instruction: {instruction[:100]}...")
    
    try:
        # Process instruction with agent
        result = agent.process_instruction(instruction)
        
        if not result.get("success", False):
            logging.error(f"Failed to process instruction: {result.get('error', 'Unknown error')}")
            return result
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        timestamp = result.get("timestamp", "unknown")
        base_filename = f"pipeline_{timestamp.replace(':', '-').replace('.', '_')}"
        
        # Save parsed instruction
        parsed_file = output_path / f"{base_filename}_parsed.json"
        with open(parsed_file, 'w') as f:
            json.dump(result.get("parsed_instruction", {}), f, indent=2)
        
        # Save pipeline plan
        plan_file = output_path / f"{base_filename}_plan.json"
        with open(plan_file, 'w') as f:
            json.dump(result.get("pipeline_plan", {}), f, indent=2)
        
        # Generate and save pipeline code
        if output_format == "kfp" and result.get("pipeline_plan"):
            pipeline_plan = result["pipeline_plan"]
            planner = PipelinePlanner(
                project_id=agent.config.gcp.project_id,
                region=agent.config.gcp.region
            )
            
            kfp_code = planner.generate_kfp_pipeline(pipeline_plan)
            kfp_file = output_path / f"{base_filename}_pipeline.py"
            with open(kfp_file, 'w') as f:
                f.write(kfp_code)
            
            result["kfp_file"] = str(kfp_file)
        
        # Execute pipeline if not dry run
        if not dry_run and result.get("pipeline_plan"):
            logging.info("Executing pipeline...")
            execution_result = agent.execute_pipeline(result["pipeline_plan"])
            result["execution"] = execution_result
            
            # Save execution results
            exec_file = output_path / f"{base_filename}_execution.json"
            with open(exec_file, 'w') as f:
                json.dump(execution_result, f, indent=2)
        
        result["output_files"] = {
            "parsed_instruction": str(parsed_file),
            "pipeline_plan": str(plan_file)
        }
        
        logging.info(f"Results saved to {output_dir}")
        return result
        
    except Exception as e:
        logging.error(f"Error processing instruction: {e}")
        return {
            "success": False,
            "error": str(e),
            "instruction": instruction
        }


def process_batch_instructions(
    agent: VertexAIAgent,
    batch_file: str,
    output_dir: str,
    output_format: str,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Process multiple instructions from a file."""
    logging.info(f"Processing batch instructions from {batch_file}")
    
    try:
        with open(batch_file, 'r') as f:
            instructions = [line.strip() for line in f if line.strip()]
        
        if not instructions:
            return {
                "success": False,
                "error": "No instructions found in batch file"
            }
        
        results = []
        for i, instruction in enumerate(instructions, 1):
            logging.info(f"Processing instruction {i}/{len(instructions)}")
            
            # Create subdirectory for each instruction
            instruction_dir = Path(output_dir) / f"instruction_{i:03d}"
            
            result = process_single_instruction(
                agent, instruction, str(instruction_dir), output_format, dry_run
            )
            results.append(result)
        
        # Save batch summary
        summary = {
            "total_instructions": len(instructions),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "results": results
        }
        
        summary_file = Path(output_dir) / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Batch processing completed. Summary saved to {summary_file}")
        return summary
        
    except Exception as e:
        logging.error(f"Error processing batch instructions: {e}")
        return {
            "success": False,
            "error": str(e),
            "batch_file": batch_file
        }


def interactive_mode(
    agent: VertexAIAgent,
    output_dir: str,
    output_format: str,
    dry_run: bool = False,
    save_conversation: Optional[str] = None
) -> None:
    """Start interactive conversation mode."""
    logging.info("Starting interactive mode")
    
    # Initialize conversation manager
    conversation_manager = ConversationManager(agent)
    
    print("\n" + "="*60)
    print("Vertex AI Pipeline Agent - Interactive Mode")
    print("="*60)
    print("Enter natural language instructions to create ML pipelines.")
    print("Type 'help' for commands, 'quit' to exit.")
    print("="*60 + "\n")
    
    try:
        while True:
            try:
                # Get user input
                user_input = input("\n🤖 Enter instruction: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    print_help()
                    continue
                elif user_input.lower() == 'history':
                    conversation_manager.print_history()
                    continue
                elif user_input.lower() == 'clear':
                    conversation_manager.clear_history()
                    print("Conversation history cleared.")
                    continue
                elif user_input.lower().startswith('save '):
                    filename = user_input[5:].strip()
                    conversation_manager.save_conversation(filename)
                    print(f"Conversation saved to {filename}")
                    continue
                elif user_input.lower().startswith('load '):
                    filename = user_input[5:].strip()
                    conversation_manager.load_conversation(filename)
                    print(f"Conversation loaded from {filename}")
                    continue
                
                # Process instruction
                print("\n🔄 Processing instruction...")
                
                result = conversation_manager.process_turn(user_input)
                
                if result.get("success", False):
                    print("\n✅ Instruction processed successfully!")
                    
                    # Display summary
                    if "parsed_instruction" in result:
                        parsed = result["parsed_instruction"]
                        print(f"\n📋 Task Type: {parsed.get('task_type', 'Unknown')}")
                        print(f"🎯 Target Column: {parsed.get('target_column', 'Not specified')}")
                        print(f"📊 Data Source: {parsed.get('data_source', {}).get('location', 'Not specified')}")
                        print(f"🔧 Framework: {parsed.get('model_config', {}).get('framework', 'Not specified')}")
                    
                    if "pipeline_plan" in result:
                        plan = result["pipeline_plan"]
                        print(f"\n🏗️  Pipeline: {plan.get('name', 'Unnamed')}")
                        print(f"📦 Components: {len(plan.get('components', []))}")
                    
                    # Save results if requested
                    if output_dir:
                        timestamp = result.get("timestamp", "unknown")
                        instruction_dir = Path(output_dir) / f"interactive_{timestamp.replace(':', '-').replace('.', '_')}"
                        
                        saved_result = process_single_instruction(
                            agent, user_input, str(instruction_dir), output_format, dry_run
                        )
                        
                        if saved_result.get("success", False):
                            print(f"\n💾 Results saved to {instruction_dir}")
                
                else:
                    print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                    
                    # Show suggestions if available
                    if "suggestions" in result:
                        print("\n💡 Suggestions:")
                        for suggestion in result["suggestions"]:
                            print(f"  • {suggestion}")
            
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Type 'quit' to exit.")
                continue
            except Exception as e:
                logging.error(f"Error in interactive mode: {e}")
                print(f"\n❌ Unexpected error: {e}")
                continue
    
    finally:
        # Save conversation if requested
        if save_conversation:
            conversation_manager.save_conversation(save_conversation)
            print(f"\n💾 Conversation saved to {save_conversation}")
        
        print("\n👋 Goodbye!")


def print_help() -> None:
    """Print help information for interactive mode."""
    help_text = """
Available Commands:
  help          - Show this help message
  history       - Show conversation history
  clear         - Clear conversation history
  save <file>   - Save conversation to file
  load <file>   - Load conversation from file
  quit/exit/q   - Exit interactive mode

Example Instructions:
  • "Create a fraud detection model using BigQuery data from project.dataset.table"
  • "Build a churn prediction pipeline with XGBoost using customer data"
  • "Set up a recommendation system with TensorFlow for product recommendations"
  • "Create a time series forecasting model for sales data"
  • "Deploy the model to an endpoint with monitoring"

Tips:
  • Be specific about your data source (BigQuery table, GCS path, etc.)
  • Mention the target column you want to predict
  • Specify the ML framework if you have a preference
  • Include deployment and monitoring requirements
    """
    print(help_text)


def main() -> None:
    """Main application entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize agent
        logging.info("Initializing Vertex AI Pipeline Agent...")
        agent = VertexAIAgent(config)
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process based on mode
        if args.interactive:
            interactive_mode(
                agent, 
                args.output_dir, 
                args.format, 
                args.dry_run,
                args.save_conversation
            )
        
        elif args.instruction:
            result = process_single_instruction(
                agent, 
                args.instruction, 
                args.output_dir, 
                args.format, 
                args.dry_run
            )
            
            if result.get("success", False):
                print("✅ Instruction processed successfully!")
                if "output_files" in result:
                    print(f"📁 Output files: {result['output_files']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        elif args.batch:
            result = process_batch_instructions(
                agent, 
                args.batch, 
                args.output_dir, 
                args.format, 
                args.dry_run
            )
            
            if result.get("success", False):
                print(f"✅ Batch processing completed!")
                print(f"📊 Processed: {result['successful']}/{result['total_instructions']} instructions")
            else:
                print(f"❌ Batch processing failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
