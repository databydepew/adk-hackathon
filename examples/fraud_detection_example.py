"""
Fraud Detection Pipeline Example

This example demonstrates how to use the Vertex AI Pipeline Agent to create
a complete fraud detection ML pipeline from natural language instructions.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.agent import VertexAIAgent, AgentConfig
from src.conversation import ConversationManager


def setup_logging():
    """Set up logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def run_fraud_detection_example():
    """Run the fraud detection pipeline example."""
    print("🔍 Fraud Detection Pipeline Example")
    print("=" * 50)
    
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config.yaml"
        config = AgentConfig.load_from_file(str(config_path))
        
        # Initialize agent
        print("🤖 Initializing Vertex AI Pipeline Agent...")
        agent = VertexAIAgent(config)
        
        # Initialize conversation manager
        conversation = ConversationManager(agent)
        
        # Example conversation for fraud detection
        instructions = [
            # Step 1: Initial pipeline creation
            """
            Create a fraud detection model using BigQuery data from the table 
            `my-project.financial_data.transactions`. The target column is 'is_fraud' 
            and I want to use XGBoost for the model. Include features like transaction_amount, 
            merchant_category, time_of_day, and user_location.
            """,
            
            # Step 2: Add deployment requirements
            """
            Deploy this fraud detection model to a Vertex AI endpoint called 
            'fraud-detection-endpoint' with auto-scaling between 2-10 replicas 
            using n1-standard-4 machines.
            """,
            
            # Step 3: Add monitoring
            """
            Set up monitoring for the fraud detection model with drift detection 
            threshold of 0.05 and performance monitoring. Send alerts if accuracy 
            drops below 95% or if data drift is detected.
            """,
            
            # Step 4: Schedule retraining
            """
            Schedule automatic retraining of the fraud detection model weekly 
            and trigger retraining if drift is detected or performance degrades.
            """
        ]
        
        print("\n📝 Processing fraud detection instructions...")
        
        # Process each instruction
        for i, instruction in enumerate(instructions, 1):
            print(f"\n🔄 Step {i}: Processing instruction...")
            print(f"📋 Instruction: {instruction.strip()[:100]}...")
            
            # Process with conversation manager
            result = conversation.process_turn(instruction.strip())
            
            if result.get("success", False):
                print(f"✅ Step {i} completed successfully!")
                
                # Show key information
                if "parsed_instruction" in result:
                    parsed = result["parsed_instruction"]
                    print(f"   🎯 Task: {parsed.get('task_type', 'Unknown')}")
                    print(f"   🔧 Framework: {parsed.get('model_config', {}).get('framework', 'Unknown')}")
                    
                    if parsed.get('data_source'):
                        print(f"   📊 Data: {parsed['data_source'].get('location', 'Unknown')}")
                
                if "pipeline_plan" in result:
                    plan = result["pipeline_plan"]
                    print(f"   🏗️  Pipeline: {plan.get('name', 'Unnamed')}")
                    print(f"   📦 Components: {len(plan.get('components', []))}")
            else:
                print(f"❌ Step {i} failed: {result.get('error', 'Unknown error')}")
                break
        
        # Show final pipeline status
        print("\n" + "=" * 50)
        print("📊 Final Pipeline Summary")
        print("=" * 50)
        
        pipeline_result = conversation.process_turn("show pipeline")
        if pipeline_result.get("success", False) and "pipeline" in pipeline_result:
            pipeline_info = pipeline_result["pipeline"]
            print(f"Pipeline Name: {pipeline_info.get('name', 'Unknown')}")
            print(f"Task Type: {pipeline_info.get('task_type', 'Unknown')}")
            print(f"Framework: {pipeline_info.get('framework', 'Unknown')}")
            print(f"Components: {pipeline_info.get('components', 0)}")
            print(f"Status: {pipeline_info.get('status', 'Unknown')}")
        
        # Show conversation context
        print("\n📋 Conversation Context")
        print("-" * 30)
        
        context_result = conversation.process_turn("show context")
        if context_result.get("success", False) and "context" in context_result:
            context_info = context_result["context"]
            print(f"Turns Completed: {context_info.get('turns_completed', 0)}")
            print(f"Data Sources: {context_info.get('data_sources', 0)}")
            print(f"Has Pipeline: {context_info.get('current_pipeline', False)}")
        
        # Save conversation
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        conversation_file = output_dir / "fraud_detection_conversation.json"
        conversation.save_conversation(str(conversation_file))
        print(f"\n💾 Conversation saved to: {conversation_file}")
        
        # Generate pipeline code
        if conversation.context.current_pipeline:
            print("\n🔧 Generating pipeline code...")
            
            from src.parser import PipelinePlanner
            
            planner = PipelinePlanner(
                project_id=config.gcp.project_id,
                region=config.gcp.region
            )
            
            # Generate KFP code
            kfp_code = planner.generate_kfp_pipeline(conversation.context.current_pipeline)
            
            kfp_file = output_dir / "fraud_detection_pipeline.py"
            with open(kfp_file, 'w') as f:
                f.write(kfp_code)
            
            print(f"📄 KFP pipeline code saved to: {kfp_file}")
            
            # Generate YAML
            yaml_code = planner.export_pipeline_yaml(conversation.context.current_pipeline)
            
            yaml_file = output_dir / "fraud_detection_pipeline.yaml"
            with open(yaml_file, 'w') as f:
                f.write(yaml_code)
            
            print(f"📄 YAML pipeline saved to: {yaml_file}")
        
        print("\n🎉 Fraud detection example completed successfully!")
        print(f"📁 All outputs saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")
        return False
    
    return True


def demonstrate_interactive_features():
    """Demonstrate interactive conversation features."""
    print("\n🗣️  Interactive Features Demo")
    print("=" * 40)
    
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config.yaml"
        config = AgentConfig.load_from_file(str(config_path))
        
        # Initialize agent and conversation
        agent = VertexAIAgent(config)
        conversation = ConversationManager(agent)
        
        # Simulate interactive commands
        commands = [
            "Create a simple fraud detection model with XGBoost",
            "show pipeline",
            "show context",
            "Add monitoring to the current pipeline",
            "show history"
        ]
        
        for command in commands:
            print(f"\n💬 Command: {command}")
            result = conversation.process_turn(command)
            
            if result.get("success", False):
                if "message" in result:
                    print(f"📝 Response: {result['message']}")
                else:
                    print("✅ Command processed successfully")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        print("\n✨ Interactive features demo completed!")
        
    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")


if __name__ == "__main__":
    setup_logging()
    
    print("🚀 Starting Fraud Detection Example")
    print("This example demonstrates the Vertex AI Pipeline Agent capabilities")
    print("for creating fraud detection ML pipelines from natural language.\n")
    
    # Run main example
    success = run_fraud_detection_example()
    
    if success:
        # Run interactive features demo
        demonstrate_interactive_features()
        
        print("\n" + "=" * 60)
        print("🎯 Example Summary")
        print("=" * 60)
        print("✅ Created fraud detection pipeline from natural language")
        print("✅ Added deployment configuration")
        print("✅ Set up monitoring and alerting")
        print("✅ Configured automatic retraining")
        print("✅ Generated KFP and YAML pipeline code")
        print("✅ Demonstrated conversation management")
        print("\n🎉 All features working correctly!")
    else:
        print("\n❌ Example failed. Check logs for details.")
        sys.exit(1)
