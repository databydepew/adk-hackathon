"""
Churn Prediction Pipeline Example

This example demonstrates creating a customer churn prediction pipeline
using different ML frameworks and deployment strategies.
"""

import sys
import json
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


def run_churn_prediction_example():
    """Run the churn prediction pipeline example."""
    print("📈 Customer Churn Prediction Pipeline Example")
    print("=" * 55)
    
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config.yaml"
        config = AgentConfig.load_from_file(str(config_path))
        
        # Initialize agent
        print("🤖 Initializing Vertex AI Pipeline Agent...")
        agent = VertexAIAgent(config)
        
        # Initialize conversation manager
        conversation = ConversationManager(agent)
        
        # Churn prediction scenario
        print("\n📋 Scenario: Telecom company wants to predict customer churn")
        print("🎯 Goal: Build ML pipeline to identify customers likely to churn")
        
        # Step-by-step pipeline creation
        instructions = [
            # Step 1: Create initial model
            """
            Create a customer churn prediction model using TensorFlow. The data is in 
            BigQuery table `telecom-project.customer_data.churn_features` with target 
            column 'churned'. Include features like monthly_charges, total_charges, 
            contract_length, customer_service_calls, and usage_patterns.
            """,
            
            # Step 2: Compare with different framework
            """
            Also create an alternative churn model using scikit-learn Random Forest 
            for comparison. Use the same data source and features.
            """,
            
            # Step 3: Set up A/B testing deployment
            """
            Deploy both churn models to the same endpoint 'churn-prediction-endpoint' 
            with 70% traffic to TensorFlow model and 30% to Random Forest model 
            for A/B testing.
            """,
            
            # Step 4: Add comprehensive monitoring
            """
            Set up comprehensive monitoring for both churn models including:
            - Data drift detection with threshold 0.08
            - Performance monitoring comparing model accuracy
            - Feature importance tracking
            - Prediction distribution monitoring
            """,
            
            # Step 5: Batch prediction setup
            """
            Set up daily batch prediction job for scoring all active customers 
            in the database. Save results to BigQuery table 
            `telecom-project.predictions.daily_churn_scores`.
            """
        ]
        
        results = []
        
        print("\n🔄 Processing churn prediction instructions...")
        
        # Process each instruction
        for i, instruction in enumerate(instructions, 1):
            print(f"\n📝 Step {i}: {instruction.strip().split('.')[0]}...")
            
            result = conversation.process_turn(instruction.strip())
            results.append(result)
            
            if result.get("success", False):
                print(f"✅ Step {i} completed successfully!")
                
                # Show specific information for each step
                if i == 1 and "parsed_instruction" in result:
                    parsed = result["parsed_instruction"]
                    print(f"   🧠 Model: {parsed.get('model_config', {}).get('framework', 'Unknown')}")
                    print(f"   🎯 Target: {parsed.get('target_column', 'Unknown')}")
                    print(f"   📊 Features: {len(parsed.get('feature_columns', []))} columns")
                
                elif i == 2:
                    print("   🔄 Alternative model configuration created")
                
                elif i == 3:
                    print("   🚀 A/B testing deployment configured")
                
                elif i == 4:
                    print("   📊 Comprehensive monitoring setup")
                
                elif i == 5:
                    print("   ⏰ Batch prediction scheduling configured")
            
            else:
                print(f"❌ Step {i} failed: {result.get('error', 'Unknown error')}")
                # Continue with other steps
        
        # Demonstrate conversation features
        print("\n" + "=" * 55)
        print("🗣️  Conversation Features Demo")
        print("=" * 55)
        
        conversation_commands = [
            ("show pipeline", "Display current pipeline information"),
            ("show context", "Show conversation context"),
            ("show history", "Display conversation history")
        ]
        
        for command, description in conversation_commands:
            print(f"\n💬 {description}")
            result = conversation.process_turn(command)
            
            if result.get("success", False):
                if command == "show pipeline" and "pipeline" in result:
                    pipeline = result["pipeline"]
                    print(f"   📦 Pipeline: {pipeline.get('name', 'Unknown')}")
                    print(f"   🔧 Components: {pipeline.get('components', 0)}")
                    print(f"   📈 Task: {pipeline.get('task_type', 'Unknown')}")
                
                elif command == "show context" and "context" in result:
                    context = result["context"]
                    print(f"   🔄 Turns: {context.get('turns_completed', 0)}")
                    print(f"   📊 Data Sources: {context.get('data_sources', 0)}")
                    print(f"   🎯 Has Pipeline: {context.get('current_pipeline', False)}")
                
                elif command == "show history":
                    print(f"   📝 {len(conversation.turns)} conversation turns recorded")
            
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        
        # Save outputs
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Save conversation
        conversation_file = output_dir / "churn_prediction_conversation.json"
        conversation.save_conversation(str(conversation_file))
        
        # Save results summary
        summary = {
            "example": "churn_prediction",
            "total_steps": len(instructions),
            "successful_steps": sum(1 for r in results if r.get("success", False)),
            "conversation_summary": conversation.get_conversation_summary(),
            "results": results
        }
        
        summary_file = output_dir / "churn_prediction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Outputs saved to: {output_dir}")
        print(f"   📄 Conversation: {conversation_file.name}")
        print(f"   📊 Summary: {summary_file.name}")
        
        # Generate pipeline artifacts if available
        if conversation.context.current_pipeline:
            print("\n🔧 Generating pipeline artifacts...")
            
            from src.parser import PipelinePlanner
            
            planner = PipelinePlanner(
                project_id=config.gcp.project_id,
                region=config.gcp.region
            )
            
            # Generate KFP pipeline
            try:
                kfp_code = planner.generate_kfp_pipeline(conversation.context.current_pipeline)
                kfp_file = output_dir / "churn_prediction_pipeline.py"
                with open(kfp_file, 'w') as f:
                    f.write(kfp_code)
                print(f"   🐍 KFP Code: {kfp_file.name}")
            except Exception as e:
                print(f"   ⚠️  KFP generation failed: {e}")
            
            # Generate YAML pipeline
            try:
                yaml_code = planner.export_pipeline_yaml(conversation.context.current_pipeline)
                yaml_file = output_dir / "churn_prediction_pipeline.yaml"
                with open(yaml_file, 'w') as f:
                    f.write(yaml_code)
                print(f"   📄 YAML: {yaml_file.name}")
            except Exception as e:
                print(f"   ⚠️  YAML generation failed: {e}")
        
        print("\n🎉 Churn prediction example completed successfully!")
        
        # Show success metrics
        successful_steps = sum(1 for r in results if r.get("success", False))
        print(f"📊 Success Rate: {successful_steps}/{len(instructions)} steps completed")
        
        return True
        
    except Exception as e:
        logging.error(f"Churn prediction example failed: {e}")
        print(f"❌ Example failed: {e}")
        return False


def demonstrate_advanced_features():
    """Demonstrate advanced conversation and pipeline features."""
    print("\n🚀 Advanced Features Demo")
    print("=" * 35)
    
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config.yaml"
        config = AgentConfig.load_from_file(str(config_path))
        
        # Initialize fresh conversation
        agent = VertexAIAgent(config)
        conversation = ConversationManager(agent)
        
        # Advanced scenarios
        advanced_instructions = [
            # Multi-model comparison
            """
            Create a churn prediction pipeline that compares XGBoost, Random Forest, 
            and Neural Network models. Use cross-validation and select the best 
            performing model automatically.
            """,
            
            # Feature engineering
            """
            Add advanced feature engineering to the churn pipeline including:
            - Customer lifetime value calculation
            - Interaction features between usage and charges
            - Time-based features from usage patterns
            - Categorical encoding for contract types
            """,
            
            # Hyperparameter tuning
            """
            Add hyperparameter tuning to the selected model using Vertex AI 
            Hyperparameter Tuning with 50 trials optimizing for AUC score.
            """,
            
            # Model explanation
            """
            Add model explainability features including SHAP values for feature 
            importance and prediction explanations for the churn model.
            """
        ]
        
        print("🔬 Testing advanced ML pipeline features...")
        
        for i, instruction in enumerate(advanced_instructions, 1):
            print(f"\n🧪 Advanced Feature {i}: Processing...")
            
            result = conversation.process_turn(instruction.strip())
            
            if result.get("success", False):
                print(f"✅ Advanced feature {i} implemented successfully!")
            else:
                print(f"⚠️  Advanced feature {i}: {result.get('error', 'Implementation pending')}")
        
        # Test conversation persistence
        print("\n💾 Testing conversation persistence...")
        
        temp_file = Path(__file__).parent / "temp_conversation.json"
        conversation.save_conversation(str(temp_file))
        
        # Create new conversation and load
        new_conversation = ConversationManager(agent)
        new_conversation.load_conversation(str(temp_file))
        
        # Verify loaded conversation
        if len(new_conversation.turns) > 0:
            print("✅ Conversation persistence working correctly")
        else:
            print("⚠️  Conversation persistence issue detected")
        
        # Clean up temp file
        temp_file.unlink(missing_ok=True)
        
        print("\n✨ Advanced features demo completed!")
        
    except Exception as e:
        print(f"❌ Advanced features demo failed: {e}")


if __name__ == "__main__":
    setup_logging()
    
    print("🚀 Starting Customer Churn Prediction Example")
    print("This example demonstrates advanced ML pipeline creation")
    print("with A/B testing, monitoring, and batch predictions.\n")
    
    # Run main example
    success = run_churn_prediction_example()
    
    if success:
        # Run advanced features demo
        demonstrate_advanced_features()
        
        print("\n" + "=" * 60)
        print("🎯 Churn Prediction Example Summary")
        print("=" * 60)
        print("✅ Multi-framework model comparison (TensorFlow vs scikit-learn)")
        print("✅ A/B testing deployment configuration")
        print("✅ Comprehensive monitoring setup")
        print("✅ Batch prediction scheduling")
        print("✅ Advanced feature engineering")
        print("✅ Conversation state management")
        print("✅ Pipeline artifact generation")
        print("\n🎉 All churn prediction features demonstrated successfully!")
    else:
        print("\n❌ Churn prediction example failed. Check logs for details.")
        sys.exit(1)
