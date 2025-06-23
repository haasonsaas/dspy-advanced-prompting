"""
Quick start examples for DSpy Advanced Prompting Techniques

Run this file to see each technique in action with sample inputs.
Requires DSpy to be configured with an API key.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts.manager_style import create_customer_support_manager
from src.techniques.thinking_traces import ThinkingTracer
from src.techniques.escape_hatches import EscapeHatchResponder

def example_manager_style():
    """Example: Using manager-style prompts"""
    print("=== Manager-Style Prompt Example ===\n")
    
    # Create a customer support manager
    support_manager = create_customer_support_manager()
    
    # Handle a customer issue
    response = support_manager(
        task="Customer's data was corrupted during migration",
        context="Enterprise customer, $100k annual contract, been with us 5 years"
    )
    
    print("Manager Response:")
    print(response)
    print("\n" + "="*50 + "\n")

def example_thinking_traces():
    """Example: Using thinking traces for debugging"""
    print("=== Thinking Traces Example ===\n")
    
    tracer = ThinkingTracer(verbose=True)
    
    problem = "Calculate the compound interest on $1000 at 5% annual rate for 3 years"
    
    result = tracer(problem)
    print(f"\nFinal Answer: {result['answer']}")
    print("\n" + "="*50 + "\n")

def example_escape_hatches():
    """Example: Handling uncertainty properly"""
    print("=== Escape Hatches Example ===\n")
    
    escaper = EscapeHatchResponder()
    
    # Ask something uncertain
    question = "What will the stock market do tomorrow?"
    result = escaper(question)
    
    print(f"Question: {question}")
    print(f"Response: {result['response']}")
    print(f"Confidence Level: {result['uncertainty_analysis'].confidence_level:.2f}")
    print(f"Uncertainty Level: {result['uncertainty_analysis'].uncertainty_level}")

def configure_dspy():
    """Configure DSpy with API key"""
    try:
        import dspy
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("ERROR: OPENAI_API_KEY not found in environment")
            print("Please set your API key in .env file")
            return False
        
        lm = dspy.LM(model="gpt-4o-mini", api_key=api_key, max_tokens=1000)
        dspy.settings.configure(lm=lm)
        print("✓ DSpy configured with OpenAI API\n")
        return True
    except Exception as e:
        print(f"Failed to configure DSpy: {e}")
        return False

if __name__ == "__main__":
    print("DSpy Advanced Prompting - Quick Start Examples\n")
    
    # Check if DSpy can be configured
    if not configure_dspy():
        sys.exit(1)
    
    try:
        # Run examples
        example_manager_style()
        example_thinking_traces() 
        example_escape_hatches()
        
        print("\n✅ All examples completed successfully!")
        print("Explore the src/ directory to see how each technique is implemented.")
        
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("\nPlease install required packages:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure your API key is configured in .env file")