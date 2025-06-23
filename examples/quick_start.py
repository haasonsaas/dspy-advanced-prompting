"""
Quick start examples for DSpy Advanced Prompting Techniques
"""

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

if __name__ == "__main__":
    print("DSpy Advanced Prompting - Quick Start Examples\n")
    print("Note: These examples require DSpy to be installed.")
    print("Install with: pip install -r requirements.txt\n")
    
    try:
        # Run examples
        example_manager_style()
        example_thinking_traces()
        example_escape_hatches()
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install required packages:")
        print("pip install -r requirements.txt")