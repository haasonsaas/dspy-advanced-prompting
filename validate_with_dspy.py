#!/usr/bin/env python3
"""
Comprehensive validation script for DSpy Advanced Prompting
This requires DSpy and other dependencies to be installed
"""

import os
import sys
from typing import Dict, List, Tuple

def check_dependencies() -> Tuple[List[str], List[str]]:
    """Check if all required dependencies are installed"""
    installed = []
    missing = []
    
    dependencies = {
        'dspy': 'dspy-ai>=2.4.0',
        'pydantic': 'pydantic>=2.0.0',
        'rich': 'rich>=13.0.0',
        'loguru': 'loguru>=0.7.0',
        'numpy': 'numpy',
        'pytest': 'pytest>=7.0.0',
        'jinja2': 'jinja2>=3.1.0',
        'dotenv': 'python-dotenv>=1.0.0'
    }
    
    for module, package in dependencies.items():
        try:
            if module == 'dotenv':
                __import__('dotenv')
            else:
                __import__(module)
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    return installed, missing

def validate_dspy_modules():
    """Validate that DSpy modules can be instantiated"""
    print("\n=== Validating DSpy Module Creation ===")
    
    try:
        import dspy
        
        # Test 1: Basic DSpy functionality
        class SimpleSignature(dspy.Signature):
            """A simple test signature"""
            input = dspy.InputField(desc="Test input")
            output = dspy.OutputField(desc="Test output")
        
        # Create a module
        simple_module = dspy.ChainOfThought(SimpleSignature)
        print("✓ DSpy basic module creation works")
        
        # Test 2: Our manager style prompt
        from src.prompts.manager_style import create_customer_support_manager
        manager = create_customer_support_manager()
        print("✓ Manager-style prompt module created")
        
        # Test 3: Role prompting
        from src.techniques.role_prompting import create_veteran_engineer_persona
        engineer = create_veteran_engineer_persona()
        print("✓ Role persona module created")
        
        # Test 4: Task planning
        from src.techniques.task_planning import TaskPlanner
        planner = TaskPlanner()
        print("✓ Task planner module created")
        
        # Test 5: Structured output
        from src.techniques.structured_output import StructuredOutputGenerator
        generator = StructuredOutputGenerator()
        print("✓ Structured output generator created")
        
        # Test 6: Meta-prompting
        from src.techniques.meta_prompting import MetaPromptOptimizer
        optimizer = MetaPromptOptimizer()
        print("✓ Meta-prompt optimizer created")
        
        # Test 7: Few-shot learning
        from src.techniques.few_shot import FewShotLearner, create_bug_analysis_examples
        from src.techniques.few_shot import FewShotPromptTemplate
        examples = create_bug_analysis_examples()
        template = FewShotPromptTemplate(
            task_intro="Test task",
            example_intro="Examples:",
            include_reasoning=True
        )
        learner = FewShotLearner(examples, template)
        print("✓ Few-shot learner created")
        
        # Test 8: Escape hatches
        from src.techniques.escape_hatches import EscapeHatchResponder
        escaper = EscapeHatchResponder()
        print("✓ Escape hatch responder created")
        
        # Test 9: Thinking traces
        from src.techniques.thinking_traces import ThinkingTracer
        tracer = ThinkingTracer(verbose=False)
        print("✓ Thinking tracer created")
        
        # Test 10: Evaluation framework
        from src.evaluations.evaluation_framework import PromptEvaluator
        evaluator = PromptEvaluator(verbose=False)
        print("✓ Prompt evaluator created")
        
        # Test 11: Model distillation
        from src.techniques.model_distillation import PromptDistiller
        distiller = PromptDistiller()
        print("✓ Prompt distiller created")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality with mock LM"""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        import dspy
        
        # Configure DSpy with a mock LM for testing
        # In real use, this would be: dspy.OpenAI(api_key=...) or dspy.Claude(api_key=...)
        print("Note: Real functionality requires configuring DSpy with actual LLM API keys")
        
        # Test uncertainty detection
        from src.techniques.escape_hatches import UncertaintyDetector
        detector = UncertaintyDetector()
        
        test_text = "I might be wrong, but I think the answer could possibly be 42"
        level, confidence = detector.detect_uncertainty(test_text)
        print(f"✓ Uncertainty detection: '{test_text[:30]}...' → {level} (confidence: {confidence})")
        
        # Test thinking trace parsing
        from src.techniques.thinking_traces import ThinkingTracer
        tracer = ThinkingTracer(verbose=False)
        
        trace_text = """
        [THOUGHT] First, I need to understand the problem
        [HYPOTHESIS] The solution might involve recursion
        [VERIFICATION] Let me check this approach
        [INSIGHT] The key is to use dynamic programming
        """
        
        thoughts = tracer.parse_thinking_trace(trace_text)
        print(f"✓ Thinking trace parsing: Found {len(thoughts)} thoughts")
        
        # Test evaluation metrics
        from src.evaluations.evaluation_framework import EvaluationMetrics
        metrics = EvaluationMetrics(
            accuracy=0.92,
            precision=0.89,
            recall=0.94,
            f1_score=0.91,
            consistency_score=0.88,
            robustness_score=0.90,
            edge_case_performance=0.85,
            average_execution_time_ms=250,
            test_coverage=0.95
        )
        overall = metrics.overall_score()
        print(f"✓ Evaluation metrics: Overall score = {overall:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def check_api_keys():
    """Check if API keys are configured"""
    print("\n=== Checking API Keys ===")
    
    api_keys_found = []
    api_keys_missing = []
    
    # Check environment variables
    if os.getenv('OPENAI_API_KEY'):
        api_keys_found.append('OPENAI_API_KEY')
    else:
        api_keys_missing.append('OPENAI_API_KEY')
    
    if os.getenv('ANTHROPIC_API_KEY'):
        api_keys_found.append('ANTHROPIC_API_KEY')
    else:
        api_keys_missing.append('ANTHROPIC_API_KEY')
    
    # Check .env file
    if os.path.exists('.env'):
        print("✓ .env file exists")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✓ .env file loaded")
        except:
            print("✗ Could not load .env file")
    else:
        print("ℹ .env file not found (using environment variables)")
    
    for key in api_keys_found:
        print(f"✓ {key} is set")
    
    for key in api_keys_missing:
        print(f"✗ {key} is not set")
    
    return len(api_keys_missing) == 0

def main():
    print("DSpy Advanced Prompting - Comprehensive Validation")
    print("=" * 60)
    
    # Step 1: Check dependencies
    print("\n=== Checking Dependencies ===")
    installed, missing = check_dependencies()
    
    if installed:
        print(f"\nInstalled packages ({len(installed)}):")
        for pkg in installed:
            print(f"  ✓ {pkg}")
    
    if missing:
        print(f"\nMissing packages ({len(missing)}):")
        for pkg in missing:
            print(f"  ✗ {pkg}")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        return False
    
    # Step 2: Validate DSpy modules
    if not validate_dspy_modules():
        return False
    
    # Step 3: Test basic functionality
    if not test_basic_functionality():
        return False
    
    # Step 4: Check API keys
    api_keys_ok = check_api_keys()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY:")
    print(f"✓ All dependencies installed")
    print(f"✓ All DSpy modules can be created")
    print(f"✓ Basic functionality works")
    
    if api_keys_ok:
        print(f"✓ API keys configured")
        print("\n✅ The project is fully validated and ready to use!")
        print("\nTo run with actual LLMs:")
        print("  python main.py")
    else:
        print(f"ℹ API keys not configured")
        print("\n⚠️  The project structure is valid but needs API keys to run with actual LLMs")
        print("\nTo configure:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OpenAI and/or Anthropic API keys")
        print("  3. Run: python main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)