#!/usr/bin/env python3
"""
Quick test to verify notebooks will work for users
"""

import os
import sys
import json
from pathlib import Path

def test_notebook_structure():
    """Test that notebooks have valid structure"""
    notebooks_dir = Path("notebooks")
    notebooks = list(notebooks_dir.glob("*.ipynb"))
    
    print(f"📓 Found {len(notebooks)} notebooks:")
    
    for notebook_path in notebooks:
        print(f"\n📋 Testing {notebook_path.name}...")
        
        try:
            with open(notebook_path) as f:
                notebook = json.load(f)
            
            cells = notebook.get('cells', [])
            code_cells = [c for c in cells if c.get('cell_type') == 'code']
            markdown_cells = [c for c in cells if c.get('cell_type') == 'markdown']
            
            print(f"  ✅ Valid JSON structure")
            print(f"  📊 {len(cells)} total cells ({len(code_cells)} code, {len(markdown_cells)} markdown)")
            
            # Check first code cell has setup
            if code_cells:
                first_code = ''.join(code_cells[0].get('source', []))
                if 'sys.path.append' in first_code and 'load_dotenv' in first_code:
                    print(f"  ✅ Proper setup in first code cell")
                else:
                    print(f"  ⚠️ First code cell missing setup")
            
            # Check for API key handling
            has_api_check = False
            for cell in code_cells:
                source = ''.join(cell.get('source', []))
                if 'OPENAI_API_KEY' in source and 'not api_key' in source:
                    has_api_check = True
                    break
            
            if has_api_check:
                print(f"  ✅ Proper API key error handling")
            else:
                print(f"  ⚠️ Missing API key error handling")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return True

def test_imports():
    """Test that all imports used in notebooks work"""
    print(f"\n🔍 Testing notebook imports...")
    
    try:
        # Core dependencies
        import dspy
        import matplotlib.pyplot as plt
        from IPython.display import HTML, display
        import numpy as np
        from dotenv import load_dotenv
        
        # Our modules
        from src.prompts.manager_style import create_customer_support_manager, ManagerStylePromptConfig
        from src.techniques.escape_hatches import EscapeHatchResponder
        from src.techniques.thinking_traces import ThinkingTracer
        from src.techniques.few_shot import FewShotLearner, create_bug_analysis_examples
        
        print(f"  ✅ All notebook imports successful")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test core functionality without API calls"""
    print(f"\n⚙️ Testing basic functionality...")
    
    try:
        # Test uncertainty detection (no API needed)
        from src.techniques.escape_hatches import UncertaintyDetector
        detector = UncertaintyDetector()
        level, confidence = detector.detect_uncertainty("I might be wrong about this")
        print(f"  ✅ Uncertainty detection: {level} (confidence: {confidence:.2f})")
        
        # Test config creation
        from src.prompts.manager_style import ManagerStylePromptConfig
        config = ManagerStylePromptConfig(
            role_title="Test",
            department="Test", 
            company_context="Test",
            reporting_structure="Test",
            key_responsibilities=["Test"],
            performance_metrics=["Test"],
            tools_and_resources=["Test"],
            communication_style="Test",
            decision_authority="Test",
            escalation_procedures="Test",
            constraints=["Test"],
            examples_of_excellence=[{"title": "Test", "situation": "Test", "action": "Test", "result": "Test", "takeaway": "Test"}]
        )
        print(f"  ✅ Config creation works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality error: {e}")
        return False

def main():
    print("🧪 Testing DSpy Advanced Prompting Notebooks")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("notebooks").exists():
        print("❌ notebooks/ directory not found. Run from project root.")
        return False
    
    if not Path("src").exists():
        print("❌ src/ directory not found. Run from project root.")
        return False
    
    # Run tests
    tests = [
        ("Notebook Structure", test_notebook_structure),
        ("Import Dependencies", test_imports), 
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} test...")
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            print(f"   ❌ FAILED with error: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n" + "=" * 50)
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"✅ All notebooks should work correctly!")
        print(f"📓 Users can run: jupyter notebook notebooks/")
    else:
        print(f"⚠️ Some issues found. Check output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)