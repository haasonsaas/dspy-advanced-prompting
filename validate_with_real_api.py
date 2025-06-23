#!/usr/bin/env python3
"""
Real API validation script for DSpy Advanced Prompting
This makes actual API calls to test the prompting techniques
"""

import os
import sys
import asyncio
from typing import Dict, List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()

def configure_dspy():
    """Configure DSpy with OpenAI API"""
    try:
        import dspy
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            console.print("[red]ERROR: OPENAI_API_KEY not found in environment[/red]")
            console.print("Please set your API key in .env file")
            return False
        
        # Configure DSpy with OpenAI
        lm = dspy.LM(
            model="gpt-4o-mini",
            api_key=api_key,
            max_tokens=2000
        )
        dspy.settings.configure(lm=lm)
        
        console.print("✓ DSpy configured with OpenAI GPT-4o-mini")
        return True
        
    except Exception as e:
        console.print(f"[red]Failed to configure DSpy: {e}[/red]")
        return False

def test_manager_style_prompts():
    """Test manager-style prompts with real API calls"""
    console.print("\n[bold blue]=== Testing Manager-Style Prompts ===[/bold blue]")
    
    try:
        from src.prompts.manager_style import create_customer_support_manager
        
        # Create manager agent
        support_manager = create_customer_support_manager()
        
        # Test case: Customer complaint
        task = "Handle a customer complaint about data loss"
        context = "Customer reports losing 2 weeks of project data after recent update. They are threatening to cancel their $50k enterprise subscription."
        
        console.print(f"[yellow]Task:[/yellow] {task}")
        console.print(f"[yellow]Context:[/yellow] {context}")
        console.print("\n[cyan]Calling OpenAI API...[/cyan]")
        
        # Make actual API call
        result = support_manager(task=task, context=context)
        
        console.print("\n[green]Manager Response:[/green]")
        console.print(Panel(result, expand=False))
        
        # Validate response quality
        response_length = len(result)
        has_empathy = any(word in result.lower() for word in ['sorry', 'understand', 'apologize', 'frustration'])
        has_solution = any(word in result.lower() for word in ['investigate', 'restore', 'backup', 'solution'])
        
        console.print(f"\n[cyan]Response Analysis:[/cyan]")
        console.print(f"• Length: {response_length} characters")
        console.print(f"• Shows empathy: {'✓' if has_empathy else '✗'}")
        console.print(f"• Offers solution: {'✓' if has_solution else '✗'}")
        
        return {
            'success': True,
            'response_length': response_length,
            'has_empathy': has_empathy,
            'has_solution': has_solution,
            'response': result
        }
        
    except Exception as e:
        console.print(f"[red]Manager-style test failed: {e}[/red]")
        return {'success': False, 'error': str(e)}

def test_escape_hatches():
    """Test escape hatches with real API calls"""
    console.print("\n[bold blue]=== Testing Escape Hatches ===[/bold blue]")
    
    try:
        from src.techniques.escape_hatches import EscapeHatchResponder
        
        # Create escape hatch responder
        escaper = EscapeHatchResponder()
        
        test_questions = [
            "What will Bitcoin's price be exactly one year from today?",
            "Explain how neural networks work",
            "What's the definitive cure for all types of cancer?"
        ]
        
        results = []
        
        for question in test_questions:
            console.print(f"\n[yellow]Question:[/yellow] {question}")
            console.print("[cyan]Calling OpenAI API...[/cyan]")
            
            # Make actual API call
            result = escaper(question)
            
            console.print(f"[green]Response:[/green] {result['response'][:200]}...")
            console.print(f"[cyan]Confidence:[/cyan] {result['uncertainty_analysis'].confidence_level:.2f}")
            console.print(f"[cyan]Uncertainty Level:[/cyan] {result['uncertainty_analysis'].uncertainty_level}")
            
            results.append({
                'question': question,
                'confidence': result['uncertainty_analysis'].confidence_level,
                'uncertainty_level': result['uncertainty_analysis'].uncertainty_level,
                'response_length': len(result['response'])
            })
        
        return {'success': True, 'results': results}
        
    except Exception as e:
        console.print(f"[red]Escape hatch test failed: {e}[/red]")
        return {'success': False, 'error': str(e)}

def test_few_shot_learning():
    """Test few-shot learning with real API calls"""
    console.print("\n[bold blue]=== Testing Few-Shot Learning ===[/bold blue]")
    
    try:
        from src.techniques.few_shot import FewShotLearner, create_bug_analysis_examples, FewShotPromptTemplate
        
        # Create few-shot learner
        examples = create_bug_analysis_examples()
        template = FewShotPromptTemplate(
            task_intro="Analyze the following bug report:",
            example_intro="Here are examples of good bug analysis:",
            include_reasoning=True
        )
        learner = FewShotLearner(examples, template)
        
        # Test bug analysis
        bug_report = """
        User reports: "The app crashes when I try to upload a large file. 
        It happens every time with files over 50MB. The error message says 
        'Memory allocation failed' but my device has plenty of storage space."
        """
        
        console.print(f"[yellow]Bug Report:[/yellow] {bug_report.strip()}")
        console.print("[cyan]Calling OpenAI API...[/cyan]")
        
        # Make actual API call
        result = learner(bug_report)
        
        console.print(f"\n[green]Analysis:[/green]")
        console.print(Panel(result, expand=False))
        
        # Validate analysis quality
        has_root_cause = any(word in result.lower() for word in ['memory', 'allocation', 'heap', 'ram'])
        has_solution = any(word in result.lower() for word in ['chunk', 'stream', 'optimize', 'limit'])
        
        console.print(f"\n[cyan]Analysis Quality:[/cyan]")
        console.print(f"• Identifies root cause: {'✓' if has_root_cause else '✗'}")
        console.print(f"• Suggests solution: {'✓' if has_solution else '✗'}")
        
        return {
            'success': True,
            'has_root_cause': has_root_cause,
            'has_solution': has_solution,
            'response': result
        }
        
    except Exception as e:
        console.print(f"[red]Few-shot test failed: {e}[/red]")
        return {'success': False, 'error': str(e)}

def test_thinking_traces():
    """Test thinking traces with real API calls"""
    console.print("\n[bold blue]=== Testing Thinking Traces ===[/bold blue]")
    
    try:
        from src.techniques.thinking_traces import ThinkingTracer
        
        # Create thinking tracer
        tracer = ThinkingTracer(verbose=True)
        
        # Logic puzzle
        problem = """
        You have 12 balls that look identical. 11 balls have the same weight, 
        but one ball is either heavier or lighter than the others. You have a 
        balance scale that can compare weights. What's the minimum number of 
        weighings needed to identify the different ball AND determine if it's 
        heavier or lighter?
        """
        
        console.print(f"[yellow]Problem:[/yellow] {problem.strip()}")
        console.print("[cyan]Calling OpenAI API...[/cyan]")
        
        # Make actual API call
        result = tracer(problem)
        
        # Extract the answer and thinking steps for display
        answer = result.get('answer', 'No answer provided')
        thinking_steps = result.get('thinking_steps', '')
        
        console.print(f"\n[green]Solution:[/green]")
        console.print(Panel(answer, title="Final Answer", expand=False))
        
        console.print(f"\n[green]Thinking Steps:[/green]")
        console.print(Panel(thinking_steps[:500] + "..." if len(thinking_steps) > 500 else thinking_steps, 
                           title="Reasoning Process", expand=False))
        
        # Check if trace contains thinking markers
        has_thinking_markers = any(marker in thinking_steps for marker in ['[THOUGHT]', '[HYPOTHESIS]', '[ANALYSIS]'])
        mentions_weighings = 'weigh' in thinking_steps.lower() or 'weigh' in answer.lower()
        
        console.print(f"\n[cyan]Trace Quality:[/cyan]")
        console.print(f"• Contains thinking markers: {'✓' if has_thinking_markers else '✗'}")
        console.print(f"• Addresses weighings: {'✓' if mentions_weighings else '✗'}")
        
        return {
            'success': True,
            'has_thinking_markers': has_thinking_markers,
            'mentions_weighings': mentions_weighings,
            'response': answer,
            'thinking_steps': thinking_steps
        }
        
    except Exception as e:
        console.print(f"[red]Thinking traces test failed: {e}[/red]")
        return {'success': False, 'error': str(e)}

def calculate_api_usage_estimate(results):
    """Estimate API usage based on results"""
    total_chars = 0
    api_calls = 0
    
    for test_name, result in results.items():
        if result.get('success'):
            api_calls += 1
            if 'response' in result:
                total_chars += len(result['response'])
            if 'thinking_steps' in result:
                total_chars += len(result['thinking_steps'])
            if 'results' in result:  # escape hatches has multiple results
                api_calls += len(result['results']) - 1  # -1 because we already counted one
                for r in result['results']:
                    total_chars += r.get('response_length', 0)
    
    # Rough token estimate (4 chars per token average)
    estimated_tokens = total_chars // 4
    # GPT-4o-mini pricing (approximate)
    estimated_cost = (estimated_tokens / 1000) * 0.0015  # $0.0015 per 1K tokens
    
    return {
        'api_calls': api_calls,
        'estimated_tokens': estimated_tokens,
        'estimated_cost_usd': estimated_cost
    }

def main():
    """Run comprehensive API validation"""
    console.print(Panel(
        "[bold cyan]DSpy Advanced Prompting - Real API Validation[/bold cyan]\n" +
        "This will make actual API calls to test the techniques",
        expand=False
    ))
    
    # Step 1: Configure DSpy
    if not configure_dspy():
        return False
    
    # Step 2: Run tests with progress tracking
    tests = [
        ("Manager-Style Prompts", test_manager_style_prompts),
        ("Escape Hatches", test_escape_hatches),
        ("Few-Shot Learning", test_few_shot_learning),
        ("Thinking Traces", test_thinking_traces)
    ]
    
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        for test_name, test_func in tests:
            task = progress.add_task(f"Running {test_name}...", total=None)
            
            try:
                result = test_func()
                results[test_name] = result
                
                if result.get('success'):
                    progress.update(task, description=f"✓ {test_name} completed")
                else:
                    progress.update(task, description=f"✗ {test_name} failed")
                    
                progress.stop_task(task)
                
            except Exception as e:
                results[test_name] = {'success': False, 'error': str(e)}
                progress.update(task, description=f"✗ {test_name} error")
                progress.stop_task(task)
    
    # Step 3: Generate summary
    console.print("\n" + "="*60)
    console.print("[bold]REAL API VALIDATION SUMMARY[/bold]")
    console.print("="*60)
    
    successful_tests = sum(1 for r in results.values() if r.get('success'))
    total_tests = len(results)
    
    console.print(f"Tests passed: {successful_tests}/{total_tests}")
    
    for test_name, result in results.items():
        status = "✓" if result.get('success') else "✗"
        console.print(f"{status} {test_name}")
        if not result.get('success') and 'error' in result:
            console.print(f"   Error: {result['error']}")
    
    # API usage summary
    usage = calculate_api_usage_estimate(results)
    console.print(f"\n[cyan]API Usage Summary:[/cyan]")
    console.print(f"• API calls made: {usage['api_calls']}")
    console.print(f"• Estimated tokens: {usage['estimated_tokens']:,}")
    console.print(f"• Estimated cost: ${usage['estimated_cost_usd']:.4f}")
    
    # Overall status
    if successful_tests == total_tests:
        console.print(f"\n✅ [bold green]All techniques validated with real API calls![/bold green]")
        console.print("The advanced prompting techniques are working correctly with OpenAI's API.")
    else:
        console.print(f"\n⚠️ [bold yellow]{total_tests - successful_tests} test(s) failed[/bold yellow]")
        console.print("Check the error messages above for details.")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)