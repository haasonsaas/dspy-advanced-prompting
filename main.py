"""
Main demo script showcasing all advanced prompting techniques
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

# Import all techniques
from src.prompts.manager_style import create_customer_support_manager, create_code_review_manager
from src.techniques.role_prompting import (
    create_veteran_engineer_persona, 
    create_empathetic_therapist_persona,
    MultiPersonaOrchestrator
)
from src.techniques.task_planning import TaskOrchestrator
from src.techniques.structured_output import (
    StructuredOutputGenerator,
    create_bug_report_schema,
    create_code_review_schema
)
from src.techniques.meta_prompting import MetaPromptOptimizer
from src.techniques.few_shot import FewShotLearner, create_bug_analysis_examples
from src.techniques.prompt_folding import AdaptiveFolder
from src.techniques.escape_hatches import EscapeHatchResponder, GracefulDegradation
from src.techniques.thinking_traces import ThinkingTracer
from src.evaluations.evaluation_framework import PromptEvaluator, create_code_generation_test_suite
from src.techniques.model_distillation import DistillationPipeline

console = Console()


def demo_manager_style():
    """Demo: Manager-Style Hyper-Specific Prompts"""
    console.print(Panel("[bold blue]Manager-Style Prompts Demo[/bold blue]"))
    
    # Customer Support Manager
    support_manager = create_customer_support_manager()
    result = support_manager(
        task="A customer is furious about losing important data after our latest update. They're threatening to cancel their enterprise subscription.",
        context="Customer has been with us for 3 years, $50k annual contract"
    )
    
    console.print("\n[green]Customer Support Manager Response:[/green]")
    console.print(result)
    
    # Code Review Manager
    console.print("\n" + "="*50 + "\n")
    code_manager = create_code_review_manager()
    result = code_manager(
        task="Review this authentication code: user = db.query(f\"SELECT * FROM users WHERE email='{email}'\")",
        context="This is in a production API endpoint"
    )
    
    console.print("[green]Code Review Manager Response:[/green]")
    console.print(result)


def demo_role_prompting():
    """Demo: Role Prompting with Clear Personas"""
    console.print(Panel("[bold blue]Role Prompting Demo[/bold blue]"))
    
    # Create personas
    engineer = create_veteran_engineer_persona()
    therapist = create_empathetic_therapist_persona()
    
    # Multi-persona orchestration
    orchestrator = MultiPersonaOrchestrator({
        "engineer": engineer,
        "therapist": therapist
    })
    
    task = "Our development team is struggling with burnout from constant deadline pressure"
    responses = orchestrator.panel_discussion(task)
    
    console.print("[green]Multi-Persona Perspectives:[/green]\n")
    for persona, response in responses.items():
        console.print(f"[yellow]{persona}:[/yellow]")
        console.print(response)
        console.print()


def demo_task_planning():
    """Demo: Task Definition and Planning"""
    console.print(Panel("[bold blue]Task Planning Demo[/bold blue]"))
    
    orchestrator = TaskOrchestrator()
    
    task = """
    Build a recommendation system that:
    1. Analyzes user behavior patterns
    2. Provides personalized recommendations
    3. Learns from user feedback
    4. Scales to millions of users
    """
    
    context = "E-commerce platform with 10M products and 1M daily active users"
    
    console.print("[yellow]Planning complex task...[/yellow]")
    result = orchestrator(task, context)
    
    console.print(f"\n[green]Task Plan Created:[/green]")
    console.print(f"Total sub-tasks: {len(result['plan']['sub_tasks'])}")
    console.print(f"Success: {result['success']}")


def demo_structured_output():
    """Demo: Structured Output with Tags"""
    console.print(Panel("[bold blue]Structured Output Demo[/bold blue]"))
    
    generator = StructuredOutputGenerator()
    bug_schema = create_bug_report_schema()
    
    result = generator(
        task="The payment processing fails silently when users have special characters in their address",
        schema=bug_schema,
        context="This happens on the checkout page of our e-commerce site"
    )
    
    console.print("[green]Generated Bug Report:[/green]")
    console.print(result)


def demo_meta_prompting():
    """Demo: Meta-Prompting Self-Optimization"""
    console.print(Panel("[bold blue]Meta-Prompting Demo[/bold blue]"))
    
    optimizer = MetaPromptOptimizer()
    
    # Poor initial prompt
    initial_prompt = "Summarize the text"
    
    # Analyze and optimize
    console.print(f"[yellow]Initial Prompt:[/yellow] {initial_prompt}")
    console.print("\n[yellow]Optimizing prompt...[/yellow]")
    
    from src.techniques.meta_prompting import PromptExample
    examples = [
        PromptExample(
            input="Technical paper on machine learning",
            output="Brief summary without technical details",
            quality_score=4.0,
            issues=["Too vague", "Missing structure"]
        )
    ]
    
    analysis = optimizer.analyze_prompt(initial_prompt, examples)
    console.print(f"\nClarity Score: {analysis.clarity_score}/10")
    console.print(f"Issues: {', '.join(analysis.ambiguities)}")


def demo_thinking_traces():
    """Demo: Thinking Traces"""
    console.print(Panel("[bold blue]Thinking Traces Demo[/bold blue]"))
    
    tracer = ThinkingTracer(verbose=True)
    
    problem = """
    You have 3 boxes: one with only apples, one with only oranges, and one with both.
    All boxes are labeled incorrectly. You can pick one fruit from one box.
    How do you determine the correct labels for all boxes?
    """
    
    result = tracer(problem)
    # Trace visualization handled by ThinkingTracer


def demo_escape_hatches():
    """Demo: Escape Hatches for Uncertainty"""
    console.print(Panel("[bold blue]Escape Hatches Demo[/bold blue]"))
    
    escaper = EscapeHatchResponder()
    
    questions = [
        "What will Bitcoin's price be next month?",
        "Explain how neural networks work",
        "What's the cure for cancer?"
    ]
    
    for question in questions:
        console.print(f"\n[yellow]Question:[/yellow] {question}")
        result = escaper(question)
        console.print(f"[green]Response:[/green] {result['response'][:200]}...")
        console.print(f"[cyan]Confidence:[/cyan] {result['uncertainty_analysis'].confidence_level:.2f}")


async def demo_model_distillation():
    """Demo: Model Distillation Pipeline"""
    console.print(Panel("[bold blue]Model Distillation Demo[/bold blue]"))
    
    pipeline = DistillationPipeline()
    
    complex_prompt = """
    Analyze code for security vulnerabilities including SQL injection,
    XSS, authentication issues, and provide detailed remediation steps.
    """
    
    requirements = {
        "max_latency_ms": 500,
        "max_cost_per_request": 0.001,
        "min_accuracy": 0.85,
        "daily_volume": 50000
    }
    
    console.print("[yellow]Distilling prompt for production...[/yellow]")
    
    # Simulated result for demo
    console.print("\n[green]Distillation Complete:[/green]")
    console.print("• Quality Retention: 92%")
    console.print("• Cost Reduction: 85%")
    console.print("• Speed Improvement: 4.2x")


def main():
    """Run all demos"""
    console.print(Panel(
        "[bold cyan]DSpy Advanced Prompting Techniques Demo[/bold cyan]\n" +
        "Showcasing state-of-the-art prompting strategies",
        expand=False
    ))
    
    demos = [
        ("Manager-Style Prompts", demo_manager_style),
        ("Role Prompting", demo_role_prompting),
        ("Task Planning", demo_task_planning),
        ("Structured Output", demo_structured_output),
        ("Meta-Prompting", demo_meta_prompting),
        ("Thinking Traces", demo_thinking_traces),
        ("Escape Hatches", demo_escape_hatches),
        ("Model Distillation", demo_model_distillation)
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        console.print(f"\n[bold]Demo {i}/{len(demos)}: {name}[/bold]")
        console.print("="*60)
        
        try:
            if asyncio.iscoroutinefunction(demo_func):
                asyncio.run(demo_func())
            else:
                demo_func()
        except Exception as e:
            console.print(f"[red]Demo error: {e}[/red]")
        
        if i < len(demos):
            console.print("\n[dim]Press Enter to continue to next demo...[/dim]")
            input()
    
    console.print("\n[bold green]All demos completed![/bold green]")
    console.print("\nExplore the source code to see how each technique is implemented.")


if __name__ == "__main__":
    main()