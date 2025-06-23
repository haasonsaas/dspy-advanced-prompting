"""
Prompt Folding for Multi-Step Workflows

This module implements prompt folding where one prompt triggers generation
of deeper, more specific prompts to manage complex multi-step AI workflows.
"""

import dspy
from typing import List, Dict, Optional, Any, Callable
from pydantic import BaseModel, Field
from dataclasses import dataclass
from enum import Enum
import json


class FoldingStrategy(str, Enum):
    RECURSIVE = "recursive"  # Each prompt generates sub-prompts recursively
    PIPELINE = "pipeline"  # Sequential prompt chain
    BRANCHING = "branching"  # Conditional branching based on outputs
    PARALLEL = "parallel"  # Multiple prompts executed in parallel
    ADAPTIVE = "adaptive"  # Strategy chosen based on task analysis


class PromptNode(BaseModel):
    """Represents a node in the prompt folding tree"""
    id: str = Field(..., description="Unique identifier")
    prompt_template: str = Field(..., description="Template for this prompt")
    node_type: str = Field(..., description="Type of processing node")
    children: List['PromptNode'] = Field(default_factory=list, description="Child nodes")
    conditions: Dict[str, str] = Field(default_factory=dict, description="Conditions for execution")
    max_depth: int = Field(default=3, description="Maximum recursion depth")
    merge_strategy: str = Field(default="concatenate", description="How to merge child results")


class FoldingContext(BaseModel):
    """Context passed through the folding process"""
    initial_task: str
    current_depth: int = 0
    accumulated_results: Dict[str, Any] = Field(default_factory=dict)
    execution_path: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PromptGenerationSignature(dspy.Signature):
    """Generate sub-prompts from a high-level prompt"""
    
    parent_prompt = dspy.InputField(desc="The parent/high-level prompt")
    context = dspy.InputField(desc="Current execution context")
    folding_strategy = dspy.InputField(desc="Strategy for generating sub-prompts")
    sub_prompts_json = dspy.OutputField(desc="JSON array of generated sub-prompts")


class PromptExecutionSignature(dspy.Signature):
    """Execute a specific prompt in the folding chain"""
    
    prompt = dspy.InputField(desc="The prompt to execute")
    context = dspy.InputField(desc="Execution context with previous results")
    result = dspy.OutputField(desc="Result of prompt execution")


class PromptFolder(dspy.Module):
    """Manages prompt folding for complex workflows"""
    
    def __init__(self, strategy: FoldingStrategy = FoldingStrategy.RECURSIVE):
        super().__init__()
        self.strategy = strategy
        self.prompt_generator = dspy.ChainOfThought(PromptGenerationSignature)
        self.prompt_executor = dspy.ChainOfThought(PromptExecutionSignature)
        
    def generate_sub_prompts(self, parent_prompt: str, context: FoldingContext) -> List[str]:
        """Generate sub-prompts from a parent prompt"""
        
        strategy_guidelines = {
            FoldingStrategy.RECURSIVE: "Break down into smaller, similar sub-tasks",
            FoldingStrategy.PIPELINE: "Create a sequence of dependent prompts",
            FoldingStrategy.BRANCHING: "Generate alternative paths based on conditions",
            FoldingStrategy.PARALLEL: "Create independent sub-tasks that can run simultaneously",
            FoldingStrategy.ADAPTIVE: "Analyze the task and choose the best decomposition"
        }
        
        result = self.prompt_generator(
            parent_prompt=parent_prompt,
            context=json.dumps(context.dict()),
            folding_strategy=strategy_guidelines[self.strategy]
        )
        
        try:
            return json.loads(result.sub_prompts_json)
        except json.JSONDecodeError:
            return [result.sub_prompts_json]
    
    def execute_prompt(self, prompt: str, context: FoldingContext) -> str:
        """Execute a single prompt with context"""
        
        result = self.prompt_executor(
            prompt=prompt,
            context=json.dumps(context.dict())
        )
        return result.result
    
    def fold(self, initial_prompt: str, max_depth: int = 3) -> Dict[str, Any]:
        """Recursively fold a prompt into sub-prompts and execute"""
        
        context = FoldingContext(initial_task=initial_prompt)
        result = self._fold_recursive(initial_prompt, context, max_depth)
        
        return {
            "final_result": result,
            "execution_path": context.execution_path,
            "accumulated_results": context.accumulated_results
        }
    
    def _fold_recursive(self, prompt: str, context: FoldingContext, 
                       remaining_depth: int) -> Any:
        """Recursive folding implementation"""
        
        context.execution_path.append(prompt[:50] + "...")
        
        if remaining_depth <= 0 or self._is_atomic_prompt(prompt):
            return self.execute_prompt(prompt, context)
        
        sub_prompts = self.generate_sub_prompts(prompt, context)
        
        if not sub_prompts:
            return self.execute_prompt(prompt, context)
        
        context.current_depth += 1
        
        if self.strategy == FoldingStrategy.PARALLEL:
            results = []
            for sub_prompt in sub_prompts:
                result = self._fold_recursive(sub_prompt, context, remaining_depth - 1)
                results.append(result)
            return self._merge_results(results)
        
        elif self.strategy == FoldingStrategy.PIPELINE:
            result = None
            for sub_prompt in sub_prompts:
                context.accumulated_results[f"step_{len(context.accumulated_results)}"] = result
                result = self._fold_recursive(sub_prompt, context, remaining_depth - 1)
            return result
        
        else:  # RECURSIVE or default
            results = []
            for sub_prompt in sub_prompts:
                result = self._fold_recursive(sub_prompt, context, remaining_depth - 1)
                results.append(result)
            return self._merge_results(results)
    
    def _is_atomic_prompt(self, prompt: str) -> bool:
        """Check if a prompt is atomic (cannot be folded further)"""
        
        atomic_indicators = [
            len(prompt.split()) < 10,
            "specific" in prompt.lower(),
            "calculate" in prompt.lower(),
            "return" in prompt.lower(),
            "?" in prompt and len(prompt) < 50
        ]
        
        return any(atomic_indicators)
    
    def _merge_results(self, results: List[Any]) -> Any:
        """Merge results from multiple sub-prompts"""
        
        if not results:
            return None
        if len(results) == 1:
            return results[0]
        
        return {
            "merged_results": results,
            "summary": f"Combined output from {len(results)} sub-tasks"
        }


class ConditionalFolder(dspy.Module):
    """Implements conditional prompt folding with branching logic"""
    
    def __init__(self):
        super().__init__()
        self.condition_evaluator = dspy.ChainOfThought("prompt, result -> condition_met")
        self.folder = PromptFolder(FoldingStrategy.BRANCHING)
        
    def create_branching_node(self, prompt: str, conditions: Dict[str, PromptNode]) -> PromptNode:
        """Create a branching node with conditions"""
        
        return PromptNode(
            id=f"branch_{hash(prompt)}",
            prompt_template=prompt,
            node_type="branching",
            conditions=conditions,
            children=list(conditions.values())
        )
    
    def evaluate_condition(self, condition: str, context: FoldingContext) -> bool:
        """Evaluate if a condition is met"""
        
        result = self.condition_evaluator(
            prompt=condition,
            result=json.dumps(context.accumulated_results)
        )
        return result.condition_met.lower() == "true"
    
    def execute_branching(self, node: PromptNode, context: FoldingContext) -> Any:
        """Execute branching logic"""
        
        for condition, child_node in node.conditions.items():
            if self.evaluate_condition(condition, context):
                return self.folder._fold_recursive(
                    child_node.prompt_template, 
                    context, 
                    node.max_depth
                )
        
        return self.folder.execute_prompt(node.prompt_template, context)


class WorkflowFolder(dspy.Module):
    """High-level workflow management using prompt folding"""
    
    def __init__(self):
        super().__init__()
        self.folders = {
            FoldingStrategy.RECURSIVE: PromptFolder(FoldingStrategy.RECURSIVE),
            FoldingStrategy.PIPELINE: PromptFolder(FoldingStrategy.PIPELINE),
            FoldingStrategy.PARALLEL: PromptFolder(FoldingStrategy.PARALLEL)
        }
        
    def create_data_analysis_workflow(self) -> PromptNode:
        """Create a data analysis workflow"""
        
        root = PromptNode(
            id="data_analysis_root",
            prompt_template="Analyze the dataset and provide insights",
            node_type="root",
            children=[
                PromptNode(
                    id="data_exploration",
                    prompt_template="Explore the data structure and basic statistics",
                    node_type="exploration",
                    children=[
                        PromptNode(
                            id="data_quality",
                            prompt_template="Check data quality and identify missing values",
                            node_type="quality_check"
                        ),
                        PromptNode(
                            id="data_stats",
                            prompt_template="Calculate descriptive statistics",
                            node_type="statistics"
                        )
                    ]
                ),
                PromptNode(
                    id="pattern_analysis",
                    prompt_template="Identify patterns and correlations",
                    node_type="analysis",
                    children=[
                        PromptNode(
                            id="correlation",
                            prompt_template="Compute correlation matrix",
                            node_type="correlation"
                        ),
                        PromptNode(
                            id="anomalies",
                            prompt_template="Detect anomalies and outliers",
                            node_type="anomaly_detection"
                        )
                    ]
                ),
                PromptNode(
                    id="insights_generation",
                    prompt_template="Generate actionable insights from the analysis",
                    node_type="insights"
                )
            ]
        )
        
        return root
    
    def create_code_generation_workflow(self) -> PromptNode:
        """Create a code generation workflow"""
        
        root = PromptNode(
            id="code_gen_root",
            prompt_template="Generate a complete implementation for the requested feature",
            node_type="root",
            max_depth=4,
            children=[
                PromptNode(
                    id="requirements_analysis",
                    prompt_template="Analyze requirements and identify components needed",
                    node_type="analysis"
                ),
                PromptNode(
                    id="architecture_design",
                    prompt_template="Design the architecture and interfaces",
                    node_type="design",
                    children=[
                        PromptNode(
                            id="api_design",
                            prompt_template="Design the API contracts",
                            node_type="api"
                        ),
                        PromptNode(
                            id="data_model",
                            prompt_template="Design the data model",
                            node_type="model"
                        )
                    ]
                ),
                PromptNode(
                    id="implementation",
                    prompt_template="Implement the components",
                    node_type="implementation",
                    children=[
                        PromptNode(
                            id="core_logic",
                            prompt_template="Implement core business logic",
                            node_type="logic"
                        ),
                        PromptNode(
                            id="error_handling",
                            prompt_template="Add error handling and validation",
                            node_type="error_handling"
                        ),
                        PromptNode(
                            id="tests",
                            prompt_template="Write comprehensive tests",
                            node_type="testing"
                        )
                    ]
                )
            ]
        )
        
        return root


class AdaptiveFolder(dspy.Module):
    """Adaptively chooses folding strategy based on task analysis"""
    
    def __init__(self):
        super().__init__()
        self.task_analyzer = dspy.ChainOfThought("task -> best_strategy")
        self.folders = {
            strategy: PromptFolder(strategy) 
            for strategy in FoldingStrategy
        }
        
    def analyze_task(self, task: str) -> FoldingStrategy:
        """Analyze task to determine best folding strategy"""
        
        task_features = {
            "has_steps": "step" in task.lower() or "first" in task.lower(),
            "has_conditions": "if" in task.lower() or "when" in task.lower(),
            "is_complex": len(task.split()) > 50,
            "has_parallel": "simultaneously" in task.lower() or "parallel" in task.lower(),
            "is_analytical": "analyze" in task.lower() or "investigate" in task.lower()
        }
        
        if task_features["has_steps"]:
            return FoldingStrategy.PIPELINE
        elif task_features["has_conditions"]:
            return FoldingStrategy.BRANCHING
        elif task_features["has_parallel"]:
            return FoldingStrategy.PARALLEL
        elif task_features["is_complex"]:
            return FoldingStrategy.RECURSIVE
        else:
            return FoldingStrategy.ADAPTIVE
    
    def fold_adaptive(self, task: str, max_depth: int = 3) -> Dict[str, Any]:
        """Adaptively fold based on task analysis"""
        
        strategy = self.analyze_task(task)
        folder = self.folders[strategy]
        
        result = folder.fold(task, max_depth)
        result["selected_strategy"] = strategy.value
        
        return result


def create_research_workflow_example():
    """Example of a research workflow using prompt folding"""
    
    workflow = WorkflowFolder()
    
    research_task = """
    Research the impact of artificial intelligence on employment in the healthcare sector.
    Include current statistics, future projections, opportunities, and challenges.
    """
    
    folder = PromptFolder(FoldingStrategy.RECURSIVE)
    result = folder.fold(research_task, max_depth=3)
    
    return result


if __name__ == "__main__":
    adaptive_folder = AdaptiveFolder()
    
    task = """
    Create a comprehensive marketing strategy for a new sustainable fashion brand.
    First, analyze the target market. Then, develop brand positioning.
    Finally, create a multi-channel campaign plan with specific tactics.
    """
    
    result = adaptive_folder.fold_adaptive(task)
    print(f"Selected Strategy: {result['selected_strategy']}")
    print(f"Execution Path: {result['execution_path']}")
    print(f"Final Result: {result['final_result']}")