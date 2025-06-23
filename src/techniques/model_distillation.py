"""
Model Distillation Pipeline

This module implements model distillation techniques to use large models
for prompt crafting, then deploy on smaller, cheaper models in production.
"""

import dspy
from typing import List, Dict, Optional, Any, Tuple, Callable
from pydantic import BaseModel, Field
from dataclasses import dataclass
import json
import time
from datetime import datetime
import numpy as np
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor


class ModelSize(str, Enum):
    LARGE = "large"  # GPT-4, Claude-3, etc.
    MEDIUM = "medium"  # GPT-3.5, Claude-2, etc.
    SMALL = "small"  # Smaller/faster models
    TINY = "tiny"  # Edge deployment models


class DistillationStrategy(str, Enum):
    DIRECT = "direct"  # Direct prompt transfer
    SYNTHETIC = "synthetic"  # Generate synthetic training data
    CHAIN = "chain"  # Chain of thought distillation
    SELECTIVE = "selective"  # Selective feature distillation
    ENSEMBLE = "ensemble"  # Ensemble distillation


@dataclass
class ModelProfile:
    """Profile of a model's capabilities"""
    name: str
    size: ModelSize
    context_window: int
    tokens_per_second: float
    cost_per_1k_tokens: float
    strengths: List[str]
    weaknesses: List[str]
    best_use_cases: List[str]


class DistillationConfig(BaseModel):
    """Configuration for distillation process"""
    teacher_model: str
    student_model: str
    strategy: DistillationStrategy
    num_examples: int = 100
    temperature: float = 0.7
    optimization_target: str = "accuracy"  # accuracy, speed, cost
    quality_threshold: float = 0.9
    max_iterations: int = 5


class DistillationResult(BaseModel):
    """Result of distillation process"""
    original_prompt: str
    distilled_prompt: str
    teacher_performance: Dict[str, float]
    student_performance: Dict[str, float]
    quality_retention: float
    cost_reduction: float
    speed_improvement: float
    synthetic_examples: Optional[List[Dict[str, str]]] = None


class TeacherSignature(dspy.Signature):
    """Teacher model generating high-quality outputs"""
    
    task = dspy.InputField(desc="The task to complete")
    context = dspy.InputField(desc="Additional context")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning")
    output = dspy.OutputField(desc="High-quality output")


class StudentSignature(dspy.Signature):
    """Student model learning from teacher"""
    
    task = dspy.InputField(desc="The task to complete")
    examples = dspy.InputField(desc="Examples from teacher model")
    output = dspy.OutputField(desc="Student's attempt at the task")


class PromptDistiller(dspy.Module):
    """Distills prompts from large to small models"""
    
    def __init__(self, teacher_model: str = "gpt-4", student_model: str = "gpt-3.5-turbo"):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher = dspy.ChainOfThought(TeacherSignature)
        self.student = dspy.Predict(StudentSignature)
        
    def generate_teacher_examples(self, base_prompt: str, 
                                num_examples: int = 10) -> List[Dict[str, str]]:
        """Generate high-quality examples using teacher model"""
        
        examples = []
        
        variations = [
            "simple case",
            "complex case",
            "edge case",
            "typical use case",
            "challenging scenario"
        ]
        
        for i in range(num_examples):
            variation = variations[i % len(variations)]
            task = f"{base_prompt} (Variation: {variation})"
            
            result = self.teacher(
                task=task,
                context=f"Generate a high-quality example for: {variation}"
            )
            
            examples.append({
                "input": task,
                "reasoning": result.reasoning,
                "output": result.output
            })
        
        return examples
    
    def distill_prompt(self, original_prompt: str, 
                      config: DistillationConfig) -> DistillationResult:
        """Distill a prompt from teacher to student model"""
        
        # Generate teacher examples
        teacher_examples = self.generate_teacher_examples(
            original_prompt, 
            config.num_examples
        )
        
        # Create distilled prompt for student
        distilled_prompt = self._create_distilled_prompt(
            original_prompt,
            teacher_examples,
            config.strategy
        )
        
        # Evaluate performance
        teacher_perf = self._evaluate_model_performance(
            self.teacher, 
            original_prompt,
            is_teacher=True
        )
        
        student_perf = self._evaluate_model_performance(
            self.student,
            distilled_prompt,
            is_teacher=False
        )
        
        # Calculate metrics
        quality_retention = student_perf["accuracy"] / teacher_perf["accuracy"]
        cost_reduction = 1 - (self._get_model_cost(self.student_model) / 
                            self._get_model_cost(self.teacher_model))
        speed_improvement = teacher_perf["avg_latency_ms"] / student_perf["avg_latency_ms"]
        
        return DistillationResult(
            original_prompt=original_prompt,
            distilled_prompt=distilled_prompt,
            teacher_performance=teacher_perf,
            student_performance=student_perf,
            quality_retention=quality_retention,
            cost_reduction=cost_reduction,
            speed_improvement=speed_improvement,
            synthetic_examples=teacher_examples
        )
    
    def _create_distilled_prompt(self, original: str, examples: List[Dict[str, str]], 
                               strategy: DistillationStrategy) -> str:
        """Create optimized prompt for student model"""
        
        if strategy == DistillationStrategy.DIRECT:
            return original
        
        elif strategy == DistillationStrategy.SYNTHETIC:
            # Include synthetic examples in prompt
            example_text = "\n\n".join([
                f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
                for i, ex in enumerate(examples[:3])
            ])
            
            return f"""{original}

Here are some examples:

{example_text}

Now complete the task following the pattern shown in the examples."""
        
        elif strategy == DistillationStrategy.CHAIN:
            # Distill chain-of-thought into simpler steps
            steps = self._extract_reasoning_steps(examples)
            
            return f"""{original}

Follow these steps:
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(steps))}"""
        
        elif strategy == DistillationStrategy.SELECTIVE:
            # Focus on most important features
            key_features = self._identify_key_features(examples)
            
            return f"""{original}

Focus on these key aspects:
{chr(10).join(f'- {feature}' for feature in key_features)}"""
        
        else:  # ENSEMBLE
            return self._create_ensemble_prompt(original, examples)
    
    def _extract_reasoning_steps(self, examples: List[Dict[str, str]]) -> List[str]:
        """Extract common reasoning steps from examples"""
        
        # Simplified extraction - in practice, use NLP techniques
        common_steps = [
            "Understand the requirements",
            "Identify key components",
            "Apply the appropriate method",
            "Verify the result"
        ]
        
        return common_steps
    
    def _identify_key_features(self, examples: List[Dict[str, str]]) -> List[str]:
        """Identify key features to focus on"""
        
        # Simplified - in practice, analyze examples for patterns
        return [
            "Accuracy in core functionality",
            "Proper error handling",
            "Clear output structure"
        ]
    
    def _create_ensemble_prompt(self, original: str, 
                              examples: List[Dict[str, str]]) -> str:
        """Create ensemble prompt combining strategies"""
        
        return f"""{original}

Approach this task by:
1. Following the examples provided
2. Using step-by-step reasoning
3. Focusing on accuracy and clarity

Remember: Quality over complexity."""
    
    def _evaluate_model_performance(self, model: dspy.Module, prompt: str,
                                  is_teacher: bool) -> Dict[str, float]:
        """Evaluate model performance metrics"""
        
        # Simplified evaluation - in practice, use comprehensive test suite
        test_inputs = [
            "Simple test case",
            "Complex test case",
            "Edge case test"
        ]
        
        latencies = []
        outputs = []
        
        for test_input in test_inputs:
            start_time = time.time()
            
            if is_teacher:
                output = model(task=test_input, context="")
            else:
                output = model(task=test_input, examples="")
            
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            outputs.append(output)
        
        return {
            "accuracy": 0.95 if is_teacher else 0.88,  # Simulated
            "avg_latency_ms": np.mean(latencies),
            "consistency": 0.92 if is_teacher else 0.85,
            "token_efficiency": 0.8
        }
    
    def _get_model_cost(self, model_name: str) -> float:
        """Get cost per 1k tokens for a model"""
        
        costs = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "claude-3": 0.025,
            "claude-2": 0.008
        }
        
        return costs.get(model_name, 0.01)


class AdaptiveDistillation(dspy.Module):
    """Adaptive distillation that improves over iterations"""
    
    def __init__(self):
        super().__init__()
        self.distiller = PromptDistiller()
        self.iteration_history: List[DistillationResult] = []
        
    def iterative_distillation(self, prompt: str, config: DistillationConfig) -> DistillationResult:
        """Iteratively improve distillation"""
        
        best_result = None
        best_score = 0.0
        
        for iteration in range(config.max_iterations):
            # Adjust strategy based on previous results
            if iteration > 0:
                config = self._adjust_config(config, self.iteration_history)
            
            result = self.distiller.distill_prompt(prompt, config)
            self.iteration_history.append(result)
            
            # Calculate composite score
            score = self._calculate_composite_score(result, config.optimization_target)
            
            if score > best_score:
                best_score = score
                best_result = result
            
            # Early stopping if quality threshold met
            if result.quality_retention >= config.quality_threshold:
                break
        
        return best_result
    
    def _adjust_config(self, config: DistillationConfig, 
                      history: List[DistillationResult]) -> DistillationConfig:
        """Adjust configuration based on history"""
        
        last_result = history[-1]
        
        # If quality is too low, try different strategy
        if last_result.quality_retention < 0.8:
            strategies = list(DistillationStrategy)
            current_idx = strategies.index(config.strategy)
            config.strategy = strategies[(current_idx + 1) % len(strategies)]
        
        # Adjust number of examples
        if last_result.quality_retention < 0.9:
            config.num_examples = min(config.num_examples + 20, 200)
        
        return config
    
    def _calculate_composite_score(self, result: DistillationResult, 
                                 target: str) -> float:
        """Calculate composite score based on optimization target"""
        
        weights = {
            "accuracy": {"quality": 0.8, "cost": 0.1, "speed": 0.1},
            "speed": {"quality": 0.3, "cost": 0.2, "speed": 0.5},
            "cost": {"quality": 0.4, "cost": 0.5, "speed": 0.1}
        }
        
        w = weights.get(target, weights["accuracy"])
        
        score = (
            w["quality"] * result.quality_retention +
            w["cost"] * result.cost_reduction +
            w["speed"] * min(result.speed_improvement / 5, 1.0)  # Cap speed score
        )
        
        return score


class ProductionOptimizer:
    """Optimizes prompts for production deployment"""
    
    def __init__(self):
        self.model_profiles = self._initialize_model_profiles()
        
    def _initialize_model_profiles(self) -> Dict[str, ModelProfile]:
        """Initialize profiles for different models"""
        
        return {
            "gpt-4": ModelProfile(
                name="gpt-4",
                size=ModelSize.LARGE,
                context_window=8192,
                tokens_per_second=20,
                cost_per_1k_tokens=0.03,
                strengths=["reasoning", "creativity", "accuracy"],
                weaknesses=["cost", "latency"],
                best_use_cases=["complex analysis", "creative tasks"]
            ),
            "gpt-3.5-turbo": ModelProfile(
                name="gpt-3.5-turbo",
                size=ModelSize.MEDIUM,
                context_window=4096,
                tokens_per_second=50,
                cost_per_1k_tokens=0.002,
                strengths=["balance", "speed", "cost"],
                weaknesses=["complex reasoning"],
                best_use_cases=["general tasks", "high volume"]
            )
        }
    
    def optimize_for_production(self, prompt: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize prompt for production requirements"""
        
        # Analyze requirements
        max_latency_ms = requirements.get("max_latency_ms", 1000)
        max_cost_per_request = requirements.get("max_cost_per_request", 0.01)
        min_accuracy = requirements.get("min_accuracy", 0.85)
        expected_volume = requirements.get("daily_volume", 10000)
        
        # Select optimal model
        optimal_model = self._select_optimal_model(
            max_latency_ms,
            max_cost_per_request,
            min_accuracy
        )
        
        # Create deployment configuration
        deployment_config = {
            "model": optimal_model.name,
            "optimizations": [],
            "caching_strategy": "aggressive" if expected_volume > 50000 else "moderate",
            "fallback_model": "gpt-3.5-turbo",
            "monitoring": {
                "latency_threshold_ms": max_latency_ms * 0.8,
                "error_rate_threshold": 0.01,
                "quality_sampling_rate": 0.05
            }
        }
        
        # Add specific optimizations
        if max_latency_ms < 500:
            deployment_config["optimizations"].append("response_streaming")
            deployment_config["optimizations"].append("prompt_compression")
        
        if expected_volume > 100000:
            deployment_config["optimizations"].append("request_batching")
            deployment_config["optimizations"].append("result_caching")
        
        return deployment_config
    
    def _select_optimal_model(self, max_latency: float, max_cost: float, 
                            min_accuracy: float) -> ModelProfile:
        """Select optimal model based on constraints"""
        
        # Simplified selection - in practice, use more sophisticated algorithm
        for model in self.model_profiles.values():
            latency = 1000 / model.tokens_per_second  # Rough estimate
            cost = model.cost_per_1k_tokens / 1000  # Per token cost
            
            if latency <= max_latency and cost <= max_cost:
                return model
        
        # Default to fastest small model
        return self.model_profiles["gpt-3.5-turbo"]


class DistillationPipeline:
    """Complete pipeline for prompt distillation"""
    
    def __init__(self):
        self.distiller = AdaptiveDistillation()
        self.optimizer = ProductionOptimizer()
        
    async def distill_and_deploy(self, prompt: str, 
                                production_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Complete distillation and deployment pipeline"""
        
        # Step 1: Create distillation config
        config = DistillationConfig(
            teacher_model="gpt-4",
            student_model="gpt-3.5-turbo",
            strategy=DistillationStrategy.SYNTHETIC,
            optimization_target="accuracy" if production_requirements.get("min_accuracy", 0) > 0.9 else "cost"
        )
        
        # Step 2: Perform distillation
        distillation_result = self.distiller.iterative_distillation(prompt, config)
        
        # Step 3: Optimize for production
        deployment_config = self.optimizer.optimize_for_production(
            distillation_result.distilled_prompt,
            production_requirements
        )
        
        # Step 4: Create deployment package
        deployment_package = {
            "prompt": distillation_result.distilled_prompt,
            "model_config": deployment_config,
            "performance_metrics": {
                "expected_accuracy": distillation_result.student_performance["accuracy"],
                "expected_latency_ms": distillation_result.student_performance["avg_latency_ms"],
                "cost_per_request": self.optimizer.model_profiles[deployment_config["model"]].cost_per_1k_tokens / 1000
            },
            "monitoring": {
                "quality_benchmarks": distillation_result.teacher_performance,
                "degradation_threshold": 0.1
            },
            "created_at": datetime.now().isoformat()
        }
        
        return deployment_package


def create_distillation_example():
    """Example of distilling a complex prompt"""
    
    pipeline = DistillationPipeline()
    
    complex_prompt = """
    Analyze the provided code for security vulnerabilities. Consider:
    1. SQL injection risks
    2. Cross-site scripting (XSS)
    3. Authentication bypasses
    4. Information disclosure
    5. Resource exhaustion
    
    Provide a detailed report with severity levels and remediation steps.
    """
    
    production_requirements = {
        "max_latency_ms": 500,
        "max_cost_per_request": 0.001,
        "min_accuracy": 0.88,
        "daily_volume": 50000
    }
    
    # Run async pipeline
    async def run_example():
        result = await pipeline.distill_and_deploy(complex_prompt, production_requirements)
        return result
    
    # For synchronous execution
    import asyncio
    return asyncio.run(run_example())


if __name__ == "__main__":
    # Test distillation
    distiller = PromptDistiller()
    
    original_prompt = "Generate a comprehensive business plan for a startup"
    
    config = DistillationConfig(
        teacher_model="gpt-4",
        student_model="gpt-3.5-turbo",
        strategy=DistillationStrategy.SYNTHETIC,
        num_examples=5
    )
    
    result = distiller.distill_prompt(original_prompt, config)
    
    print("Distillation Complete!")
    print(f"Quality Retention: {result.quality_retention:.2%}")
    print(f"Cost Reduction: {result.cost_reduction:.2%}")
    print(f"Speed Improvement: {result.speed_improvement:.1f}x")
    print(f"\nDistilled Prompt:\n{result.distilled_prompt}")