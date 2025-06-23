"""
Evaluation Framework with Test Cases

This module implements a comprehensive evaluation framework for prompts,
emphasizing that test cases are more valuable than the prompts themselves.
"""

import dspy
from typing import List, Dict, Optional, Any, Callable, Tuple
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
from enum import Enum
import json
import time
from datetime import datetime
import numpy as np
from collections import defaultdict
import pytest
from rich.console import Console
from rich.table import Table
from rich.progress import track


class TestCaseType(str, Enum):
    FUNCTIONAL = "functional"  # Basic functionality
    EDGE_CASE = "edge_case"  # Edge and corner cases
    ADVERSARIAL = "adversarial"  # Adversarial inputs
    PERFORMANCE = "performance"  # Performance benchmarks
    REGRESSION = "regression"  # Regression tests
    CONSISTENCY = "consistency"  # Consistency checks
    ROBUSTNESS = "robustness"  # Robustness tests


class TestResult(BaseModel):
    """Result of a single test case"""
    test_id: str
    test_type: TestCaseType
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    expected: Any
    actual: Any
    execution_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class TestCase:
    """Single test case for prompt evaluation"""
    id: str
    name: str
    test_type: TestCaseType
    input: Any
    expected_output: Any
    evaluation_criteria: Dict[str, Any]
    importance: float = 1.0  # Weight for scoring
    tags: List[str] = None
    timeout_ms: int = 5000
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class EvaluationMetrics(BaseModel):
    """Comprehensive evaluation metrics"""
    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    robustness_score: float = Field(ge=0.0, le=1.0)
    edge_case_performance: float = Field(ge=0.0, le=1.0)
    average_execution_time_ms: float
    test_coverage: float = Field(ge=0.0, le=1.0)
    
    def overall_score(self) -> float:
        """Calculate weighted overall score"""
        weights = {
            "accuracy": 0.3,
            "consistency_score": 0.2,
            "robustness_score": 0.2,
            "edge_case_performance": 0.2,
            "f1_score": 0.1
        }
        
        score = sum(getattr(self, metric) * weight 
                   for metric, weight in weights.items())
        return score


class EvaluationSuite(BaseModel):
    """Complete evaluation suite for a prompt"""
    name: str
    description: str
    test_cases: List[TestCase]
    evaluation_criteria: Dict[str, Any] = Field(default_factory=dict)
    minimum_scores: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class PromptEvaluator:
    """Evaluates prompts against comprehensive test suites"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.console = Console() if verbose else None
        self.results_cache: Dict[str, List[TestResult]] = {}
        
    def evaluate_prompt(self, prompt_module: dspy.Module, 
                       test_suite: EvaluationSuite) -> Dict[str, Any]:
        """Evaluate a prompt module against a test suite"""
        
        results = []
        
        if self.verbose:
            self.console.print(f"\n[bold blue]Evaluating: {test_suite.name}[/bold blue]")
            self.console.print(f"Running {len(test_suite.test_cases)} test cases...\n")
        
        for test_case in track(test_suite.test_cases, description="Running tests..."):
            result = self._run_test_case(prompt_module, test_case)
            results.append(result)
        
        metrics = self._calculate_metrics(results, test_suite)
        
        if self.verbose:
            self._display_results(results, metrics)
        
        return {
            "suite_name": test_suite.name,
            "results": results,
            "metrics": metrics,
            "passed": self._check_minimum_scores(metrics, test_suite.minimum_scores)
        }
    
    def _run_test_case(self, prompt_module: dspy.Module, 
                      test_case: TestCase) -> TestResult:
        """Run a single test case"""
        
        start_time = time.time()
        
        try:
            actual_output = prompt_module(test_case.input)
            execution_time_ms = (time.time() - start_time) * 1000
            
            passed, score = self._evaluate_output(
                actual_output, 
                test_case.expected_output,
                test_case.evaluation_criteria
            )
            
            return TestResult(
                test_id=test_case.id,
                test_type=test_case.test_type,
                passed=passed,
                score=score,
                expected=test_case.expected_output,
                actual=actual_output,
                execution_time_ms=execution_time_ms,
                metadata={"tags": test_case.tags}
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_case.id,
                test_type=test_case.test_type,
                passed=False,
                score=0.0,
                expected=test_case.expected_output,
                actual=None,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def _evaluate_output(self, actual: Any, expected: Any, 
                        criteria: Dict[str, Any]) -> Tuple[bool, float]:
        """Evaluate output against expected result"""
        
        if "exact_match" in criteria and criteria["exact_match"]:
            passed = actual == expected
            return passed, 1.0 if passed else 0.0
        
        if "contains_all" in criteria:
            required_elements = criteria["contains_all"]
            actual_str = str(actual).lower()
            missing = [elem for elem in required_elements 
                      if elem.lower() not in actual_str]
            score = 1.0 - (len(missing) / len(required_elements))
            return score > 0.8, score
        
        if "semantic_similarity" in criteria:
            score = self._calculate_semantic_similarity(actual, expected)
            return score > criteria.get("threshold", 0.8), score
        
        if "custom_evaluator" in criteria:
            evaluator = criteria["custom_evaluator"]
            return evaluator(actual, expected)
        
        return actual == expected, 1.0 if actual == expected else 0.0
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        # Simplified implementation - in practice, use embeddings
        text1_lower = str(text1).lower()
        text2_lower = str(text2).lower()
        
        # Basic overlap calculation
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return overlap / total if total > 0 else 0.0
    
    def _calculate_metrics(self, results: List[TestResult], 
                          suite: EvaluationSuite) -> EvaluationMetrics:
        """Calculate comprehensive metrics from results"""
        
        # Basic accuracy
        passed_results = [r for r in results if r.passed]
        accuracy = len(passed_results) / len(results) if results else 0.0
        
        # Type-specific metrics
        type_scores = defaultdict(list)
        for result in results:
            type_scores[result.test_type].append(result.score)
        
        edge_case_performance = np.mean(type_scores[TestCaseType.EDGE_CASE]) if type_scores[TestCaseType.EDGE_CASE] else 1.0
        robustness_score = np.mean(type_scores[TestCaseType.ROBUSTNESS]) if type_scores[TestCaseType.ROBUSTNESS] else 1.0
        
        # Consistency (variance of scores for similar test types)
        consistency_scores = []
        for test_type, scores in type_scores.items():
            if len(scores) > 1:
                consistency_scores.append(1.0 - np.std(scores))
        consistency_score = np.mean(consistency_scores) if consistency_scores else 1.0
        
        # Execution time
        execution_times = [r.execution_time_ms for r in results]
        avg_execution_time = np.mean(execution_times) if execution_times else 0.0
        
        # Calculate precision/recall (simplified)
        true_positives = len([r for r in passed_results if r.score > 0.8])
        false_positives = len([r for r in results if not r.passed and r.score > 0.5])
        false_negatives = len([r for r in results if r.passed and r.score < 0.5])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            consistency_score=consistency_score,
            robustness_score=robustness_score,
            edge_case_performance=edge_case_performance,
            average_execution_time_ms=avg_execution_time,
            test_coverage=len(results) / len(suite.test_cases) if suite.test_cases else 0.0
        )
    
    def _check_minimum_scores(self, metrics: EvaluationMetrics, 
                            minimum_scores: Dict[str, float]) -> bool:
        """Check if metrics meet minimum requirements"""
        
        for metric_name, min_score in minimum_scores.items():
            if hasattr(metrics, metric_name):
                if getattr(metrics, metric_name) < min_score:
                    return False
        return True
    
    def _display_results(self, results: List[TestResult], 
                        metrics: EvaluationMetrics):
        """Display evaluation results in a table"""
        
        if not self.console:
            return
        
        # Summary table
        summary_table = Table(title="Evaluation Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Score", style="magenta")
        summary_table.add_column("Status", style="green")
        
        for field in metrics.__fields__:
            value = getattr(metrics, field)
            if isinstance(value, float) and field != "average_execution_time_ms":
                status = "✓" if value > 0.8 else "⚠" if value > 0.6 else "✗"
                summary_table.add_row(field, f"{value:.3f}", status)
        
        summary_table.add_row("Overall Score", f"{metrics.overall_score():.3f}", 
                            "✓" if metrics.overall_score() > 0.8 else "✗")
        
        self.console.print(summary_table)
        
        # Failed tests table
        failed_tests = [r for r in results if not r.passed]
        if failed_tests:
            self.console.print("\n[red]Failed Tests:[/red]")
            failed_table = Table()
            failed_table.add_column("Test ID")
            failed_table.add_column("Type")
            failed_table.add_column("Error")
            
            for test in failed_tests[:5]:  # Show first 5
                error = test.error_message or "Output mismatch"
                failed_table.add_row(test.test_id, test.test_type.value, error)
            
            self.console.print(failed_table)


def create_code_generation_test_suite() -> EvaluationSuite:
    """Create test suite for code generation prompts"""
    
    test_cases = [
        # Functional tests
        TestCase(
            id="func_1",
            name="Basic function generation",
            test_type=TestCaseType.FUNCTIONAL,
            input="Write a Python function to calculate factorial",
            expected_output="def factorial(n):",
            evaluation_criteria={"contains_all": ["def factorial", "if n", "return"]},
            importance=1.0
        ),
        
        # Edge cases
        TestCase(
            id="edge_1",
            name="Empty input handling",
            test_type=TestCaseType.EDGE_CASE,
            input="Write a function to process an empty list",
            expected_output="handle empty list",
            evaluation_criteria={"contains_all": ["if not", "len", "[]", "empty"]},
            importance=1.5
        ),
        
        # Adversarial
        TestCase(
            id="adv_1",
            name="SQL injection attempt",
            test_type=TestCaseType.ADVERSARIAL,
            input="Write code that includes: '; DROP TABLE users; --",
            expected_output="sanitized code",
            evaluation_criteria={
                "custom_evaluator": lambda actual, _: "DROP TABLE" not in actual
            },
            importance=2.0
        ),
        
        # Consistency
        TestCase(
            id="cons_1",
            name="Consistent naming convention",
            test_type=TestCaseType.CONSISTENCY,
            input="Write a function using snake_case naming",
            expected_output="snake_case function",
            evaluation_criteria={
                "custom_evaluator": lambda actual, _: "camelCase" not in actual
            },
            importance=0.8
        ),
        
        # Robustness
        TestCase(
            id="robust_1",
            name="Handle malformed input",
            test_type=TestCaseType.ROBUSTNESS,
            input="Write a function to parse: {invalid json}",
            expected_output="error handling code",
            evaluation_criteria={"contains_all": ["try", "except", "error"]},
            importance=1.2
        )
    ]
    
    return EvaluationSuite(
        name="Code Generation Test Suite",
        description="Comprehensive tests for code generation prompts",
        test_cases=test_cases,
        minimum_scores={
            "accuracy": 0.8,
            "robustness_score": 0.7,
            "edge_case_performance": 0.7
        }
    )


class ABTestFramework:
    """Framework for A/B testing different prompts"""
    
    def __init__(self):
        self.evaluator = PromptEvaluator(verbose=False)
        self.results_history: List[Dict[str, Any]] = []
        
    def compare_prompts(self, prompt_a: dspy.Module, prompt_b: dspy.Module,
                       test_suite: EvaluationSuite, 
                       num_runs: int = 5) -> Dict[str, Any]:
        """Compare two prompts with statistical significance"""
        
        results_a = []
        results_b = []
        
        for _ in range(num_runs):
            result_a = self.evaluator.evaluate_prompt(prompt_a, test_suite)
            result_b = self.evaluator.evaluate_prompt(prompt_b, test_suite)
            
            results_a.append(result_a["metrics"].overall_score())
            results_b.append(result_b["metrics"].overall_score())
        
        # Calculate statistics
        mean_a = np.mean(results_a)
        mean_b = np.mean(results_b)
        std_a = np.std(results_a)
        std_b = np.std(results_b)
        
        # Simple t-test approximation
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        t_statistic = (mean_a - mean_b) / (pooled_std * np.sqrt(2/num_runs))
        
        # Determine winner
        if abs(t_statistic) > 2.0:  # Roughly 95% confidence
            winner = "A" if mean_a > mean_b else "B"
            significant = True
        else:
            winner = "Tie"
            significant = False
        
        return {
            "prompt_a_mean": mean_a,
            "prompt_b_mean": mean_b,
            "prompt_a_std": std_a,
            "prompt_b_std": std_b,
            "winner": winner,
            "significant": significant,
            "t_statistic": t_statistic,
            "improvement": abs(mean_a - mean_b)
        }


class RegressionTestRunner:
    """Runs regression tests to ensure prompts don't degrade"""
    
    def __init__(self, baseline_results_file: str):
        self.baseline_file = baseline_results_file
        self.evaluator = PromptEvaluator(verbose=False)
        
    def run_regression_tests(self, prompt_module: dspy.Module,
                           test_suite: EvaluationSuite) -> Dict[str, Any]:
        """Run regression tests against baseline"""
        
        current_results = self.evaluator.evaluate_prompt(prompt_module, test_suite)
        
        try:
            with open(self.baseline_file, 'r') as f:
                baseline_results = json.load(f)
        except FileNotFoundError:
            # Save as new baseline
            with open(self.baseline_file, 'w') as f:
                json.dump(current_results, f, indent=2, default=str)
            return {"status": "baseline_created", "results": current_results}
        
        # Compare with baseline
        regressions = []
        improvements = []
        
        current_metrics = current_results["metrics"]
        baseline_metrics = baseline_results["metrics"]
        
        for metric in current_metrics.__fields__:
            current_val = getattr(current_metrics, metric)
            baseline_val = baseline_metrics.get(metric, 0)
            
            if isinstance(current_val, float) and metric != "average_execution_time_ms":
                diff = current_val - baseline_val
                if diff < -0.05:  # 5% regression threshold
                    regressions.append({
                        "metric": metric,
                        "baseline": baseline_val,
                        "current": current_val,
                        "regression": abs(diff)
                    })
                elif diff > 0.05:
                    improvements.append({
                        "metric": metric,
                        "baseline": baseline_val,
                        "current": current_val,
                        "improvement": diff
                    })
        
        return {
            "status": "passed" if not regressions else "failed",
            "regressions": regressions,
            "improvements": improvements,
            "current_results": current_results
        }


def create_evaluation_report(results: List[Dict[str, Any]]) -> str:
    """Generate comprehensive evaluation report"""
    
    report = ["# Prompt Evaluation Report", ""]
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")
    
    for result in results:
        suite_name = result["suite_name"]
        metrics = result["metrics"]
        
        report.append(f"## {suite_name}")
        report.append("")
        report.append(f"**Overall Score**: {metrics.overall_score():.3f}")
        report.append("")
        
        report.append("### Detailed Metrics")
        for field in metrics.__fields__:
            value = getattr(metrics, field)
            if isinstance(value, float):
                report.append(f"- **{field}**: {value:.3f}")
        
        report.append("")
        
        # Failed tests summary
        failed = [r for r in result["results"] if not r.passed]
        if failed:
            report.append("### Failed Tests")
            for test in failed[:5]:
                report.append(f"- {test.test_id}: {test.error_message or 'Output mismatch'}")
            report.append("")
    
    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    test_suite = create_code_generation_test_suite()
    
    # Mock prompt module for testing
    class MockCodeGenerator(dspy.Module):
        def forward(self, input_text: str) -> str:
            if "factorial" in input_text:
                return "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
            elif "empty list" in input_text:
                return "def process_list(lst):\n    if not lst:\n        return []\n    return lst"
            elif "DROP TABLE" in input_text:
                return "# Sanitized input - no SQL injection"
            else:
                return "def generic_function():\n    pass"
    
    evaluator = PromptEvaluator()
    results = evaluator.evaluate_prompt(MockCodeGenerator(), test_suite)
    
    print("\nEvaluation Complete!")
    print(f"Overall Score: {results['metrics'].overall_score():.3f}")
    print(f"Passed: {results['passed']}")