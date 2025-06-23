"""
Thinking Traces and Debug Logging

This module implements thinking traces that expose the model's internal logic
and debug information for troubleshooting and iteration.
"""

import dspy
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
from enum import Enum
from loguru import logger
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.syntax import Syntax


class ThoughtType(str, Enum):
    ANALYSIS = "analysis"
    HYPOTHESIS = "hypothesis"
    VERIFICATION = "verification"
    DECISION = "decision"
    QUESTION = "question"
    OBSERVATION = "observation"
    CONCLUSION = "conclusion"
    ERROR = "error"


@dataclass
class ThoughtNode:
    """Represents a single thought in the reasoning chain"""
    id: str
    thought_type: ThoughtType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    parent_id: Optional[str] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'ThoughtNode'):
        """Add a child thought"""
        child.parent_id = self.id
        self.children.append(child)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.thought_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "parent_id": self.parent_id,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata
        }


class ThinkingTrace(BaseModel):
    """Complete thinking trace for a task"""
    task: str
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    root_thoughts: List[ThoughtNode] = Field(default_factory=list)
    decision_points: List[Dict[str, Any]] = Field(default_factory=list)
    errors_encountered: List[Dict[str, Any]] = Field(default_factory=list)
    final_output: Optional[str] = None
    total_thoughts: int = 0
    
    class Config:
        arbitrary_types_allowed = True


class DebugInfo(BaseModel):
    """Debug information for a reasoning step"""
    step_number: int
    step_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    duration_ms: float
    memory_usage: Optional[Dict[str, int]] = None
    token_counts: Optional[Dict[str, int]] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ThinkingTraceSignature(dspy.Signature):
    """Generate response with explicit thinking trace"""
    
    task = dspy.InputField(desc="The task to complete")
    trace_instructions = dspy.InputField(desc="Instructions for generating thinking trace")
    thinking_trace = dspy.OutputField(desc="Step-by-step thinking process")
    final_answer = dspy.OutputField(desc="Final answer based on thinking")


class DebuggedSignature(dspy.Signature):
    """Execute with debug information"""
    
    task = dspy.InputField(desc="The task to execute")
    debug_level = dspy.InputField(desc="Level of debug detail required")
    result = dspy.OutputField(desc="Task result")
    debug_info = dspy.OutputField(desc="Debug information in JSON format")


class ThinkingTracer(dspy.Module):
    """Generates detailed thinking traces for reasoning tasks"""
    
    def __init__(self, verbose: bool = True):
        super().__init__()
        self.tracer = dspy.ChainOfThought(ThinkingTraceSignature)
        self.verbose = verbose
        self.console = Console() if verbose else None
        self.current_trace: Optional[ThinkingTrace] = None
        
    def create_trace_instructions(self) -> str:
        """Create instructions for thinking trace generation"""
        
        return """
Generate a detailed thinking trace following this format:

1. UNDERSTANDING THE PROBLEM
   - What exactly is being asked?
   - What are the key components?
   - What constraints exist?

2. BREAKING DOWN THE TASK
   - What sub-problems need to be solved?
   - What is the logical sequence?
   - What dependencies exist?

3. EXPLORING APPROACHES
   - What are possible solutions?
   - What are pros/cons of each?
   - Which approach seems best?

4. STEP-BY-STEP REASONING
   - Work through the chosen approach
   - Show intermediate results
   - Verify each step

5. CHECKING THE WORK
   - Does the answer make sense?
   - Are there edge cases?
   - Could anything be improved?

6. FINAL ANSWER
   - State the conclusion clearly
   - Summarize key insights

Use these markers in your trace:
- [THOUGHT]: A reasoning step
- [QUESTION]: Something to investigate
- [HYPOTHESIS]: A tentative conclusion
- [VERIFICATION]: Checking a result
- [DECISION]: Choosing between options
- [ERROR]: Mistake found and corrected
- [INSIGHT]: Key realization
"""
        
    def parse_thinking_trace(self, trace_text: str) -> List[ThoughtNode]:
        """Parse thinking trace text into thought nodes"""
        
        thoughts = []
        lines = trace_text.split('\n')
        
        current_id = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            thought_type = ThoughtType.OBSERVATION
            confidence = 1.0
            
            if line.startswith('[THOUGHT]'):
                thought_type = ThoughtType.ANALYSIS
                line = line[9:].strip()
            elif line.startswith('[QUESTION]'):
                thought_type = ThoughtType.QUESTION
                line = line[10:].strip()
            elif line.startswith('[HYPOTHESIS]'):
                thought_type = ThoughtType.HYPOTHESIS
                confidence = 0.7
                line = line[12:].strip()
            elif line.startswith('[VERIFICATION]'):
                thought_type = ThoughtType.VERIFICATION
                line = line[14:].strip()
            elif line.startswith('[DECISION]'):
                thought_type = ThoughtType.DECISION
                line = line[10:].strip()
            elif line.startswith('[ERROR]'):
                thought_type = ThoughtType.ERROR
                line = line[7:].strip()
            elif line.startswith('[INSIGHT]'):
                thought_type = ThoughtType.CONCLUSION
                confidence = 0.9
                line = line[9:].strip()
            
            if line:
                thought = ThoughtNode(
                    id=f"thought_{current_id}",
                    thought_type=thought_type,
                    content=line,
                    confidence=confidence
                )
                thoughts.append(thought)
                current_id += 1
                
        return thoughts
    
    def visualize_trace(self, trace: ThinkingTrace):
        """Visualize thinking trace in console"""
        
        if not self.console:
            return
            
        tree = Tree("🧠 Thinking Trace")
        
        for thought in trace.root_thoughts:
            self._add_thought_to_tree(tree, thought)
        
        self.console.print(Panel(tree, title=f"Task: {trace.task}"))
        
        if trace.decision_points:
            self.console.print("\n📊 Decision Points:")
            for dp in trace.decision_points:
                self.console.print(f"  • {dp['description']}: {dp['choice']}")
        
        if trace.errors_encountered:
            self.console.print("\n⚠️  Errors Encountered:")
            for error in trace.errors_encountered:
                self.console.print(f"  • {error['description']}")
    
    def _add_thought_to_tree(self, tree: Tree, thought: ThoughtNode):
        """Recursively add thought nodes to tree"""
        
        icon_map = {
            ThoughtType.ANALYSIS: "🔍",
            ThoughtType.HYPOTHESIS: "💡",
            ThoughtType.VERIFICATION: "✓",
            ThoughtType.DECISION: "🎯",
            ThoughtType.QUESTION: "❓",
            ThoughtType.OBSERVATION: "👁",
            ThoughtType.CONCLUSION: "📌",
            ThoughtType.ERROR: "❌"
        }
        
        icon = icon_map.get(thought.thought_type, "•")
        confidence_indicator = "⚡" if thought.confidence > 0.8 else "?" if thought.confidence < 0.5 else ""
        
        branch = tree.add(f"{icon} {thought.content} {confidence_indicator}")
        
        for child in thought.children:
            self._add_thought_to_tree(branch, child)
    
    def forward(self, task: str) -> Dict[str, Any]:
        """Execute task with thinking trace"""
        
        trace = ThinkingTrace(task=task)
        self.current_trace = trace
        
        instructions = self.create_trace_instructions()
        
        result = self.tracer(
            task=task,
            trace_instructions=instructions
        )
        
        thoughts = self.parse_thinking_trace(result.thinking_trace)
        trace.root_thoughts = thoughts
        trace.total_thoughts = len(thoughts)
        trace.final_output = result.final_answer
        trace.end_time = datetime.now()
        
        if self.verbose:
            self.visualize_trace(trace)
        
        return {
            "answer": result.final_answer,
            "trace": trace,
            "thinking_steps": result.thinking_trace
        }


class DebugLogger(dspy.Module):
    """Provides detailed debug logging for prompt execution"""
    
    def __init__(self, log_file: Optional[str] = None):
        super().__init__()
        self.debugger = dspy.ChainOfThought(DebuggedSignature)
        self.log_file = log_file
        self.step_counter = 0
        
        if log_file:
            logger.add(log_file, rotation="10 MB")
        
    def log_step(self, step_name: str, inputs: Dict[str, Any], 
                 outputs: Dict[str, Any], duration_ms: float):
        """Log a single execution step"""
        
        self.step_counter += 1
        
        debug_info = DebugInfo(
            step_number=self.step_counter,
            step_name=step_name,
            inputs=inputs,
            outputs=outputs,
            duration_ms=duration_ms
        )
        
        logger.debug(f"Step {self.step_counter}: {step_name}")
        logger.debug(f"Inputs: {json.dumps(inputs, indent=2)}")
        logger.debug(f"Outputs: {json.dumps(outputs, indent=2)}")
        logger.debug(f"Duration: {duration_ms}ms")
        
        return debug_info
    
    def forward(self, task: str, debug_level: str = "detailed") -> Dict[str, Any]:
        """Execute with debug logging"""
        
        start_time = time.time()
        
        result = self.debugger(
            task=task,
            debug_level=debug_level
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        debug_info = self.log_step(
            "main_execution",
            {"task": task, "debug_level": debug_level},
            {"result": result.result},
            duration_ms
        )
        
        try:
            parsed_debug = json.loads(result.debug_info)
        except:
            parsed_debug = {"raw": result.debug_info}
        
        return {
            "result": result.result,
            "debug_info": debug_info,
            "model_debug": parsed_debug
        }


class ReasoningDebugger(dspy.Module):
    """Advanced debugger for complex reasoning chains"""
    
    def __init__(self):
        super().__init__()
        self.tracer = ThinkingTracer(verbose=True)
        self.checkpoints: List[Dict[str, Any]] = []
        
    def checkpoint(self, name: str, state: Dict[str, Any]):
        """Create a debugging checkpoint"""
        
        checkpoint = {
            "name": name,
            "timestamp": datetime.now(),
            "state": state,
            "stack_depth": len(self.checkpoints)
        }
        
        self.checkpoints.append(checkpoint)
        logger.info(f"Checkpoint: {name}")
        
    def get_reasoning_path(self) -> List[str]:
        """Get the path of reasoning steps"""
        
        return [cp["name"] for cp in self.checkpoints]
    
    def replay_from_checkpoint(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """Replay reasoning from a specific checkpoint"""
        
        for i, cp in enumerate(self.checkpoints):
            if cp["name"] == checkpoint_name:
                return {
                    "checkpoint": cp,
                    "remaining_path": self.get_reasoning_path()[i+1:]
                }
        
        return None
    
    def analyze_reasoning_errors(self, trace: ThinkingTrace) -> List[Dict[str, Any]]:
        """Analyze errors in reasoning trace"""
        
        errors = []
        
        for thought in trace.root_thoughts:
            if thought.thought_type == ThoughtType.ERROR:
                errors.append({
                    "thought_id": thought.id,
                    "content": thought.content,
                    "timestamp": thought.timestamp
                })
        
        for i in range(len(self.checkpoints) - 1):
            curr = self.checkpoints[i]
            next_cp = self.checkpoints[i + 1]
            
            if "error" in str(next_cp["state"]).lower():
                errors.append({
                    "between": f"{curr['name']} -> {next_cp['name']}",
                    "state_change": {
                        "from": curr["state"],
                        "to": next_cp["state"]
                    }
                })
        
        return errors


class InteractiveDebugger(dspy.Module):
    """Interactive debugging interface for prompt development"""
    
    def __init__(self):
        super().__init__()
        self.console = Console()
        self.history: List[Dict[str, Any]] = []
        
    def debug_prompt(self, prompt: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Interactively debug a prompt"""
        
        self.console.print(Panel(
            f"[bold blue]Debugging Prompt[/bold blue]\n\n{prompt}",
            title="🐛 Debug Session"
        ))
        
        self.console.print("\n[yellow]Inputs:[/yellow]")
        self.console.print(Syntax(
            json.dumps(inputs, indent=2),
            "json",
            theme="monokai"
        ))
        
        trace_points = []
        
        self.console.print("\n[cyan]Starting execution trace...[/cyan]")
        
        result = {
            "prompt": prompt,
            "inputs": inputs,
            "trace_points": trace_points,
            "output": "Execution result"
        }
        
        self.history.append(result)
        
        return result
    
    def show_execution_history(self):
        """Display execution history"""
        
        for i, execution in enumerate(self.history):
            self.console.print(f"\n[bold]Execution {i + 1}:[/bold]")
            self.console.print(f"Prompt: {execution['prompt'][:100]}...")
            self.console.print(f"Trace points: {len(execution['trace_points'])}")


def create_math_problem_trace():
    """Example of thinking trace for a math problem"""
    
    tracer = ThinkingTracer(verbose=True)
    
    problem = """
    A farmer has chickens and cows. 
    The animals have a total of 35 heads and 94 legs.
    How many chickens and how many cows does the farmer have?
    """
    
    result = tracer(problem)
    
    return result


def create_code_debugging_trace():
    """Example of debugging trace for code analysis"""
    
    debugger = ReasoningDebugger()
    
    code_problem = """
    Debug this Python function that's supposed to find the median of a list
    but is returning incorrect results:
    
    def find_median(nums):
        nums.sort()
        n = len(nums)
        if n % 2 == 0:
            return nums[n//2]
        else:
            return (nums[n//2] + nums[n//2 + 1]) / 2
    """
    
    debugger.checkpoint("problem_analysis", {"task": "debug median function"})
    
    result = debugger.tracer(code_problem)
    
    debugger.checkpoint("solution_found", {"error": "even/odd logic reversed"})
    
    errors = debugger.analyze_reasoning_errors(result["trace"])
    
    return {
        "result": result,
        "reasoning_path": debugger.get_reasoning_path(),
        "errors_found": errors
    }


if __name__ == "__main__":
    math_result = create_math_problem_trace()
    print("\nFinal Answer:", math_result["answer"])
    
    print("\n" + "="*50 + "\n")
    
    code_result = create_code_debugging_trace()
    print("\nDebugging Result:", code_result["result"]["answer"])
    print("\nReasoning Path:", code_result["reasoning_path"])