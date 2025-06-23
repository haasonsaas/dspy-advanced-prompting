"""
Task Definition and Planning System

This module implements structured task decomposition and planning
to help LLMs handle complex workflows through explicit sub-task guidance.
"""

import dspy
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import json


class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class SubTask(BaseModel):
    """Represents a single sub-task in a plan"""
    id: str = Field(..., description="Unique identifier for the task")
    title: str = Field(..., description="Brief task title")
    description: str = Field(..., description="Detailed task description")
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks that must complete first")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    estimated_complexity: int = Field(..., description="Complexity score 1-10")
    acceptance_criteria: List[str] = Field(..., description="Criteria for task completion")
    potential_issues: List[str] = Field(default_factory=list, description="Potential blockers or issues")
    status: TaskStatus = Field(default=TaskStatus.NOT_STARTED)
    result: Optional[str] = Field(None, description="Task execution result")


class TaskPlan(BaseModel):
    """Complete task execution plan"""
    goal: str = Field(..., description="Overall goal to achieve")
    context: str = Field(..., description="Context and constraints")
    sub_tasks: List[SubTask] = Field(..., description="Ordered list of sub-tasks")
    success_metrics: List[str] = Field(..., description="How to measure overall success")
    risk_mitigation: Dict[str, str] = Field(default_factory=dict, description="Risk -> Mitigation strategy")
    

class TaskPlanningSignature(dspy.Signature):
    """Decompose a complex task into a structured plan"""
    
    task_description = dspy.InputField(desc="Description of the complex task")
    context = dspy.InputField(desc="Additional context, constraints, and requirements")
    planning_guidelines = dspy.InputField(desc="Guidelines for creating the plan")
    task_plan_json = dspy.OutputField(desc="JSON representation of the detailed task plan")


class TaskExecutionSignature(dspy.Signature):
    """Execute a specific sub-task according to plan"""
    
    sub_task = dspy.InputField(desc="The sub-task to execute (JSON)")
    previous_results = dspy.InputField(desc="Results from completed dependencies")
    execution_context = dspy.InputField(desc="Overall context and constraints")
    result = dspy.OutputField(desc="Detailed result of sub-task execution")


class TaskPlanner(dspy.Module):
    """Plans complex tasks by breaking them into manageable sub-tasks"""
    
    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(TaskPlanningSignature)
        
    def forward(self, task_description: str, context: str = "") -> TaskPlan:
        """Create a detailed execution plan for a complex task"""
        
        planning_guidelines = """
Create a comprehensive task plan following these guidelines:

1. DECOMPOSITION:
   - Break the main task into atomic, executable sub-tasks
   - Each sub-task should be completable in a single step
   - Identify clear dependencies between tasks

2. PRIORITIZATION:
   - Assign priorities based on criticality and dependencies
   - Critical path tasks should be marked as high/critical priority

3. COMPLEXITY ASSESSMENT:
   - Rate each task's complexity (1-10)
   - Consider technical difficulty, time required, and uncertainty

4. ACCEPTANCE CRITERIA:
   - Define clear, measurable criteria for each sub-task
   - Include both positive outcomes and quality checks

5. RISK IDENTIFICATION:
   - Identify potential blockers or issues for each task
   - Create mitigation strategies for major risks

6. SUCCESS METRICS:
   - Define how to measure overall success
   - Include both quantitative and qualitative metrics

The plan should be detailed enough that someone else could execute it.
"""
        
        result = self.planner(
            task_description=task_description,
            context=context,
            planning_guidelines=planning_guidelines
        )
        
        try:
            plan_data = json.loads(result.task_plan_json)
            return TaskPlan(**plan_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse task plan: {e}")


class TaskExecutor(dspy.Module):
    """Executes tasks according to a plan"""
    
    def __init__(self):
        super().__init__()
        self.executor = dspy.ChainOfThought(TaskExecutionSignature)
        
    def forward(self, sub_task: SubTask, previous_results: Dict[str, str], context: str) -> str:
        """Execute a single sub-task"""
        
        result = self.executor(
            sub_task=sub_task.json(),
            previous_results=json.dumps(previous_results),
            execution_context=context
        )
        
        return result.result


class TaskOrchestrator(dspy.Module):
    """Orchestrates planning and execution of complex tasks"""
    
    def __init__(self):
        super().__init__()
        self.planner = TaskPlanner()
        self.executor = TaskExecutor()
        
    def _get_executable_tasks(self, plan: TaskPlan) -> List[SubTask]:
        """Get tasks that can be executed (dependencies met)"""
        completed_ids = {t.id for t in plan.sub_tasks if t.status == TaskStatus.COMPLETED}
        
        executable = []
        for task in plan.sub_tasks:
            if task.status == TaskStatus.NOT_STARTED:
                if all(dep_id in completed_ids for dep_id in task.dependencies):
                    executable.append(task)
                    
        return executable
    
    def _execute_task(self, task: SubTask, plan: TaskPlan, results: Dict[str, str]) -> None:
        """Execute a single task and update its status"""
        task.status = TaskStatus.IN_PROGRESS
        
        try:
            dependency_results = {
                dep_id: results.get(dep_id, "")
                for dep_id in task.dependencies
            }
            
            result = self.executor(
                sub_task=task,
                previous_results=dependency_results,
                context=plan.context
            )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            results[task.id] = result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = f"Failed: {str(e)}"
            raise
    
    def forward(self, task_description: str, context: str = "", 
                max_parallel: int = 1) -> Dict[str, Any]:
        """Plan and execute a complex task"""
        
        plan = self.planner(task_description, context)
        
        results = {}
        completed_count = 0
        
        while completed_count < len(plan.sub_tasks):
            executable_tasks = self._get_executable_tasks(plan)
            
            if not executable_tasks:
                blocked_tasks = [
                    t for t in plan.sub_tasks 
                    if t.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                ]
                if blocked_tasks:
                    raise RuntimeError(f"Tasks blocked: {[t.id for t in blocked_tasks]}")
                break
            
            for task in executable_tasks[:max_parallel]:
                self._execute_task(task, plan, results)
                completed_count += 1
        
        return {
            "plan": plan.dict(),
            "results": results,
            "success": all(t.status == TaskStatus.COMPLETED for t in plan.sub_tasks)
        }


def create_code_refactoring_example():
    """Example: Planning a code refactoring task"""
    
    orchestrator = TaskOrchestrator()
    
    task = """
Refactor a legacy authentication module that has the following issues:
1. Passwords stored in plain text
2. No rate limiting on login attempts  
3. Session tokens never expire
4. Mixed authentication logic with business logic
5. No logging of security events
"""
    
    context = """
- The application is written in Python using Flask
- Currently has 10,000 active users
- Must maintain backward compatibility during migration
- Zero downtime requirement
- Must comply with OWASP security standards
"""
    
    return orchestrator(task, context)


def create_data_pipeline_example():
    """Example: Planning a data pipeline implementation"""
    
    planner = TaskPlanner()
    
    task = """
Build a real-time data pipeline that:
1. Ingests streaming data from multiple IoT sensors
2. Validates and cleanses the data
3. Applies ML models for anomaly detection
4. Stores processed data in a data warehouse
5. Triggers alerts for critical anomalies
"""
    
    context = """
- Expected data volume: 1 million events per minute
- Sensors send data in different formats (JSON, CSV, binary)
- Must detect anomalies within 30 seconds
- 99.9% uptime requirement
- Budget constraint: Use existing AWS infrastructure
"""
    
    return planner(task, context)


class RecursiveTaskPlanner(dspy.Module):
    """Plans tasks recursively, breaking down complex sub-tasks further"""
    
    def __init__(self, max_depth: int = 3):
        super().__init__()
        self.planner = TaskPlanner()
        self.max_depth = max_depth
        
    def _is_task_complex(self, task: SubTask) -> bool:
        """Determine if a task needs further decomposition"""
        return (
            task.estimated_complexity >= 7 or
            len(task.description.split()) > 50 or
            "multiple steps" in task.description.lower() or
            "complex" in task.description.lower()
        )
    
    def _decompose_task(self, task: SubTask, depth: int) -> List[SubTask]:
        """Recursively decompose a complex task"""
        if depth >= self.max_depth:
            return [task]
        
        if not self._is_task_complex(task):
            return [task]
        
        sub_plan = self.planner(
            task_description=task.description,
            context=f"This is a sub-task of: {task.title}"
        )
        
        decomposed = []
        for sub_task in sub_plan.sub_tasks:
            sub_task.id = f"{task.id}.{sub_task.id}"
            sub_sub_tasks = self._decompose_task(sub_task, depth + 1)
            decomposed.extend(sub_sub_tasks)
            
        return decomposed
    
    def forward(self, task_description: str, context: str = "") -> TaskPlan:
        """Create a recursively decomposed task plan"""
        
        initial_plan = self.planner(task_description, context)
        
        all_tasks = []
        for task in initial_plan.sub_tasks:
            decomposed = self._decompose_task(task, 0)
            all_tasks.extend(decomposed)
        
        initial_plan.sub_tasks = all_tasks
        return initial_plan


if __name__ == "__main__":
    result = create_code_refactoring_example()
    print("Refactoring Plan:")
    print(json.dumps(result, indent=2))