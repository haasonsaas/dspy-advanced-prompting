"""
Few-Shot Prompting with Real Examples

This module implements few-shot prompting techniques using real, challenging
examples to shape LLM behavior and improve accuracy.
"""

import dspy
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass
import json
from enum import Enum


class ExampleQuality(str, Enum):
    GOLD = "gold"  # Perfect examples
    SILVER = "silver"  # Good examples with minor issues
    BRONZE = "bronze"  # Acceptable examples
    CHALLENGING = "challenging"  # Edge cases and difficult examples


@dataclass
class FewShotExample:
    """Represents a single few-shot example"""
    input: str
    output: str
    reasoning: Optional[str] = None
    quality: ExampleQuality = ExampleQuality.SILVER
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def format(self, include_reasoning: bool = True) -> str:
        """Format example for prompt inclusion"""
        formatted = f"Input: {self.input}\n"
        if include_reasoning and self.reasoning:
            formatted += f"Reasoning: {self.reasoning}\n"
        formatted += f"Output: {self.output}"
        return formatted


class ExampleSelector(BaseModel):
    """Selects relevant examples for a given task"""
    strategy: str = Field(default="similarity", description="Selection strategy")
    max_examples: int = Field(default=5, description="Maximum examples to include")
    prioritize_challenging: bool = Field(default=True, description="Include challenging cases")
    balance_categories: bool = Field(default=True, description="Balance example categories")


class FewShotSignature(dspy.Signature):
    """Execute task using few-shot examples"""
    
    task_description = dspy.InputField(desc="Description of the task to perform")
    examples = dspy.InputField(desc="Carefully selected examples demonstrating the task")
    input = dspy.InputField(desc="The specific input to process")
    output = dspy.OutputField(desc="Output following the pattern shown in examples")


class FewShotPromptTemplate(BaseModel):
    """Template for few-shot prompts"""
    task_intro: str = Field(..., description="Introduction explaining the task")
    example_intro: str = Field(default="Here are some examples:", description="Introduction to examples")
    example_separator: str = Field(default="\n---\n", description="Separator between examples")
    task_prompt: str = Field(default="Now, process this input:", description="Prompt before user input")
    include_reasoning: bool = Field(default=True, description="Include reasoning in examples")


class FewShotLearner(dspy.Module):
    """Implements few-shot learning with intelligent example selection"""
    
    def __init__(self, examples: List[FewShotExample], template: FewShotPromptTemplate):
        super().__init__()
        self.examples = examples
        self.template = template
        self.executor = dspy.ChainOfThought(FewShotSignature)
        self.example_selector = ExampleSelector()
        
    def select_examples(self, input_text: str, selector: ExampleSelector) -> List[FewShotExample]:
        """Select relevant examples for the given input"""
        
        selected = []
        
        if selector.prioritize_challenging:
            challenging = [ex for ex in self.examples if ex.quality == ExampleQuality.CHALLENGING]
            selected.extend(challenging[:2])
        
        gold_examples = [ex for ex in self.examples if ex.quality == ExampleQuality.GOLD]
        selected.extend(gold_examples[:selector.max_examples - len(selected)])
        
        if len(selected) < selector.max_examples:
            other_examples = [ex for ex in self.examples 
                            if ex not in selected and ex.quality != ExampleQuality.BRONZE]
            selected.extend(other_examples[:selector.max_examples - len(selected)])
        
        return selected[:selector.max_examples]
    
    def format_examples(self, examples: List[FewShotExample]) -> str:
        """Format examples for inclusion in prompt"""
        
        formatted_examples = []
        for i, example in enumerate(examples, 1):
            example_text = f"Example {i}:\n{example.format(self.template.include_reasoning)}"
            if example.quality == ExampleQuality.CHALLENGING:
                example_text += "\n(Note: This is a challenging case)"
            formatted_examples.append(example_text)
        
        return self.template.example_separator.join(formatted_examples)
    
    def forward(self, input_text: str, custom_selector: Optional[ExampleSelector] = None) -> str:
        """Process input using few-shot examples"""
        
        selector = custom_selector or self.example_selector
        selected_examples = self.select_examples(input_text, selector)
        formatted_examples = self.format_examples(selected_examples)
        
        task_with_examples = f"{self.template.task_intro}\n\n{self.template.example_intro}\n\n{formatted_examples}"
        
        result = self.executor(
            task_description=self.template.task_intro,
            examples=formatted_examples,
            input=input_text
        )
        
        return result.output


def create_bug_analysis_examples() -> List[FewShotExample]:
    """Create few-shot examples for bug analysis (Jazzberry-style)"""
    
    return [
        FewShotExample(
            input="TypeError: Cannot read property 'map' of undefined at UserList.render()",
            output="Root cause: The component is trying to map over a users array that is undefined. This typically happens when: 1) The data hasn't loaded yet (add loading state check), 2) The API returned null/undefined instead of empty array (add default value), or 3) Props are not being passed correctly (verify prop types). Fix: Add `const users = this.props.users || []` before mapping.",
            reasoning="undefined errors on array methods indicate missing null/undefined checks",
            quality=ExampleQuality.GOLD,
            tags=["null-check", "react", "defensive-coding"]
        ),
        FewShotExample(
            input="CORS error: Access to XMLHttpRequest at 'api.example.com' from origin 'localhost:3000' blocked",
            output="This is a Cross-Origin Resource Sharing (CORS) issue. Solutions: 1) Backend fix: Add CORS headers (`Access-Control-Allow-Origin: http://localhost:3000`), 2) Development proxy: Configure webpack devServer proxy, 3) For production: Ensure backend allows your domain. Never use `Access-Control-Allow-Origin: *` in production. If using credentials, also set `Access-Control-Allow-Credentials: true`.",
            reasoning="CORS errors require backend configuration, not frontend fixes",
            quality=ExampleQuality.GOLD,
            tags=["cors", "api", "security"]
        ),
        FewShotExample(
            input="Maximum call stack size exceeded in recursive function calculateFactorial()",
            output="Stack overflow due to infinite recursion. Common causes: 1) Missing base case (add `if (n <= 1) return 1`), 2) Base case never reached due to wrong decrement, 3) Negative input causing infinite loop. Fix: Add proper base case and input validation. For large numbers, consider iterative approach or memoization to prevent stack issues.",
            reasoning="Stack overflow = check recursion base case and termination condition",
            quality=ExampleQuality.GOLD,
            tags=["recursion", "stack-overflow", "algorithm"]
        ),
        FewShotExample(
            input="React Hook useEffect has missing dependency: 'userId'",
            output="React's exhaustive deps rule detected that useEffect uses 'userId' but doesn't list it in the dependency array. This can cause stale closures. Solutions: 1) Add to deps: `useEffect(() => {...}, [userId])`, 2) If intentional, use eslint-disable comment with explanation, 3) Consider useCallback/useMemo for function dependencies. Warning: Empty deps array means effect only runs once (mount), not on userId changes.",
            reasoning="React hooks require all dependencies to prevent stale closures",
            quality=ExampleQuality.SILVER,
            tags=["react-hooks", "dependencies", "best-practices"]
        ),
        FewShotExample(
            input="Database connection timeout after 30 seconds when app is under heavy load",
            output="Connection pool exhaustion under load. Diagnosis: 1) Check pool size (increase if needed), 2) Look for connection leaks (connections not being released), 3) Verify query performance (slow queries hold connections longer), 4) Check for N+1 queries. Solutions: Increase pool size, implement connection pooling properly, add connection timeout handling, use read replicas for scaling, implement circuit breaker pattern.",
            reasoning="Timeouts under load usually indicate resource exhaustion, not network issues",
            quality=ExampleQuality.CHALLENGING,
            tags=["database", "performance", "scaling", "connection-pool"]
        ),
        FewShotExample(
            input="Memory leak detected: heap size growing from 50MB to 2GB over 24 hours",
            output="Progressive memory leak indication. Common JS causes: 1) Event listeners not removed (add cleanup in useEffect return), 2) Closures holding references, 3) Large objects in global scope, 4) Circular references, 5) SetInterval without clearInterval. Debugging: Use Chrome DevTools heap snapshots, compare allocations over time, look for detached DOM nodes. Focus on components that mount/unmount frequently.",
            reasoning="Gradual memory growth indicates retained references, not initial allocation",
            quality=ExampleQuality.CHALLENGING,
            tags=["memory-leak", "performance", "debugging", "profiling"]
        )
    ]


def create_code_review_examples() -> List[FewShotExample]:
    """Create few-shot examples for code review"""
    
    return [
        FewShotExample(
            input="""
def authenticate_user(username, password):
    user = db.query(f"SELECT * FROM users WHERE username='{username}' AND password='{password}'")
    if user:
        return {"success": True, "user": user}
    return {"success": False}
""",
            output="""CRITICAL Security Issues:
1. **SQL Injection**: Never use f-strings for SQL. Use parameterized queries: `db.query("SELECT * FROM users WHERE username=? AND password=?", (username, password))`
2. **Plain text passwords**: Passwords must be hashed. Use bcrypt: `bcrypt.checkpw(password.encode(), user.password_hash)`
3. **Information disclosure**: Don't reveal if username exists. Return generic "Invalid credentials"
4. **No rate limiting**: Add brute force protection
5. **No logging**: Log failed attempts for security monitoring

Secure version:
```python
def authenticate_user(username, password):
    # Rate limiting check here
    user = db.query("SELECT * FROM users WHERE username = ?", (username,))
    if user and bcrypt.checkpw(password.encode(), user.password_hash):
        log_successful_login(username)
        return {"success": True, "user_id": user.id}  # Don't return sensitive data
    log_failed_attempt(username)
    return {"success": False, "error": "Invalid credentials"}
```""",
            reasoning="Authentication code requires multiple security layers",
            quality=ExampleQuality.GOLD,
            tags=["security", "sql-injection", "authentication"]
        ),
        FewShotExample(
            input="""
async function processPayments(payments) {
    for (let payment of payments) {
        await processSinglePayment(payment);
        await updateDatabase(payment);
        await sendNotification(payment);
    }
}
""",
            output="""Performance Issues:
1. **Sequential processing**: Processing payments one-by-one is inefficient. Use concurrent processing with proper error handling
2. **No transaction management**: Database updates should be atomic
3. **No error handling**: One failure stops all processing
4. **No idempotency**: Re-running could double-charge

Improved version:
```javascript
async function processPayments(payments) {
    const results = await Promise.allSettled(
        payments.map(payment => processPaymentWithRetry(payment))
    );
    
    const successful = results.filter(r => r.status === 'fulfilled');
    const failed = results.filter(r => r.status === 'rejected');
    
    // Log failures for manual review
    if (failed.length > 0) {
        await logFailedPayments(failed);
    }
    
    return { successful: successful.length, failed: failed.length };
}

async function processPaymentWithRetry(payment, retries = 3) {
    const idempotencyKey = generateIdempotencyKey(payment);
    
    try {
        await db.transaction(async (trx) => {
            await processS inglePayment(payment, idempotencyKey);
            await updateDatabase(payment, trx);
        });
        await sendNotification(payment); // Outside transaction
    } catch (error) {
        if (retries > 0 && isRetriable(error)) {
            await delay(1000 * (4 - retries)); // Exponential backoff
            return processPaymentWithRetry(payment, retries - 1);
        }
        throw error;
    }
}
```""",
            reasoning="Payment processing requires reliability, idempotency, and performance",
            quality=ExampleQuality.GOLD,
            tags=["performance", "reliability", "payments", "async"]
        )
    ]


def create_api_design_examples() -> List[FewShotExample]:
    """Create few-shot examples for API design feedback"""
    
    return [
        FewShotExample(
            input="""
POST /api/users/123/update-email
Body: { "new_email": "user@example.com" }
""",
            output="""RESTful design issues:
1. **Verb in URL**: Use HTTP verbs, not URL verbs. Should be `PATCH /api/users/123`
2. **Unclear field naming**: Use consistent field names: `email` not `new_email`
3. **No validation info**: Add email format validation
4. **Missing response spec**: Define success/error responses

Improved design:
```
PATCH /api/users/123
Content-Type: application/json

{
  "email": "user@example.com"
}

Response (200 OK):
{
  "id": "123",
  "email": "user@example.com",
  "emailVerified": false,
  "updatedAt": "2024-01-15T10:30:00Z"
}

Response (400 Bad Request):
{
  "error": {
    "code": "INVALID_EMAIL",
    "message": "Email format is invalid",
    "field": "email"
  }
}
```

Additional considerations:
- Require authentication
- Send verification email
- Rate limit email changes
- Audit log the change""",
            reasoning="REST APIs should follow conventions and handle edge cases",
            quality=ExampleQuality.GOLD,
            tags=["api-design", "rest", "best-practices"]
        )
    ]


class AdaptiveFewShotLearner(dspy.Module):
    """Learns which examples work best for different types of inputs"""
    
    def __init__(self, example_pools: Dict[str, List[FewShotExample]]):
        super().__init__()
        self.example_pools = example_pools
        self.performance_history: Dict[str, List[float]] = {}
        
    def categorize_input(self, input_text: str) -> str:
        """Categorize input to select appropriate example pool"""
        
        categories = {
            "error": ["error", "exception", "failed", "undefined"],
            "security": ["password", "auth", "token", "sql", "inject"],
            "performance": ["slow", "timeout", "memory", "scale"],
            "api": ["endpoint", "rest", "api", "http"],
            "code": ["function", "class", "method", "variable"]
        }
        
        input_lower = input_text.lower()
        for category, keywords in categories.items():
            if any(keyword in input_lower for keyword in keywords):
                return category
                
        return "general"
    
    def select_adaptive_examples(self, input_text: str, category: str) -> List[FewShotExample]:
        """Select examples based on past performance"""
        
        if category not in self.example_pools:
            category = "general"
            
        examples = self.example_pools[category]
        
        if category in self.performance_history:
            scores = self.performance_history[category]
            example_scores = list(zip(examples, scores))
            example_scores.sort(key=lambda x: x[1], reverse=True)
            return [ex for ex, _ in example_scores[:5]]
        
        return examples[:5]
    
    def update_performance(self, category: str, example_index: int, score: float):
        """Update performance history for adaptive selection"""
        
        if category not in self.performance_history:
            self.performance_history[category] = [0.5] * len(self.example_pools.get(category, []))
            
        if 0 <= example_index < len(self.performance_history[category]):
            old_score = self.performance_history[category][example_index]
            self.performance_history[category][example_index] = 0.8 * old_score + 0.2 * score


class ChainOfThoughtFewShot(dspy.Module):
    """Combines few-shot learning with chain-of-thought reasoning"""
    
    def __init__(self, examples: List[FewShotExample]):
        super().__init__()
        self.examples = examples
        
    def forward(self, input_text: str) -> Tuple[str, str]:
        """Generate output with explicit reasoning"""
        
        cot_examples = []
        for ex in self.examples[:3]:
            if ex.reasoning:
                cot_examples.append(
                    f"Input: {ex.input}\n"
                    f"Let me think step by step: {ex.reasoning}\n"
                    f"Therefore: {ex.output}"
                )
        
        prompt = f"""
{chr(10).join(cot_examples)}

Now for your input:
Input: {input_text}
Let me think step by step:"""
        
        reasoning = "Breaking down the problem systematically..."
        output = "Based on the analysis..."
        
        return reasoning, output


if __name__ == "__main__":
    bug_examples = create_bug_analysis_examples()
    
    template = FewShotPromptTemplate(
        task_intro="Analyze the following bug report and provide a detailed solution:",
        example_intro="Here are examples of bug analyses with solutions:",
        include_reasoning=True
    )
    
    learner = FewShotLearner(bug_examples, template)
    
    result = learner("Error: Cannot connect to Redis server after deploying to production")
    print("Bug Analysis Result:")
    print(result)