# DSpy Advanced Prompting Techniques

A comprehensive implementation of state-of-the-art prompting techniques used by top AI startups, built with DSpy.

> **DSpy** is a framework for algorithmically optimizing LM prompts and weights. Instead of manually crafting prompts, DSpy allows you to define high-level signatures and automatically optimize them.

## 🚀 Overview

This project demonstrates advanced prompting strategies that go beyond simple prompt engineering. Each technique is implemented as a reusable DSpy module with real-world examples.

## 📋 Implemented Techniques

### 1. **Manager-Style Hyper-Specific Prompts**
- 6+ page detailed prompts structured like onboarding documents
- Complete role definitions, responsibilities, and performance metrics
- Example: Customer Support Manager, Code Review Manager

### 2. **Role Prompting with Clear Personas**
- LLMs adopt specific personas for better task alignment
- Includes veteran engineer, therapist, data scientist personas
- Multi-persona orchestration for complex tasks

### 3. **Task Definition and Planning System**
- Breaks complex workflows into predictable steps
- Recursive task decomposition
- Dependency management and execution orchestration

### 4. **Structured Output (XML/Markdown Tags)**
- Enforces consistent response formats
- Supports XML, Markdown, JSON, and hybrid formats
- Parahelp-style verification tags

### 5. **Meta-Prompting for Self-Optimization**
- LLMs analyze and improve their own prompts
- Iterative refinement based on output quality
- Prompt evolution using genetic algorithms

### 6. **Few-Shot Prompting with Real Examples**
- Challenging bug analysis examples (Jazzberry-style)
- Adaptive example selection
- Chain-of-thought few-shot learning

### 7. **Prompt Folding for Multi-Step Workflows**
- One prompt triggers generation of deeper prompts
- Supports recursive, pipeline, branching strategies
- Workflow management for complex tasks

### 8. **Escape Hatches for Uncertainty**
- Prevents hallucination through uncertainty admission
- Graceful degradation when confidence is low
- Domain-specific disclaimers

### 9. **Thinking Traces and Debug Logging**
- Exposes model's internal reasoning
- Visual thinking trace representation
- Comprehensive debug information

### 10. **Evaluation Framework**
- Test cases more valuable than prompts
- A/B testing framework
- Regression testing and performance metrics

### 11. **Model Distillation Pipeline**
- Use large models for prompt crafting
- Deploy on smaller, cheaper models
- Production optimization strategies

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/haasonsaas/dspy-advanced-prompting.git
cd dspy-advanced-prompting

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## ✅ Testing

The project structure has been verified to have:
- ✅ Valid Python syntax in all files
- ✅ Proper class and module organization
- ✅ All required files present

To run the examples after installing dependencies:
```bash
python main.py  # Interactive demo of all techniques
python examples/quick_start.py  # Quick start examples
```

## 📖 Quick Start

```python
from src.prompts.manager_style import create_customer_support_manager
from src.techniques.role_prompting import create_veteran_engineer_persona
from src.techniques.thinking_traces import ThinkingTracer

# Use manager-style prompts
support_manager = create_customer_support_manager()
result = support_manager(
    task="Handle a customer complaint about data loss",
    context="Customer reports losing 2 weeks of project data"
)

# Use role-based personas
engineer = create_veteran_engineer_persona()
code_review = engineer(
    task="Review this code for performance issues",
    context="High-traffic production endpoint"
)

# Use thinking traces
tracer = ThinkingTracer(verbose=True)
math_solution = tracer("Solve: A farmer has 35 heads and 94 legs...")
```

## 🏗️ Project Structure

```
dspy-advanced-prompting/
├── src/
│   ├── prompts/
│   │   └── manager_style.py      # Manager-style prompts
│   ├── techniques/
│   │   ├── role_prompting.py     # Role personas
│   │   ├── task_planning.py      # Task decomposition
│   │   ├── structured_output.py  # Output formatting
│   │   ├── meta_prompting.py     # Self-optimization
│   │   ├── few_shot.py          # Few-shot learning
│   │   ├── prompt_folding.py    # Workflow folding
│   │   ├── escape_hatches.py    # Uncertainty handling
│   │   ├── thinking_traces.py   # Debug traces
│   │   └── model_distillation.py # Distillation
│   └── evaluations/
│       └── evaluation_framework.py # Testing framework
├── examples/                     # Usage examples
└── tests/                       # Unit tests
```

## 🧪 Running Examples

```bash
# Run manager-style prompt example
python -m src.prompts.manager_style

# Run thinking traces demo
python -m src.techniques.thinking_traces

# Run evaluation framework
python -m src.evaluations.evaluation_framework
```

## 📊 Performance Metrics

Each technique includes built-in evaluation metrics:
- **Accuracy**: How well the prompt performs its intended task
- **Consistency**: Stability across different inputs
- **Robustness**: Performance on edge cases
- **Efficiency**: Token usage and execution time

## 🔧 Advanced Usage

### Creating Custom Manager-Style Prompts

```python
from src.prompts.manager_style import ManagerStylePromptConfig, ManagerStyleAgent

config = ManagerStylePromptConfig(
    role_title="Senior Data Analyst",
    department="Business Intelligence",
    key_responsibilities=[
        "Analyze business metrics",
        "Create actionable insights",
        "Build dashboards"
    ],
    # ... more configuration
)

analyst = ManagerStyleAgent(config)
```

### Building Evaluation Suites

```python
from src.evaluations.evaluation_framework import TestCase, EvaluationSuite

test_suite = EvaluationSuite(
    name="Custom Test Suite",
    test_cases=[
        TestCase(
            id="test_1",
            input="Your input",
            expected_output="Expected output",
            evaluation_criteria={"contains_all": ["key", "terms"]}
        )
    ]
)
```

## 🚀 Production Deployment

The model distillation pipeline helps optimize prompts for production:

```python
from src.techniques.model_distillation import DistillationPipeline

pipeline = DistillationPipeline()
deployment = await pipeline.distill_and_deploy(
    prompt="Your complex prompt",
    production_requirements={
        "max_latency_ms": 500,
        "min_accuracy": 0.9,
        "daily_volume": 100000
    }
)
```

## 📚 Key Insights

1. **Prompts as Onboarding Docs**: Treat prompts like you're onboarding a new employee
2. **Test Cases > Prompts**: Evaluation frameworks are more valuable than the prompts themselves
3. **Uncertainty is Good**: Better to admit uncertainty than hallucinate
4. **Debug Everything**: Thinking traces reveal model reasoning
5. **Start Big, Deploy Small**: Use large models to craft, small models to serve

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

Created by Jonathan Haas (jonathan@haas.holdings)

## 🙏 Acknowledgments

Inspired by prompting techniques from leading AI startups including:
- Parahelp (manager-style prompts)
- Jazzberry (few-shot bug analysis)
- And many others pushing the boundaries of prompt engineering