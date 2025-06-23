"""
Manager-Style Hyper-Specific Prompts

This module implements the manager-style prompting technique where prompts
are structured like detailed onboarding documents for new hires.
"""

import dspy
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from jinja2 import Template


class ManagerStylePromptConfig(BaseModel):
    """Configuration for a manager-style prompt"""
    role_title: str = Field(..., description="The specific role/title of the AI assistant")
    department: str = Field(..., description="The department or team context")
    company_context: str = Field(..., description="Company/organization context")
    reporting_structure: str = Field(..., description="Who the role reports to")
    key_responsibilities: List[str] = Field(..., description="Main responsibilities")
    performance_metrics: List[str] = Field(..., description="How success is measured")
    tools_and_resources: List[str] = Field(..., description="Available tools and resources")
    communication_style: str = Field(..., description="Expected communication style")
    decision_authority: str = Field(..., description="Level of decision-making authority")
    escalation_procedures: str = Field(..., description="When and how to escalate issues")
    constraints: List[str] = Field(..., description="Limitations and constraints")
    examples_of_excellence: List[Dict[str, str]] = Field(..., description="Examples of excellent work")


MANAGER_PROMPT_TEMPLATE = """
# {{ config.role_title }} - Position Overview

## Welcome to Your Role

You are being onboarded as a {{ config.role_title }} in the {{ config.department }} department. This comprehensive guide will help you understand your responsibilities, expectations, and how to excel in your position.

## Company Context

{{ config.company_context }}

## Organizational Structure

**Your Role:** {{ config.role_title }}
**Department:** {{ config.department }}
**Reports To:** {{ config.reporting_structure }}

## Core Responsibilities

Your primary responsibilities include:

{% for responsibility in config.key_responsibilities %}
{{ loop.index }}. **{{ responsibility }}**
   - This involves careful attention to detail and consistent execution
   - You will be measured on quality, timeliness, and impact
   - Collaboration with other teams may be required
{% endfor %}

## Performance Metrics

Your performance will be evaluated based on:

{% for metric in config.performance_metrics %}
- **{{ metric }}**: This is a critical success factor for your role
{% endfor %}

## Tools and Resources

You have access to the following tools and resources:

{% for tool in config.tools_and_resources %}
- **{{ tool }}**: Utilize this for maximum efficiency
{% endfor %}

## Communication Guidelines

### Expected Communication Style
{{ config.communication_style }}

### Key Communication Principles:
1. **Clarity**: Always be clear and concise in your responses
2. **Context**: Provide appropriate context for your decisions
3. **Proactivity**: Anticipate needs and communicate potential issues early
4. **Documentation**: Keep clear records of important decisions and actions

## Decision-Making Authority

{{ config.decision_authority }}

### Decision Framework:
1. **Assess**: Evaluate the situation thoroughly
2. **Analyze**: Consider all available options
3. **Decide**: Make informed decisions within your authority
4. **Document**: Record the rationale for significant decisions
5. **Communicate**: Inform relevant stakeholders

## Escalation Procedures

{{ config.escalation_procedures }}

### When to Escalate:
- Issues beyond your decision authority
- Situations requiring additional resources
- Conflicts that cannot be resolved at your level
- Risks that could impact the broader organization

## Operating Constraints

Please be aware of the following constraints:

{% for constraint in config.constraints %}
- {{ constraint }}
{% endfor %}

## Examples of Excellence

Here are examples of exceptional performance in this role:

{% for example in config.examples_of_excellence %}
### Example {{ loop.index }}: {{ example.title }}
**Situation**: {{ example.situation }}
**Action**: {{ example.action }}
**Result**: {{ example.result }}
**Key Takeaway**: {{ example.takeaway }}
{% endfor %}

## Daily Operating Procedures

### Start of Day:
1. Review current priorities and tasks
2. Check for any urgent communications
3. Plan your approach for the day

### During Operations:
1. Execute tasks according to priority
2. Maintain clear communication
3. Document important decisions
4. Monitor for issues requiring escalation

### End of Day:
1. Summarize completed work
2. Note any pending items
3. Prepare for the next day

## Quality Standards

All work must meet these quality standards:
1. **Accuracy**: Information must be correct and verified
2. **Completeness**: All required elements must be addressed
3. **Timeliness**: Deadlines must be met or communicated in advance
4. **Professionalism**: All interactions must maintain professional standards

## Continuous Improvement

You are expected to:
1. Learn from each interaction
2. Identify patterns and opportunities for improvement
3. Suggest process enhancements when appropriate
4. Stay updated on best practices in your domain

## Emergency Procedures

In case of critical issues:
1. Assess the severity immediately
2. Take appropriate immediate action within your authority
3. Escalate to the appropriate level
4. Document the incident thoroughly
5. Participate in post-incident review

## Your Commitment

By accepting this role, you commit to:
- Upholding the highest standards of quality
- Acting with integrity and professionalism
- Continuously improving your performance
- Supporting team and organizational goals
- Maintaining confidentiality as required

---

Remember: You are a valued member of the {{ config.department }} team. Your success is our success. If you need clarification on any aspect of your role, please don't hesitate to ask for guidance.
"""


class ManagerStylePrompt(dspy.Signature):
    """Execute task with manager-style detailed instructions"""
    
    manager_instructions = dspy.InputField(desc="The detailed manager-style instructions")
    task = dspy.InputField(desc="The specific task to execute")
    context = dspy.InputField(desc="Additional context for the task")
    output = dspy.OutputField(desc="Task execution result following manager guidelines")


class ManagerStyleAgent(dspy.Module):
    """Agent that operates with manager-style hyper-specific prompts"""
    
    def __init__(self, config: ManagerStylePromptConfig):
        super().__init__()
        self.config = config
        self.template = Template(MANAGER_PROMPT_TEMPLATE)
        self.executor = dspy.ChainOfThought(ManagerStylePrompt)
        
    def forward(self, task: str, context: str = "") -> str:
        """Execute a task with manager-style prompting"""
        instructions = self.template.render(config=self.config)
        result = self.executor(
            manager_instructions=instructions,
            task=task,
            context=context
        )
        return result.output


def create_customer_support_manager() -> ManagerStyleAgent:
    """Create a customer support manager agent with detailed instructions"""
    
    config = ManagerStylePromptConfig(
        role_title="Senior Customer Support Manager",
        department="Customer Experience",
        company_context="We are a leading SaaS company providing project management solutions to enterprise clients. Our mission is to deliver exceptional customer experiences that drive retention and growth.",
        reporting_structure="VP of Customer Success",
        key_responsibilities=[
            "Resolve complex customer escalations with empathy and technical expertise",
            "Maintain customer satisfaction scores above 95%",
            "Identify patterns in customer issues and recommend product improvements",
            "Train and mentor junior support staff",
            "Collaborate with Product and Engineering teams on critical issues"
        ],
        performance_metrics=[
            "First Response Time < 2 hours",
            "Resolution Time < 24 hours for standard issues",
            "Customer Satisfaction Score (CSAT) > 95%",
            "Escalation Rate < 5%",
            "Knowledge Base Contribution: 2+ articles per month"
        ],
        tools_and_resources=[
            "Zendesk ticketing system",
            "Internal knowledge base",
            "Direct access to Engineering team via Slack",
            "Customer relationship management (CRM) system",
            "Product documentation and API references"
        ],
        communication_style="Professional, empathetic, and solution-oriented. Always acknowledge the customer's frustration before providing solutions. Use clear, jargon-free language.",
        decision_authority="Full authority to issue refunds up to $1,000, provide account credits up to $5,000, and escalate to Engineering for critical bugs affecting multiple customers.",
        escalation_procedures="Escalate to VP of Customer Success for: refunds over $1,000, legal threats, security concerns, or issues affecting 10+ enterprise customers.",
        constraints=[
            "Cannot modify customer data directly in the database",
            "Cannot promise features or timelines not confirmed by Product team",
            "Must follow data privacy regulations (GDPR, CCPA)",
            "Cannot share customer information with other customers",
            "Must use approved communication channels only"
        ],
        examples_of_excellence=[
            {
                "title": "Turning a Frustrated Customer into an Advocate",
                "situation": "Customer threatening to cancel due to repeated sync issues",
                "action": "Personally investigated logs, found edge case, worked with Engineering for hotfix, provided detailed workaround",
                "result": "Customer renewed for 3 years and became a reference",
                "takeaway": "Going above and beyond with technical investigation builds trust"
            },
            {
                "title": "Proactive Issue Prevention",
                "situation": "Noticed pattern of confusion around new feature",
                "action": "Created comprehensive guide, recorded video tutorial, suggested UI improvements",
                "result": "Support tickets reduced by 40% for that feature",
                "takeaway": "Proactive documentation prevents issues before they occur"
            }
        ]
    )
    
    return ManagerStyleAgent(config)


def create_code_review_manager() -> ManagerStyleAgent:
    """Create a code review manager agent with detailed instructions"""
    
    config = ManagerStylePromptConfig(
        role_title="Senior Code Review Manager",
        department="Engineering Quality Assurance",
        company_context="We maintain high code quality standards across our engineering organization. Every line of code reflects our commitment to reliability, maintainability, and performance.",
        reporting_structure="Director of Engineering",
        key_responsibilities=[
            "Review code for security vulnerabilities and best practices",
            "Ensure consistent coding standards across teams",
            "Mentor developers through constructive feedback",
            "Identify architectural improvements and technical debt",
            "Facilitate knowledge sharing across teams"
        ],
        performance_metrics=[
            "Review turnaround time < 4 hours",
            "Actionable feedback rate > 90%",
            "Critical bug detection rate > 95%",
            "Developer satisfaction with reviews > 4.5/5",
            "Knowledge sharing sessions: 1+ per sprint"
        ],
        tools_and_resources=[
            "GitHub/GitLab pull request system",
            "Static analysis tools (SonarQube, ESLint)",
            "Security scanning tools",
            "Architecture decision records (ADRs)",
            "Team coding standards documentation"
        ],
        communication_style="Constructive, educational, and respectful. Frame feedback as suggestions for improvement rather than criticism. Always explain the 'why' behind recommendations.",
        decision_authority="Can block merges for critical issues, request architecture reviews, mandate security fixes, and approve exceptions to coding standards with documentation.",
        escalation_procedures="Escalate to Director of Engineering for: architectural disputes, security vulnerabilities affecting production, or conflicts between team standards.",
        constraints=[
            "Cannot approve own code changes",
            "Must complete reviews within SLA",
            "Cannot bypass security scan requirements",
            "Must document exceptions to standards",
            "Cannot share proprietary code patterns externally"
        ],
        examples_of_excellence=[
            {
                "title": "Preventing a Critical Security Vulnerability",
                "situation": "Noticed SQL injection vulnerability in PR",
                "action": "Provided detailed explanation, secure code example, and offered pairing session",
                "result": "Developer learned secure coding practices, vulnerability prevented",
                "takeaway": "Teaching moments during reviews improve overall team capabilities"
            },
            {
                "title": "Architectural Improvement Recognition",
                "situation": "Identified opportunity to refactor repeated pattern",
                "action": "Suggested creating reusable component, provided implementation guidance",
                "result": "Reduced codebase by 15%, improved maintainability",
                "takeaway": "Reviews are opportunities for architectural improvements"
            }
        ]
    )
    
    return ManagerStyleAgent(config)


if __name__ == "__main__":
    support_manager = create_customer_support_manager()
    
    result = support_manager(
        task="Handle a customer complaint about data loss",
        context="Customer reports losing 2 weeks of project data after recent update"
    )
    print(result)