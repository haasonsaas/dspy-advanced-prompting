"""
Structured Output with XML/Markdown Tags

This module implements structured output formatting using XML-style tags
and markdown to improve response consistency and parseability.
"""

import dspy
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from xml.etree import ElementTree as ET
import re
import json
from enum import Enum


class OutputFormat(str, Enum):
    XML = "xml"
    MARKDOWN = "markdown"
    JSON = "json"
    HYBRID = "hybrid"


class StructuredSection(BaseModel):
    """Represents a section in structured output"""
    tag: str = Field(..., description="Tag name for the section")
    content: Union[str, List['StructuredSection']] = Field(..., description="Section content")
    attributes: Dict[str, str] = Field(default_factory=dict, description="Section attributes")
    required: bool = Field(default=True, description="Whether this section is required")
    

class OutputSchema(BaseModel):
    """Defines the expected structure of output"""
    format: OutputFormat = Field(..., description="Output format type")
    sections: List[StructuredSection] = Field(..., description="Expected sections")
    validation_rules: List[str] = Field(default_factory=list, description="Validation rules")
    example: str = Field(..., description="Example of correctly formatted output")


class StructuredOutputSignature(dspy.Signature):
    """Generate output in a specific structured format"""
    
    task = dspy.InputField(desc="The task to complete")
    output_schema = dspy.InputField(desc="The required output structure and format")
    context = dspy.InputField(desc="Additional context")
    structured_output = dspy.OutputField(desc="Output following the specified structure")


class ValidationResult(BaseModel):
    """Result of output validation"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    parsed_content: Optional[Dict[str, Any]] = None


class StructuredOutputGenerator(dspy.Module):
    """Generates output in structured formats"""
    
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(StructuredOutputSignature)
        
    def forward(self, task: str, schema: OutputSchema, context: str = "") -> str:
        """Generate structured output according to schema"""
        
        schema_description = self._build_schema_description(schema)
        result = self.generator(
            task=task,
            output_schema=schema_description,
            context=context
        )
        return result.structured_output
    
    def _build_schema_description(self, schema: OutputSchema) -> str:
        """Build a detailed description of the output schema"""
        
        format_guidelines = {
            OutputFormat.XML: "Use XML tags like <tag>content</tag>",
            OutputFormat.MARKDOWN: "Use Markdown headers (##) and formatting",
            OutputFormat.JSON: "Return valid JSON with specified structure",
            OutputFormat.HYBRID: "Combine XML tags within Markdown formatting"
        }
        
        sections_desc = "\n".join([
            f"- <{s.tag}> {'(required)' if s.required else '(optional)'}: {s.tag}"
            for s in schema.sections
        ])
        
        return f"""
Output Format: {schema.format.value}
Format Guidelines: {format_guidelines[schema.format]}

Required Sections:
{sections_desc}

Validation Rules:
{chr(10).join(f'- {rule}' for rule in schema.validation_rules)}

Example Output:
{schema.example}

Ensure all required sections are present and follow the exact format shown.
"""


class OutputValidator:
    """Validates structured output against schemas"""
    
    @staticmethod
    def validate_xml(output: str, schema: OutputSchema) -> ValidationResult:
        """Validate XML-formatted output"""
        errors = []
        warnings = []
        parsed = {}
        
        try:
            root = ET.fromstring(f"<root>{output}</root>")
            
            for section in schema.sections:
                elements = root.findall(section.tag)
                if not elements and section.required:
                    errors.append(f"Missing required section: <{section.tag}>")
                elif elements:
                    parsed[section.tag] = [elem.text for elem in elements]
                    
        except ET.ParseError as e:
            errors.append(f"Invalid XML: {str(e)}")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            parsed_content=parsed
        )
    
    @staticmethod
    def validate_markdown(output: str, schema: OutputSchema) -> ValidationResult:
        """Validate Markdown-formatted output"""
        errors = []
        warnings = []
        parsed = {}
        
        for section in schema.sections:
            pattern = rf"##\s*{section.tag}\s*\n(.*?)(?=\n##|\Z)"
            matches = re.findall(pattern, output, re.DOTALL)
            
            if not matches and section.required:
                errors.append(f"Missing required section: ## {section.tag}")
            elif matches:
                parsed[section.tag] = matches[0].strip()
                
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            parsed_content=parsed
        )
    
    @staticmethod
    def validate(output: str, schema: OutputSchema) -> ValidationResult:
        """Validate output based on schema format"""
        validators = {
            OutputFormat.XML: OutputValidator.validate_xml,
            OutputFormat.MARKDOWN: OutputValidator.validate_markdown,
            OutputFormat.JSON: OutputValidator.validate_json,
            OutputFormat.HYBRID: OutputValidator.validate_hybrid
        }
        
        validator = validators.get(schema.format, OutputValidator.validate_xml)
        return validator(output, schema)
    
    @staticmethod
    def validate_json(output: str, schema: OutputSchema) -> ValidationResult:
        """Validate JSON-formatted output"""
        errors = []
        warnings = []
        
        try:
            parsed = json.loads(output)
            
            for section in schema.sections:
                if section.required and section.tag not in parsed:
                    errors.append(f"Missing required field: {section.tag}")
                    
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                parsed_content=parsed
            )
            
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid JSON: {str(e)}"],
                warnings=warnings
            )
    
    @staticmethod
    def validate_hybrid(output: str, schema: OutputSchema) -> ValidationResult:
        """Validate hybrid XML/Markdown output"""
        errors = []
        warnings = []
        parsed = {}
        
        for section in schema.sections:
            md_pattern = rf"##\s*{section.tag}.*?<{section.tag}>(.*?)</{section.tag}>"
            matches = re.findall(md_pattern, output, re.DOTALL)
            
            if not matches and section.required:
                errors.append(f"Missing required section: ## {section.tag} with <{section.tag}> tags")
            elif matches:
                parsed[section.tag] = matches[0].strip()
                
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            parsed_content=parsed
        )


def create_bug_report_schema() -> OutputSchema:
    """Create schema for structured bug reports"""
    
    return OutputSchema(
        format=OutputFormat.XML,
        sections=[
            StructuredSection(tag="summary", content="Brief description of the bug"),
            StructuredSection(tag="environment", content="System and version information"),
            StructuredSection(tag="steps_to_reproduce", content="Numbered steps"),
            StructuredSection(tag="expected_behavior", content="What should happen"),
            StructuredSection(tag="actual_behavior", content="What actually happens"),
            StructuredSection(tag="error_logs", content="Relevant error messages", required=False),
            StructuredSection(tag="severity", content="critical|high|medium|low"),
            StructuredSection(tag="suggested_fix", content="Potential solution", required=False)
        ],
        validation_rules=[
            "Summary must be under 100 characters",
            "Steps to reproduce must be numbered",
            "Severity must be one of: critical, high, medium, low",
            "Include stack traces in error_logs if available"
        ],
        example="""<summary>Login button unresponsive on mobile devices</summary>
<environment>iOS 15.0, Safari, iPhone 12</environment>
<steps_to_reproduce>
1. Navigate to login page on mobile device
2. Enter valid credentials
3. Tap the login button
</steps_to_reproduce>
<expected_behavior>User should be logged in and redirected to dashboard</expected_behavior>
<actual_behavior>Button appears to be tapped but no action occurs</actual_behavior>
<error_logs>Console: Uncaught TypeError: Cannot read property 'submit' of null</error_logs>
<severity>high</severity>
<suggested_fix>Check if form element exists before calling submit()</suggested_fix>"""
    )


def create_code_review_schema() -> OutputSchema:
    """Create schema for structured code reviews"""
    
    return OutputSchema(
        format=OutputFormat.HYBRID,
        sections=[
            StructuredSection(tag="overall_assessment", content="High-level review summary"),
            StructuredSection(tag="security_issues", content="Security vulnerabilities found"),
            StructuredSection(tag="performance_concerns", content="Performance issues"),
            StructuredSection(tag="code_quality", content="Style and maintainability issues"),
            StructuredSection(tag="positive_aspects", content="What was done well"),
            StructuredSection(tag="required_changes", content="Must-fix issues"),
            StructuredSection(tag="suggestions", content="Optional improvements", required=False),
            StructuredSection(tag="learning_resources", content="Helpful references", required=False)
        ],
        validation_rules=[
            "Use <critical>, <high>, <medium>, <low> tags for issue severity",
            "Include line numbers for specific issues",
            "Provide code examples for suggested improvements",
            "Be constructive and educational in tone"
        ],
        example="""## Overall Assessment
<overall_assessment>
The code implements the feature correctly but has several security and performance concerns that need addressing before merge.
</overall_assessment>

## Security Issues
<security_issues>
<critical>SQL injection vulnerability on line 45: User input directly concatenated into query</critical>
<high>Missing input validation on lines 23-25: Email format not verified</high>
</security_issues>

## Performance Concerns
<performance_concerns>
<medium>N+1 query problem in getUserPosts() - consider eager loading</medium>
<low>Unnecessary array copying on line 78 - use slice() instead</low>
</performance_concerns>"""
    )


class ParahelpStyleFormatter(dspy.Module):
    """Formatter using Parahelp-style verification tags"""
    
    def __init__(self):
        super().__init__()
        self.sections = {
            "analysis": "Initial analysis and understanding",
            "approach": "Planned approach to solve the problem",
            "implementation": "Actual implementation or solution",
            "verification": "Double-check of the solution",
            "manager_verify": "Final manager-level quality check"
        }
        
    def format_response(self, content: Dict[str, str]) -> str:
        """Format response with Parahelp-style tags"""
        
        formatted = []
        for tag, description in self.sections.items():
            if tag in content:
                formatted.append(f"<{tag}>")
                formatted.append(f"<!-- {description} -->")
                formatted.append(content[tag])
                formatted.append(f"</{tag}>")
                formatted.append("")
                
        return "\n".join(formatted)
    
    def parse_response(self, response: str) -> Dict[str, str]:
        """Parse Parahelp-style tagged response"""
        
        parsed = {}
        for tag in self.sections:
            pattern = rf"<{tag}>(.*?)</{tag}>"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL).strip()
                parsed[tag] = content
                
        return parsed


class StructuredDialogueFormatter(dspy.Module):
    """Formats multi-turn dialogues with clear structure"""
    
    def __init__(self):
        super().__init__()
        self.turn_template = """
<turn number="{number}" speaker="{speaker}" timestamp="{timestamp}">
  <intent>{intent}</intent>
  <content>{content}</content>
  <metadata>
    <sentiment>{sentiment}</sentiment>
    <topics>{topics}</topics>
    <action_items>{action_items}</action_items>
  </metadata>
</turn>
"""
    
    def format_turn(self, number: int, speaker: str, content: str,
                    intent: str = "", sentiment: str = "neutral",
                    topics: List[str] = None, action_items: List[str] = None,
                    timestamp: str = "") -> str:
        """Format a single dialogue turn"""
        
        return self.turn_template.format(
            number=number,
            speaker=speaker,
            timestamp=timestamp,
            intent=intent,
            content=content,
            sentiment=sentiment,
            topics=", ".join(topics or []),
            action_items=", ".join(action_items or [])
        )


if __name__ == "__main__":
    generator = StructuredOutputGenerator()
    
    bug_schema = create_bug_report_schema()
    result = generator(
        task="Report a bug where the search function returns no results for queries containing special characters",
        schema=bug_schema,
        context="This happens on the e-commerce website's product search"
    )
    
    print("Generated Bug Report:")
    print(result)
    print("\nValidation:")
    validation = OutputValidator.validate(result, bug_schema)
    print(f"Valid: {validation.is_valid}")
    if validation.errors:
        print(f"Errors: {validation.errors}")