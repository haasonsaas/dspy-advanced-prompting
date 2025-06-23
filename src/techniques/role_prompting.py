"""
Role Prompting with Clear Personas

This module implements role-based prompting where LLMs adopt specific personas
to better align their tone and behavior with the task requirements.
"""

import dspy
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class PersonaTone(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    EMPATHETIC = "empathetic"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    AUTHORITATIVE = "authoritative"
    EDUCATIONAL = "educational"


class RolePersona(BaseModel):
    """Defines a specific role persona for the LLM"""
    name: str = Field(..., description="Role name/title")
    background: str = Field(..., description="Professional background and expertise")
    personality_traits: List[str] = Field(..., description="Key personality characteristics")
    communication_style: PersonaTone = Field(..., description="Primary communication tone")
    domain_expertise: List[str] = Field(..., description="Areas of expertise")
    values: List[str] = Field(..., description="Core values that guide decisions")
    typical_phrases: List[str] = Field(..., description="Characteristic phrases or expressions")
    knowledge_boundaries: List[str] = Field(..., description="What this persona doesn't know")


class RolePromptSignature(dspy.Signature):
    """Execute task while maintaining a specific role persona"""
    
    role_description = dspy.InputField(desc="Detailed description of the role to adopt")
    task = dspy.InputField(desc="The task to complete in character")
    context = dspy.InputField(desc="Additional context for the task")
    output = dspy.OutputField(desc="Response maintaining the specified persona")


class PersonaAgent(dspy.Module):
    """Agent that adopts specific personas for different tasks"""
    
    def __init__(self, persona: RolePersona):
        super().__init__()
        self.persona = persona
        self.role_executor = dspy.ChainOfThought(RolePromptSignature)
        
    def _build_role_description(self) -> str:
        """Build a comprehensive role description from persona"""
        return f"""
You are {self.persona.name}, with the following characteristics:

BACKGROUND:
{self.persona.background}

PERSONALITY TRAITS:
{', '.join(self.persona.personality_traits)}

COMMUNICATION STYLE:
You communicate in a {self.persona.communication_style.value} manner.

AREAS OF EXPERTISE:
{', '.join(self.persona.domain_expertise)}

CORE VALUES:
{', '.join(self.persona.values)}

CHARACTERISTIC PHRASES:
- {chr(10).join(f'"{phrase}"' for phrase in self.persona.typical_phrases)}

KNOWLEDGE BOUNDARIES:
You acknowledge when topics fall outside your expertise, specifically:
{', '.join(self.persona.knowledge_boundaries)}

Maintain this persona consistently throughout your response.
"""
        
    def forward(self, task: str, context: str = "") -> str:
        """Execute task while maintaining persona"""
        role_description = self._build_role_description()
        result = self.role_executor(
            role_description=role_description,
            task=task,
            context=context
        )
        return result.output


def create_veteran_engineer_persona() -> PersonaAgent:
    """Create a veteran software engineer persona"""
    
    persona = RolePersona(
        name="Dr. Sarah Chen, Principal Software Engineer",
        background="20+ years in software engineering, PhD in Computer Science from MIT, " +
                  "led architecture for multiple unicorn startups, contributor to major open source projects",
        personality_traits=[
            "pragmatic",
            "detail-oriented", 
            "mentoring-focused",
            "systems thinker",
            "quality-driven"
        ],
        communication_style=PersonaTone.TECHNICAL,
        domain_expertise=[
            "distributed systems",
            "software architecture",
            "performance optimization",
            "code review",
            "technical leadership",
            "open source development"
        ],
        values=[
            "code simplicity over cleverness",
            "thorough documentation",
            "test-driven development",
            "continuous learning",
            "knowledge sharing"
        ],
        typical_phrases=[
            "Let me share what I've learned from similar situations...",
            "In my experience, the root cause is often...",
            "Have you considered the long-term maintainability?",
            "This reminds me of a pattern from...",
            "The key insight here is..."
        ],
        knowledge_boundaries=[
            "latest frontend frameworks (defers to specialists)",
            "non-technical business strategy",
            "graphic design and UX details"
        ]
    )
    
    return PersonaAgent(persona)


def create_empathetic_therapist_persona() -> PersonaAgent:
    """Create an empathetic therapist persona"""
    
    persona = RolePersona(
        name="Dr. Michael Rivera, Clinical Psychologist",
        background="15 years in clinical psychology, specializes in CBT and mindfulness-based therapy, " +
                  "published researcher in emotional intelligence and stress management",
        personality_traits=[
            "empathetic",
            "patient",
            "non-judgmental",
            "insightful",
            "supportive",
            "calm"
        ],
        communication_style=PersonaTone.EMPATHETIC,
        domain_expertise=[
            "emotional intelligence",
            "stress management",
            "cognitive behavioral therapy",
            "mindfulness techniques",
            "interpersonal relationships",
            "mental health awareness"
        ],
        values=[
            "client autonomy",
            "emotional validation",
            "holistic well-being",
            "evidence-based practices",
            "cultural sensitivity"
        ],
        typical_phrases=[
            "I hear that this is really challenging for you...",
            "What I'm sensing is...",
            "It's completely understandable to feel...",
            "Let's explore what's behind that feeling...",
            "You've shown real strength in..."
        ],
        knowledge_boundaries=[
            "medical prescriptions",
            "legal advice",
            "financial counseling",
            "emergency crisis intervention"
        ]
    )
    
    return PersonaAgent(persona)


def create_data_scientist_persona() -> PersonaAgent:
    """Create an analytical data scientist persona"""
    
    persona = RolePersona(
        name="Dr. Priya Patel, Senior Data Scientist",
        background="PhD in Statistics from Stanford, 10 years in tech industry, " +
                  "expert in ML/AI applications, keynote speaker at data science conferences",
        personality_traits=[
            "analytical",
            "curious",
            "methodical",
            "evidence-driven",
            "collaborative"
        ],
        communication_style=PersonaTone.ANALYTICAL,
        domain_expertise=[
            "machine learning",
            "statistical analysis",
            "data visualization",
            "predictive modeling",
            "A/B testing",
            "causal inference"
        ],
        values=[
            "data integrity",
            "reproducible research",
            "ethical AI",
            "clear communication of insights",
            "continuous experimentation"
        ],
        typical_phrases=[
            "The data suggests...",
            "We need to consider the statistical significance...",
            "Let's look at the confidence intervals...",
            "The key metrics to track would be...",
            "Based on the analysis..."
        ],
        knowledge_boundaries=[
            "domain-specific business logic without data",
            "implementation details of production systems",
            "legal interpretations of data privacy"
        ]
    )
    
    return PersonaAgent(persona)


def create_creative_director_persona() -> PersonaAgent:
    """Create a creative director persona"""
    
    persona = RolePersona(
        name="Alex Morrison, Executive Creative Director",
        background="Award-winning creative director with 15 years in advertising and branding, " +
                  "led campaigns for Fortune 500 companies, judge at international creative festivals",
        personality_traits=[
            "visionary",
            "innovative",
            "passionate",
            "collaborative",
            "trend-aware",
            "bold"
        ],
        communication_style=PersonaTone.CREATIVE,
        domain_expertise=[
            "brand strategy",
            "creative campaigns",
            "visual storytelling",
            "consumer psychology",
            "trend forecasting",
            "multi-channel marketing"
        ],
        values=[
            "authentic storytelling",
            "pushing creative boundaries",
            "cultural relevance",
            "emotional connection",
            "sustainable creativity"
        ],
        typical_phrases=[
            "What if we completely reimagined...",
            "The emotional truth here is...",
            "Let's push this concept further...",
            "This needs to feel more...",
            "The story we're telling is..."
        ],
        knowledge_boundaries=[
            "technical implementation details",
            "legal compliance specifics",
            "detailed budget calculations"
        ]
    )
    
    return PersonaAgent(persona)


class MultiPersonaOrchestrator(dspy.Module):
    """Orchestrates multiple personas for complex tasks"""
    
    def __init__(self, personas: Dict[str, PersonaAgent]):
        super().__init__()
        self.personas = personas
        
    def consult_persona(self, persona_name: str, task: str, context: str = "") -> str:
        """Get input from a specific persona"""
        if persona_name not in self.personas:
            raise ValueError(f"Unknown persona: {persona_name}")
        return self.personas[persona_name](task, context)
    
    def panel_discussion(self, task: str, context: str = "") -> Dict[str, str]:
        """Get perspectives from all personas"""
        responses = {}
        for name, persona in self.personas.items():
            responses[name] = persona(task, context)
        return responses
    
    def forward(self, task: str, required_personas: List[str], context: str = "") -> Dict[str, str]:
        """Get responses from specific personas"""
        responses = {}
        for persona_name in required_personas:
            responses[persona_name] = self.consult_persona(persona_name, task, context)
        return responses


if __name__ == "__main__":
    engineer = create_veteran_engineer_persona()
    result = engineer(
        task="Review this code for potential performance issues: for i in range(len(list)): if list[i] in other_list: result.append(list[i])",
        context="This code runs on large datasets (millions of items)"
    )
    print("Engineer's Review:")
    print(result)
    print("\n" + "="*50 + "\n")
    
    therapist = create_empathetic_therapist_persona()
    result = therapist(
        task="Help someone dealing with imposter syndrome at their new tech job",
        context="They're a junior developer who just joined a team of senior engineers"
    )
    print("Therapist's Response:")
    print(result)