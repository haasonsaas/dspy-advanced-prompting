"""
Escape Hatches for Uncertainty Handling

This module implements escape hatches that instruct LLMs to admit uncertainty,
preventing hallucination and improving trust.
"""

import dspy
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import re


class UncertaintyLevel(str, Enum):
    NONE = "none"  # Fully confident
    LOW = "low"  # Slightly uncertain
    MEDIUM = "medium"  # Moderately uncertain
    HIGH = "high"  # Very uncertain
    UNABLE = "unable"  # Cannot provide answer


class UncertaintyType(str, Enum):
    FACTUAL = "factual"  # Uncertainty about facts
    INTERPRETATION = "interpretation"  # Uncertainty about meaning
    PREDICTION = "prediction"  # Uncertainty about future events
    TECHNICAL = "technical"  # Uncertainty about technical details
    CONTEXTUAL = "contextual"  # Missing context
    EXPERTISE = "expertise"  # Outside area of expertise


class UncertaintyResponse(BaseModel):
    """Structured uncertainty response"""
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    uncertainty_level: UncertaintyLevel
    uncertainty_types: List[UncertaintyType] = Field(default_factory=list)
    certain_aspects: List[str] = Field(default_factory=list, description="What we know")
    uncertain_aspects: List[str] = Field(default_factory=list, description="What we're unsure about")
    clarifying_questions: List[str] = Field(default_factory=list, description="Questions to reduce uncertainty")
    caveats: List[str] = Field(default_factory=list, description="Important limitations")
    alternative_interpretations: List[str] = Field(default_factory=list)
    suggested_resources: List[str] = Field(default_factory=list, description="Where to find authoritative info")


class EscapeHatchSignature(dspy.Signature):
    """Generate response with appropriate uncertainty handling"""
    
    question = dspy.InputField(desc="The question or task")
    uncertainty_guidelines = dspy.InputField(desc="Guidelines for expressing uncertainty")
    context = dspy.InputField(desc="Available context")
    response = dspy.OutputField(desc="Response with appropriate uncertainty indicators")
    uncertainty_analysis = dspy.OutputField(desc="Analysis of uncertainty in JSON format")


class UncertaintyDetector(dspy.Module):
    """Detects and quantifies uncertainty in responses"""
    
    def __init__(self):
        super().__init__()
        self.uncertainty_phrases = {
            UncertaintyLevel.LOW: [
                "likely", "probably", "seems to", "appears to", "suggests that",
                "generally", "typically", "usually", "often", "tends to"
            ],
            UncertaintyLevel.MEDIUM: [
                "might", "could", "possibly", "perhaps", "may", "potentially",
                "it's possible", "uncertain", "unclear", "debatable"
            ],
            UncertaintyLevel.HIGH: [
                "I'm not sure", "I don't know", "unknown", "cannot determine",
                "insufficient information", "no clear answer", "highly uncertain"
            ],
            UncertaintyLevel.UNABLE: [
                "I cannot", "I'm unable", "beyond my knowledge", "outside my expertise",
                "I don't have access", "I lack the information"
            ]
        }
        
    def detect_uncertainty(self, text: str) -> Tuple[UncertaintyLevel, float]:
        """Detect uncertainty level in text"""
        
        text_lower = text.lower()
        
        for level in [UncertaintyLevel.UNABLE, UncertaintyLevel.HIGH, 
                     UncertaintyLevel.MEDIUM, UncertaintyLevel.LOW]:
            if any(phrase in text_lower for phrase in self.uncertainty_phrases[level]):
                confidence_map = {
                    UncertaintyLevel.UNABLE: 0.0,
                    UncertaintyLevel.HIGH: 0.2,
                    UncertaintyLevel.MEDIUM: 0.5,
                    UncertaintyLevel.LOW: 0.7,
                    UncertaintyLevel.NONE: 0.95
                }
                return level, confidence_map[level]
        
        return UncertaintyLevel.NONE, 0.95
    
    def analyze_response(self, response: str, question: str) -> UncertaintyResponse:
        """Analyze a response for uncertainty indicators"""
        
        level, confidence = self.detect_uncertainty(response)
        
        certain_aspects = []
        uncertain_aspects = []
        
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sent_level, _ = self.detect_uncertainty(sentence)
            if sent_level in [UncertaintyLevel.NONE, UncertaintyLevel.LOW]:
                certain_aspects.append(sentence)
            else:
                uncertain_aspects.append(sentence)
        
        return UncertaintyResponse(
            confidence_level=confidence,
            uncertainty_level=level,
            certain_aspects=certain_aspects[:3],
            uncertain_aspects=uncertain_aspects[:3],
            uncertainty_types=self._identify_uncertainty_types(response, question)
        )
    
    def _identify_uncertainty_types(self, response: str, question: str) -> List[UncertaintyType]:
        """Identify types of uncertainty present"""
        
        types = []
        response_lower = response.lower()
        
        if any(word in response_lower for word in ["fact", "data", "statistic", "number"]):
            types.append(UncertaintyType.FACTUAL)
        
        if any(word in response_lower for word in ["interpret", "meaning", "understand"]):
            types.append(UncertaintyType.INTERPRETATION)
        
        if any(word in response_lower for word in ["future", "will", "predict", "forecast"]):
            types.append(UncertaintyType.PREDICTION)
        
        if any(word in response_lower for word in ["technical", "specific", "detail"]):
            types.append(UncertaintyType.TECHNICAL)
        
        if any(word in response_lower for word in ["context", "background", "situation"]):
            types.append(UncertaintyType.CONTEXTUAL)
        
        return types


class EscapeHatchResponder(dspy.Module):
    """Generates responses with built-in escape hatches"""
    
    def __init__(self):
        super().__init__()
        self.responder = dspy.ChainOfThought(EscapeHatchSignature)
        self.detector = UncertaintyDetector()
        
    def create_uncertainty_guidelines(self) -> str:
        """Create comprehensive uncertainty guidelines"""
        
        return """
UNCERTAINTY HANDLING GUIDELINES:

1. ADMISSION OF UNCERTAINTY:
   - When you don't know something, say so clearly
   - Use phrases like "I don't have enough information to answer that definitively"
   - Never make up facts or fill gaps with plausible-sounding information

2. CONFIDENCE LEVELS:
   - High confidence: State facts directly
   - Medium confidence: Use "likely", "probably", "appears to be"
   - Low confidence: Use "might", "possibly", "it's unclear"
   - No confidence: "I don't know", "I cannot determine"

3. PARTIAL KNOWLEDGE:
   - Share what you DO know: "While I can't answer X, I can tell you Y"
   - Identify specific missing information: "To answer this, I would need..."
   - Suggest alternative approaches: "Instead, you might consider..."

4. CLARIFYING QUESTIONS:
   - Ask for clarification when the question is ambiguous
   - Request specific details that would help provide a better answer
   - Example: "Could you specify whether you mean X or Y?"

5. CAVEATS AND LIMITATIONS:
   - State assumptions explicitly: "Assuming X, then..."
   - Note temporal limitations: "As of my last update..."
   - Acknowledge expertise boundaries: "This is outside my area of expertise"

6. ALTERNATIVE INTERPRETATIONS:
   - If multiple interpretations exist, present them
   - "This could mean either A or B, depending on..."

7. RESOURCE SUGGESTIONS:
   - Point to authoritative sources when uncertain
   - "For definitive information, consult..."
   - "The official documentation would have..."

8. AVOID THESE PATTERNS:
   - Don't use hedging to cover up ignorance
   - Don't provide "common sense" answers to specific questions
   - Don't extrapolate beyond available information
"""
        
    def forward(self, question: str, context: str = "", 
                require_sources: bool = False) -> Dict[str, Any]:
        """Generate response with uncertainty handling"""
        
        guidelines = self.create_uncertainty_guidelines()
        
        if require_sources:
            guidelines += "\n\n9. SOURCES: Always cite sources or indicate when making claims without sources."
        
        result = self.responder(
            question=question,
            uncertainty_guidelines=guidelines,
            context=context
        )
        
        uncertainty_analysis = self.detector.analyze_response(result.response, question)
        
        return {
            "response": result.response,
            "uncertainty_analysis": uncertainty_analysis,
            "raw_analysis": result.uncertainty_analysis
        }


class GracefulDegradation(dspy.Module):
    """Implements graceful degradation when confidence is low"""
    
    def __init__(self, confidence_threshold: float = 0.6):
        super().__init__()
        self.threshold = confidence_threshold
        self.escaper = EscapeHatchResponder()
        
    def create_degraded_response(self, question: str, 
                               uncertainty: UncertaintyResponse) -> str:
        """Create a degraded but helpful response"""
        
        response_parts = []
        
        if uncertainty.confidence_level < 0.3:
            response_parts.append(
                "I don't have enough information to answer this question reliably."
            )
        else:
            response_parts.append(
                "I have limited confidence in my response to this question."
            )
        
        if uncertainty.certain_aspects:
            response_parts.append("\nWhat I can tell you:")
            for aspect in uncertainty.certain_aspects:
                response_parts.append(f"• {aspect}")
        
        if uncertainty.uncertain_aspects:
            response_parts.append("\nWhat I'm uncertain about:")
            for aspect in uncertainty.uncertain_aspects:
                response_parts.append(f"• {aspect}")
        
        if uncertainty.clarifying_questions:
            response_parts.append("\nTo provide a better answer, I would need to know:")
            for question in uncertainty.clarifying_questions:
                response_parts.append(f"• {question}")
        
        if uncertainty.suggested_resources:
            response_parts.append("\nFor authoritative information, consider:")
            for resource in uncertainty.suggested_resources:
                response_parts.append(f"• {resource}")
        
        return "\n".join(response_parts)
    
    def forward(self, question: str, context: str = "") -> Dict[str, Any]:
        """Generate response with graceful degradation"""
        
        result = self.escaper(question, context)
        uncertainty = result["uncertainty_analysis"]
        
        if uncertainty.confidence_level < self.threshold:
            degraded_response = self.create_degraded_response(question, uncertainty)
            return {
                "response": degraded_response,
                "degraded": True,
                "original_response": result["response"],
                "uncertainty_analysis": uncertainty
            }
        
        return {
            "response": result["response"],
            "degraded": False,
            "uncertainty_analysis": uncertainty
        }


class HallucinationPreventer(dspy.Module):
    """Prevents hallucination through proactive uncertainty checking"""
    
    def __init__(self):
        super().__init__()
        self.fact_patterns = [
            r'\d{4}',  # Years
            r'\d+%',  # Percentages
            r'\$[\d,]+',  # Dollar amounts
            r'[A-Z][a-z]+ \d+, \d{4}',  # Dates
            r'according to',  # Citations
            r'study shows',  # Research claims
            r'research indicates',  # Research claims
        ]
        
    def check_for_potential_hallucinations(self, response: str) -> List[str]:
        """Check for patterns that often indicate hallucination"""
        
        warnings = []
        
        for pattern in self.fact_patterns:
            matches = re.findall(pattern, response)
            if matches:
                warnings.append(f"Contains specific claims: {matches[:3]}")
        
        if "everyone knows" in response.lower():
            warnings.append("Uses 'everyone knows' - potential overgeneralization")
        
        if len(re.findall(r'always|never|all|none', response, re.I)) > 2:
            warnings.append("Multiple absolute statements - check for overgeneralization")
        
        return warnings
    
    def add_uncertainty_markers(self, response: str, warnings: List[str]) -> str:
        """Add uncertainty markers to potentially hallucinated content"""
        
        if not warnings:
            return response
        
        disclaimer = "\n\n⚠️ Note: This response contains specific claims that should be verified:\n"
        for warning in warnings:
            disclaimer += f"• {warning}\n"
        
        return response + disclaimer


def create_medical_advice_example():
    """Example of medical advice with strong escape hatches"""
    
    escaper = EscapeHatchResponder()
    
    question = "What's the best treatment for chronic migraines?"
    
    medical_context = """
    IMPORTANT: You are not a medical professional. Always include appropriate disclaimers
    and encourage consulting healthcare providers for medical advice.
    """
    
    result = escaper(question, medical_context)
    
    return result


def create_financial_advice_example():
    """Example of financial advice with uncertainty handling"""
    
    degrader = GracefulDegradation(confidence_threshold=0.7)
    
    question = "Should I invest in cryptocurrency right now?"
    
    result = degrader(question, "User asking about investment timing")
    
    return result


class ContextualEscapeHatch(dspy.Module):
    """Provides context-specific escape hatches"""
    
    def __init__(self):
        super().__init__()
        self.domain_disclaimers = {
            "medical": "I'm not a medical professional. Please consult a healthcare provider for medical advice.",
            "legal": "I cannot provide legal advice. Please consult a qualified attorney.",
            "financial": "This is not financial advice. Consult a financial advisor for personalized guidance.",
            "safety": "For safety-critical decisions, please consult relevant experts and official guidelines.",
            "personal": "I don't have access to your personal information or specific circumstances."
        }
        
    def detect_domain(self, question: str) -> Optional[str]:
        """Detect the domain of the question"""
        
        domain_keywords = {
            "medical": ["health", "medical", "symptom", "treatment", "doctor", "pain"],
            "legal": ["legal", "law", "sue", "contract", "rights", "court"],
            "financial": ["invest", "money", "stock", "crypto", "finance", "loan"],
            "safety": ["safe", "danger", "hazard", "emergency", "risk"],
            "personal": ["my", "personal", "private", "specific situation"]
        }
        
        question_lower = question.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return domain
                
        return None
    
    def add_domain_disclaimer(self, response: str, domain: str) -> str:
        """Add domain-specific disclaimer"""
        
        if domain in self.domain_disclaimers:
            return f"{self.domain_disclaimers[domain]}\n\n{response}"
        return response


if __name__ == "__main__":
    escaper = EscapeHatchResponder()
    
    questions = [
        "What will the stock market do next year?",
        "Explain how quantum computing works",
        "What's the population of Mars?",
        "How do I implement a binary search tree?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = escaper(question)
        print(f"Response: {result['response']}")
        print(f"Confidence: {result['uncertainty_analysis'].confidence_level}")
        print(f"Level: {result['uncertainty_analysis'].uncertainty_level}")