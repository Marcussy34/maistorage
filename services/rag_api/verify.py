"""
Verification component for agentic RAG system in Phase 5.

This module provides advanced verification capabilities for validating
answer quality, faithfulness, and determining refinement needs.
"""

import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from pydantic import BaseModel

from llm_client import LLMClient, LLMConfig
from prompts.verifier import format_verifier_prompt, format_faithfulness_check

logger = logging.getLogger(__name__)


class VerificationLevel(str, Enum):
    """Levels of verification depth."""
    BASIC = "basic"          # Simple faithfulness check
    STANDARD = "standard"    # Multi-criteria evaluation
    COMPREHENSIVE = "comprehensive"  # Full RAGAS-style evaluation


class VerificationCriterion(str, Enum):
    """Individual verification criteria."""
    FAITHFULNESS = "faithfulness"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    CITATION_QUALITY = "citation_quality"
    CLARITY = "clarity"


class VerificationResult(BaseModel):
    """Result of answer verification."""
    overall_score: float  # 1-5 scale
    criteria_scores: Dict[VerificationCriterion, float]
    passed: bool
    needs_refinement: bool
    issues: List[str]
    suggestions: List[str]
    verification_time_ms: float
    verification_level: VerificationLevel
    raw_response: str
    confidence: float  # Confidence in the verification result
    
    def __init__(self, **data):
        super().__init__(**data)
        # Auto-calculate passed status if not provided
        if 'passed' not in data:
            self.passed = self.overall_score >= 4.0
        if 'needs_refinement' not in data:
            self.needs_refinement = not self.passed


class Verifier:
    """
    Advanced verification component for agentic RAG responses.
    
    Provides multiple levels of verification from basic faithfulness
    checking to comprehensive RAGAS-style evaluation.
    """
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """
        Initialize the verifier.
        
        Args:
            llm_config: Configuration for LLM client
        """
        self.llm = LLMClient(llm_config or LLMConfig())
        
        # Verification thresholds
        self.pass_threshold = 4.0
        self.criteria_threshold = 3.0
        self.confidence_threshold = 0.7
        
        logger.info("Verifier initialized")
    
    async def verify(
        self,
        query: str,
        answer: str,
        context: str,
        level: VerificationLevel = VerificationLevel.STANDARD,
        citations: Optional[List[Dict[str, Any]]] = None
    ) -> VerificationResult:
        """
        Verify answer quality and determine if refinement is needed.
        
        Args:
            query: Original user query
            answer: Generated answer to verify
            context: Context documents used for generation
            level: Level of verification to perform
            citations: Citation information if available
            
        Returns:
            Verification result with scores and recommendations
        """
        start_time = time.time()
        
        logger.info(f"Verifying answer with {level} level verification")
        
        if level == VerificationLevel.BASIC:
            result = await self._basic_verification(query, answer, context)
        elif level == VerificationLevel.STANDARD:
            result = await self._standard_verification(query, answer, context, citations)
        else:  # COMPREHENSIVE
            result = await self._comprehensive_verification(query, answer, context, citations)
        
        # Set verification metadata
        result.verification_time_ms = (time.time() - start_time) * 1000
        result.verification_level = level
        
        logger.info(f"Verification completed: score={result.overall_score:.1f}, passed={result.passed}")
        
        return result
    
    async def _basic_verification(
        self,
        query: str,
        answer: str,
        context: str
    ) -> VerificationResult:
        """
        Basic faithfulness verification using simple prompt.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Context documents
            
        Returns:
            Basic verification result
        """
        try:
            # Use simple faithfulness check
            messages = format_faithfulness_check(answer, context)
            response = await self.llm.achat_completion(messages)
            
            # Parse basic result
            result_text = response.content.lower()
            
            if "faithful" in result_text and "unfaithful" not in result_text:
                score = 5.0 if "fully" in result_text else 4.0
                passed = True
                issues = []
            elif "partially_faithful" in result_text:
                score = 3.0
                passed = False
                issues = ["Some claims not fully supported by context"]
            elif "unfaithful" in result_text:
                score = 2.0
                passed = False
                issues = ["Answer contradicts or goes beyond context"]
            else:
                score = 2.0
                passed = False
                issues = ["Insufficient context to verify claims"]
            
            # Basic suggestions
            suggestions = []
            if not passed:
                if score < 3.0:
                    suggestions.append("Revise answer to better align with provided context")
                else:
                    suggestions.append("Strengthen support for unsupported claims")
            
            return VerificationResult(
                overall_score=score,
                criteria_scores={VerificationCriterion.FAITHFULNESS: score},
                passed=passed,
                needs_refinement=not passed,
                issues=issues,
                suggestions=suggestions,
                verification_time_ms=0.0,  # Will be set by caller
                verification_level=VerificationLevel.BASIC,
                raw_response=response.content,
                confidence=0.8  # High confidence for basic check
            )
            
        except Exception as e:
            logger.error(f"Basic verification failed: {e}")
            return self._create_error_result(str(e), VerificationLevel.BASIC)
    
    async def _standard_verification(
        self,
        query: str,
        answer: str,
        context: str,
        citations: Optional[List[Dict[str, Any]]] = None
    ) -> VerificationResult:
        """
        Standard multi-criteria verification.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Context documents
            citations: Citation information
            
        Returns:
            Standard verification result
        """
        try:
            # Use comprehensive verifier prompt
            messages = format_verifier_prompt(query, answer, context)
            response = await self.llm.achat_completion(messages)
            
            # Parse structured response
            scores, issues, suggestions = self._parse_verifier_response(response.content)
            
            # Calculate overall score
            overall_score = sum(scores.values()) / len(scores) if scores else 2.0
            
            # Additional citation verification if available
            if citations:
                citation_score = self._verify_citations(answer, citations)
                scores[VerificationCriterion.CITATION_QUALITY] = citation_score
                overall_score = sum(scores.values()) / len(scores)
            
            # Determine confidence based on consistency of scores
            confidence = self._calculate_confidence(scores)
            
            return VerificationResult(
                overall_score=overall_score,
                criteria_scores=scores,
                passed=overall_score >= self.pass_threshold,
                needs_refinement=overall_score < self.pass_threshold,
                issues=issues,
                suggestions=suggestions,
                verification_time_ms=0.0,  # Will be set by caller
                verification_level=VerificationLevel.STANDARD,
                raw_response=response.content,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Standard verification failed: {e}")
            return self._create_error_result(str(e), VerificationLevel.STANDARD)
    
    async def _comprehensive_verification(
        self,
        query: str,
        answer: str,
        context: str,
        citations: Optional[List[Dict[str, Any]]] = None
    ) -> VerificationResult:
        """
        Comprehensive RAGAS-style verification.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Context documents
            citations: Citation information
            
        Returns:
            Comprehensive verification result
        """
        try:
            # Run multiple verification checks in parallel
            tasks = []
            
            # Basic faithfulness
            basic_result = await self._basic_verification(query, answer, context)
            
            # Standard verification
            standard_result = await self._standard_verification(query, answer, context, citations)
            
            # Additional semantic coherence check
            coherence_score = await self._check_semantic_coherence(answer)
            
            # Combine results
            all_scores = {**standard_result.criteria_scores}
            all_scores[VerificationCriterion.FAITHFULNESS] = max(
                basic_result.overall_score,
                all_scores.get(VerificationCriterion.FAITHFULNESS, 0)
            )
            
            # Add coherence as clarity score
            all_scores[VerificationCriterion.CLARITY] = coherence_score
            
            # Calculate comprehensive score with weighted criteria
            weights = {
                VerificationCriterion.FAITHFULNESS: 0.3,
                VerificationCriterion.RELEVANCE: 0.25,
                VerificationCriterion.COMPLETENESS: 0.2,
                VerificationCriterion.CITATION_QUALITY: 0.15,
                VerificationCriterion.CLARITY: 0.1
            }
            
            weighted_score = sum(
                all_scores.get(criterion, 2.0) * weight
                for criterion, weight in weights.items()
            )
            
            # Combine issues and suggestions
            all_issues = list(set(basic_result.issues + standard_result.issues))
            all_suggestions = list(set(basic_result.suggestions + standard_result.suggestions))
            
            # Add comprehensive suggestions
            if weighted_score < 3.5:
                all_suggestions.append("Consider retrieving additional context")
            if coherence_score < 3.5:
                all_suggestions.append("Improve answer structure and flow")
            
            confidence = self._calculate_confidence(all_scores)
            
            return VerificationResult(
                overall_score=weighted_score,
                criteria_scores=all_scores,
                passed=weighted_score >= self.pass_threshold,
                needs_refinement=weighted_score < self.pass_threshold,
                issues=all_issues,
                suggestions=all_suggestions,
                verification_time_ms=0.0,  # Will be set by caller
                verification_level=VerificationLevel.COMPREHENSIVE,
                raw_response=f"Basic: {basic_result.raw_response}\n\nStandard: {standard_result.raw_response}",
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Comprehensive verification failed: {e}")
            return self._create_error_result(str(e), VerificationLevel.COMPREHENSIVE)
    
    def _parse_verifier_response(self, response: str) -> Tuple[Dict[VerificationCriterion, float], List[str], List[str]]:
        """Parse structured verifier response."""
        scores = {}
        issues = []
        suggestions = []
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse overall score
            if line.lower().startswith('overall score:'):
                match = re.search(r'(\d+(?:\.\d+)?)/5', line)
                if match:
                    # Overall score will be calculated from criteria
                    pass
            
            # Parse criteria scores
            for criterion in VerificationCriterion:
                if line.lower().startswith(criterion.value.lower() + ':'):
                    match = re.search(r'(\d+(?:\.\d+)?)/5', line)
                    if match:
                        scores[criterion] = float(match.group(1))
            
            # Parse issues
            if line.lower().startswith('issues:'):
                current_section = 'issues'
                continue
            elif line.lower().startswith('suggestions:'):
                current_section = 'suggestions'
                continue
            elif line.startswith('-') and current_section:
                content = line[1:].strip()
                if current_section == 'issues':
                    issues.append(content)
                elif current_section == 'suggestions':
                    suggestions.append(content)
        
        # Default scores if not found
        default_criteria = [
            VerificationCriterion.FAITHFULNESS,
            VerificationCriterion.RELEVANCE,
            VerificationCriterion.COMPLETENESS
        ]
        
        for criterion in default_criteria:
            if criterion not in scores:
                scores[criterion] = 3.0  # Neutral score
        
        return scores, issues, suggestions
    
    def _verify_citations(self, answer: str, citations: List[Dict[str, Any]]) -> float:
        """Verify citation quality and coverage."""
        if not citations:
            return 2.0
        
        # Count citation markers in answer
        citation_markers = len(re.findall(r'\[Source:', answer))
        
        # Basic citation quality scoring
        if citation_markers == 0:
            return 1.0  # No citations
        elif citation_markers >= len(citations):
            return 5.0  # Good citation coverage
        elif citation_markers >= len(citations) // 2:
            return 4.0  # Adequate citations
        else:
            return 3.0  # Some citations
    
    async def _check_semantic_coherence(self, answer: str) -> float:
        """Check semantic coherence and clarity of the answer."""
        try:
            coherence_prompt = f"""
            Rate the semantic coherence and clarity of this answer on a scale of 1-5:
            
            Answer: {answer}
            
            Consider:
            - Logical flow and structure
            - Clarity of expression
            - Consistency of information
            - Overall readability
            
            Respond with just a number from 1-5 and a brief explanation.
            """
            
            messages = [{"role": "user", "content": coherence_prompt}]
            response = await self.llm.achat_completion(messages)
            
            # Extract score
            match = re.search(r'(\d+(?:\.\d+)?)', response.content)
            if match:
                return min(5.0, max(1.0, float(match.group(1))))
            
            return 3.0  # Default if parsing fails
            
        except Exception as e:
            logger.warning(f"Coherence check failed: {e}")
            return 3.0
    
    def _calculate_confidence(self, scores: Dict[VerificationCriterion, float]) -> float:
        """Calculate confidence in verification based on score consistency."""
        if not scores:
            return 0.5
        
        score_values = list(scores.values())
        if len(score_values) == 1:
            return 0.8
        
        # Calculate variance
        mean_score = sum(score_values) / len(score_values)
        variance = sum((score - mean_score) ** 2 for score in score_values) / len(score_values)
        
        # High variance = low confidence
        confidence = max(0.3, 1.0 - (variance / 2.0))
        
        return min(0.95, confidence)
    
    def _create_error_result(self, error_msg: str, level: VerificationLevel) -> VerificationResult:
        """Create error verification result."""
        return VerificationResult(
            overall_score=2.0,
            criteria_scores={VerificationCriterion.FAITHFULNESS: 2.0},
            passed=False,
            needs_refinement=True,
            issues=[f"Verification failed: {error_msg}"],
            suggestions=["Retry verification or use manual review"],
            verification_time_ms=0.0,
            verification_level=level,
            raw_response=f"Error: {error_msg}",
            confidence=0.1
        )
    
    def update_thresholds(
        self,
        pass_threshold: Optional[float] = None,
        criteria_threshold: Optional[float] = None,
        confidence_threshold: Optional[float] = None
    ):
        """Update verification thresholds."""
        if pass_threshold is not None:
            self.pass_threshold = pass_threshold
            logger.info(f"Updated pass threshold to {pass_threshold}")
        
        if criteria_threshold is not None:
            self.criteria_threshold = criteria_threshold
            logger.info(f"Updated criteria threshold to {criteria_threshold}")
        
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            logger.info(f"Updated confidence threshold to {confidence_threshold}")


# Convenience functions
async def quick_verify(
    query: str,
    answer: str,
    context: str,
    llm_config: Optional[LLMConfig] = None
) -> bool:
    """
    Quick verification check for simple pass/fail.
    
    Args:
        query: Original query
        answer: Generated answer
        context: Context documents
        llm_config: LLM configuration
        
    Returns:
        True if answer passes verification, False otherwise
    """
    verifier = Verifier(llm_config)
    result = await verifier.verify(query, answer, context, VerificationLevel.BASIC)
    return result.passed


async def detailed_verify(
    query: str,
    answer: str,
    context: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    llm_config: Optional[LLMConfig] = None
) -> VerificationResult:
    """
    Detailed verification with full scoring.
    
    Args:
        query: Original query
        answer: Generated answer
        context: Context documents
        citations: Citation information
        llm_config: LLM configuration
        
    Returns:
        Complete verification result
    """
    verifier = Verifier(llm_config)
    return await verifier.verify(
        query, answer, context, 
        VerificationLevel.STANDARD, 
        citations
    )


# Factory function
def create_verifier(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    **kwargs
) -> Verifier:
    """
    Factory function to create a verifier instance.
    
    Args:
        model: LLM model name
        api_key: OpenAI API key
        **kwargs: Additional LLM configuration
        
    Returns:
        Configured Verifier instance
    """
    llm_config = LLMConfig(
        model=model,
        api_key=api_key,
        **kwargs
    )
    
    return Verifier(llm_config)
