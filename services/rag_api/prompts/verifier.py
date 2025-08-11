"""
Verifier prompt templates for validating and improving RAG responses.
"""

VERIFIER_SYSTEM_PROMPT = """You are a response quality verifier for a RAG system. Your job is to evaluate generated answers for accuracy, completeness, and proper citation.

Evaluation criteria:
1. **Faithfulness**: Is the answer supported by the provided context?
2. **Completeness**: Does the answer fully address the user's question?
3. **Citation Quality**: Are sources properly referenced?
4. **Clarity**: Is the response clear and well-structured?
5. **Relevance**: Does the answer stay on topic?

Scoring:
- Rate each criterion on a scale of 1-5 (5 = excellent)
- Provide an overall score (1-5)
- Identify specific issues that need improvement
- Suggest concrete improvements if score < 4

Response format:
- Overall Score: X/5
- Faithfulness: X/5 - [brief explanation]
- Completeness: X/5 - [brief explanation]  
- Citation Quality: X/5 - [brief explanation]
- Clarity: X/5 - [brief explanation]
- Relevance: X/5 - [brief explanation]
- Issues: [list major problems]
- Suggestions: [specific improvements needed]
- Verification: PASS/FAIL (pass if overall score >= 4)
"""

VERIFIER_USER_TEMPLATE = """Please evaluate this RAG response:

Original Query: {query}

Generated Answer: {answer}

Context Used:
{context}

Evaluate the answer quality and provide your assessment."""

def format_verifier_prompt(query: str, answer: str, context: str) -> list:
    """
    Format the verifier prompt for response evaluation.
    
    Args:
        query: Original user question
        answer: Generated response to evaluate
        context: Context documents used for generation
        
    Returns:
        List of message dictionaries for OpenAI chat completion
    """
    return [
        {
            "role": "system",
            "content": VERIFIER_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": VERIFIER_USER_TEMPLATE.format(
                query=query,
                answer=answer,
                context=context
            )
        }
    ]

# Simple faithfulness check
FAITHFULNESS_CHECK_PROMPT = """Given this context and generated answer, is the answer faithful to the context?

Context: {context}

Generated Answer: {answer}

Respond with:
- FAITHFUL: if the answer is fully supported by the context
- PARTIALLY_FAITHFUL: if mostly supported but has some unsupported claims
- UNFAITHFUL: if the answer contradicts or goes beyond the context
- NOT_ENOUGH_CONTEXT: if the context is insufficient to verify

Then provide a brief explanation (1-2 sentences)."""

def format_faithfulness_check(answer: str, context: str) -> list:
    """
    Format a simple faithfulness check prompt.
    
    Args:
        answer: Generated response to check
        context: Context documents used
        
    Returns:
        List of message dictionaries for OpenAI chat completion
    """
    return [
        {
            "role": "user",
            "content": FAITHFULNESS_CHECK_PROMPT.format(
                answer=answer,
                context=context
            )
        }
    ]
