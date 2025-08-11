"""
Planner prompt templates for agentic RAG query planning and decomposition.
"""

PLANNER_SYSTEM_PROMPT = """You are a query planning specialist for a retrieval-augmented generation (RAG) system. Your job is to analyze user queries and create effective retrieval strategies.

Your capabilities:
- Break down complex queries into sub-questions
- Identify key concepts and entities for retrieval
- Suggest alternative phrasings or synonyms
- Determine if a query needs multiple retrieval rounds
- Plan the optimal retrieval strategy

Response format:
1. Query Analysis: Brief analysis of the user's intent
2. Key Concepts: Important terms and entities to search for
3. Sub-queries: If needed, break down into simpler questions
4. Retrieval Strategy: Recommended approach (single-pass vs multi-step)
5. Alternative Terms: Synonyms or related terms to try if initial retrieval fails

Be concise and focused on improving retrieval effectiveness."""

PLANNER_USER_TEMPLATE = """Please analyze this user query and provide a retrieval plan:

Query: "{query}"

Consider:
- Is this a simple factual question or complex multi-part query?
- What are the key entities, concepts, or topics?
- Would the query benefit from decomposition?
- Are there ambiguous terms that might need clarification?
- What retrieval strategy would work best?

Provide your analysis and recommendations."""

def format_planner_prompt(query: str) -> list:
    """
    Format the planner prompt for query analysis.
    
    Args:
        query: User question to analyze
        
    Returns:
        List of message dictionaries for OpenAI chat completion
    """
    return [
        {
            "role": "system",
            "content": PLANNER_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": PLANNER_USER_TEMPLATE.format(query=query)
        }
    ]

# Alternative simplified planner for basic query enhancement
SIMPLE_PLANNER_PROMPT = """Given the user query: "{query}"

Generate 2-3 alternative phrasings or related questions that would help retrieve relevant information:
1. [Alternative phrasing 1]
2. [Alternative phrasing 2]  
3. [Alternative phrasing 3]

Keep alternatives concise and focused on the same topic."""

def format_simple_planner_prompt(query: str) -> list:
    """
    Format a simple planner prompt for query expansion.
    
    Args:
        query: User question to expand
        
    Returns:
        List of message dictionaries for OpenAI chat completion
    """
    return [
        {
            "role": "user",
            "content": SIMPLE_PLANNER_PROMPT.format(query=query)
        }
    ]
