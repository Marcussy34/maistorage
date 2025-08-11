"""
Baseline RAG prompt templates for traditional single-pass retrieval.
"""

BASELINE_SYSTEM_PROMPT = """You are an AI assistant that provides accurate and helpful answers based on the provided context documents. 

Key guidelines:
- Answer questions using ONLY the information provided in the context documents
- If the context doesn't contain enough information to answer the question, say so clearly
- Cite your sources by referencing the document chunks used
- Be concise but thorough in your responses
- Maintain a professional and helpful tone
- Do not hallucinate or add information not present in the context

When citing sources, use this format: [Source: doc_name, chunk_index]
"""

BASELINE_USER_TEMPLATE = """Context Documents:
{context}

Question: {query}

Please provide a helpful answer based on the context provided above. Remember to cite your sources."""

def format_baseline_prompt(query: str, context: str) -> list:
    """
    Format the baseline RAG prompt with query and retrieved context.
    
    Args:
        query: User question
        context: Retrieved document context
        
    Returns:
        List of message dictionaries for OpenAI chat completion
    """
    return [
        {
            "role": "system",
            "content": BASELINE_SYSTEM_PROMPT
        },
        {
            "role": "user", 
            "content": BASELINE_USER_TEMPLATE.format(
                context=context,
                query=query
            )
        }
    ]


def format_context_from_results(results: list) -> str:
    """
    Format retrieval results into context string for the prompt.
    
    Args:
        results: List of RetrievalResult objects
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, result in enumerate(results, 1):
        doc = result.document
        context_part = f"""Document {i}:
Title: {doc.doc_name or 'Unknown'}
Content: {doc.text}
Chunk: {doc.chunk_index or 'N/A'}/{doc.total_chunks or 'N/A'}
---"""
        context_parts.append(context_part)
    
    return "\n\n".join(context_parts)
