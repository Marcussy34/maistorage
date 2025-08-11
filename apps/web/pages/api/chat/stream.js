/**
 * Next.js API route for streaming chat responses
 * Proxies requests to the FastAPI backend and streams NDJSON responses
 */

const API_BASE_URL = process.env.RAG_API_URL || 'http://localhost:8000'

export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const { message } = req.body

  // Validate request body
  if (!message || typeof message !== 'string' || !message.trim()) {
    return res.status(400).json({ error: 'Message is required' })
  }

  // Set up streaming response headers
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST',
    'Access-Control-Allow-Headers': 'Content-Type',
  })

  try {
    // Forward request to FastAPI backend
    const backendResponse = await fetch(`${API_BASE_URL}/rag`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: message.trim()
      })
    })

    if (!backendResponse.ok) {
      console.error(`Backend API error: ${backendResponse.status} ${backendResponse.statusText}`)
      
      // Send error as NDJSON
      const errorData = {
        type: 'error',
        error: `Backend API error: ${backendResponse.status} ${backendResponse.statusText}`
      }
      res.write(JSON.stringify(errorData) + '\n')
      res.write(JSON.stringify({ type: 'done' }) + '\n')
      return res.end()
    }

    // Parse the response from FastAPI (non-streaming baseline RAG)
    const data = await backendResponse.json()

    // Convert the response to streaming format
    // Stream the tokens one by one to simulate streaming
    if (data.answer) {
      const words = data.answer.split(' ')
      
      for (let i = 0; i < words.length; i++) {
        const tokenData = {
          type: 'token',
          content: (i === 0 ? '' : ' ') + words[i]
        }
        res.write(JSON.stringify(tokenData) + '\n')
        
        // Add small delay to simulate realistic streaming
        await new Promise(resolve => setTimeout(resolve, 50))
      }
    }

    // Send citations if available
    if (data.citations && data.citations.length > 0) {
      const citationsData = {
        type: 'sources',
        citations: data.citations.map(citation => ({
          title: citation.metadata?.title || citation.metadata?.doc_id || 'Unknown Source',
          content: citation.content,
          score: citation.score,
          metadata: citation.metadata
        }))
      }
      res.write(JSON.stringify(citationsData) + '\n')
    }

    // Send metrics if available
    if (data.metrics) {
      const metricsData = {
        type: 'metrics',
        metrics: {
          retrieval_time_ms: data.metrics.retrieval_time_ms,
          llm_time_ms: data.metrics.llm_time_ms,
          total_tokens: data.metrics.total_tokens,
          total_time_ms: data.metrics.total_time_ms
        }
      }
      res.write(JSON.stringify(metricsData) + '\n')
    }

    // Send completion signal
    res.write(JSON.stringify({ type: 'done' }) + '\n')
    res.end()

  } catch (error) {
    console.error('API proxy error:', error)
    
    // Send error as NDJSON
    const errorData = {
      type: 'error',
      error: `Connection failed: ${error.message}`
    }
    res.write(JSON.stringify(errorData) + '\n')
    res.write(JSON.stringify({ type: 'done' }) + '\n')
    res.end()
  }
}

// Disable body parser to handle streaming
export const config = {
  api: {
    responseLimit: false, // Disable response size limit for streaming
  },
}
