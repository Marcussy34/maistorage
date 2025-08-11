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

  const { message, agentic = false, top_k = 10 } = req.body

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
    // Forward request to FastAPI backend - use new streaming endpoint
    const endpoint = `${API_BASE_URL}/chat/stream${agentic ? '?agentic=true' : '?agentic=false'}`
    const backendResponse = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: message.trim(),
        top_k: top_k,
        enable_verification: true,
        max_refinements: 2,
        stream_traces: true,
        stream_tokens: false
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

    // Stream the response directly from FastAPI
    if (!backendResponse.body) {
      throw new Error('No response body from backend')
    }

    const reader = backendResponse.body.getReader()
    const decoder = new TextDecoder()

    try {
      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        // Decode and process the chunk
        const chunk = decoder.decode(value, { stream: true })
        
        // Handle Server-Sent Events format from FastAPI
        const lines = chunk.split('\n')
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonData = line.substring(6) // Remove "data: " prefix
            if (jsonData.trim()) {
              res.write(jsonData + '\n') // Convert to NDJSON format
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }

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
