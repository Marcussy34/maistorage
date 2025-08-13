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
    let sseBuffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        // Append decoded text to buffer (normalize CRLF to LF)
        sseBuffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')

        // SSE events are separated by a blank line (\n\n)
        const events = sseBuffer.split('\n\n')
        sseBuffer = events.pop() || '' // keep residual partial event

        for (const rawEvent of events) {
          if (!rawEvent) continue
          // Each event can have multiple data: lines; concatenate them
          const dataLines = []
          for (const line of rawEvent.split('\n')) {
            if (line.startsWith('data:')) {
              // Remove the leading 'data:' and optional space
              dataLines.push(line.slice(5).replace(/^\s/, ''))
            }
          }
          if (dataLines.length === 0) continue

          // Join all data lines; trim stray wrapping quotes if present
          let jsonLine = dataLines.join('')
          const trimmed = jsonLine.trim()
          const normalized = (trimmed.startsWith('"') && trimmed.endsWith('"'))
            ? trimmed.slice(1, -1)
            : trimmed

          if (normalized) {
            res.write(normalized + '\n')
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
