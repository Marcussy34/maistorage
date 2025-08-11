// Next.js API route for comparing evaluations
// Proxies requests to the FastAPI backend

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const RAG_API_URL = process.env.RAG_API_URL || 'http://localhost:8000'
    
    const { traditional_file, agentic_file } = req.query
    const queryParams = new URLSearchParams()
    
    if (traditional_file) queryParams.append('traditional_file', traditional_file)
    if (agentic_file) queryParams.append('agentic_file', agentic_file)
    
    const queryString = queryParams.toString()
    const url = `${RAG_API_URL}/eval/compare${queryString ? '?' + queryString : ''}`
    
    const response = await fetch(url)
    const data = await response.json()

    if (!response.ok) {
      return res.status(response.status).json(data)
    }

    res.status(200).json(data)
  } catch (error) {
    console.error('Evaluation comparison proxy error:', error)
    res.status(500).json({ 
      error: 'Failed to connect to evaluation service',
      message: error.message 
    })
  }
}
