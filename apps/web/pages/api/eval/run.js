// Next.js API route for running evaluations
// Proxies requests to the FastAPI backend

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const RAG_API_URL = process.env.RAG_API_URL || 'http://localhost:8000'
    
    const response = await fetch(`${RAG_API_URL}/eval/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body)
    })

    const data = await response.json()

    if (!response.ok) {
      return res.status(response.status).json(data)
    }

    res.status(200).json(data)
  } catch (error) {
    console.error('Evaluation proxy error:', error)
    res.status(500).json({ 
      error: 'Failed to connect to evaluation service',
      message: error.message 
    })
  }
}
