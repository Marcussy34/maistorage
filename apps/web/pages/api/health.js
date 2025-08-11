/**
 * Health check endpoint for Next.js frontend.
 * 
 * Provides status information about the frontend application
 * and its connection to backend services.
 */

const RAG_API_URL = process.env.RAG_API_URL || 'http://localhost:8000';

export default async function handler(req, res) {
  const startTime = Date.now();
  
  try {
    // Check RAG API health
    let ragApiHealthy = true;
    let ragApiStatus = null;
    
    try {
      const ragResponse = await fetch(`${RAG_API_URL}/health`, {
        method: 'GET',
        timeout: 5000, // 5 second timeout
      });
      
      if (ragResponse.ok) {
        ragApiStatus = await ragResponse.json();
      } else {
        ragApiHealthy = false;
        ragApiStatus = { status: 'unhealthy', statusCode: ragResponse.status };
      }
    } catch (error) {
      ragApiHealthy = false;
      ragApiStatus = { status: 'unreachable', error: error.message };
    }
    
    const responseTime = Date.now() - startTime;
    const overallHealthy = ragApiHealthy;
    
    const health = {
      status: overallHealthy ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      version: '0.3.0',
      environment: process.env.NODE_ENV || 'development',
      
      // Service dependencies
      dependencies: {
        rag_api: {
          healthy: ragApiHealthy,
          url: RAG_API_URL,
          status: ragApiStatus
        }
      },
      
      // Performance metrics
      metrics: {
        response_time_ms: responseTime,
        uptime_seconds: process.uptime(),
        memory_usage: process.memoryUsage(),
        node_version: process.version
      }
    };
    
    // Return appropriate HTTP status
    const statusCode = overallHealthy ? 200 : 503;
    
    res.status(statusCode).json(health);
    
  } catch (error) {
    console.error('Health check failed:', error);
    
    res.status(500).json({
      status: 'error',
      timestamp: new Date().toISOString(),
      error: {
        code: 'HEALTH_CHECK_FAILED',
        message: 'Health check endpoint failed',
        details: error.message
      }
    });
  }
}
