import React, { useEffect, useState, useRef } from 'react';
import Head from "next/head";
import Link from "next/link";
import { motion } from "framer-motion";
import { Button } from "../src/components/ui/button";
import { Badge } from "../src/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../src/components/ui/tabs";
import { 
  Database, 
  MessageSquare, 
  ArrowLeft, 
  Server, 
  Zap, 
  Search,
  FileText,
  Code,
  Play,
  CheckCircle,
  Moon,
  Sun
} from "lucide-react";

export default function ApiDocs() {
  const [darkMode, setDarkMode] = useState(false)
  const mountedRef = useRef(false)

  // Initialize dark mode from localStorage or system preference
  useEffect(() => {
    if (mountedRef.current) return // Prevent double execution in StrictMode
    mountedRef.current = true
    
    const savedDarkMode = localStorage.getItem('darkMode')
    const systemDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches
    
    const shouldUseDarkMode = savedDarkMode 
      ? JSON.parse(savedDarkMode) 
      : systemDarkMode

    setDarkMode(shouldUseDarkMode)
    updateDarkModeClass(shouldUseDarkMode)
  }, [])

  // Update dark mode class on document
  const updateDarkModeClass = (isDark) => {
    if (isDark) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }

  // Toggle dark mode
  const toggleDarkMode = () => {
    const newDarkMode = !darkMode
    setDarkMode(newDarkMode)
    localStorage.setItem('darkMode', JSON.stringify(newDarkMode))
    updateDarkModeClass(newDarkMode)
  }

  return (
    <>
      <Head>
        <title>API Documentation - MaiStorage RAG</title>
        <meta name="description" content="Complete API documentation for MaiStorage Agentic RAG System" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <div className="min-h-screen bg-background text-foreground">
        {/* Header */}
        <motion.header 
          className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <motion.div 
                className="flex items-center gap-3"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                <Database className="h-6 w-6 text-primary" />
                <div>
                  <h1 className="text-lg font-semibold">MaiStorage API</h1>
                  <p className="text-sm text-muted-foreground">Agentic RAG System Documentation</p>
                </div>
              </motion.div>
              
              <motion.div 
                className="flex items-center gap-2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                {/* Dark mode toggle */}
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={toggleDarkMode}
                  >
                    {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                  </Button>
                </motion.div>
                
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Link href="/">
                    <Button variant="outline" size="sm" className="gap-2">
                      <ArrowLeft className="h-4 w-4" />
                      Back to Home
                    </Button>
                  </Link>
                </motion.div>
              </motion.div>
            </div>
          </div>
        </motion.header>

        {/* Main Content */}
        <motion.main 
          className="container mx-auto px-4 py-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          {/* Overview Section */}
          <div className="text-center mb-12">
            <motion.div 
              className="flex justify-center mb-6"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <div className="p-3 bg-primary/10 rounded-full">
                <Server className="h-12 w-12 text-primary" />
              </div>
            </motion.div>
            
            <motion.h1 
              className="text-4xl font-bold tracking-tight mb-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              API Documentation
            </motion.h1>
            <motion.p 
              className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
            >
              Complete reference for the MaiStorage Agentic RAG API
            </motion.p>
            <motion.div 
              className="flex gap-4 justify-center mb-12"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <Badge variant="success" className="gap-1">
                <CheckCircle className="h-3 w-3" />
                API v1.0
              </Badge>
              <Badge variant="info">FastAPI</Badge>
              <Badge variant="outline">OpenAPI 3.0</Badge>
            </motion.div>
          </div>

          {/* Quick Links */}
          <motion.div 
            className="grid md:grid-cols-3 gap-6 mb-12"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.7 }}
          >
            <motion.div 
              className="text-center p-6 rounded-lg border bg-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.8 }}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
            >
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Play className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Quick Start</h3>
              <p className="text-muted-foreground">
                Get started with basic API usage and authentication
              </p>
            </motion.div>

            <motion.div 
              className="text-center p-6 rounded-lg border bg-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.9 }}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
            >
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <MessageSquare className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Chat Endpoints</h3>
              <p className="text-muted-foreground">
                Streaming chat and Q&A functionality
              </p>
            </motion.div>

            <motion.div 
              className="text-center p-6 rounded-lg border bg-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.0 }}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
            >
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Search className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Search API</h3>
              <p className="text-muted-foreground">
                Hybrid retrieval and document search
              </p>
            </motion.div>
          </motion.div>

          {/* API Documentation Tabs */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.1 }}
          >
            <Tabs defaultValue="endpoints" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="endpoints">Endpoints</TabsTrigger>
              <TabsTrigger value="authentication">Auth</TabsTrigger>
              <TabsTrigger value="examples">Examples</TabsTrigger>
              <TabsTrigger value="schemas">Schemas</TabsTrigger>
            </TabsList>

            {/* Endpoints Tab */}
            <TabsContent value="endpoints" className="space-y-6">
              <div className="grid gap-6">
                {/* Chat Endpoints */}
                <div className="rounded-lg border bg-card p-6">
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold flex items-center gap-2 mb-2">
                      <MessageSquare className="h-5 w-5" />
                      Chat Endpoints
                    </h3>
                    <p className="text-muted-foreground">
                      Real-time chat and Q&A with streaming responses
                    </p>
                  </div>
                  <div className="space-y-4">
                    <div className="border rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="success">POST</Badge>
                        <code className="text-sm font-mono">/api/chat/stream</code>
                      </div>
                      <p className="text-sm text-muted-foreground mb-3">
                        Send a question and receive streaming responses with citations
                      </p>
                      <div className="bg-muted rounded p-3">
                        <pre className="text-sm"><code>{`{
  "query": "What is the main topic?",
  "agentic_mode": false,
  "context_limit": 5
}`}</code></pre>
                      </div>
                    </div>

                    <div className="border rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="info">GET</Badge>
                        <code className="text-sm font-mono">/api/health</code>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Check API health and system status
                      </p>
                    </div>
                  </div>
                </div>

                {/* Search Endpoints */}
                <div className="rounded-lg border bg-card p-6">
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold flex items-center gap-2 mb-2">
                      <Search className="h-5 w-5" />
                      Search Endpoints
                    </h3>
                    <p className="text-muted-foreground">
                      Document retrieval and hybrid search functionality
                    </p>
                  </div>
                  <div className="space-y-4">
                    <div className="border rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="success">POST</Badge>
                        <code className="text-sm font-mono">/api/search</code>
                      </div>
                      <p className="text-sm text-muted-foreground mb-3">
                        Perform hybrid search across documents
                      </p>
                      <div className="bg-muted rounded p-3">
                        <pre className="text-sm"><code>{`{
  "query": "search terms",
  "top_k": 10,
  "rerank": true
}`}</code></pre>
                      </div>
                    </div>

                    <div className="border rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="success">POST</Badge>
                        <code className="text-sm font-mono">/api/documents/ingest</code>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Upload and index new documents
                      </p>
                    </div>
                  </div>
                </div>

                {/* Evaluation Endpoints */}
                <div className="rounded-lg border bg-card p-6">
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold flex items-center gap-2 mb-2">
                      <Zap className="h-5 w-5" />
                      Evaluation Endpoints
                    </h3>
                    <p className="text-muted-foreground">
                      System evaluation and metrics
                    </p>
                  </div>
                  <div className="space-y-4">
                    <div className="border rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="success">POST</Badge>
                        <code className="text-sm font-mono">/api/eval/run</code>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Run evaluation on test dataset
                      </p>
                    </div>

                    <div className="border rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="info">GET</Badge>
                        <code className="text-sm font-mono">/api/eval/results</code>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Get latest evaluation results
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Authentication Tab */}
            <TabsContent value="authentication" className="space-y-6">
              <div className="rounded-lg border bg-card p-6">
                <div className="mb-6">
                  <h3 className="text-lg font-semibold mb-2">Authentication</h3>
                  <p className="text-muted-foreground">
                    Currently, the API is in development mode with no authentication required
                  </p>
                </div>
                <div>
                  <div className="space-y-4">
                    <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="warning">Development Mode</Badge>
                      </div>
                      <p className="text-sm">
                        The API is currently running in development mode. All endpoints are publicly accessible.
                        Authentication will be added in future phases.
                      </p>
                    </div>

                    <div className="border rounded-lg p-4">
                      <h4 className="font-semibold mb-2">Base URL</h4>
                      <code className="bg-muted px-2 py-1 rounded text-sm">http://localhost:8000</code>
                    </div>

                    <div className="border rounded-lg p-4">
                      <h4 className="font-semibold mb-2">Headers</h4>
                      <div className="bg-muted rounded p-3">
                        <pre className="text-sm"><code>{`Content-Type: application/json
Accept: application/json`}</code></pre>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Examples Tab */}
            <TabsContent value="examples" className="space-y-6">
              <div className="rounded-lg border bg-card p-6">
                <div className="mb-6">
                  <h3 className="text-lg font-semibold mb-2">Usage Examples</h3>
                  <p className="text-muted-foreground">
                    Common API usage patterns and code examples
                  </p>
                </div>
                <div className="space-y-6">
                  {/* Streaming Chat Example */}
                  <div>
                    <h4 className="font-semibold mb-3 flex items-center gap-2">
                      <Code className="h-4 w-4" />
                      Streaming Chat
                    </h4>
                    <div className="bg-muted rounded-lg p-4">
                      <pre className="text-sm overflow-x-auto"><code>{`// JavaScript/Node.js example
const response = await fetch('http://localhost:8000/api/chat/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: "What are the main features of this system?",
    agentic_mode: false
  })
});

const reader = response.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = new TextDecoder().decode(value);
  const lines = chunk.split('\\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      console.log(data);
    }
  }
}`}</code></pre>
                    </div>
                  </div>

                  {/* Search Example */}
                  <div>
                    <h4 className="font-semibold mb-3 flex items-center gap-2">
                      <Search className="h-4 w-4" />
                      Document Search
                    </h4>
                    <div className="bg-muted rounded-lg p-4">
                      <pre className="text-sm overflow-x-auto"><code>{`// Python example
import requests

response = requests.post('http://localhost:8000/api/search', json={
    "query": "machine learning algorithms",
    "top_k": 5,
    "rerank": True
})

results = response.json()
for result in results['documents']:
    print(f"Score: {result['score']}")
    print(f"Content: {result['content']}")
    print(f"Source: {result['metadata']['source']}")
    print("---")`}</code></pre>
                    </div>
                  </div>

                  {/* cURL Example */}
                  <div>
                    <h4 className="font-semibold mb-3">cURL Examples</h4>
                    <div className="bg-muted rounded-lg p-4">
                      <pre className="text-sm overflow-x-auto"><code>{`# Health check
curl -X GET http://localhost:8000/api/health

# Chat query
curl -X POST http://localhost:8000/api/chat/stream \\
  -H "Content-Type: application/json" \\
  -d '{"query": "How does the system work?", "agentic_mode": false}'

# Search documents
curl -X POST http://localhost:8000/api/search \\
  -H "Content-Type: application/json" \\
  -d '{"query": "search terms", "top_k": 10}'`}</code></pre>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Schemas Tab */}
            <TabsContent value="schemas" className="space-y-6">
              <div className="rounded-lg border bg-card p-6">
                <div className="mb-6">
                  <h3 className="text-lg font-semibold flex items-center gap-2 mb-2">
                    <FileText className="h-5 w-5" />
                    Response Schemas
                  </h3>
                  <p className="text-muted-foreground">
                    Data structures returned by the API
                  </p>
                </div>
                <div className="space-y-6">
                  {/* Chat Response Schema */}
                  <div>
                    <h4 className="font-semibold mb-3">Chat Stream Response</h4>
                    <div className="bg-muted rounded-lg p-4">
                      <pre className="text-sm overflow-x-auto"><code>{`{
  "type": "token" | "citation" | "metrics" | "error",
  "content": "string",
  "citation": {
    "chunk_id": "string",
    "source": "string", 
    "page": "number",
    "relevance_score": "number",
    "content": "string"
  },
  "metrics": {
    "total_time_ms": "number",
    "total_tokens": "number",
    "retrieval_time_ms": "number",
    "context_chunks": "number"
  }
}`}</code></pre>
                    </div>
                  </div>

                  {/* Search Response Schema */}
                  <div>
                    <h4 className="font-semibold mb-3">Search Response</h4>
                    <div className="bg-muted rounded-lg p-4">
                      <pre className="text-sm overflow-x-auto"><code>{`{
  "documents": [
    {
      "id": "string",
      "content": "string",
      "score": "number",
      "metadata": {
        "source": "string",
        "page": "number",
        "chunk_index": "number"
      }
    }
  ],
  "total_time_ms": "number",
  "retrieval_method": "hybrid" | "dense" | "sparse"
}`}</code></pre>
                    </div>
                  </div>

                  {/* Error Response Schema */}
                  <div>
                    <h4 className="font-semibold mb-3">Error Response</h4>
                    <div className="bg-muted rounded-lg p-4">
                      <pre className="text-sm overflow-x-auto"><code>{`{
  "error": {
    "code": "string",
    "message": "string",
    "details": "string"
  },
  "timestamp": "string"
}`}</code></pre>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
          </motion.div>

          {/* Interactive Demo Section */}
          <motion.div 
            className="mt-12 rounded-lg border bg-card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.2 }}
          >
            <div className="text-center mb-6">
              <h3 className="text-lg font-semibold mb-2">Try the API</h3>
              <p className="text-muted-foreground">
                Test the API endpoints directly from your browser
              </p>
            </div>
            <div className="text-center">
              <div className="space-y-4">
                <p className="text-muted-foreground">
                  Use our interactive chat interface to test the streaming API in real-time.
                </p>
                <div className="flex gap-4 justify-center">
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Link href="/chat">
                      <Button size="lg" className="gap-2">
                        <MessageSquare className="h-5 w-5" />
                        Try Chat Interface
                      </Button>
                    </Link>
                  </motion.div>
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Button variant="outline" size="lg" asChild>
                      <a href="https://maistorage-backend-mytfom2tca-as.a.run.app/docs" target="_blank" rel="noopener noreferrer">
                        <Server className="h-5 w-5 mr-2" />
                        Interactive API Docs
                      </a>
                    </Button>
                  </motion.div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.main>

        {/* Footer */}
        <motion.footer 
          className="border-t py-6 mt-12 text-center text-sm text-muted-foreground"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 1.3 }}
        >
          <p>
            MaiStorage API Documentation - Built with FastAPI and OpenAPI 3.0
          </p>
        </motion.footer>
      </div>
    </>
  );
}