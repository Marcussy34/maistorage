import React, { useState, useCallback, useRef, useEffect } from 'react'
import { Button } from './ui/button'
import { RefreshCw, Copy, AlertCircle } from 'lucide-react'

/**
 * Parses NDJSON stream data line by line
 * Each line should be a valid JSON object
 */
function parseNDJSONLine(line) {
  try {
    return JSON.parse(line.trim())
  } catch (error) {
    console.warn('Failed to parse NDJSON line:', line, error)
    return null
  }
}

/**
 * ChatStream component for displaying streaming chat responses
 * Handles NDJSON parsing and real-time token display
 */
export function ChatStream({ messages, onRetry, isLoading }) {
  const messagesEndRef = useRef(null)
  const [copiedIndex, setCopiedIndex] = useState(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Copy message to clipboard
  const handleCopy = useCallback(async (content, index) => {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedIndex(index)
      setTimeout(() => setCopiedIndex(null), 2000)
    } catch (error) {
      console.error('Failed to copy to clipboard:', error)
    }
  }, [])

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.length === 0 && !isLoading && (
        <div className="text-center text-muted-foreground py-12">
          <div className="text-lg font-medium mb-2">Welcome to MAI Storage RAG</div>
          <div>Ask questions about your documents to get started.</div>
        </div>
      )}

      {messages.map((msg, index) => (
        <div key={index} className="space-y-2">
          {/* User Message */}
          {msg.type === 'user' && (
            <div className="flex justify-end">
              <div className="bg-primary text-primary-foreground px-4 py-2 rounded-lg max-w-[80%]">
                {msg.content}
              </div>
            </div>
          )}

          {/* Assistant Message */}
          {msg.type === 'assistant' && (
            <div className="flex justify-start">
              <div className="bg-muted px-4 py-2 rounded-lg max-w-[80%] relative group">
                <div className="whitespace-pre-wrap">{msg.content}</div>
                
                {/* Copy button */}
                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => handleCopy(msg.content, index)}
                  >
                    <Copy className="h-3 w-3" />
                  </Button>
                </div>

                {/* Copy confirmation */}
                {copiedIndex === index && (
                  <div className="absolute top-8 right-2 bg-background border rounded px-2 py-1 text-xs text-muted-foreground">
                    Copied!
                  </div>
                )}

                {/* Citations if available */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-border/50">
                    <div className="text-xs text-muted-foreground mb-2">Sources:</div>
                    <div className="space-y-1">
                      {msg.citations.map((citation, citIndex) => (
                        <div key={citIndex} className="text-xs bg-background/50 rounded px-2 py-1">
                          <div className="font-medium">{citation.title || `Source ${citIndex + 1}`}</div>
                          {citation.content && (
                            <div className="text-muted-foreground mt-1">
                              &quot;{citation.content.substring(0, 150)}...&quot;
                            </div>
                          )}
                          {citation.score && (
                            <div className="text-muted-foreground">Score: {citation.score.toFixed(3)}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Metrics if available */}
                {msg.metrics && (
                  <div className="mt-2 text-xs text-muted-foreground">
                    {msg.metrics.retrieval_time_ms && 
                      `Retrieval: ${msg.metrics.retrieval_time_ms}ms`}
                    {msg.metrics.llm_time_ms && 
                      ` • LLM: ${msg.metrics.llm_time_ms}ms`}
                    {msg.metrics.total_tokens && 
                      ` • Tokens: ${msg.metrics.total_tokens}`}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error Message */}
          {msg.type === 'error' && (
            <div className="flex justify-start">
              <div className="bg-destructive/10 border border-destructive/20 text-destructive px-4 py-2 rounded-lg max-w-[80%]">
                <div className="flex items-center gap-2 mb-2">
                  <AlertCircle className="h-4 w-4" />
                  <span className="font-medium">Error</span>
                </div>
                <div className="text-sm">{msg.content}</div>
                {onRetry && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onRetry(msg.originalQuery)}
                    className="mt-2"
                  >
                    <RefreshCw className="h-3 w-3 mr-1" />
                    Retry
                  </Button>
                )}
              </div>
            </div>
          )}

          {/* Loading indicator */}
          {msg.type === 'loading' && (
            <div className="flex justify-start">
              <div className="bg-muted px-4 py-2 rounded-lg">
                <div className="flex items-center gap-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                    <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"></div>
                  </div>
                  <span className="text-sm text-muted-foreground">Thinking...</span>
                </div>
              </div>
            </div>
          )}
        </div>
      ))}

      <div ref={messagesEndRef} />
    </div>
  )
}

/**
 * Hook for handling NDJSON streaming responses
 * Manages the streaming lifecycle and message parsing
 */
export function useStreamingChat() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const abortControllerRef = useRef(null)

  const sendMessage = useCallback(async (query) => {
    if (isLoading) return

    // Add user message immediately
    const userMessage = { type: 'user', content: query, id: Date.now() }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    // Add loading indicator
    const loadingMessage = { type: 'loading', content: '', id: Date.now() + 1 }
    setMessages(prev => [...prev, loadingMessage])

    try {
      // Create abort controller for this request
      abortControllerRef.current = new AbortController()

      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: query }),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      // Remove loading indicator and prepare for assistant response
      setMessages(prev => prev.slice(0, -1))
      
      let assistantMessage = {
        type: 'assistant',
        content: '',
        citations: null,
        metrics: null,
        id: Date.now() + 2
      }

      setMessages(prev => [...prev, assistantMessage])

      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        // Decode the chunk and add to buffer
        buffer += decoder.decode(value, { stream: true })
        
        // Process complete lines
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer

        for (const line of lines) {
          if (!line.trim()) continue

          const data = parseNDJSONLine(line)
          if (!data) continue

          // Handle different types of streaming events
          if (data.type === 'token') {
            // Append token to current assistant message
            setMessages(prev => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage.type === 'assistant') {
                lastMessage.content += data.content
              }
              return newMessages
            })
          } else if (data.type === 'sources') {
            // Add citations to current assistant message
            setMessages(prev => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage.type === 'assistant') {
                lastMessage.citations = data.citations
              }
              return newMessages
            })
          } else if (data.type === 'metrics') {
            // Add metrics to current assistant message
            setMessages(prev => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage.type === 'assistant') {
                lastMessage.metrics = data.metrics
              }
              return newMessages
            })
          } else if (data.type === 'done') {
            // Streaming complete
            break
          }
        }
      }

    } catch (error) {
      console.error('Streaming error:', error)
      
      // Remove loading message and add error message
      setMessages(prev => {
        const newMessages = prev.slice(0, -1)
        return [...newMessages, {
          type: 'error',
          content: error.name === 'AbortError' 
            ? 'Request was cancelled' 
            : `Failed to get response: ${error.message}`,
          originalQuery: query
        }]
      })
    } finally {
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }, [isLoading])

  const retryMessage = useCallback((query) => {
    sendMessage(query)
  }, [sendMessage])

  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return {
    messages,
    isLoading,
    sendMessage,
    retryMessage,
    stopGeneration,
    clearMessages
  }
}
