import React, { useState, useCallback, useRef, useEffect } from 'react'
import { Button } from './ui/button'
import { RefreshCw, Copy, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react'
import { SourceBadges, TextWithCitations } from './SourceBadge'
import { AgentTrace, CompactAgentTrace } from './AgentTrace'
import { ContextPanel, CompactContextPanel } from './ContextPanel'
import { MetricsChips, CompactMetrics } from './MetricsChips'

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
export function ChatStream({ messages, onRetry, isLoading, ragMode = false }) {
  const messagesEndRef = useRef(null)
  const [copiedIndex, setCopiedIndex] = useState(null)
  const [expandedSources, setExpandedSources] = useState(new Set())
  const [expandedTrace, setExpandedTrace] = useState(new Set())
  
  const toggleSourcesExpanded = (messageId) => {
    setExpandedSources(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
      } else {
        newSet.add(messageId)
      }
      return newSet
    })
  }
  
  const toggleTraceExpanded = (messageId) => {
    setExpandedTrace(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
      } else {
        newSet.add(messageId)
      }
      return newSet
    })
  }

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
        <div key={msg.id || index} className="space-y-2">
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
              <div className="bg-muted px-4 py-2 rounded-lg max-w-[80%] relative group space-y-3">
                {/* Main content with citations */}
                <div>
                  <TextWithCitations
                    text={msg.content}
                    citations={msg.citations}
                    sentenceAttributions={msg.sentence_attribution}
                    onSourceClick={(citation) => {
                      console.log('Source clicked:', citation)
                    }}
                  />
                </div>
                
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

                {/* Agent Trace for agentic responses */}
                {msg.traceEvents && msg.traceEvents.length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-muted-foreground">Agent Trace</span>
                        <CompactAgentTrace
                          traceEvents={msg.traceEvents}
                          totalTime={msg.metrics?.total_time_ms}
                        />
                      </div>
                      
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => toggleTraceExpanded(msg.id)}
                        className="h-6 text-xs"
                      >
                        {expandedTrace.has(msg.id) ? (
                          <ChevronUp className="h-3 w-3" />
                        ) : (
                          <ChevronDown className="h-3 w-3" />
                        )}
                      </Button>
                    </div>
                    
                    {expandedTrace.has(msg.id) && (
                      <AgentTrace
                        traceEvents={msg.traceEvents}
                        totalTime={msg.metrics?.total_time_ms}
                        refinementCount={msg.refinementCount}
                        className="mt-2"
                      />
                    )}
                  </div>
                )}

                {/* Context Panel for sources */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-muted-foreground">Sources</span>
                        <CompactContextPanel sources={msg.sources} />
                      </div>
                      
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => toggleSourcesExpanded(msg.id)}
                        className="h-6 text-xs"
                      >
                        {expandedSources.has(msg.id) ? (
                          <ChevronUp className="h-3 w-3" />
                        ) : (
                          <ChevronDown className="h-3 w-3" />
                        )}
                      </Button>
                    </div>
                    
                    {expandedSources.has(msg.id) && (
                      <ContextPanel
                        sources={msg.sources}
                        query={msg.originalQuery}
                        retrievalTime={msg.metrics?.retrieval_time_ms}
                        chunksRetrieved={msg.metrics?.chunks_retrieved}
                        className="mt-2"
                      />
                    )}
                  </div>
                )}

                {/* Metrics display */}
                {msg.metrics && (
                  <MetricsChips 
                    metrics={msg.metrics}
                    layout="inline"
                    showLabels={false}
                    className="text-xs"
                  />
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
  const requestInProgressRef = useRef(false)

  const sendMessage = useCallback(async (query, ragMode = false) => {
    if (isLoading || requestInProgressRef.current) return
    
    console.log('Sending message:', query, 'RAG Mode:', ragMode)
    requestInProgressRef.current = true
    
    // Prevent duplicate requests by aborting any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

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
        body: JSON.stringify({ 
          message: query,
          agentic: ragMode,
          top_k: 10
        }),
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
        sources: null,
        metrics: null,
        traceEvents: [],
        sentence_attribution: null,
        refinementCount: 0,
        originalQuery: query,
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
          console.log('Received streaming data:', data)
          
          if (data.type === 'token') {
            // Append token to current assistant message
            setMessages(prev => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage && lastMessage.type === 'assistant') {
                newMessages[newMessages.length - 1] = {
                  ...lastMessage,
                  content: lastMessage.content + (data.content || '')
                }
              }
              return newMessages
            })
          } else if (data.type === 'step_start' || data.type === 'step_complete' || data.type === 'verification') {
            // Add trace events for agentic workflow
            setMessages(prev => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage && lastMessage.type === 'assistant') {
                newMessages[newMessages.length - 1] = {
                  ...lastMessage,
                  traceEvents: [...lastMessage.traceEvents, data]
                }
              }
              return newMessages
            })
          } else if (data.type === 'sources') {
            // Add sources/citations to current assistant message
            setMessages(prev => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage && lastMessage.type === 'assistant') {
                newMessages[newMessages.length - 1] = {
                  ...lastMessage,
                  citations: data.citations,
                  sources: data.data?.sources || data.citations
                }
              }
              return newMessages
            })
          } else if (data.type === 'answer') {
            // Handle complete answer with all metadata
            setMessages(prev => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage && lastMessage.type === 'assistant') {
                newMessages[newMessages.length - 1] = {
                  ...lastMessage,
                  content: data.content || lastMessage.content,
                  citations: data.citations || lastMessage.citations,
                  sources: data.citations || lastMessage.sources,
                  sentence_attribution: data.sentence_attribution,
                  metrics: data.metadata || lastMessage.metrics,
                  refinementCount: data.metadata?.refinement_count || 0
                }
              }
              return newMessages
            })
          } else if (data.type === 'metrics') {
            // Add metrics to current assistant message
            setMessages(prev => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage && lastMessage.type === 'assistant') {
                newMessages[newMessages.length - 1] = {
                  ...lastMessage,
                  metrics: data.metrics
                }
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
          originalQuery: query,
          id: Date.now() + 3
        }]
      })
    } finally {
      setIsLoading(false)
      requestInProgressRef.current = false
      abortControllerRef.current = null
    }
  }, [isLoading])

  const retryMessage = useCallback((query, ragMode = false) => {
    sendMessage(query, ragMode)
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
