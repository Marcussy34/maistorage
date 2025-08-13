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
    const trimmedLine = line.trim()
    if (!trimmedLine) return null
    
    return JSON.parse(trimmedLine)
  } catch (error) {
    console.warn('Failed to parse NDJSON line:', {
      line: line.substring(0, 200) + (line.length > 200 ? '...' : ''),
      length: line.length,
      error: error.message
    })
    
    // Try to handle incomplete JSON by checking if it looks like it was truncated
    if (error.message.includes('Unterminated string') || error.message.includes('Unexpected end')) {
      console.warn('JSON appears to be truncated, this might be a streaming issue')
    }
    
    return null
  }
}

// Best-effort extraction of an answer string from any event shape
function extractAnswerFromEvent(evt) {
  try {
    const candidates = []
    if (!evt || typeof evt !== 'object') return ''

    // Direct fields
    if (typeof evt.content === 'string') candidates.push(evt.content)
    if (typeof evt.answer === 'string') candidates.push(evt.answer)

    // Nested common containers
    const d = evt.data || {}
    if (typeof d.answer === 'string') candidates.push(d.answer)
    if (typeof d.final_answer === 'string') candidates.push(d.final_answer)
    if (typeof d.output === 'string') candidates.push(d.output)
    if (typeof d.message === 'string') candidates.push(d.message)
    if (typeof d.result === 'string') candidates.push(d.result)

    // Pick the longest non-empty string to be safe
    const nonEmpty = candidates.filter(s => typeof s === 'string' && s.trim().length > 0)
    if (nonEmpty.length === 0) return ''
    return nonEmpty.sort((a, b) => b.length - a.length)[0]
  } catch {
    return ''
  }
}

// Dedupe helper for sources/citations while preserving order
function dedupeSources(list) {
  if (!Array.isArray(list)) return list
  const seen = new Set()
  const keyOf = (s) => {
    const name = (s.doc_name || '').toLowerCase().trim()
    const idx = s.chunk_index ?? s.chunkIndex ?? 0
    const snippet = (s.text_snippet || s.snippet || '').toLowerCase().trim()
    return `${name}#${idx}#${snippet}`
  }
  return list.filter((s) => {
    const k = keyOf(s)
    if (seen.has(k)) return false
    seen.add(k)
    return true
  })
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
          <div className="text-lg font-medium mb-2">Welcome to MaiStorage RAG</div>
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

      // Keep loading indicator until we start receiving content
      // We'll remove it when we get the first token or other content
      
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

      let assistantMessageAdded = false

      let buffer = ''

      // Track the latest finalized payload so we can commit it on 'done'
      // This prevents an empty message when only step events were streamed
      // and the final answer arrived just before completion.
      let latestAnswer = ''
      let latestCitations = null
      let latestSources = null
      let latestMetrics = null
      let latestSentenceAttribution = null
      let latestRefinementCount = 0

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
            // On first token, remove loading indicator and add assistant message
            if (!assistantMessageAdded) {
              setMessages(prev => {
                // Remove loading message and add assistant message
                const newMessages = prev.slice(0, -1)
                return [...newMessages, assistantMessage]
              })
              assistantMessageAdded = true
            }
            
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
            // On first event, remove loading indicator and add assistant message if not already done
            if (!assistantMessageAdded) {
              setMessages(prev => {
                // Remove loading message and add assistant message
                const newMessages = prev.slice(0, -1)
                return [...newMessages, assistantMessage]
              })
              assistantMessageAdded = true
            }
            
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
            // On first event, remove loading indicator and add assistant message if not already done
            if (!assistantMessageAdded) {
              setMessages(prev => {
                // Remove loading message and add assistant message
                const newMessages = prev.slice(0, -1)
                return [...newMessages, assistantMessage]
              })
              assistantMessageAdded = true
            }
            
            // Add sources/citations to current assistant message (deduped)
            setMessages(prev => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage && lastMessage.type === 'assistant') {
                newMessages[newMessages.length - 1] = {
                  ...lastMessage,
                  citations: dedupeSources(data.citations),
                  sources: dedupeSources(data.data?.sources || data.citations)
                }
              }
              return newMessages
            })
          } else if (data.type === 'answer') {
            console.log('🎯 Received answer event:', data)
            console.log('🎯 Answer content:', data.content)
            console.log('🎯 Answer content length:', data.content?.length)
            
            // On first event, remove loading indicator and add assistant message if not already done
            if (!assistantMessageAdded) {
              setMessages(prev => {
                // Remove loading message and add assistant message
                const newMessages = prev.slice(0, -1)
                return [...newMessages, assistantMessage]
              })
              assistantMessageAdded = true
            }
            
            // Handle complete answer with all metadata
            setMessages(prev => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage && lastMessage.type === 'assistant') {
                // Process metrics and extract total_tokens from tokens_used
                const metrics = data.metadata || lastMessage.metrics || {}
                if (metrics.tokens_used && typeof metrics.tokens_used === 'object') {
                  metrics.total_tokens = metrics.tokens_used.total_tokens || metrics.tokens_used.total || 0
                }
                
                console.log('🎯 Before update - lastMessage content:', lastMessage.content)
                console.log('🎯 Setting content to:', data.content || lastMessage.content)
                
                newMessages[newMessages.length - 1] = {
                  ...lastMessage,
                  content: data.content || lastMessage.content,
                  citations: dedupeSources(data.citations || lastMessage.citations),
                  sources: dedupeSources(data.citations || lastMessage.sources),
                  sentence_attribution: data.sentence_attribution,
                  metrics: metrics,
                  refinementCount: data.metadata?.refinement_count || 0
                }
                
                console.log('🎯 After update - message content:', newMessages[newMessages.length - 1].content)
              }
              return newMessages
            })

            // Capture latest payload to commit again on 'done' if needed
            latestAnswer = extractAnswerFromEvent(data) || latestAnswer
            latestCitations = dedupeSources(data.citations) || latestCitations
            latestSources = dedupeSources(data.citations) || latestSources
            latestSentenceAttribution = data.sentence_attribution || latestSentenceAttribution
            latestRefinementCount = data.metadata?.refinement_count ?? latestRefinementCount
            latestMetrics = (() => {
              const m = data.metadata || latestMetrics || {}
              if (m.tokens_used && typeof m.tokens_used === 'object') {
                m.total_tokens = m.tokens_used.total_tokens || m.tokens_used.total || 0
              }
              return m
            })()
          } else if (data.type === 'metrics') {
            // On first event, remove loading indicator and add assistant message if not already done
            if (!assistantMessageAdded) {
              setMessages(prev => {
                // Remove loading message and add assistant message
                const newMessages = prev.slice(0, -1)
                return [...newMessages, assistantMessage]
              })
              assistantMessageAdded = true
            }
            
            // Add metrics to current assistant message
            setMessages(prev => {
              const newMessages = [...prev]
              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage && lastMessage.type === 'assistant') {
                // Process metrics and extract total_tokens from tokens_used
                const metrics = data.metrics || {}
                if (metrics.tokens_used && typeof metrics.tokens_used === 'object') {
                  metrics.total_tokens = metrics.tokens_used.total_tokens || metrics.tokens_used.total || 0
                }
                
                newMessages[newMessages.length - 1] = {
                  ...lastMessage,
                  metrics: metrics
                }
              }
              return newMessages
            })
          } else if (data.type === 'done') {
            // Streaming complete. Ensure assistant message has the final answer.
            setMessages(prev => {
              const newMessages = [...prev]
              // If last message is loading, replace it with assistant message
              if (!assistantMessageAdded) {
                const withoutLoading = newMessages.slice(0, -1)
                withoutLoading.push({
                  ...assistantMessage,
                  content: latestAnswer || extractAnswerFromEvent(data) || '',
                  citations: latestCitations,
                  sources: latestSources,
                  sentence_attribution: latestSentenceAttribution,
                  metrics: latestMetrics,
                  refinementCount: latestRefinementCount
                })
                assistantMessageAdded = true
                return withoutLoading
              }

              const lastMessage = newMessages[newMessages.length - 1]
              if (lastMessage && lastMessage.type === 'assistant') {
                const extracted = extractAnswerFromEvent(data)
                const finalContent = (lastMessage.content && lastMessage.content.length > 0)
                  ? lastMessage.content
                  : (latestAnswer || extracted || '')
                newMessages[newMessages.length - 1] = {
                  ...lastMessage,
                  content: finalContent,
                  citations: dedupeSources(lastMessage.citations || latestCitations),
                  sources: dedupeSources(lastMessage.sources || latestSources),
                  sentence_attribution: lastMessage.sentence_attribution || latestSentenceAttribution,
                  metrics: lastMessage.metrics || latestMetrics,
                  refinementCount: lastMessage.refinementCount || latestRefinementCount
                }
              }
              return newMessages
            })
            // Do NOT break here; some backends may emit an 'answer' after an early 'done'.
            // We keep reading until the stream naturally ends.
          }
        }
      }

    } catch (error) {
      console.error('Streaming error:', error)
      
      // Remove loading message (if still present) and add error message
      setMessages(prev => {
        // Find and remove loading message if it exists
        const hasLoadingMessage = prev[prev.length - 1]?.type === 'loading'
        const newMessages = hasLoadingMessage ? prev.slice(0, -1) : prev
        
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
