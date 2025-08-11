import React, { useEffect, useState, useRef } from 'react'
import Head from 'next/head'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { ChatInput } from '../src/components/ChatInput'
import { ChatStream, useStreamingChat } from '../src/components/ChatStream'
import { Button } from '../src/components/ui/button'
import { CompactModeToggle, ModeStatus } from '../src/components/ModeToggle'
import { Trash2, Moon, Sun, MessageSquare, Upload, BarChart3 } from 'lucide-react'

/**
 * Main chat page for the MaiStorage RAG system
 * Provides a streaming chat interface with dark mode support
 */
export default function ChatPage() {
  const {
    messages,
    isLoading,
    sendMessage,
    retryMessage,
    stopGeneration,
    clearMessages
  } = useStreamingChat()

  const [darkMode, setDarkMode] = useState(false)
  const [agenticMode, setAgenticMode] = useState(false)
  const mountedRef = useRef(false)

  // Initialize dark mode and agentic mode from localStorage or defaults
  useEffect(() => {
    if (mountedRef.current) return // Prevent double execution in StrictMode
    mountedRef.current = true
    
    // Dark mode initialization
    const savedDarkMode = localStorage.getItem('darkMode')
    const systemDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches
    
    const shouldUseDarkMode = savedDarkMode 
      ? JSON.parse(savedDarkMode) 
      : systemDarkMode

    setDarkMode(shouldUseDarkMode)
    updateDarkModeClass(shouldUseDarkMode)
    
    // Agentic mode initialization
    const savedAgenticMode = localStorage.getItem('agenticMode')
    if (savedAgenticMode) {
      setAgenticMode(JSON.parse(savedAgenticMode))
    }
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

  // Toggle agentic mode
  const toggleAgenticMode = () => {
    const newAgenticMode = !agenticMode
    setAgenticMode(newAgenticMode)
    localStorage.setItem('agenticMode', JSON.stringify(newAgenticMode))
  }

  // Send message with current mode
  const handleSendMessage = (message) => {
    sendMessage(message, agenticMode)
  }

  // Retry message with current mode
  const handleRetryMessage = (query) => {
    retryMessage(query, agenticMode)
  }
  
  // Get last response metrics for status display
  const lastAssistantMessage = messages.filter(m => m.type === 'assistant').pop()
  const lastResponseTime = lastAssistantMessage?.metrics?.total_time_ms
  const lastTokenCount = lastAssistantMessage?.metrics?.total_tokens

  // Handle clear chat with confirmation
  const handleClearChat = () => {
    if (messages.length > 0) {
      const confirmed = window.confirm('Are you sure you want to clear all messages?')
      if (confirmed) {
        clearMessages()
      }
    }
  }

  return (
    <>
      <Head>
        <title>MaiStorage - RAG Chat</title>
        <meta name="description" content="Intelligent document search and Q&A with MaiStorage RAG system" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="flex flex-col h-screen bg-background text-foreground">
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
            <MessageSquare className="h-6 w-6 text-primary" />
            <div>
              <h1 className="text-lg font-semibold">MaiStorage</h1>
              <div className="flex items-center gap-2">
                <p className="text-sm text-muted-foreground">Agentic RAG Chat Interface</p>
                <ModeStatus 
                  isAgentic={agenticMode}
                  lastResponseTime={lastResponseTime}
                  lastTokenCount={lastTokenCount}
                />
              </div>
            </div>
          </motion.div>
          
          <motion.div 
            className="flex items-center gap-2"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            {/* Upload button */}
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link href="/upload">
                <Button variant="outline" size="sm" className="hidden sm:flex">
                  <Upload className="h-4 w-4 mr-2" />
                  Upload
                </Button>
              </Link>
            </motion.div>

            {/* Eval button */}
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link href="/eval">
                <Button variant="outline" size="sm" className="hidden sm:flex">
                  <BarChart3 className="h-4 w-4 mr-2" />
                  Eval
                </Button>
              </Link>
            </motion.div>

            {/* Mode toggle */}
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <CompactModeToggle
                isAgentic={agenticMode}
                onToggle={toggleAgenticMode}
                disabled={isLoading}
              />
            </motion.div>

            {/* Clear chat button */}
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button
                variant="outline"
                size="sm"
                onClick={handleClearChat}
                disabled={messages.length === 0}
                className="hidden sm:flex"
              >
                <Trash2 className="h-4 w-4" />
                <span className="ml-2">Clear</span>
              </Button>
            </motion.div>

            {/* Mobile clear button */}
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button
                variant="outline"
                size="icon"
                onClick={handleClearChat}
                disabled={messages.length === 0}
                className="sm:hidden"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </motion.div>

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

            {/* Stop generation button */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={stopGeneration}
                >
                  Stop
                </Button>
              </motion.div>
            )}
              </motion.div>
            </div>
          </div>
        </motion.header>

        {/* Chat content area */}
        <motion.main 
          className="flex-1 flex flex-col min-h-0 bg-muted/30"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="container mx-auto px-4 py-6 flex-1 flex flex-col min-h-0">
            {/* Chat container with visual styling */}
            <motion.div 
              className="flex-1 flex flex-col min-h-0 bg-background rounded-lg border shadow-sm"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              {/* Messages area */}
              <motion.div 
                className="flex-1 min-h-0 p-4 pr-12 overflow-y-auto"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4, delay: 0.4 }}
              >
                <ChatStream
                  messages={messages}
                  onRetry={handleRetryMessage}
                  isLoading={isLoading}
                  ragMode={agenticMode}
                />
              </motion.div>

              {/* Input area with border separator */}
              <motion.div 
                className="border-t bg-background/50 p-4 rounded-b-lg"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.5 }}
              >
                <ChatInput
                  onSendMessage={handleSendMessage}
                  isLoading={isLoading}
                  disabled={false}
                  placeholder={`Ask a question using ${agenticMode ? 'Agentic' : 'Traditional'} RAG...`}
                />
              </motion.div>
            </motion.div>
          </div>
        </motion.main>

        {/* Footer */}
        <motion.footer 
          className="px-4 py-2 text-center text-xs text-muted-foreground border-t"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <p>
            MaiStorage 
            {' • '}
            <span className="font-medium">
              {messages.filter(m => m.type === 'user').length} messages sent
            </span>
            {agenticMode && ' • Agentic Mode Active'}
          </p>
        </motion.footer>
      </div>
    </>
  )
}
