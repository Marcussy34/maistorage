import React, { useEffect, useState } from 'react'
import Head from 'next/head'
import { ChatInput } from '../src/components/ChatInput'
import { ChatStream, useStreamingChat } from '../src/components/ChatStream'
import { Button } from '../src/components/ui/button'
import { Trash2, Moon, Sun, MessageSquare } from 'lucide-react'

/**
 * Main chat page for the MAI Storage RAG system
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

  // Initialize dark mode from localStorage or system preference
  useEffect(() => {
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
        <title>MAI Storage - RAG Chat</title>
        <meta name="description" content="Intelligent document search and Q&A with MAI Storage RAG system" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="flex flex-col h-screen bg-background text-foreground">
        {/* Header */}
        <header className="flex items-center justify-between p-4 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="flex items-center gap-3">
            <MessageSquare className="h-6 w-6 text-primary" />
            <div>
              <h1 className="text-lg font-semibold">MAI Storage</h1>
              <p className="text-sm text-muted-foreground">Agentic RAG System</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {/* Clear chat button */}
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

            {/* Mobile clear button */}
            <Button
              variant="outline"
              size="icon"
              onClick={handleClearChat}
              disabled={messages.length === 0}
              className="sm:hidden"
            >
              <Trash2 className="h-4 w-4" />
            </Button>

            {/* Dark mode toggle */}
            <Button
              variant="outline"
              size="icon"
              onClick={toggleDarkMode}
            >
              {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>

            {/* Stop generation button */}
            {isLoading && (
              <Button
                variant="destructive"
                size="sm"
                onClick={stopGeneration}
              >
                Stop
              </Button>
            )}
          </div>
        </header>

        {/* Chat content area */}
        <main className="flex-1 flex flex-col min-h-0">
          {/* Messages area */}
          <ChatStream
            messages={messages}
            onRetry={retryMessage}
            isLoading={isLoading}
          />

          {/* Input area */}
          <ChatInput
            onSendMessage={sendMessage}
            isLoading={isLoading}
            disabled={false}
          />
        </main>

        {/* Footer */}
        <footer className="px-4 py-2 text-center text-xs text-muted-foreground border-t">
          <p>
            MAI Storage Phase 4 - Next.js Streaming Chat Interface
            {' • '}
            <span className="font-medium">
              {messages.filter(m => m.type === 'user').length} messages sent
            </span>
          </p>
        </footer>
      </div>
    </>
  )
}
