import React, { useState } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Send } from 'lucide-react'

/**
 * ChatInput component for sending messages
 * Handles user input and message submission with loading states
 */
export function ChatInput({ onSendMessage, isLoading, disabled = false }) {
  const [message, setMessage] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (message.trim() && !isLoading) {
      onSendMessage(message.trim())
      setMessage('')
    }
  }

  const handleKeyDown = (e) => {
    // Send message on Ctrl+Enter or Cmd+Enter
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSubmit(e)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 p-4 border-t bg-background">
      <div className="flex-1">
        <Input
          type="text"
          placeholder="Ask a question about your documents..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled || isLoading}
          className="min-h-[44px]"
        />
      </div>
      <Button 
        type="submit" 
        disabled={!message.trim() || isLoading || disabled}
        size="icon"
        className="h-[44px] w-[44px]"
      >
        <Send className="h-4 w-4" />
      </Button>
    </form>
  )
}
