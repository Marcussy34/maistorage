import React from 'react'
import { HoverCard, HoverCardTrigger, HoverCardContent } from './ui/hover-card'
import { Badge } from './ui/badge'
import { ExternalLink, FileText, AlertTriangle } from 'lucide-react'

/**
 * Converts a number to superscript numbers (¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹)
 */
function numberToSuperscript(num) {
  const superscriptMap = {
    0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴',
    5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹'
  }
  
  return num.toString()
    .split('')
    .map(digit => superscriptMap[digit] || digit)
    .join('')
}

/**
 * Replaces placeholder citations with actual document names
 * Converts [Source: doc_name, chunk_index] to readable citation format
 */
function replacePlaceholderCitations(text, citations = []) {
  if (!text || !citations.length) return text
  
  // Replace patterns like [Source: doc_name, chunk_index] with actual names
  return text.replace(/\[Source:\s*([^,\]]+),\s*([^\]]+)\]/g, (match, docNamePlaceholder, chunkPlaceholder) => {
    // Try to find a matching citation
    // Look for citations that might match this pattern
    const matchingCitation = citations.find(citation => {
      // Try different possible field names
      const docName = citation.doc_name || citation.title || citation.document_id || citation.source_doc_name
      const chunkIndex = citation.chunk_index !== undefined ? citation.chunk_index : citation.source_chunk_index
      
      // Check if this citation could be the source for this placeholder
      return docName || chunkIndex !== undefined
    })
    
    if (matchingCitation) {
      const docName = matchingCitation.doc_name || 
                     matchingCitation.title || 
                     matchingCitation.document_id || 
                     matchingCitation.source_doc_name || 
                     'Unknown Document'
      
      // Create a cleaner, more readable format
      const cleanDocName = docName
        .replace(/\.(md|txt|pdf|docx?)$/i, '') // Remove file extensions
        .replace(/[-_]/g, ' ') // Replace hyphens and underscores with spaces
        .replace(/\b\w/g, l => l.toUpperCase()) // Title case
      
      return `[Source: ${cleanDocName}]`
    }
    
    // If no matching citation found, try to clean up the placeholder
    const cleanDocName = docNamePlaceholder
      .replace(/\.(md|txt|pdf|docx?)$/i, '')
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase())
    
    return `[Source: ${cleanDocName}]`
  })
}

/**
 * Formats text with markdown-like syntax support:
 * - *text* becomes bold
 * - Line breaks and bullet points are preserved with proper formatting
 * - Replaces placeholder citations with actual document names
 */
function formatTextWithMarkdown(text, citations = []) {
  if (!text) return []
  
  // First, replace placeholder citations with actual document names
  const textWithReplacedCitations = replacePlaceholderCitations(text, citations)
  
  // Split text by lines first to handle structure
  const lines = textWithReplacedCitations.split('\n')
  const formattedElements = []
  
  lines.forEach((line, lineIndex) => {
    if (line.trim() === '') {
      // Empty line - add space
      formattedElements.push(<br key={`br-${lineIndex}`} />)
      return
    }
    
    // Check if line is a bullet point or numbered list
    const isBulletPoint = /^\s*[-*•]\s/.test(line)
    const isNumberedList = /^\s*\d+\.\s/.test(line)
    const isSubPoint = /^\s{2,}[-*•]\s/.test(line)
    
    // Process asterisk formatting within the line
    const parts = []
    let currentText = line
    let partIndex = 0
    
    // Find all *text* patterns and replace with bold
    const asteriskRegex = /\*([^*]+)\*/g
    let lastIndex = 0
    let match
    
    while ((match = asteriskRegex.exec(currentText)) !== null) {
      // Add text before the match
      if (match.index > lastIndex) {
        const beforeText = currentText.substring(lastIndex, match.index)
        if (beforeText) {
          parts.push(beforeText)
        }
      }
      
      // Add bold text
      parts.push(
        <strong key={`bold-${lineIndex}-${partIndex++}`} className="font-semibold">
          {match[1]}
        </strong>
      )
      
      lastIndex = match.index + match[0].length
    }
    
    // Add remaining text after last match
    if (lastIndex < currentText.length) {
      const remainingText = currentText.substring(lastIndex)
      if (remainingText) {
        parts.push(remainingText)
      }
    }
    
    // If no asterisks found, use the original line
    if (parts.length === 0) {
      parts.push(currentText)
    }
    
    // Wrap in appropriate container based on line type
    if (isBulletPoint) {
      formattedElements.push(
        <div key={`bullet-${lineIndex}`} className="flex items-start gap-2 my-1">
          <span className="text-primary font-bold mt-0.5">•</span>
          <span className="flex-1">{parts}</span>
        </div>
      )
    } else if (isNumberedList) {
      const numberMatch = line.match(/^\s*(\d+)\.\s(.*)/)
      if (numberMatch) {
        const number = numberMatch[1]
        const content = numberMatch[2]
        
        // Re-process the content part for asterisks
        const contentParts = []
        let contentText = content
        let contentPartIndex = 0
        const contentAsteriskRegex = /\*([^*]+)\*/g
        let contentLastIndex = 0
        let contentMatch
        
        while ((contentMatch = contentAsteriskRegex.exec(contentText)) !== null) {
          if (contentMatch.index > contentLastIndex) {
            const beforeText = contentText.substring(contentLastIndex, contentMatch.index)
            if (beforeText) {
              contentParts.push(beforeText)
            }
          }
          
          contentParts.push(
            <strong key={`content-bold-${lineIndex}-${contentPartIndex++}`} className="font-semibold">
              {contentMatch[1]}
            </strong>
          )
          
          contentLastIndex = contentMatch.index + contentMatch[0].length
        }
        
        if (contentLastIndex < contentText.length) {
          const remainingText = contentText.substring(contentLastIndex)
          if (remainingText) {
            contentParts.push(remainingText)
          }
        }
        
        if (contentParts.length === 0) {
          contentParts.push(content)
        }
        
        formattedElements.push(
          <div key={`numbered-${lineIndex}`} className="flex items-start gap-2 my-1">
            <span className="text-primary font-semibold mt-0.5 min-w-[1.5rem]">{number}.</span>
            <span className="flex-1">{contentParts}</span>
          </div>
        )
      }
    } else if (isSubPoint) {
      formattedElements.push(
        <div key={`sub-${lineIndex}`} className="flex items-start gap-2 my-1 ml-6">
          <span className="text-muted-foreground font-bold mt-0.5">◦</span>
          <span className="flex-1">{parts}</span>
        </div>
      )
    } else {
      // Regular text
      formattedElements.push(
        <div key={`text-${lineIndex}`} className="my-1">
          {parts}
        </div>
      )
    }
  })
  
  return formattedElements
}

/**
 * Individual source badge component with hover card
 */
export function SourceBadge({ 
  citation, 
  index, 
  onSourceClick 
}) {
  const superscriptNumber = numberToSuperscript(index + 1)
  
  // Determine confidence level styling
  const getConfidenceColor = (score) => {
    if (score >= 0.8) return 'success'
    if (score >= 0.6) return 'info'
    if (score >= 0.4) return 'warning'
    return 'destructive'
  }
  
  const confidenceVariant = citation.score 
    ? getConfidenceColor(citation.score)
    : 'secondary'
  
  const needsWarning = citation.score && citation.score < 0.4

  return (
    <HoverCard>
      <HoverCardTrigger asChild>
        <Badge 
          variant={confidenceVariant}
          className="cursor-pointer hover:scale-110 transition-transform mx-0.5 relative"
          onClick={() => onSourceClick && onSourceClick(citation)}
        >
          {superscriptNumber}
          {needsWarning && (
            <AlertTriangle className="h-2 w-2 ml-1" />
          )}
        </Badge>
      </HoverCardTrigger>
      
      <HoverCardContent className="w-80">
        <div className="space-y-3">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4 text-muted-foreground" />
              <span className="font-medium text-sm">
                {citation.title || citation.doc_name || `Source ${index + 1}`}
              </span>
            </div>
            
            {citation.score && (
              <Badge variant={confidenceVariant} className="text-xs">
                {Math.round(citation.score * 100)}%
              </Badge>
            )}
          </div>

          {/* Content preview */}
          <div className="text-sm text-muted-foreground">
            <p className="leading-relaxed">
              &ldquo;{citation.content || citation.text_snippet || 'No preview available'}&rdquo;
            </p>
          </div>

          {/* Metadata */}
          <div className="space-y-1 text-xs text-muted-foreground">
            {citation.chunk_index !== undefined && (
              <div>Chunk {citation.chunk_index}</div>
            )}
            
            {citation.document_id && (
              <div>Document ID: {citation.document_id}</div>
            )}
            
            {citation.supporting_span && (
              <div>
                Span: {citation.supporting_span.start}-{citation.supporting_span.end}
              </div>
            )}
            
            {citation.attribution_method && (
              <div>Method: {citation.attribution_method}</div>
            )}
          </div>

          {/* Warning for low confidence */}
          {needsWarning && (
            <div className="flex items-center gap-2 text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/20 rounded p-2">
              <AlertTriangle className="h-3 w-3" />
              Low confidence attribution
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-between pt-2 border-t">
            <span className="text-xs text-muted-foreground">
              Citation {index + 1}
            </span>
            
            {onSourceClick && (
              <button
                onClick={() => onSourceClick(citation)}
                className="flex items-center gap-1 text-xs text-primary hover:underline"
              >
                <ExternalLink className="h-3 w-3" />
                View Source
              </button>
            )}
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  )
}

/**
 * Container for multiple source badges
 */
export function SourceBadges({ 
  citations = [], 
  onSourceClick,
  className = "" 
}) {
  if (!citations || citations.length === 0) {
    return null
  }

  return (
    <span className={`inline-flex items-center gap-0.5 ${className}`}>
      {citations.map((citation, index) => (
        <SourceBadge
          key={`${citation.document_id}-${index}`}
          citation={citation}
          index={index}
          onSourceClick={onSourceClick}
        />
      ))}
    </span>
  )
}

/**
 * Enhanced text component that displays text with inline citations
 * Now supports markdown-like formatting with *bold* text and structured point forms
 */
export function TextWithCitations({ 
  text, 
  citations = [], 
  sentenceAttributions = null,
  onSourceClick 
}) {
  // If we have sentence-level attributions, use those for more granular citations
  if (sentenceAttributions && sentenceAttributions.sentence_citations) {
    const sentences = sentenceAttributions.sentences || []
    const sentenceCitations = sentenceAttributions.sentence_citations || []
    
    return (
      <div className="space-y-2">
        {sentences.map((sentence, index) => {
          // Find citations for this sentence
          const sentenceCits = sentenceCitations.filter(sc => sc.sentence_index === index)
          
          // Format the sentence with markdown support and citation replacement
          const formattedSentence = formatTextWithMarkdown(sentence, sentenceCits)
          
          return (
            <div key={index} className="inline-block w-full">
              <span className="inline">
                {formattedSentence}
                {sentenceCits.length > 0 && (
                  <SourceBadges 
                    citations={sentenceCits.map(sc => ({
                      ...sc,
                      title: sc.source_doc_name || 'Unknown Source',
                      content: sc.supporting_span?.text || 'No preview',
                      score: sc.attribution_score
                    }))}
                    onSourceClick={onSourceClick}
                    className="ml-1"
                  />
                )}
                {' '}
              </span>
            </div>
          )
        })}
      </div>
    )
  }

  // Format the main text with markdown support and citation replacement
  const formattedText = formatTextWithMarkdown(text, citations)

  // Fallback to formatted text with citations at the end
  return (
    <div className="space-y-1">
      <div className="leading-relaxed">
        {formattedText}
      </div>
      {citations && citations.length > 0 && (
        <div className="pt-2">
          <SourceBadges 
            citations={citations}
            onSourceClick={onSourceClick}
            className="inline-flex"
          />
        </div>
      )}
    </div>
  )
}
