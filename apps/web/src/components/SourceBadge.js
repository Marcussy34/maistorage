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
              "{citation.content || citation.text_snippet || 'No preview available'}"
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
          
          return (
            <span key={index} className="inline">
              {sentence}
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
          )
        })}
      </div>
    )
  }

  // Fallback to simple text with citations at the end
  return (
    <div>
      <span>{text}</span>
      {citations && citations.length > 0 && (
        <SourceBadges 
          citations={citations}
          onSourceClick={onSourceClick}
          className="ml-2"
        />
      )}
    </div>
  )
}
