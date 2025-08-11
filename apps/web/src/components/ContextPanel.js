import React, { useState } from 'react'
import { Badge } from './ui/badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs'
import { 
  FileText, 
  Search, 
  TrendingUp, 
  Clock,
  ChevronDown,
  ChevronUp,
  Copy,
  ExternalLink
} from 'lucide-react'
import { Button } from './ui/button'

/**
 * Individual source document component
 */
function SourceDocument({ 
  source, 
  index, 
  isExpanded = false, 
  onToggle,
  onCopy,
  highlightQuery = ""
}) {
  const {
    doc_name,
    chunk_index,
    text_snippet,
    content,
    relevance_score,
    score,
    document_id
  } = source
  
  const displayScore = relevance_score || score || 0
  const displayContent = content || text_snippet || 'No content available'
  
  // Simple text highlighting
  const highlightText = (text, query) => {
    if (!query || query.length < 2) return text
    
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
    const parts = text.split(regex)
    
    return parts.map((part, i) => 
      regex.test(part) ? (
        <mark key={i} className="bg-yellow-200 dark:bg-yellow-800/30 px-0.5 rounded">
          {part}
        </mark>
      ) : part
    )
  }

  const getScoreColor = (score) => {
    if (score >= 0.8) return 'success'
    if (score >= 0.6) return 'info'
    if (score >= 0.4) return 'warning'
    return 'secondary'
  }

  return (
    <div className="border rounded-lg p-3 space-y-2">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <div className="min-w-0 flex-1">
            <h4 className="font-medium text-sm truncate">
              {doc_name || `Document ${document_id || index + 1}`}
            </h4>
            {chunk_index !== undefined && (
              <p className="text-xs text-muted-foreground">
                Chunk {chunk_index}
              </p>
            )}
          </div>
        </div>
        
        <div className="flex items-center gap-2 flex-shrink-0">
          <Badge variant={getScoreColor(displayScore)} className="text-xs">
            {Math.round(displayScore * 100)}%
          </Badge>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onToggle && onToggle(index)}
            className="h-6 w-6 p-0"
          >
            {isExpanded ? (
              <ChevronUp className="h-3 w-3" />
            ) : (
              <ChevronDown className="h-3 w-3" />
            )}
          </Button>
        </div>
      </div>

      {/* Content preview */}
      <div className="text-sm text-muted-foreground">
        <p className="line-clamp-2">
          {highlightText(displayContent.substring(0, 150), highlightQuery)}
          {displayContent.length > 150 && '...'}
        </p>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="space-y-3 pt-2 border-t">
          <div className="text-sm leading-relaxed">
            {highlightText(displayContent, highlightQuery)}
          </div>
          
          {/* Actions */}
          <div className="flex items-center justify-between">
            <div className="text-xs text-muted-foreground">
              Score: {displayScore.toFixed(3)}
              {document_id && ` • ID: ${document_id.substring(0, 8)}...`}
            </div>
            
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onCopy && onCopy(displayContent)}
                className="h-6 text-xs"
              >
                <Copy className="h-3 w-3 mr-1" />
                Copy
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * Main context panel component
 */
export function ContextPanel({
  sources = [],
  query = "",
  retrievalTime,
  chunksRetrieved,
  isVisible = true,
  className = ""
}) {
  const [expandedItems, setExpandedItems] = useState(new Set([0])) // First item expanded by default
  const [activeTab, setActiveTab] = useState("sources")
  const [copiedIndex, setCopiedIndex] = useState(null)

  const toggleExpanded = (index) => {
    const newExpanded = new Set(expandedItems)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedItems(newExpanded)
  }

  const handleCopy = async (content, index) => {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedIndex(index)
      setTimeout(() => setCopiedIndex(null), 2000)
    } catch (error) {
      console.error('Failed to copy to clipboard:', error)
    }
  }

  if (!isVisible) {
    return null
  }

  // Process sources for different views
  const topSources = sources.slice(0, 10) // Limit to top 10
  const uniqueDocuments = sources.reduce((acc, source) => {
    const docKey = source.doc_name || source.document_id
    if (docKey && !acc.find(doc => (doc.doc_name || doc.document_id) === docKey)) {
      acc.push(source)
    }
    return acc
  }, [])
  
  // Calculate top score for display
  const topScore = sources.length > 0
    ? Math.max(...sources.map(s => s.score || s.relevance_score || 0))
    : 0

  return (
    <div className={`bg-card rounded-lg border ${className}`}>
      <div className="p-4 border-b">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Search className="h-5 w-5 text-primary" />
            <h3 className="font-semibold">Retrieved Context</h3>
          </div>
          
          <div className="flex items-center gap-2">
            {topScore > 0 && (
              <Badge variant="outline" className="text-xs">
                <TrendingUp className="h-3 w-3 mr-1" />
                Top: {Math.round(topScore * 100)}%
              </Badge>
            )}
            
            {retrievalTime && (
              <Badge variant="outline" className="text-xs">
                <Clock className="h-3 w-3 mr-1" />
                {Math.round(retrievalTime)}ms
              </Badge>
            )}
            
            {chunksRetrieved && (
              <Badge variant="secondary" className="text-xs">
                {chunksRetrieved} chunks
              </Badge>
            )}
          </div>
        </div>

        {query && (
          <div className="text-sm text-muted-foreground">
            Query: "{query}"
          </div>
        )}
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <div className="px-4 pt-3">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="sources">
              Sources ({topSources.length})
            </TabsTrigger>
            <TabsTrigger value="documents">
              Documents ({uniqueDocuments.length})
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="sources" className="p-4 space-y-3">
          {topSources.length === 0 ? (
            <div className="text-center text-muted-foreground py-8">
              <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No sources retrieved</p>
            </div>
          ) : (
            topSources.map((source, index) => (
              <SourceDocument
                key={`source-${index}`}
                source={source}
                index={index}
                isExpanded={expandedItems.has(index)}
                onToggle={toggleExpanded}
                onCopy={(content) => handleCopy(content, index)}
                highlightQuery={query}
              />
            ))
          )}
        </TabsContent>

        <TabsContent value="documents" className="p-4 space-y-3">
          {uniqueDocuments.length === 0 ? (
            <div className="text-center text-muted-foreground py-8">
              <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No documents found</p>
            </div>
          ) : (
            uniqueDocuments.map((doc, index) => (
              <SourceDocument
                key={`doc-${index}`}
                source={doc}
                index={index}
                isExpanded={expandedItems.has(index)}
                onToggle={toggleExpanded}
                onCopy={(content) => handleCopy(content, index)}
                highlightQuery={query}
              />
            ))
          )}
        </TabsContent>
      </Tabs>

      {/* Copy notification */}
      {copiedIndex !== null && (
        <div className="absolute top-2 right-2 bg-background border rounded px-2 py-1 text-xs text-muted-foreground shadow-lg">
          Copied!
        </div>
      )}
    </div>
  )
}

/**
 * Compact context panel for sidebar display
 */
export function CompactContextPanel({
  sources = [],
  className = ""
}) {
  const totalSources = sources.length
  const topScore = sources.length > 0
    ? Math.max(...sources.map(s => s.score || s.relevance_score || 0))
    : 0

  return (
    <div className={`bg-card rounded border p-3 ${className}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Search className="h-4 w-4 text-primary" />
          <span className="font-medium text-sm">Context</span>
        </div>
        
        <Badge variant="secondary" className="text-xs">
          {totalSources}
        </Badge>
      </div>

      <div className="space-y-1 text-xs text-muted-foreground">
        <div>Sources: {totalSources}</div>
        {topScore > 0 && (
          <div>Top Score: {Math.round(topScore * 100)}%</div>
        )}
      </div>

      {totalSources > 0 && (
        <div className="mt-2 space-y-1">
          {sources.slice(0, 3).map((source, index) => (
            <div key={index} className="text-xs truncate">
              {source.doc_name || `Source ${index + 1}`}
            </div>
          ))}
          {totalSources > 3 && (
            <div className="text-xs text-muted-foreground">
              +{totalSources - 3} more...
            </div>
          )}
        </div>
      )}
    </div>
  )
}
