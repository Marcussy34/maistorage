import React from 'react'
import { Switch } from './ui/switch'
import { Badge } from './ui/badge'
import { 
  Target, 
  Brain, 
  Info,
  Zap,
  Clock
} from 'lucide-react'

/**
 * Mode selector component with visual toggle
 */
export function ModeToggle({ 
  isAgentic = false, 
  onToggle,
  disabled = false,
  showDescription = true,
  className = ""
}) {
  return (
    <div className={`space-y-3 ${className}`}>
      {/* Toggle control */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            {isAgentic ? (
              <Brain className="h-4 w-4 text-primary" />
            ) : (
              <Target className="h-4 w-4 text-muted-foreground" />
            )}
            
            <span className="font-medium">
              {isAgentic ? 'Agentic RAG' : 'Traditional RAG'}
            </span>
          </div>

          <Badge 
            variant={isAgentic ? "default" : "secondary"}
            className="text-xs"
          >
            {isAgentic ? 'Multi-step' : 'Single-pass'}
          </Badge>
        </div>

        <Switch
          checked={isAgentic}
          onCheckedChange={onToggle}
          disabled={disabled}
          aria-label="Toggle between Traditional and Agentic RAG"
        />
      </div>

      {/* Mode descriptions */}
      {showDescription && (
        <div className="text-sm text-muted-foreground">
          {isAgentic ? (
            <div className="space-y-1">
              <p>
                <strong>Agentic RAG:</strong> Multi-step reasoning with planning, retrieval, synthesis, and verification.
              </p>
              <div className="flex items-center gap-1 text-xs">
                <Info className="h-3 w-3" />
                <span>Provides detailed trace and higher accuracy</span>
              </div>
            </div>
          ) : (
            <div className="space-y-1">
              <p>
                <strong>Traditional RAG:</strong> Single-pass retrieve and generate approach.
              </p>
              <div className="flex items-center gap-1 text-xs">
                <Clock className="h-3 w-3" />
                <span>Faster response times</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * Compact mode indicator for header/toolbar
 */
export function CompactModeToggle({ 
  isAgentic = false, 
  onToggle,
  disabled = false,
  className = ""
}) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <Switch
        checked={isAgentic}
        onCheckedChange={onToggle}
        disabled={disabled}
        aria-label="Toggle RAG mode"
      />
      
      <div className="flex items-center gap-1">
        {isAgentic ? (
          <Brain className="h-4 w-4 text-primary" />
        ) : (
          <Target className="h-4 w-4 text-muted-foreground" />
        )}
        
        <span className="text-sm font-medium">
          {isAgentic ? 'Agentic' : 'Traditional'}
        </span>
      </div>
    </div>
  )
}

/**
 * Mode comparison component showing differences
 */
export function ModeComparison({ 
  className = ""
}) {
  const features = [
    {
      feature: "Planning",
      traditional: "❌ No",
      agentic: "✅ Query decomposition"
    },
    {
      feature: "Retrieval",
      traditional: "🔍 Single pass",
      agentic: "🔍 Multi-step with refinement"
    },
    {
      feature: "Verification",
      traditional: "❌ No",
      agentic: "✅ Answer verification"
    },
    {
      feature: "Citations",
      traditional: "📄 Chunk-level",
      agentic: "📄 Sentence-level"
    },
    {
      feature: "Speed",
      traditional: "⚡ Fast",
      agentic: "🐌 Slower"
    },
    {
      feature: "Accuracy",
      traditional: "📊 Good",
      agentic: "📊 Higher"
    }
  ]

  return (
    <div className={`bg-card rounded-lg border p-4 ${className}`}>
      <h3 className="font-semibold mb-3 flex items-center gap-2">
        <Info className="h-4 w-4" />
        Mode Comparison
      </h3>
      
      <div className="space-y-3">
        <div className="grid grid-cols-3 gap-3 text-xs font-medium text-muted-foreground">
          <div>Feature</div>
          <div>Traditional</div>
          <div>Agentic</div>
        </div>
        
        {features.map((item, index) => (
          <div key={index} className="grid grid-cols-3 gap-3 text-sm">
            <div className="font-medium">{item.feature}</div>
            <div className="text-muted-foreground">{item.traditional}</div>
            <div className="text-muted-foreground">{item.agentic}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

/**
 * Mode status indicator with current configuration
 */
export function ModeStatus({ 
  isAgentic = false,
  lastResponseTime,
  lastTokenCount,
  className = ""
}) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <Badge 
        variant={isAgentic ? "default" : "secondary"}
        className="flex items-center gap-1"
      >
        {isAgentic ? (
          <Brain className="h-3 w-3" />
        ) : (
          <Target className="h-3 w-3" />
        )}
        {isAgentic ? 'Agentic' : 'Traditional'}
      </Badge>
      
      {lastResponseTime && (
        <Badge variant="outline" className="text-xs">
          <Clock className="h-3 w-3 mr-1" />
          {lastResponseTime < 1000 
            ? `${Math.round(lastResponseTime)}ms`
            : `${(lastResponseTime / 1000).toFixed(1)}s`
          }
        </Badge>
      )}
      
      {lastTokenCount && (
        <Badge variant="outline" className="text-xs">
          <Zap className="h-3 w-3 mr-1" />
          {lastTokenCount}
        </Badge>
      )}
    </div>
  )
}
