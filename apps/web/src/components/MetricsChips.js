import React from 'react'
import { Badge } from './ui/badge'
import { 
  Clock, 
  Zap, 
  Search, 
  Brain, 
  Database,
  Timer,
  TrendingUp,
  BarChart3
} from 'lucide-react'

/**
 * Format timing values consistently
 */
function formatTiming(timeMs) {
  if (!timeMs || timeMs === 0) return null
  
  if (timeMs < 1) {
    return '<1ms'
  } else if (timeMs < 1000) {
    return `${Math.round(timeMs)}ms`
  } else {
    return `${(timeMs / 1000).toFixed(1)}s`
  }
}

/**
 * Format token counts
 */
function formatTokens(tokens) {
  if (!tokens || tokens === 0) return null
  
  if (tokens < 1000) {
    return `${tokens}`
  } else if (tokens < 1000000) {
    return `${(tokens / 1000).toFixed(1)}k`
  } else {
    return `${(tokens / 1000000).toFixed(1)}M`
  }
}

/**
 * Individual metric chip component
 */
function MetricChip({ 
  icon: Icon, 
  label, 
  value, 
  variant = "outline",
  className = ""
}) {
  if (!value && value !== 0) return null

  return (
    <Badge variant={variant} className={`flex items-center gap-1 ${className}`}>
      {Icon && <Icon className="h-3 w-3" />}
      <span className="text-xs">
        {label && `${label}: `}{value}
      </span>
    </Badge>
  )
}

/**
 * Main metrics display component
 */
export function MetricsChips({ 
  metrics = {}, 
  layout = "inline", // "inline" | "grid" | "compact"
  showLabels = true,
  className = ""
}) {
  const {
    retrieval_time_ms,
    llm_time_ms,
    generation_time_ms,
    total_time_ms,
    total_tokens,
    chunks_retrieved,
    model_used,
    refinement_count,
    step_times = {}
  } = metrics

  // Use generation_time_ms as fallback for llm_time_ms
  const actualLlmTime = llm_time_ms || generation_time_ms

  const chips = [
    {
      icon: Clock,
      label: showLabels ? "Total" : null,
      value: formatTiming(total_time_ms),
      variant: "secondary"
    },
    {
      icon: Search,
      label: showLabels ? "Retrieval" : null,
      value: formatTiming(retrieval_time_ms),
      variant: "outline"
    },
    {
      icon: Brain,
      label: showLabels ? "LLM" : null,
      value: formatTiming(actualLlmTime),
      variant: "outline"
    },
    {
      icon: Zap,
      label: showLabels ? "Tokens" : null,
      value: formatTokens(total_tokens),
      variant: "outline"
    },
    {
      icon: Database,
      label: showLabels ? "Chunks" : null,
      value: chunks_retrieved ? `${chunks_retrieved}` : null,
      variant: "outline"
    }
  ].filter(chip => chip.value !== null)

  // Add refinement count if present
  if (refinement_count && refinement_count > 0) {
    chips.push({
      icon: TrendingUp,
      label: showLabels ? "Refinements" : null,
      value: `${refinement_count}`,
      variant: "info"
    })
  }

  if (chips.length === 0) {
    return null
  }

  const containerClass = {
    inline: "flex items-center gap-2 flex-wrap",
    grid: "grid grid-cols-2 gap-2",
    compact: "flex items-center gap-1"
  }[layout]

  return (
    <div className={`${containerClass} ${className}`}>
      {chips.map((chip, index) => (
        <MetricChip
          key={`${chip.label}-${index}`}
          icon={chip.icon}
          label={chip.label}
          value={chip.value}
          variant={chip.variant}
        />
      ))}
    </div>
  )
}

/**
 * Detailed metrics panel with step breakdown
 */
export function DetailedMetrics({ 
  metrics = {},
  traceEvents = [],
  className = ""
}) {
  const {
    total_time_ms,
    retrieval_time_ms,
    llm_time_ms,
    generation_time_ms,
    total_tokens,
    model_used,
    step_times = {}
  } = metrics

  // Extract step timings from trace events or step_times
  const stepTimings = {}
  
  // From step_times object
  Object.entries(step_times).forEach(([step, time]) => {
    stepTimings[step] = time
  })

  // From trace events (if available)
  const stepEvents = traceEvents.filter(event => 
    event.type === 'step_complete' && event.data?.time_ms
  )
  
  stepEvents.forEach(event => {
    if (event.step && event.data.time_ms) {
      stepTimings[event.step] = event.data.time_ms
    }
  })

  return (
    <div className={`bg-card rounded-lg border p-4 space-y-4 ${className}`}>
      <div className="flex items-center gap-2 mb-3">
        <BarChart3 className="h-5 w-5 text-primary" />
        <h3 className="font-semibold">Performance Metrics</h3>
      </div>

      {/* Main metrics */}
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <div className="text-sm font-medium">Total Time</div>
            <Badge variant="secondary" className="text-sm">
              <Clock className="h-3 w-3 mr-1" />
              {formatTiming(total_time_ms) || 'N/A'}
            </Badge>
          </div>

          <div className="space-y-1">
            <div className="text-sm font-medium">Model</div>
            <Badge variant="outline" className="text-sm">
              {model_used || 'Unknown'}
            </Badge>
          </div>

          <div className="space-y-1">
            <div className="text-sm font-medium">Retrieval</div>
            <Badge variant="outline" className="text-sm">
              <Search className="h-3 w-3 mr-1" />
              {formatTiming(retrieval_time_ms) || 'N/A'}
            </Badge>
          </div>

          <div className="space-y-1">
            <div className="text-sm font-medium">Generation</div>
            <Badge variant="outline" className="text-sm">
              <Brain className="h-3 w-3 mr-1" />
              {formatTiming(llm_time_ms || generation_time_ms) || 'N/A'}
            </Badge>
          </div>
        </div>

        {total_tokens && (
          <div className="space-y-1">
            <div className="text-sm font-medium">Tokens Used</div>
            <Badge variant="outline" className="text-sm">
              <Zap className="h-3 w-3 mr-1" />
              {formatTokens(total_tokens)}
            </Badge>
          </div>
        )}
      </div>

      {/* Step breakdown */}
      {Object.keys(stepTimings).length > 0 && (
        <div className="pt-3 border-t">
          <div className="text-sm font-medium mb-2">Step Breakdown</div>
          <div className="space-y-2">
            {Object.entries(stepTimings).map(([step, time]) => (
              <div key={step} className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground capitalize">
                  {step.replace('_', ' ')}
                </span>
                <Badge variant="outline" className="text-xs">
                  {formatTiming(time)}
                </Badge>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * Compact metrics display for inline use
 */
export function CompactMetrics({ 
  metrics = {},
  className = ""
}) {
  const {
    total_time_ms,
    total_tokens,
    chunks_retrieved
  } = metrics

  const items = [
    total_time_ms && formatTiming(total_time_ms),
    total_tokens && `${formatTokens(total_tokens)} tokens`,
    chunks_retrieved && `${chunks_retrieved} chunks`
  ].filter(Boolean)

  if (items.length === 0) {
    return null
  }

  return (
    <div className={`text-xs text-muted-foreground ${className}`}>
      {items.join(' • ')}
    </div>
  )
}
