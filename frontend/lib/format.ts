/**
 * Formatting utilities for StateCraft display
 */

/**
 * Format a number as a percentage with 1 decimal place
 */
export function formatPercent(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format a number for tabular display (with fixed width)
 */
export function formatTabular(value: number, decimals = 2): string {
  return value.toFixed(decimals);
}

/**
 * Format delta values (changes) with +/- sign
 */
export function formatDelta(value: number, decimals = 2): string {
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(decimals)}`;
}

/**
 * Format delta as percentage with sign
 */
export function formatDeltaPercent(value: number, decimals = 1): string {
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${(value * 100).toFixed(decimals)}%`;
}

/**
 * Determine color tier for vital metrics
 * Returns tuple of [colorHex, tierLabel, tierRaw]
 */
export function getVitalTier(value: number, metricType: 'high' | 'low' = 'high'): {
  hex: string;
  label: string;
  tier: 'critical' | 'warning' | 'nominal';
} {
  // High metrics: GDP, Inflation (low is bad), Resources, Stability, Public Trust - want them high
  // Low metrics: Mortality, GINI - want them low
  let isGood: boolean;

  if (metricType === 'high') {
    // Higher is better (GDP, Resources, Stability, Public Trust)
    isGood = value >= 0.65;
  } else {
    // Lower is better (Mortality, GINI)
    isGood = value <= 0.35;
  }

  if (metricType === 'high') {
    if (value >= 0.75) {
      return { hex: '#34D399', label: 'Nominal', tier: 'nominal' };
    } else if (value >= 0.55) {
      return { hex: '#F59E0B', label: 'Caution', tier: 'warning' };
    } else {
      return { hex: '#EF4444', label: 'Critical', tier: 'critical' };
    }
  } else {
    // Lower is better
    if (value <= 0.25) {
      return { hex: '#34D399', label: 'Nominal', tier: 'nominal' };
    } else if (value <= 0.50) {
      return { hex: '#F59E0B', label: 'Caution', tier: 'warning' };
    } else {
      return { hex: '#EF4444', label: 'Critical', tier: 'critical' };
    }
  }
}

/**
 * Get color tier for Society Score (0-100)
 */
export function getScoreTier(score: number): {
  hex: string;
  label: string;
  tier: 'critical' | 'warning' | 'nominal';
} {
  if (score >= 75) {
    return { hex: '#34D399', label: 'Excellent', tier: 'nominal' };
  } else if (score >= 60) {
    return { hex: '#F59E0B', label: 'Fair', tier: 'warning' };
  } else if (score >= 40) {
    return { hex: '#F59E0B', label: 'Poor', tier: 'warning' };
  } else {
    return { hex: '#EF4444', label: 'Critical', tier: 'critical' };
  }
}

/**
 * Format date for display
 */
export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

/**
 * Format time for display
 */
export function formatTime(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Clamp trust matrix value to [0, 1]
 */
export function clampTrust(value: number): number {
  return clamp(value, 0, 1);
}
