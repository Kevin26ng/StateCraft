'use client';

import { EpisodeMetrics } from '@/lib/types';
import { getScoreTier } from '@/lib/format';

interface SocietyScoreHeroProps {
  metrics: EpisodeMetrics | null;
}

export default function SocietyScoreHero({ metrics }: SocietyScoreHeroProps) {
  if (!metrics) {
    return (
      <div className="py-16 px-12 text-center">
        <p className="text-[#6B7280]">Loading metrics...</p>
      </div>
    );
  }

  const score = metrics.society_score;
  const tier = getScoreTier(score);

  // Generate a sparkline (7-point array for trend visualization)
  // In a real scenario, this would come from historical data
  const sparklineData = [
    score * 0.85,
    score * 0.88,
    score * 0.92,
    score * 0.95,
    score,
    score * 0.98,
    score,
  ];

  // Normalize sparkline to 0-100 for SVG height calculation
  const maxScore = 100;
  const sparklinePoints = sparklineData.map((val, i) => ({
    x: (i / (sparklineData.length - 1)) * 120,
    y: 40 - (val / maxScore) * 40,
  }));

  const pathD = sparklinePoints
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
    .join(' ');

  return (
    <div className="py-16 px-12 text-center border-b border-[#1F2023]">
      {/* Big score number */}
      <div
        className="text-9xl font-bold mb-2"
        style={{ color: tier.hex }}
      >
        {Math.round(score)}
      </div>

      {/* Label */}
      <p className="uppercase-label mb-12">Society Score</p>

      {/* Sparkline chart */}
      <div className="flex justify-center mb-8">
        <svg
          width="140"
          height="60"
          viewBox="0 0 140 60"
          className="fill-none stroke-[#7DD3FC]"
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
        >
          {/* Grid lines */}
          <line x1="0" y1="20" x2="140" y2="20" stroke="#1F2023" strokeWidth="1" />
          <line x1="0" y1="40" x2="140" y2="40" stroke="#1F2023" strokeWidth="1" />

          {/* Sparkline path */}
          <path d={pathD} />

          {/* Current point */}
          <circle
            cx={sparklinePoints[sparklinePoints.length - 1].x}
            cy={sparklinePoints[sparklinePoints.length - 1].y}
            r="2"
            fill="#7DD3FC"
          />
        </svg>
      </div>

      {/* Tier label */}
      <p
        className="text-sm font-medium"
        style={{ color: tier.hex }}
      >
        {tier.label}
      </p>
    </div>
  );
}
