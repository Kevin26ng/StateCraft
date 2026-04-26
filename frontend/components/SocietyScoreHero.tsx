'use client';

import { EpisodeMetrics } from '@/lib/types';
import { getScoreTier, formatPercent } from '@/lib/format';

interface SocietyScoreHeroProps {
  metrics: EpisodeMetrics | null;
}

const SECONDARY = [
  { label: 'Betrayal', key: 'betrayal_rate' as const, isPercent: true },
  { label: 'Negotiation', key: 'negotiation_success' as const, isPercent: true },
  { label: 'Alliance', key: 'alliance_stability' as const, isPercent: true },
  { label: 'Trust Avg', key: 'trust_network_avg' as const, isPercent: false },
  { label: 'Auditor Acc', key: 'auditor_accuracy' as const, isPercent: true },
];

export default function SocietyScoreHero({ metrics }: SocietyScoreHeroProps) {
  const score = metrics?.society_score ?? 0;
  const { hex, label } = getScoreTier(score);

  return (
    <div className="py-8 border-b border-[#1F2023]">
      <div className="flex items-end gap-12 flex-wrap">
        <div>
          <p className="uppercase-label mb-1">Society Score</p>
          <div className="flex items-baseline gap-3">
            <span
              className="text-[64px] leading-none font-medium tabular-numbers"
              style={{ color: hex }}
            >
              {Math.round(score)}
            </span>
            <span
              className="text-xs uppercase tracking-[0.1em]"
              style={{ color: hex }}
            >
              {label}
            </span>
          </div>
        </div>

        {metrics && (
          <div className="flex gap-8 pb-2 flex-wrap">
            {SECONDARY.map(({ label: l, key, isPercent }) => (
              <div key={key}>
                <p className="text-[10px] uppercase tracking-wider text-[#4B5563] mb-0.5">{l}</p>
                <p className="text-sm text-[#F3F4F6] tabular-numbers">
                  {isPercent ? formatPercent(metrics[key]) : metrics[key].toFixed(2)}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
