'use client';

import { WorldState } from '@/lib/types';
import { formatPercent, getVitalTier, formatDeltaPercent } from '@/lib/format';

type VitalConfig = { key: keyof WorldState; label: string; type: 'high' | 'low' };

const VITALS: VitalConfig[] = [
  { key: 'gdp', label: 'GDP', type: 'high' },
  { key: 'stability', label: 'Stability', type: 'high' },
  { key: 'resources', label: 'Resources', type: 'high' },
  { key: 'public_trust', label: 'Trust', type: 'high' },
  { key: 'inflation', label: 'Inflation', type: 'low' },
  { key: 'mortality', label: 'Mortality', type: 'low' },
  { key: 'gini', label: 'Gini', type: 'low' },
];

interface StateVitalsProps {
  state: WorldState | null;
  previousState: WorldState | null;
}

export default function StateVitals({ state, previousState }: StateVitalsProps) {
  return (
    <div className="py-6 border-b border-[#1F2023]">
      <p className="uppercase-label mb-3">State Vitals</p>
      <div className="grid grid-cols-7 gap-px bg-[#1F2023]">
        {VITALS.map(({ key, label, type }) => {
          const value = (state?.[key] as number) ?? 0;
          const prevValue = previousState?.[key] as number | undefined;
          const { hex } = getVitalTier(value, type);
          const delta = prevValue !== undefined ? value - prevValue : null;

          return (
            <div key={key} className="bg-[#0A0A0A] px-4 py-3">
              <p className="text-[10px] uppercase tracking-wider text-[#4B5563] mb-1.5">{label}</p>
              <p className="text-xl font-medium tabular-numbers leading-none" style={{ color: hex }}>
                {formatPercent(value, 0)}
              </p>
              {delta !== null && Math.abs(delta) > 0.001 && (
                <p className={`text-[10px] tabular-numbers mt-1 ${delta > 0 ? 'text-[#34D399]' : 'text-[#EF4444]'}`}>
                  {formatDeltaPercent(delta, 1)}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
