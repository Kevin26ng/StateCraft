'use client';

import { WorldState } from '@/lib/types';
import { formatPercent, formatDelta, getVitalTier } from '@/lib/format';

interface StateVitalsProps {
  state: WorldState | null;
  previousState: WorldState | null;
}

export default function StateVitals({ state, previousState }: StateVitalsProps) {
  if (!state) {
    return (
      <div className="px-12 py-12">
        <p className="text-[#6B7280] text-center">Loading vitals...</p>
      </div>
    );
  }

  const calculateDelta = (current: number, previous: number | null | undefined): number => {
    if (!previous) return 0;
    return current - previous;
  };

  const vitals = [
    {
      label: 'GDP',
      value: state.gdp,
      delta: calculateDelta(state.gdp, previousState?.gdp),
      type: 'high' as const,
    },
    {
      label: 'Inflation',
      value: state.inflation,
      delta: calculateDelta(state.inflation, previousState?.inflation),
      type: 'high' as const, // Note: Lower inflation is better, but displayed as percentage
    },
    {
      label: 'Resources',
      value: state.resources,
      delta: calculateDelta(state.resources, previousState?.resources),
      type: 'high' as const,
    },
    {
      label: 'Stability',
      value: state.stability,
      delta: calculateDelta(state.stability, previousState?.stability),
      type: 'high' as const,
    },
    {
      label: 'Mortality',
      value: state.mortality,
      delta: calculateDelta(state.mortality, previousState?.mortality),
      type: 'low' as const,
    },
    {
      label: 'GINI',
      value: state.gini,
      delta: calculateDelta(state.gini, previousState?.gini),
      type: 'low' as const,
    },
    {
      label: 'Public Trust',
      value: state.public_trust,
      delta: calculateDelta(state.public_trust, previousState?.public_trust),
      type: 'high' as const,
    },
  ];

  return (
    <div className="px-12 py-12 border-b border-[#1F2023]">
      <p className="uppercase-label mb-8">State Vitals</p>

      <div className="grid grid-cols-7 gap-4">
        {vitals.map((vital) => {
          const tier = getVitalTier(vital.value, vital.type);
          const deltaSign = vital.delta > 0 ? '+' : '';

          return (
            <div
              key={vital.label}
              className="border border-[#1F2023] p-4"
            >
              <p className="uppercase-label mb-2">{vital.label}</p>

              <p
                className="text-2xl font-medium font-variant-numeric: tabular-nums mb-2"
                style={{ color: tier.hex }}
              >
                {formatPercent(vital.value)}
              </p>

              <p className="text-xs text-[#6B7280]">
                {tier.label}
              </p>

              {previousState && vital.delta !== 0 && (
                <p
                  className="text-xs mt-2"
                  style={{
                    color:
                      vital.delta > 0 ? '#34D399' : '#EF4444',
                  }}
                >
                  {deltaSign}{formatPercent(Math.abs(vital.delta), 1)}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
