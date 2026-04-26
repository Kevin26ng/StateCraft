'use client';

import { WorldState } from '@/lib/types';

interface TopStripProps {
  state: WorldState | null;
  wsConnected: boolean;
  done: boolean;
  collapsed: boolean;
  headline?: string;
}

export default function TopStrip({ state, wsConnected, done, collapsed, headline }: TopStripProps) {
  return (
    <div className="py-4 border-b border-[#1F2023] fade-in-up">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-6">
          <div>
            <span className="uppercase-label mr-2">Turn</span>
            <span className="text-[#F3F4F6] font-medium tabular-numbers text-sm">
              {state?.turn ?? 0}
              <span className="text-[#4B5563]"> / 30</span>
            </span>
          </div>

          {state?.difficulty_tier != null && (
            <div>
              <span className="uppercase-label mr-2">Tier</span>
              <span className="text-[#F3F4F6] text-sm tabular-numbers">{state.difficulty_tier}</span>
            </div>
          )}

          {done && (
            <span className="px-2 py-0.5 text-[10px] uppercase tracking-wider font-semibold text-[#34D399] border border-[#34D399]/30">
              Completed
            </span>
          )}
          {collapsed && !done && (
            <span className="px-2 py-0.5 text-[10px] uppercase tracking-wider font-semibold text-[#EF4444] border border-[#EF4444]/30">
              Collapsed
            </span>
          )}
        </div>

        <div className={`flex items-center gap-1.5 text-xs ${wsConnected ? 'text-[#34D399]' : 'text-[#6B7280]'}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${wsConnected ? 'bg-[#34D399] animate-pulse' : 'bg-[#6B7280]'}`} />
          {wsConnected ? 'Live' : 'Offline'}
        </div>
      </div>

      {headline && (
        <p className="mt-2 text-xs text-[#6B7280] italic">{headline}</p>
      )}
    </div>
  );
}
