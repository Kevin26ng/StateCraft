'use client';

import { WorldState } from '@/lib/types';

interface TopStripProps {
  state: WorldState | null;
  wsConnected: boolean;
  done: boolean;
  collapsed: boolean;
}

export default function TopStrip({
  state,
  wsConnected,
  done,
  collapsed,
}: TopStripProps) {
  if (!state) {
    return (
      <div className="px-12 py-4 border-b border-[#1F2023] flex items-center justify-between">
        <div className="text-[#6B7280] text-sm">Loading run...</div>
      </div>
    );
  }

  const borderColor = collapsed ? 'border-[#EF4444]' : 'border-[#1F2023]';
  const topBorderColor = collapsed ? 'border-t border-t-[#EF4444]' : '';

  return (
    <div className={`${borderColor} ${topBorderColor} border-b`}>
      <div className="px-12 py-4 flex items-center justify-between">
        {/* Left: Turn counter */}
        <div className="flex items-center gap-2">
          <span className="uppercase-label">Turn</span>
          <span className="text-[#F3F4F6] font-medium text-sm">
            {state.turn}
          </span>
        </div>

        {/* Center: Status messages */}
        <div className="flex-1 text-center">
          {collapsed && (
            <div className="text-[#EF4444] text-sm font-medium">
              Run ended — societal collapse at turn {state.turn}.
            </div>
          )}
          {done && !collapsed && (
            <div className="text-[#F3F4F6] text-sm font-medium">
              Run complete
            </div>
          )}
        </div>

        {/* Right: Live dot */}
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase-label">Connection</span>
          <div
            className="w-2 h-2 rounded-full"
            style={{
              backgroundColor: wsConnected ? '#34D399' : '#F59E0B',
              animation: wsConnected ? 'none' : 'pulse 2s infinite',
            }}
          />
        </div>
      </div>
    </div>
  );
}
