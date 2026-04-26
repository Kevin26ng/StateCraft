'use client';

import { AGENT_ROLES, AGENT_IDS } from '@/lib/agents';

export default function AgentRosterCarousel() {
  return (
    <div>
      <p className="uppercase-label mb-4">Agent Roster</p>
      <div className="flex gap-3 overflow-x-auto pb-3">
        {AGENT_IDS.map((id) => {
          const role = AGENT_ROLES[id];
          return (
            <div
              key={id}
              className="flex-shrink-0 w-[176px] border border-[#1F2023] bg-[#111113] hover:border-[#2A2A2E] transition-colors duration-150"
            >
              {/* Color accent bar */}
              <div className="h-0.5" style={{ background: role.paletteHex }} />

              {/* Portrait placeholder */}
              <div
                className="h-20 flex items-center justify-center"
                style={{
                  background: `linear-gradient(160deg, ${role.paletteHex}12 0%, transparent 70%)`,
                }}
              >
                <span
                  className="text-4xl font-semibold leading-none"
                  style={{ color: role.paletteHex, opacity: 0.7 }}
                >
                  {role.name.charAt(0)}
                </span>
              </div>

              {/* Info */}
              <div className="px-3 pb-3">
                <p className="text-sm font-medium text-[#F3F4F6] mb-0.5">{role.name}</p>
                <p className="text-[10px] uppercase tracking-wider text-[#4B5563] mb-2">{role.publicRole}</p>
                <p className="text-xs text-[#6B7280] leading-relaxed" style={{ display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                  {role.goalText}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
