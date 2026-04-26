'use client';

import { CoalitionGraph as CoalitionGraphType, AgentId } from '@/lib/types';
import { AGENT_ROLES, AGENT_IDS } from '@/lib/agents';

// Circular layout: 6 nodes evenly spaced, radius 110 from center 160,160
const NODE_POSITIONS = AGENT_IDS.map((_, i) => {
  const angle = (i * Math.PI * 2) / 6 - Math.PI / 2;
  return { x: 160 + 110 * Math.cos(angle), y: 160 + 110 * Math.sin(angle) };
});

const AGENT_INDEX: Record<AgentId, number> = {
  agent_0: 0, agent_1: 1, agent_2: 2, agent_3: 3, agent_4: 4, agent_5: 5,
};

interface CoalitionGraphProps {
  coalitionGraph: CoalitionGraphType | null;
}

export default function CoalitionGraph({ coalitionGraph }: CoalitionGraphProps) {
  const edges = coalitionGraph?.edges ?? [];

  return (
    <div className="py-6 border-b border-[#1F2023]">
      <p className="uppercase-label mb-4">Coalition Graph</p>
      <div className="flex gap-10 items-start">
        <svg width="320" height="320" viewBox="0 0 320 320">
          {/* Background circle */}
          <circle cx="160" cy="160" r="115" fill="none" stroke="#1F2023" strokeWidth="1" strokeDasharray="2 4" />

          {/* Edges */}
          {edges.map((edge, i) => {
            const aIdx = AGENT_INDEX[edge.a];
            const bIdx = AGENT_INDEX[edge.b];
            if (aIdx === undefined || bIdx === undefined) return null;
            const a = NODE_POSITIONS[aIdx];
            const b = NODE_POSITIONS[bIdx];
            return (
              <line
                key={i}
                x1={a.x} y1={a.y}
                x2={b.x} y2={b.y}
                stroke="#7DD3FC"
                strokeWidth={Math.max(0.5, edge.weight * 2)}
                strokeOpacity={Math.max(0.06, edge.weight * 0.55)}
              />
            );
          })}

          {/* Nodes */}
          {AGENT_IDS.map((id, i) => {
            const pos = NODE_POSITIONS[i];
            const role = AGENT_ROLES[id];
            return (
              <g key={id}>
                <circle cx={pos.x} cy={pos.y} r={20} fill="#111113" stroke={role.paletteHex} strokeWidth="1.5" />
                <text
                  x={pos.x} y={pos.y + 4}
                  textAnchor="middle"
                  fill={role.paletteHex}
                  fontSize="11"
                  fontWeight="600"
                  fontFamily="Inter, sans-serif"
                >
                  {i}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Legend */}
        <div className="pt-2 space-y-2.5">
          {AGENT_IDS.map((id) => (
            <div key={id} className="flex items-center gap-2.5">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: AGENT_ROLES[id].paletteHex }} />
              <div>
                <p className="text-xs text-[#9CA3AF]">{AGENT_ROLES[id].name}</p>
                <p className="text-[10px] text-[#4B5563]">{AGENT_ROLES[id].publicRole}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
