'use client';

import { getAgentName } from '@/lib/agents';

interface TrustMatrixProps {
  trustMatrix: number[][] | null;
}

export default function TrustMatrix({ trustMatrix }: TrustMatrixProps) {
  if (!trustMatrix || trustMatrix.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-[#6B7280]">Loading trust matrix...</p>
      </div>
    );
  }

  const size = trustMatrix.length;
  const cellSize = 40;
  const labelWidth = 140;
  const topLabelHeight = 50;
  const padding = 20;

  // Convert trust value to color (0 = dark, 1 = bright cyan)
  const getColor = (value: number): string => {
    // Interpolate between #1F2023 (dark) and #7DD3FC (cyan)
    const r = Math.round(31 + (125 - 31) * value);
    const g = Math.round(32 + (211 - 32) * value);
    const b = Math.round(35 + (252 - 35) * value);
    return `rgb(${r}, ${g}, ${b})`;
  };

  const agentIds = ['agent_0', 'agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5'] as const;

  return (
    <div className="overflow-x-auto p-6">
      <svg
        width={labelWidth + size * cellSize + padding * 2}
        height={topLabelHeight + size * cellSize + padding}
        className="mx-auto"
      >
        {/* Top agent labels */}
        {agentIds.map((id, i) => (
          <text
            key={`top-${id}`}
            x={labelWidth + i * cellSize + cellSize / 2}
            y={padding + 12}
            textAnchor="middle"
            fontSize="11"
            fill="#6B7280"
            fontFamily="var(--font-inter, 'Inter', sans-serif)"
          >
            {i}
          </text>
        ))}

        {/* Left agent labels and cells */}
        {agentIds.map((fromId, i) => (
          <g key={`row-${fromId}`}>
            {/* Row label */}
            <text
              x={padding + labelWidth - 12}
              y={padding + topLabelHeight + i * cellSize + cellSize / 2 + 4}
              textAnchor="end"
              fontSize="11"
              fill="#6B7280"
              fontFamily="var(--font-inter, 'Inter', sans-serif)"
            >
              {i}
            </text>

            {/* Cells in row */}
            {trustMatrix[i].map((value, j) => (
              <rect
                key={`cell-${i}-${j}`}
                x={labelWidth + j * cellSize}
                y={padding + topLabelHeight + i * cellSize}
                width={cellSize}
                height={cellSize}
                fill={getColor(value)}
                stroke="#1F2023"
                strokeWidth="1"
              />
            ))}
          </g>
        ))}

        {/* Value text in cells */}
        {trustMatrix.map((row, i) =>
          row.map((value, j) => (
            <text
              key={`value-${i}-${j}`}
              x={labelWidth + j * cellSize + cellSize / 2}
              y={padding + topLabelHeight + i * cellSize + cellSize / 2 + 4}
              textAnchor="middle"
              fontSize="10"
              fill={value > 0.5 ? '#0A0A0A' : '#F3F4F6'}
              fontFamily="var(--font-inter, 'Inter', sans-serif)"
              fontWeight="500"
            >
              {value.toFixed(2)}
            </text>
          ))
        )}
      </svg>

      {/* Legend */}
      <div className="mt-6 flex items-center justify-center gap-4 text-xs">
        <div className="flex items-center gap-2">
          <div
            className="w-4 h-4"
            style={{ backgroundColor: '#1F2023' }}
          />
          <span className="text-[#9CA3AF]">Low trust</span>
        </div>
        <div className="flex items-center gap-2">
          <div
            className="w-4 h-4"
            style={{ backgroundColor: '#7DD3FC' }}
          />
          <span className="text-[#9CA3AF]">High trust</span>
        </div>
      </div>
    </div>
  );
}
