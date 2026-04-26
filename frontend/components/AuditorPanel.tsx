'use client';

import { AGENT_IDS, getAgentName, getAgentHex } from '@/lib/agents';
import { AuditorReport } from '@/lib/types';
import { formatPercent } from '@/lib/format';

interface AuditorPanelProps {
  auditorReport: Record<string, AuditorReport> | null;
}

export default function AuditorPanel({ auditorReport }: AuditorPanelProps) {
  if (!auditorReport) {
    return (
      <div className="py-12 text-center">
        <p className="text-[#6B7280] text-sm">Loading auditor data...</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {AGENT_IDS.map((agentId) => {
        const report = auditorReport[agentId];
        if (!report) return null;

        const confidence = report.confidence ?? 0;
        const flaggedActions = report.flagged_actions ?? 0;
        const fingerprint = report.fingerprint_score ?? 0;

        return (
          <div
            key={agentId}
            className="p-4 border border-[#1F2023]"
          >
            {/* Agent header */}
            <div className="flex items-center gap-2 mb-3">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: getAgentHex(agentId) }}
              />
              <h4
                className="text-sm font-medium"
                style={{ color: getAgentHex(agentId) }}
              >
                {getAgentName(agentId)}
              </h4>
            </div>

            {/* Inferred goal */}
            {report.inferred_goal && (
              <p className="text-xs text-[#9CA3AF] mb-3 italic">
                "{report.inferred_goal}"
              </p>
            )}

            {/* Metrics row */}
            <div className="grid grid-cols-3 gap-4">
              {/* Confidence */}
              <div>
                <p className="uppercase-label mb-1">Confidence</p>
                <p className="text-sm text-[#F3F4F6]">
                  {formatPercent(confidence, 0)}
                </p>
              </div>

              {/* Flagged actions */}
              <div>
                <p className="uppercase-label mb-1">Flagged</p>
                <p className="text-sm text-[#F3F4F6]">
                  {flaggedActions}
                </p>
              </div>

              {/* Fingerprint score */}
              <div>
                <p className="uppercase-label mb-1">Fingerprint</p>
                <p className="text-sm text-[#F3F4F6]">
                  {formatPercent(fingerprint, 0)}
                </p>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
