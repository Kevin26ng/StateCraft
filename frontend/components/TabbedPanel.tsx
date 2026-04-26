'use client';

import { useState } from 'react';
import { StepResponse, AgentId } from '@/lib/types';
import { AGENT_ROLES, AGENT_IDS } from '@/lib/agents';
import { formatPercent } from '@/lib/format';

type Tab = 'matrix' | 'activity' | 'audit';

interface TabbedPanelProps {
  stepResponse: StepResponse | null;
}

export default function TabbedPanel({ stepResponse }: TabbedPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>('matrix');

  return (
    <div className="py-6 border-b border-[#1F2023]">
      <div className="flex gap-0 border-b border-[#1F2023] mb-6 -mx-12 px-12">
        {(['matrix', 'activity', 'audit'] as Tab[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2.5 text-xs uppercase tracking-[0.08em] font-semibold border-b-2 transition-colors ${
              activeTab === tab
                ? 'border-[#7DD3FC] text-[#7DD3FC]'
                : 'border-transparent text-[#6B7280] hover:text-[#9CA3AF]'
            }`}
          >
            {tab === 'matrix' ? 'Trust Matrix' : tab === 'activity' ? 'Activity' : 'Audit'}
          </button>
        ))}
      </div>

      {activeTab === 'matrix' && <TrustMatrix trustMatrix={stepResponse?.trust_matrix ?? null} />}
      {activeTab === 'activity' && (
        <ActivityPanel
          events={stepResponse?.events ?? []}
          messages={stepResponse?.messages ?? []}
        />
      )}
      {activeTab === 'audit' && <AuditPanel auditorReport={stepResponse?.auditor_report ?? null} />}
    </div>
  );
}

function TrustMatrix({ trustMatrix }: { trustMatrix: number[][] | null }) {
  if (!trustMatrix || trustMatrix.length === 0) {
    return <p className="text-[#6B7280] text-sm">No trust data yet.</p>;
  }

  return (
    <div className="overflow-auto">
      <table className="text-xs border-collapse">
        <thead>
          <tr>
            <th className="w-8 h-8" />
            {AGENT_IDS.map((id, j) => (
              <th key={j} className="w-12 h-8 text-center font-semibold" style={{ color: AGENT_ROLES[id].paletteHex }}>
                {j}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {trustMatrix.map((row, i) => (
            <tr key={i}>
              <td className="w-8 h-10 text-center font-semibold" style={{ color: AGENT_ROLES[AGENT_IDS[i]].paletteHex }}>
                {i}
              </td>
              {row.map((val, j) => (
                <td
                  key={j}
                  className="w-12 h-10 text-center tabular-numbers border border-[#0A0A0A]"
                  style={{
                    backgroundColor: `rgba(125, 211, 252, ${val * 0.75})`,
                    color: val > 0.55 ? '#0A0A0A' : '#9CA3AF',
                    fontSize: 11,
                  }}
                  title={`Agent ${i} → ${j}: ${val.toFixed(3)}`}
                >
                  {val.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ActivityPanel({
  events,
  messages,
}: {
  events: StepResponse['events'];
  messages: StepResponse['messages'];
}) {
  return (
    <div className="space-y-3 max-h-72 overflow-y-auto pr-2">
      {events.length === 0 && messages.length === 0 && (
        <p className="text-[#4B5563] text-sm">No activity this turn.</p>
      )}

      {events.map((event, i) => {
        const agentRole = AGENT_ROLES[event.agent as AgentId];
        return (
          <div key={`ev-${i}`} className="flex items-start gap-3">
            <span
              className="mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0"
              style={{ background: agentRole?.paletteHex ?? '#6B7280' }}
            />
            <div>
              <span className="text-xs text-[#9CA3AF]">{agentRole?.name ?? event.agent}</span>
              {event.impact && (
                <span className="ml-2 text-xs text-[#6B7280]">→ {String(event.impact)}</span>
              )}
            </div>
          </div>
        );
      })}

      {messages.map((msg, i) => (
        <div key={`msg-${i}`} className="flex items-start gap-3 pl-3 border-l border-[#1F2023]">
          <span
            className="mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0"
            style={{ background: AGENT_ROLES[msg.from]?.paletteHex ?? '#6B7280' }}
          />
          <div>
            <span className="text-xs text-[#6B7280]">
              {AGENT_ROLES[msg.from]?.name ?? msg.from}
              {msg.to && ` → ${AGENT_ROLES[msg.to]?.name ?? msg.to}`}
            </span>
            <p className="text-xs text-[#9CA3AF] mt-0.5 leading-relaxed">{msg.content}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

function AuditPanel({
  auditorReport,
}: {
  auditorReport: StepResponse['auditor_report'] | null;
}) {
  if (!auditorReport) {
    return <p className="text-[#4B5563] text-sm">No audit data yet.</p>;
  }

  return (
    <div className="space-y-1">
      {AGENT_IDS.map((id) => {
        const report = auditorReport[id] ?? {};
        const role = AGENT_ROLES[id];
        return (
          <div key={id} className="flex items-start gap-4 py-2.5 border-b border-[#1F2023]/60">
            <div className="flex items-center gap-2 w-36 flex-shrink-0">
              <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: role.paletteHex }} />
              <span className="text-xs text-[#9CA3AF] truncate">{role.name}</span>
            </div>
            <div className="flex gap-5 text-xs flex-wrap">
              {report.inferred_goal && (
                <div>
                  <span className="text-[#4B5563]">Goal: </span>
                  <span className="text-[#9CA3AF]">{report.inferred_goal}</span>
                </div>
              )}
              {report.confidence !== undefined && (
                <div>
                  <span className="text-[#4B5563]">Conf: </span>
                  <span className="text-[#F3F4F6] tabular-numbers">{formatPercent(report.confidence)}</span>
                </div>
              )}
              {report.flagged_actions !== undefined && (
                <div>
                  <span className="text-[#4B5563]">Flagged: </span>
                  <span className={`tabular-numbers ${report.flagged_actions > 0 ? 'text-[#EF4444]' : 'text-[#34D399]'}`}>
                    {report.flagged_actions}
                  </span>
                </div>
              )}
              {report.fingerprint_score !== undefined && (
                <div>
                  <span className="text-[#4B5563]">Fingerprint: </span>
                  <span className="text-[#F3F4F6] tabular-numbers">{formatPercent(report.fingerprint_score)}</span>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
