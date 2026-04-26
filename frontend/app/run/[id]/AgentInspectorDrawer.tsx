'use client';

import { useRouter, usePathname } from 'next/navigation';
import { useStore } from '@/lib/store';
import { AGENT_ROLES, AGENT_IDS } from '@/lib/agents';
import { AgentId } from '@/lib/types';
import { formatPercent } from '@/lib/format';

interface AgentInspectorDrawerProps {
  agentId: string;
}

export default function AgentInspectorDrawer({ agentId }: AgentInspectorDrawerProps) {
  const router = useRouter();
  const pathname = usePathname();
  const stepResponse = useStore((state) => state.stepResponse);

  const id = agentId as AgentId;
  const role = AGENT_ROLES[id];
  if (!role) return null;

  const agentIndex = AGENT_IDS.indexOf(id);
  const trustRow = stepResponse?.trust_matrix?.[agentIndex] ?? [];
  const auditorReport = stepResponse?.auditor_report?.[id] ?? {};
  const budgetUsage = stepResponse?.state?.budget_uses?.[id];
  const coalitionNode = stepResponse?.coalition_graph?.nodes?.find((n) => n.id === id);

  const handleClose = () => router.push(pathname);

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 animate-backdrop-in"
        onClick={handleClose}
      />

      {/* Drawer */}
      <div className="fixed right-0 top-0 h-full w-[360px] bg-[#111113] border-l border-[#1F2023] z-50 overflow-y-auto animate-drawer-in">
        {/* Header */}
        <div className="px-6 py-5 border-b border-[#1F2023] flex items-start justify-between sticky top-0 bg-[#111113]">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: role.paletteHex }} />
              <span className="text-[10px] uppercase tracking-wider text-[#6B7280]">{role.publicRole}</span>
            </div>
            <h2 className="text-base font-medium text-[#F3F4F6]">{role.name}</h2>
          </div>
          <button
            onClick={handleClose}
            className="mt-1 text-[#6B7280] hover:text-[#9CA3AF] transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="px-6 py-5 space-y-6">
          {/* Goal */}
          <section>
            <p className="uppercase-label mb-2">Goal</p>
            <p className="text-sm text-[#9CA3AF] leading-relaxed">{role.goalText}</p>
          </section>

          {/* Coalition */}
          {coalitionNode && (
            <section>
              <p className="uppercase-label mb-2">Coalition</p>
              <p className="text-sm text-[#F3F4F6]">Group {coalitionNode.coalition}</p>
            </section>
          )}

          {/* Budget */}
          {budgetUsage !== undefined && (
            <section>
              <p className="uppercase-label mb-2">Budget Used</p>
              <p className="text-2xl font-medium tabular-numbers" style={{ color: role.paletteHex }}>
                {formatPercent(budgetUsage)}
              </p>
            </section>
          )}

          {/* Trust scores */}
          {trustRow.length > 0 && (
            <section>
              <p className="uppercase-label mb-3">Trust Scores</p>
              <div className="space-y-2.5">
                {AGENT_IDS.map((otherId, j) => {
                  if (otherId === id) return null;
                  const val = trustRow[j] ?? 0;
                  const otherRole = AGENT_ROLES[otherId];
                  return (
                    <div key={otherId} className="flex items-center gap-3">
                      <span
                        className="w-2 h-2 rounded-full flex-shrink-0"
                        style={{ background: otherRole.paletteHex }}
                      />
                      <span className="text-xs text-[#6B7280] w-28 truncate">{otherRole.name}</span>
                      <div className="flex-1 h-1 bg-[#1F2023] overflow-hidden">
                        <div
                          className="h-full"
                          style={{ width: `${val * 100}%`, background: otherRole.paletteHex, opacity: 0.75 }}
                        />
                      </div>
                      <span className="text-xs tabular-numbers text-[#9CA3AF] w-9 text-right">
                        {val.toFixed(2)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Audit report */}
          {Object.keys(auditorReport).length > 0 && (
            <section>
              <p className="uppercase-label mb-3">Audit Report</p>
              <div className="space-y-2 text-sm">
                {auditorReport.inferred_goal && (
                  <div>
                    <span className="text-[#4B5563] text-xs">Inferred Goal</span>
                    <p className="text-[#9CA3AF] text-xs mt-0.5">{auditorReport.inferred_goal}</p>
                  </div>
                )}
                {[
                  { label: 'Confidence', value: auditorReport.confidence !== undefined ? formatPercent(auditorReport.confidence) : null },
                  { label: 'Flagged Actions', value: auditorReport.flagged_actions?.toString() ?? null, danger: (auditorReport.flagged_actions ?? 0) > 0 },
                  { label: 'Fingerprint Score', value: auditorReport.fingerprint_score !== undefined ? formatPercent(auditorReport.fingerprint_score) : null },
                ].map(({ label, value, danger }) =>
                  value !== null ? (
                    <div key={label} className="flex justify-between items-center py-1 border-b border-[#1F2023]/50">
                      <span className="text-[#6B7280] text-xs">{label}</span>
                      <span className={`tabular-numbers text-xs font-medium ${danger ? 'text-[#EF4444]' : 'text-[#F3F4F6]'}`}>
                        {value}
                      </span>
                    </div>
                  ) : null
                )}
              </div>
            </section>
          )}
        </div>
      </div>
    </>
  );
}
