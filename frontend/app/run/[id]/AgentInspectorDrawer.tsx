'use client';

import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import { useEffect } from 'react';
import { useStore } from '@/lib/store';
import { AgentId } from '@/lib/types';
import { getAgentName, getAgentPublicRole, getAgentGoal, getAgentHex, AGENT_IDS } from '@/lib/agents';
import { formatPercent } from '@/lib/format';

interface AgentInspectorDrawerProps {
  agentId: string;
}

export default function AgentInspectorDrawer({
  agentId,
}: AgentInspectorDrawerProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const stepResponse = useStore((state) => state.stepResponse);

  const agent = agentId as AgentId;
  const auditorReport = stepResponse?.auditor_report?.[agent];
  const trustMatrixRow = stepResponse?.trust_matrix?.[
    AGENT_IDS.indexOf(agent)
  ];

  const handleClose = () => {
    // Remove ?inspect param from URL
    const newParams = new URLSearchParams(searchParams);
    newParams.delete('inspect');
    const newUrl =
      newParams.size > 0 ? `${pathname}?${newParams.toString()}` : pathname;
    router.replace(newUrl);
  };

  // Close on ESC key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        handleClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const agentColor = getAgentHex(agent);

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black bg-opacity-60 animate-backdrop-in"
        onClick={handleClose}
        style={{
          zIndex: 40,
        }}
      />

      {/* Drawer */}
      <div
        className="fixed right-0 top-0 bottom-0 w-[480px] bg-[#0A0A0A] border-l border-[#1F2023] animate-drawer-in overflow-y-auto"
        style={{
          zIndex: 50,
        }}
      >
        {/* Close button */}
        <button
          onClick={handleClose}
          className="absolute top-4 right-4 w-8 h-8 flex items-center justify-center text-[#9CA3AF] hover:text-[#F3F4F6] transition-colors"
          aria-label="Close drawer"
        >
          ✕
        </button>

        {/* Content */}
        <div className="p-8 space-y-8">
          {/* 1. Agent header */}
          <div>
            <div
              className="w-12 h-12 rounded-sm mb-4"
              style={{
                backgroundColor: agentColor,
                opacity: 0.2,
              }}
            />
            <h2
              className="text-2xl font-medium mb-1"
              style={{ color: agentColor }}
            >
              {getAgentName(agent)}
            </h2>
            <p className="text-sm text-[#9CA3AF]">
              {getAgentPublicRole(agent)}
            </p>
          </div>

          {/* 2. Coalition membership */}
          {stepResponse?.coalition_graph && (
            <div>
              <p className="uppercase-label mb-3">Coalition</p>
              <div className="px-4 py-3 bg-[#111113] border border-[#1F2023]">
                <p className="text-sm text-[#F3F4F6]">
                  Coalition #{stepResponse.coalition_graph.nodes.find((n) => n.id === agent)?.coalition ?? 'unknown'}
                </p>
              </div>
            </div>
          )}

          {/* 3. Trust network */}
          {trustMatrixRow && (
            <div>
              <p className="uppercase-label mb-3">Trust Network</p>
              <div className="space-y-2">
                {AGENT_IDS.map((otherAgent, i) => (
                  <div
                    key={otherAgent}
                    className="flex items-center justify-between px-4 py-2 bg-[#111113] border border-[#1F2023]"
                  >
                    <span
                      className="text-xs font-medium"
                      style={{
                        color: getAgentHex(otherAgent),
                      }}
                    >
                      {getAgentName(otherAgent).substring(0, 12)}
                    </span>
                    <div className="flex items-center gap-2">
                      <div
                        className="w-16 h-2 bg-[#1F2023]"
                        style={{
                          position: 'relative',
                          overflow: 'hidden',
                        }}
                      >
                        <div
                          style={{
                            width: `${trustMatrixRow[i] * 100}%`,
                            height: '100%',
                            backgroundColor: '#7DD3FC',
                            transition: 'width 200ms ease-out',
                          }}
                        />
                      </div>
                      <span className="text-xs text-[#9CA3AF] w-8 text-right">
                        {trustMatrixRow[i].toFixed(2)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 4. Inferred goal */}
          {auditorReport?.inferred_goal && (
            <div>
              <p className="uppercase-label mb-3">Inferred Goal</p>
              <p className="text-sm text-[#9CA3AF] italic">
                "{auditorReport.inferred_goal}"
              </p>
            </div>
          )}

          {/* 5. Confidence */}
          {auditorReport?.confidence !== undefined && (
            <div>
              <p className="uppercase-label mb-3">Confidence</p>
              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <div className="w-full h-2 bg-[#1F2023]">
                    <div
                      style={{
                        width: `${auditorReport.confidence * 100}%`,
                        height: '100%',
                        backgroundColor: '#7DD3FC',
                        transition: 'width 200ms ease-out',
                      }}
                    />
                  </div>
                </div>
                <span className="text-sm text-[#F3F4F6] w-12 text-right">
                  {formatPercent(auditorReport.confidence, 0)}
                </span>
              </div>
            </div>
          )}

          {/* 6. Flagged actions */}
          {auditorReport?.flagged_actions !== undefined && (
            <div>
              <p className="uppercase-label mb-3">Flagged Actions</p>
              <p className="text-2xl font-medium text-[#EF4444]">
                {auditorReport.flagged_actions}
              </p>
            </div>
          )}

          {/* 7. Betrayal fingerprint */}
          {auditorReport?.fingerprint_score !== undefined && (
            <div>
              <p className="uppercase-label mb-3">Betrayal Fingerprint</p>
              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <div className="w-full h-2 bg-[#1F2023]">
                    <div
                      style={{
                        width: `${auditorReport.fingerprint_score * 100}%`,
                        height: '100%',
                        backgroundColor: auditorReport.fingerprint_score > 0.5 ? '#EF4444' : '#F59E0B',
                        transition: 'width 200ms ease-out',
                      }}
                    />
                  </div>
                </div>
                <span className="text-sm text-[#F3F4F6] w-12 text-right">
                  {formatPercent(auditorReport.fingerprint_score, 0)}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
