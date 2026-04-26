'use client';

import { useState } from 'react';
import { StepResponse } from '@/lib/types';
import TrustMatrix from './TrustMatrix';
import ActivityLog from './ActivityLog';
import AuditorPanel from './AuditorPanel';

interface TabbedPanelProps {
  stepResponse: StepResponse | null;
}

type TabId = 'matrix' | 'activity' | 'audit';

export default function TabbedPanel({ stepResponse }: TabbedPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>('matrix');

  if (!stepResponse) {
    return (
      <div className="py-12 text-center">
        <p className="text-[#6B7280]">Loading panel data...</p>
      </div>
    );
  }

  const tabs: Array<{ id: TabId; label: string }> = [
    { id: 'matrix', label: 'Matrix' },
    { id: 'activity', label: 'Activity' },
    { id: 'audit', label: 'Audit' },
  ];

  return (
    <div>
      {/* Tab navigation */}
      <div className="px-12 border-b border-[#1F2023] flex gap-0">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                isActive
                  ? 'border-b-[#7DD3FC] text-[#7DD3FC]'
                  : 'border-b-transparent text-[#9CA3AF] hover:text-[#F3F4F6]'
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <div className="px-12 py-8">
        {activeTab === 'matrix' && (
          <TrustMatrix trustMatrix={stepResponse.trust_matrix} />
        )}

        {activeTab === 'activity' && (
          <ActivityLog
            events={stepResponse.events}
            turn={stepResponse.state.turn}
          />
        )}

        {activeTab === 'audit' && (
          <AuditorPanel auditorReport={stepResponse.auditor_report} />
        )}
      </div>
    </div>
  );
}
