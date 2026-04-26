'use client';

import { useEffect, useState } from 'react';
import { apiGetHistory } from '@/lib/api';
import { HistoryRun } from '@/lib/types';

export default function RecentRunsStrip() {
  const [runs, setRuns] = useState<HistoryRun[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiGetHistory()
      .then((res) => {
        const all = res.runs ?? [];
        setRuns(all.slice(-5).reverse());
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="py-8 border-t border-[#1F2023]">
      <p className="uppercase-label mb-4">Recent Runs</p>

      {loading ? (
        <p className="text-[#4B5563] text-sm animate-pulse">Loading...</p>
      ) : runs.length === 0 ? (
        <p className="text-[#4B5563] text-sm">No runs yet. Launch your first simulation above.</p>
      ) : (
        <div className="flex gap-3 overflow-x-auto pb-2">
          {runs.map((run) => {
            const status = run.status ?? 'completed';
            return (
              <div
                key={run.id}
                className="flex-shrink-0 w-[160px] border border-[#1F2023] bg-[#111113] p-4 hover:border-[#2A2A2E] transition-colors"
              >
                <p className="text-[10px] uppercase tracking-wider text-[#4B5563] mb-2 capitalize">
                  {run.scenario}
                </p>
                <p className="text-2xl font-medium tabular-numbers text-[#F3F4F6] mb-1">
                  {Math.round(run.society_score)}
                </p>
                <p className="text-xs text-[#6B7280] mb-2">{run.turns_survived} turns</p>
                <span
                  className={`text-[10px] uppercase font-semibold tracking-wider ${
                    status === 'collapsed' ? 'text-[#EF4444]' : 'text-[#34D399]'
                  }`}
                >
                  {status}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
