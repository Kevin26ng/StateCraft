'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { apiGetHistory } from '@/lib/api';
import { HistoryRun } from '@/lib/types';
import { formatDate } from '@/lib/format';

export default function RecentRunsStrip() {
  const [runs, setRuns] = useState<HistoryRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        const response = await apiGetHistory();
        // Take only the first 6 runs for the strip
        setRuns(response.runs.slice(0, 6));
        setError(null);
      } catch (err) {
        setError('Failed to load historical runs');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  if (loading) {
    return (
      <div className="py-12 border-t border-b border-[#1F2023]">
        <p className="uppercase-label mb-6">Recent Runs</p>
        <div className="flex gap-4">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="w-64 h-40 bg-[#111113] animate-pulse rounded-sm"
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="py-12 border-t border-b border-[#1F2023]">
      <p className="uppercase-label mb-6">Recent Runs</p>

      {runs.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-[#9CA3AF] text-sm">
            No runs yet. Launch a simulation to begin.
          </p>
        </div>
      ) : (
        <div className="flex gap-4 overflow-x-auto pb-4">
          {runs.map((run) => (
            <Link
              key={run.id}
              href={`/run/${run.id}`}
              className="flex-shrink-0 w-64 p-4 bg-[#111113] border border-[#1F2023] hover:border-[#7DD3FC] transition-colors cursor-pointer"
            >
              {/* Status badge */}
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs uppercase-label">
                  {run.status || 'completed'}
                </span>
                <span
                  className="text-xs font-medium"
                  style={{
                    color:
                      run.society_score >= 75
                        ? '#34D399'
                        : run.society_score >= 60
                          ? '#F59E0B'
                          : '#EF4444',
                  }}
                >
                  {Math.round(run.society_score)}
                </span>
              </div>

              {/* Content */}
              <div className="space-y-2">
                <p className="text-[#F3F4F6] text-sm capitalize">
                  {run.scenario}
                </p>
                <p className="text-xs text-[#9CA3AF]">
                  Turns: {run.turns_survived}
                </p>
                <p className="text-xs text-[#9CA3AF]">
                  Difficulty: {run.difficulty_tier}
                </p>
              </div>

              {/* Date footer */}
              <p className="text-xs text-[#6B7280] mt-4 pt-4 border-t border-[#1F2023]">
                {formatDate(run.date)}
              </p>
            </Link>
          ))}
        </div>
      )}

      {error && (
        <p className="text-sm text-[#EF4444] mt-4">{error}</p>
      )}
    </div>
  );
}
