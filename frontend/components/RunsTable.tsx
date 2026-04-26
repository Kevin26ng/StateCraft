'use client';

import { HistoryRun } from '@/lib/types';
import { formatDate, getScoreTier } from '@/lib/format';

interface RunsTableProps {
  runs: HistoryRun[];
}

export default function RunsTable({ runs }: RunsTableProps) {
  if (runs.length === 0) {
    return (
      <div className="py-16 text-center">
        <p className="text-[#4B5563] text-sm">No runs match the current filters.</p>
      </div>
    );
  }

  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Date</th>
          <th>Scenario</th>
          <th>Mode</th>
          <th className="text-right">Turns</th>
          <th className="text-right">Difficulty</th>
          <th className="text-right">Score</th>
          <th>Outcome</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => {
          const { hex } = getScoreTier(run.society_score);
          const status = run.status ?? 'completed';
          return (
            <tr key={run.id}>
              <td className="text-[#6B7280] text-xs">{formatDate(run.date)}</td>
              <td className="capitalize text-[#9CA3AF]">{run.scenario}</td>
              <td className="text-[#6B7280] text-xs">{run.episode_mode}</td>
              <td className="text-right tabular-numbers text-[#9CA3AF]">{run.turns_survived}</td>
              <td className="text-right tabular-numbers text-[#6B7280]">{run.difficulty_tier}</td>
              <td className="text-right tabular-numbers font-medium" style={{ color: hex }}>
                {Math.round(run.society_score)}
              </td>
              <td>
                <span
                  className={`text-xs uppercase font-semibold tracking-wider ${
                    status === 'collapsed' ? 'text-[#EF4444]' : 'text-[#34D399]'
                  }`}
                >
                  {status}
                </span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
