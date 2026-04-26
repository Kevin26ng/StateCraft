'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import { HistoryRun } from '@/lib/types';
import { formatDate } from '@/lib/format';

interface RunsTableProps {
  runs: HistoryRun[];
}

type SortKey = 'date' | 'scenario' | 'society_score' | 'turns_survived' | 'difficulty_tier' | 'status';

export default function RunsTable({ runs }: RunsTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const sorted = useMemo(() => {
    const copy = [...runs];
    copy.sort((a, b) => {
      let aVal: any = a[sortKey];
      let bVal: any = b[sortKey];

      if (sortKey === 'date') {
        aVal = new Date(a.date).getTime();
        bVal = new Date(b.date).getTime();
      }

      if (aVal < bVal) return sortOrder === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortOrder === 'asc' ? 1 : -1;
      return 0;
    });
    return copy;
  }, [runs, sortKey, sortOrder]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortOrder('desc');
    }
  };

  const SortIndicator = ({ active }: { active: boolean }) => {
    if (!active) return null;
    return (
      <span className="text-[#7DD3FC]">
        {sortOrder === 'asc' ? ' ↑' : ' ↓'}
      </span>
    );
  };

  const getScoreColor = (score: number): string => {
    if (score >= 75) return '#34D399';
    if (score >= 60) return '#F59E0B';
    return '#EF4444';
  };

  if (runs.length === 0) {
    return (
      <div className="py-12 text-center">
        <p className="text-[#9CA3AF]">No runs recorded yet.</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-[#1F2023]">
            <th className="px-4 py-3 text-left">
              <button
                onClick={() => handleSort('status')}
                className="text-xs uppercase font-semibold text-[#6B7280] hover:text-[#9CA3AF] transition-colors"
              >
                Status
                <SortIndicator active={sortKey === 'status'} />
              </button>
            </th>
            <th className="px-4 py-3 text-left">
              <button
                onClick={() => handleSort('scenario')}
                className="text-xs uppercase font-semibold text-[#6B7280] hover:text-[#9CA3AF] transition-colors"
              >
                Scenario
                <SortIndicator active={sortKey === 'scenario'} />
              </button>
            </th>
            <th className="px-4 py-3 text-left">
              <button
                onClick={() => handleSort('turns_survived')}
                className="text-xs uppercase font-semibold text-[#6B7280] hover:text-[#9CA3AF] transition-colors"
              >
                Turns
                <SortIndicator active={sortKey === 'turns_survived'} />
              </button>
            </th>
            <th className="px-4 py-3 text-left">
              <button
                onClick={() => handleSort('society_score')}
                className="text-xs uppercase font-semibold text-[#6B7280] hover:text-[#9CA3AF] transition-colors"
              >
                Score
                <SortIndicator active={sortKey === 'society_score'} />
              </button>
            </th>
            <th className="px-4 py-3 text-left">
              <button
                onClick={() => handleSort('difficulty_tier')}
                className="text-xs uppercase font-semibold text-[#6B7280] hover:text-[#9CA3AF] transition-colors"
              >
                Difficulty
                <SortIndicator active={sortKey === 'difficulty_tier'} />
              </button>
            </th>
            <th className="px-4 py-3 text-left">
              <button
                onClick={() => handleSort('date')}
                className="text-xs uppercase font-semibold text-[#6B7280] hover:text-[#9CA3AF] transition-colors"
              >
                Date
                <SortIndicator active={sortKey === 'date'} />
              </button>
            </th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((run) => (
            <tr
              key={run.id}
              className="border-b border-[#1F2023] hover:bg-[#111113] transition-colors cursor-pointer"
            >
              <td className="px-4 py-4 text-sm">
                <Link href={`/run/${run.id}`} className="text-[#9CA3AF] hover:text-[#7DD3FC]">
                  {run.status || 'completed'}
                </Link>
              </td>
              <td className="px-4 py-4 text-sm">
                <Link href={`/run/${run.id}`} className="text-[#F3F4F6] hover:text-[#7DD3FC] capitalize">
                  {run.scenario}
                </Link>
              </td>
              <td className="px-4 py-4 text-sm">
                <Link href={`/run/${run.id}`} className="text-[#9CA3AF] hover:text-[#7DD3FC]">
                  {run.turns_survived}
                </Link>
              </td>
              <td className="px-4 py-4 text-sm font-medium">
                <Link href={`/run/${run.id}`} className="hover:opacity-80" style={{ color: getScoreColor(run.society_score) }}>
                  {Math.round(run.society_score)}
                </Link>
              </td>
              <td className="px-4 py-4 text-sm">
                <Link href={`/run/${run.id}`} className="text-[#9CA3AF] hover:text-[#7DD3FC]">
                  {run.difficulty_tier}
                </Link>
              </td>
              <td className="px-4 py-4 text-sm">
                <Link href={`/run/${run.id}`} className="text-[#9CA3AF] hover:text-[#7DD3FC]">
                  {formatDate(run.date)}
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
