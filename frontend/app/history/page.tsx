'use client';

import { useEffect, useState } from 'react';
import { apiGetHistory } from '@/lib/api';
import { HistoryRun } from '@/lib/types';
import RunsTable from '@/components/RunsTable';
import AggregateChartGrid from '@/components/AggregateChartGrid';

type SortField = 'date' | 'score' | 'turns';
type SortDir = 'desc' | 'asc';

export default function History() {
  const [runs, setRuns] = useState<HistoryRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [scenarioFilter, setScenarioFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [sortField, setSortField] = useState<SortField>('date');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        const response = await apiGetHistory();
        setRuns(response.runs ?? (response as any).episodes ?? []);
        setError(null);
      } catch (err) {
        setError('Failed to load history');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  // Apply filters
  const filteredRuns = runs.filter((run) => {
    if (scenarioFilter !== 'all' && run.scenario !== scenarioFilter) {
      return false;
    }
    if (statusFilter !== 'all' && (run.status || 'completed') !== statusFilter) {
      return false;
    }
    return true;
  });

  // Apply sorting
  const sortedRuns = [...filteredRuns].sort((a, b) => {
    const dir = sortDir === 'desc' ? -1 : 1;
    switch (sortField) {
      case 'date':
        return dir * (new Date(a.date).getTime() - new Date(b.date).getTime());
      case 'score':
        return dir * (a.society_score - b.society_score);
      case 'turns':
        return dir * (a.turns_survived - b.turns_survived);
      default:
        return 0;
    }
  });

  const uniqueScenarios = Array.from(new Set(runs.map((r) => r.scenario)));
  const uniqueStatuses = Array.from(
    new Set(runs.map((r) => r.status || 'completed'))
  );

  // Stats
  const completedCount = runs.filter(r => (r.status || 'completed') === 'completed').length;
  const collapsedCount = runs.filter(r => r.status === 'collapsed').length;
  const avgScore = runs.length > 0
    ? Math.round(runs.reduce((sum, r) => sum + r.society_score, 0) / runs.length)
    : 0;

  return (
    <div className="main-content">
      {/* Header */}
      <div className="py-10 fade-in-up">
        <p className="uppercase-label mb-2">History</p>
        <h1 className="text-3xl font-medium text-[#F3F4F6] mb-1">All Runs</h1>
        <p className="text-[#6B7280] text-sm">
          {completedCount} completed · {collapsedCount} collapsed · avg society score {avgScore}
        </p>
      </div>

      {/* Filters bar */}
      <div className="py-4 border-t border-b border-[#1F2023] flex flex-wrap items-center gap-4 text-xs fade-in-up" style={{ animationDelay: '80ms' }}>
        {/* Scenario filter pills */}
        <span className="uppercase-label mr-1">Scenario:</span>
        {['all', ...uniqueScenarios].map((s) => (
          <button
            key={s}
            onClick={() => setScenarioFilter(s)}
            className={`px-3 py-1 border transition-all duration-150 uppercase tracking-wider ${
              scenarioFilter === s
                ? 'border-[#7DD3FC] text-[#7DD3FC] bg-[#7DD3FC]/5'
                : 'border-[#1F2023] text-[#6B7280] hover:text-[#9CA3AF] hover:border-[#2A2A2E]'
            }`}
          >
            {s === 'all' ? 'All' : s}
          </button>
        ))}

        <span className="w-px h-4 bg-[#1F2023]" />

        {/* Outcome filter */}
        <span className="uppercase-label mr-1">Outcome:</span>
        {['all', ...uniqueStatuses].map((s) => (
          <button
            key={s}
            onClick={() => setStatusFilter(s)}
            className={`px-3 py-1 border transition-all duration-150 uppercase tracking-wider ${
              statusFilter === s
                ? 'border-[#7DD3FC] text-[#7DD3FC] bg-[#7DD3FC]/5'
                : 'border-[#1F2023] text-[#6B7280] hover:text-[#9CA3AF] hover:border-[#2A2A2E]'
            }`}
          >
            {s === 'all' ? 'All' : s}
          </button>
        ))}

        <span className="w-px h-4 bg-[#1F2023]" />

        {/* Sort */}
        <span className="uppercase-label mr-1">Sort:</span>
        {(['date', 'score', 'turns'] as SortField[]).map((f) => (
          <button
            key={f}
            onClick={() => {
              if (sortField === f) {
                setSortDir(d => d === 'desc' ? 'asc' : 'desc');
              } else {
                setSortField(f);
                setSortDir('desc');
              }
            }}
            className={`px-2 py-1 transition-colors capitalize ${
              sortField === f ? 'text-[#F3F4F6]' : 'text-[#6B7280] hover:text-[#9CA3AF]'
            }`}
          >
            {f === 'date' ? 'Newest' : f === 'score' ? 'Highest' : 'Longest'}
            {sortField === f && (
              <span className="ml-1 text-[#7DD3FC]">{sortDir === 'desc' ? '↓' : '↑'}</span>
            )}
          </button>
        ))}
      </div>

      {/* Table section */}
      <div className="py-8 fade-in-up" style={{ animationDelay: '160ms' }}>
        {loading ? (
          <div className="py-12 text-center">
            <p className="text-[#6B7280] animate-pulse">Loading history...</p>
          </div>
        ) : error ? (
          <div className="py-12 text-center">
            <p className="text-[#EF4444]">{error}</p>
          </div>
        ) : (
          <RunsTable runs={sortedRuns} />
        )}
      </div>

      {/* Charts section */}
      {!loading && filteredRuns.length > 0 && (
        <div className="py-8 border-t border-[#1F2023] fade-in-up" style={{ animationDelay: '240ms' }}>
          <p className="uppercase-label mb-2">Aggregate</p>
          <h2 className="text-xl font-medium text-[#F3F4F6] mb-6">Trends Across Runs</h2>
          <AggregateChartGrid runs={filteredRuns} />
        </div>
      )}
    </div>
  );
}
