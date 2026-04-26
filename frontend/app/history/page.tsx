'use client';

import { useEffect, useState } from 'react';
import { apiGetHistory } from '@/lib/api';
import { HistoryRun } from '@/lib/types';
import RunsTable from '@/components/RunsTable';
import AggregateChartGrid from '@/components/AggregateChartGrid';

export default function History() {
  const [runs, setRuns] = useState<HistoryRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [scenarioFilter, setScenarioFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        const response = await apiGetHistory();
        setRuns(response.runs);
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

  const uniqueScenarios = Array.from(new Set(runs.map((r) => r.scenario)));
  const uniqueStatuses = Array.from(
    new Set(runs.map((r) => r.status || 'completed'))
  );

  return (
    <div className="main-content">
      {/* Header */}
      <div className="py-12 border-b border-[#1F2023]">
        <h1 className="text-3xl font-medium text-[#F3F4F6] mb-2">History</h1>
        <p className="text-[#9CA3AF] text-sm">
          {filteredRuns.length} run{filteredRuns.length !== 1 ? 's' : ''} recorded
        </p>
      </div>

      {/* Filters */}
      <div className="py-8 border-b border-[#1F2023]">
        <p className="uppercase-label mb-6">Filters</p>

        <div className="grid grid-cols-2 gap-6">
          {/* Scenario filter */}
          <div>
            <label className="block uppercase-label mb-2">Scenario</label>
            <select
              value={scenarioFilter}
              onChange={(e) => setScenarioFilter(e.target.value)}
              className="w-full px-4 py-2 bg-[#111113] border border-[#1F2023] text-[#F3F4F6] text-sm focus:outline-none focus:border-[#7DD3FC] transition-colors"
            >
              <option value="all">All Scenarios</option>
              {uniqueScenarios.map((scenario) => (
                <option key={scenario} value={scenario}>
                  {scenario.charAt(0).toUpperCase() + scenario.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* Status filter */}
          <div>
            <label className="block uppercase-label mb-2">Status</label>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="w-full px-4 py-2 bg-[#111113] border border-[#1F2023] text-[#F3F4F6] text-sm focus:outline-none focus:border-[#7DD3FC] transition-colors"
            >
              <option value="all">All Statuses</option>
              {uniqueStatuses.map((status) => (
                <option key={status} value={status}>
                  {status.charAt(0).toUpperCase() + status.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Table section */}
      <div className="py-12 border-b border-[#1F2023]">
        <p className="uppercase-label mb-6">All Runs</p>
        {loading ? (
          <div className="py-12 text-center">
            <p className="text-[#6B7280]">Loading history...</p>
          </div>
        ) : error ? (
          <div className="py-12 text-center">
            <p className="text-[#EF4444]">{error}</p>
          </div>
        ) : (
          <RunsTable runs={filteredRuns} />
        )}
      </div>

      {/* Charts section */}
      {!loading && filteredRuns.length > 0 && (
        <div className="py-12">
          <p className="uppercase-label mb-6">Trends Across Runs</p>
          <AggregateChartGrid runs={filteredRuns} />
        </div>
      )}
    </div>
  );
}
