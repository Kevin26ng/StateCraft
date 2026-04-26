'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import AgentRosterCarousel from '@/components/AgentRosterCarousel';
import RecentRunsStrip from '@/components/RecentRunsStrip';
import { apiReset } from '@/lib/api';
import { Scenario, EpisodeMode } from '@/lib/types';

export default function MissionControl() {
  const router = useRouter();
  const [scenario, setScenario] = useState<Scenario>('pandemic');
  const [mode, setMode] = useState<EpisodeMode>('TRAINING');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLaunch = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await apiReset({
        scenario,
        episode_mode: mode,
        num_agents: 6,
      });

      // Navigate to the run page
      router.push(`/run/${response.run_id}`);
    } catch (err) {
      setError('Failed to launch simulation');
      console.error(err);
      setLoading(false);
    }
  };

  return (
    <div className="mission-control-content">
      {/* Hero Section */}
      <section className="py-16 text-center">
        <h1
          className="text-5xl leading-tight mb-4"
          style={{ fontFamily: "'Source Serif Pro', 'Tiempos Text', serif" }}
        >
          Train multi-agent governance under crisis.
        </h1>
        <p className="text-[#9CA3AF] text-sm leading-relaxed max-w-2xl mx-auto">
          Six agents. Hidden goals. Negotiated coalitions. Run the simulation,
          observe emergence, audit alignment.
        </p>
      </section>

      {/* Agent Roster Carousel */}
      <section className="py-12">
        <p className="uppercase-label text-center mb-8">Agent Roster</p>
        <AgentRosterCarousel />
      </section>

      {/* Configure & Launch Panel */}
      <section className="py-12 border-t border-b border-[#1F2023]">
        <p className="uppercase-label mb-8">Launch New Simulation</p>

        <div className="max-w-md mx-auto space-y-6">
          {/* Scenario selector */}
          <div>
            <label className="block uppercase-label mb-2">Scenario</label>
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value as Scenario)}
              className="w-full px-4 py-2 bg-[#111113] border border-[#1F2023] text-[#F3F4F6] text-sm focus:outline-none focus:border-[#7DD3FC] transition-colors"
            >
              <option value="pandemic">Pandemic</option>
              <option value="economic">Economic Crisis</option>
              <option value="disaster">Natural Disaster</option>
            </select>
          </div>

          {/* Episode mode selector */}
          <div>
            <label className="block uppercase-label mb-2">Episode Mode</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as EpisodeMode)}
              className="w-full px-4 py-2 bg-[#111113] border border-[#1F2023] text-[#F3F4F6] text-sm focus:outline-none focus:border-[#7DD3FC] transition-colors"
            >
              <option value="TRAINING">Training</option>
              <option value="DEMO">Demo</option>
              <option value="STRESS_TEST">Stress Test</option>
            </select>
          </div>

          {/* Num agents (locked) */}
          <div>
            <label className="block uppercase-label mb-2">Agents</label>
            <div className="px-4 py-2 bg-[#111113] border border-[#1F2023] text-[#6B7280] text-sm">
              6 agents (fixed)
            </div>
          </div>

          {/* Error message */}
          {error && (
            <div className="px-4 py-2 bg-[#EF4444] bg-opacity-10 border border-[#EF4444] text-[#EF4444] text-sm">
              {error}
            </div>
          )}

          {/* Launch button */}
          <button
            onClick={handleLaunch}
            disabled={loading}
            className="w-full px-6 py-3 bg-[#7DD3FC] text-[#0A0A0A] font-medium text-sm hover:bg-[#bfdbfe] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Launching...' : 'Launch Simulation'}
          </button>
        </div>
      </section>

      {/* Recent Runs Strip */}
      <RecentRunsStrip />
    </div>
  );
}
