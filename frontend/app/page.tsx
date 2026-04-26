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

      await apiReset({
        scenario,
        episode_mode: mode,
        num_agents: 6,
      });

      // Generate run ID client-side and navigate
      const runId = crypto.randomUUID();
      router.push(`/run/${runId}`);
    } catch (err) {
      setError('Failed to launch simulation');
      console.error(err);
      setLoading(false);
    }
  };

  return (
    <div className="mission-control-content">
      {/* Hero Section */}
      <section className="py-16 fade-in-up">
        <h1
          className="text-5xl leading-tight mb-4 text-balance"
          style={{ fontFamily: "'Source Serif Pro', 'Georgia', serif" }}
        >
          Train multi-agent governance
          <br />
          under crisis.
        </h1>
        <p className="text-[#9CA3AF] text-sm leading-relaxed max-w-2xl mb-6">
          Six agents. Hidden goals. Negotiated coalitions. Run the simulation,
          observe emergence, audit alignment.
        </p>
        <button
          onClick={() => document.getElementById('launch-panel')?.scrollIntoView({ behavior: 'smooth' })}
          className="text-[#7DD3FC] text-sm font-semibold uppercase tracking-[0.08em] hover:text-[#BAE6FD] transition-colors group"
        >
          Launch New Run
          <span className="inline-block ml-1 transition-transform group-hover:translate-x-1">→</span>
        </button>
      </section>

      {/* Agent Roster Carousel */}
      <section className="py-8 fade-in-up" style={{ animationDelay: '100ms' }}>
        <AgentRosterCarousel />
      </section>

      {/* Configure & Launch Panel */}
      <section id="launch-panel" className="py-10 border-t border-[#1F2023] fade-in-up" style={{ animationDelay: '200ms' }}>
        <p className="uppercase-label mb-8">Launch Panel</p>

        <div className="grid grid-cols-2 gap-6 mb-6">
          {/* Scenario selector */}
          <div>
            <label className="block text-xs text-[#6B7280] mb-1.5">Scenario:</label>
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value as Scenario)}
              className="select-styled"
            >
              <option value="pandemic">Pandemic</option>
              <option value="economic">Economic Crisis</option>
              <option value="disaster">Natural Disaster</option>
            </select>
          </div>

          {/* Episode mode selector */}
          <div>
            <label className="block text-xs text-[#6B7280] mb-1.5">Mode:</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as EpisodeMode)}
              className="select-styled"
            >
              <option value="TRAINING">Training</option>
              <option value="DEMO">Demo</option>
              <option value="STRESS_TEST">Stress Test</option>
            </select>
          </div>

          {/* Steps (locked) */}
          <div>
            <label className="block text-xs text-[#6B7280] mb-1.5">Steps:</label>
            <div className="select-styled text-[#6B7280] cursor-default">
              30 turns
            </div>
          </div>

          {/* Speed indicator */}
          <div>
            <label className="block text-xs text-[#6B7280] mb-1.5">Speed: <span className="text-[#7DD3FC]">500ms</span></label>
            <div className="flex items-center gap-3 mt-1">
              <input
                type="range"
                min="50"
                max="1000"
                step="50"
                defaultValue="500"
                className="flex-1 accent-[#7DD3FC] h-1"
              />
            </div>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-4 px-4 py-2 bg-[#EF4444]/10 border border-[#EF4444]/30 text-[#EF4444] text-sm">
            {error}
          </div>
        )}

        {/* Launch button */}
        <div className="flex justify-end">
          <button
            onClick={handleLaunch}
            disabled={loading}
            className="text-[#7DD3FC] text-sm font-semibold uppercase tracking-[0.08em] hover:text-[#BAE6FD] transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Launching...' : 'Launch'}
            <span className="inline-block ml-1 transition-transform group-hover:translate-x-1">→</span>
          </button>
        </div>
      </section>

      {/* Recent Runs Strip */}
      <div className="fade-in-up" style={{ animationDelay: '300ms' }}>
        <RecentRunsStrip />
      </div>
    </div>
  );
}
