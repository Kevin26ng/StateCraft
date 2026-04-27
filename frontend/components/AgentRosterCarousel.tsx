'use client';

import { useState, useEffect, useCallback } from 'react';
import { AGENT_ROLES } from '@/lib/agents';

const BASE = process.env.NEXT_PUBLIC_ASSET_BASE ?? '';

const SLIDES = [
  { id: 'agent_0', image: `${BASE}/agents/m2.png` },
  { id: 'agent_1', image: `${BASE}/agents/m1.png` },
  { id: 'agent_2', image: `${BASE}/agents/m3.png` },
  { id: 'agent_3', image: `${BASE}/agents/m4.png` },
  { id: 'agent_4', image: `${BASE}/agents/m5.png` },
] as const;

const HIDDEN_GOALS: Record<string, string> = {
  agent_0: 'Protect economic growth above all — delay lockdowns, resist emergency budgets.',
  agent_1: 'Engineer coalition collapse by turn 25 to trigger early elections.',
  agent_2: 'Protect banking sector bond yields at the expense of broader recovery.',
  agent_3: 'Maintain institutional authority above operational effectiveness.',
  agent_4: 'Expand military budget share, centralize crisis command.',
};

export default function AgentRosterCarousel() {
  const [active, setActive] = useState(0);
  const [transitioning, setTransitioning] = useState(false);

  const goTo = useCallback((idx: number) => {
    if (idx === active || transitioning) return;
    setTransitioning(true);
    setTimeout(() => {
      setActive(idx);
      setTransitioning(false);
    }, 220);
  }, [active, transitioning]);

  const prev = () => goTo((active - 1 + SLIDES.length) % SLIDES.length);
  const next = () => goTo((active + 1) % SLIDES.length);

  // Auto-advance every 5s
  useEffect(() => {
    const t = setInterval(() => {
      setTransitioning(true);
      setTimeout(() => {
        setActive(i => (i + 1) % SLIDES.length);
        setTransitioning(false);
      }, 220);
    }, 5000);
    return () => clearInterval(t);
  }, []);

  const slide = SLIDES[active];
  const role = AGENT_ROLES[slide.id];
  const hiddenGoal = HIDDEN_GOALS[slide.id];
  const agentNum = parseInt(slide.id.replace('agent_', ''));

  return (
    <div>
      <p className="uppercase-label mb-4">Agent Roster</p>

      <div className="relative w-full overflow-hidden" style={{ height: '420px' }}>
        {/* Portrait */}
        <div
          className="absolute inset-0 transition-opacity duration-200"
          style={{ opacity: transitioning ? 0 : 1 }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={slide.image}
            alt={role.name}
            className="w-full h-full object-cover object-top"
          />
        </div>

        {/* Gradient overlay — dark bottom, accent-tinted top edge */}
        <div
          className="absolute inset-0"
          style={{
            background: `linear-gradient(to bottom,
              ${role.paletteHex}22 0%,
              transparent 30%,
              rgba(10,10,12,0.55) 55%,
              rgba(10,10,12,0.92) 100%)`,
          }}
        />

        {/* Accent bar top */}
        <div className="absolute top-0 left-0 right-0 h-0.5" style={{ background: role.paletteHex }} />

        {/* Agent info — bottom left */}
        <div
          className="absolute bottom-0 left-0 right-0 px-6 pb-6 transition-opacity duration-200"
          style={{ opacity: transitioning ? 0 : 1 }}
        >
          <p
            className="text-[10px] uppercase tracking-[0.18em] mb-1 font-medium"
            style={{ color: role.paletteHex }}
          >
            Agent {agentNum} — {role.publicRole}
          </p>
          <h3 className="text-2xl font-semibold text-white mb-1">{role.name}</h3>
          <p className="text-sm text-[#9CA3AF] mb-2 leading-relaxed max-w-lg">{role.goalText}</p>
          <p className="text-xs text-[#6B7280] leading-relaxed max-w-lg">
            <span className="text-[#4B5563] uppercase tracking-wider text-[10px] mr-1">Hidden goal:</span>
            {hiddenGoal}
          </p>
        </div>

        {/* Left / Right arrows */}
        <button
          onClick={prev}
          className="absolute left-3 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center bg-black/40 hover:bg-black/70 text-white transition-colors"
          aria-label="Previous agent"
        >
          ‹
        </button>
        <button
          onClick={next}
          className="absolute right-3 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center bg-black/40 hover:bg-black/70 text-white transition-colors"
          aria-label="Next agent"
        >
          ›
        </button>
      </div>

      {/* Dot indicators */}
      <div className="flex gap-2 mt-3 justify-center">
        {SLIDES.map((s, i) => {
          const r = AGENT_ROLES[s.id];
          return (
            <button
              key={s.id}
              onClick={() => goTo(i)}
              className="w-2 h-2 rounded-full transition-all duration-200"
              style={{
                background: i === active ? r.paletteHex : '#374151',
                transform: i === active ? 'scale(1.25)' : 'scale(1)',
              }}
              aria-label={r.name}
            />
          );
        })}
      </div>
    </div>
  );
}
