'use client';

import { useState } from 'react';
import { AGENT_IDS, AGENT_ROLES } from '@/lib/agents';

export default function AgentRosterCarousel() {
  const [currentIndex, setCurrentIndex] = useState(0);

  const agent = AGENT_ROLES[AGENT_IDS[currentIndex]];

  const handlePrev = () => {
    setCurrentIndex((prev) => (prev === 0 ? AGENT_IDS.length - 1 : prev - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prev) => (prev === AGENT_IDS.length - 1 ? 0 : prev + 1));
  };

  return (
    <div className="flex flex-col items-center gap-8 py-12">
      {/* Agent portrait/gradient */}
      <div className="relative w-80 h-80">
        {/* Gradient placeholder for agent portrait */}
        <div
          className="w-full h-full rounded-sm flex items-center justify-center text-white font-medium"
          style={{
            background: `linear-gradient(135deg, ${agent.paletteHex}20 0%, ${agent.paletteHex}40 100%)`,
            border: `1px solid ${agent.paletteHex}40`,
          }}
        >
          {agent.name}
        </div>
      </div>

      {/* Agent info */}
      <div className="text-center max-w-md">
        <h3 className="text-2xl font-medium text-[#F3F4F6] mb-1">
          {agent.name}
        </h3>
        <p className="text-sm text-[#9CA3AF] mb-4">{agent.publicRole}</p>
        <p className="text-[13px] text-[#9CA3AF] leading-relaxed">
          {agent.goalText}
        </p>
      </div>

      {/* Navigation buttons */}
      <div className="flex gap-4">
        <button
          onClick={handlePrev}
          className="px-6 py-2 border border-[#1F2023] text-[#9CA3AF] hover:text-[#F3F4F6] hover:border-[#7DD3FC] transition-colors text-sm"
          aria-label="Previous agent"
        >
          ← Previous
        </button>
        <div className="flex items-center gap-2">
          {AGENT_IDS.map((_, index) => (
            <button
              key={index}
              onClick={() => setCurrentIndex(index)}
              className={`w-2 h-2 rounded-full transition-colors ${
                index === currentIndex
                  ? 'bg-[#7DD3FC]'
                  : 'bg-[#1F2023] hover:bg-[#374151]'
              }`}
              aria-label={`Go to agent ${index}`}
            />
          ))}
        </div>
        <button
          onClick={handleNext}
          className="px-6 py-2 border border-[#1F2023] text-[#9CA3AF] hover:text-[#F3F4F6] hover:border-[#7DD3FC] transition-colors text-sm"
          aria-label="Next agent"
        >
          Next →
        </button>
      </div>

      {/* Counter */}
      <p className="text-xs text-[#6B7280]">
        Agent {currentIndex + 1} of {AGENT_IDS.length}
      </p>
    </div>
  );
}
