'use client';

import { Event } from '@/lib/types';
import { getAgentName, getAgentHex } from '@/lib/agents';

interface ActivityLogProps {
  events: Event[];
  turn: number;
}

export default function ActivityLog({ events, turn }: ActivityLogProps) {
  // Dedupe events by (turn, agent, type, content)
  const deduped = Array.from(
    new Map(
      events.map((event) => [
        `${turn}:${event.agent}:${event.type || 'unknown'}:${event.impact || ''}`,
        event,
      ])
    ).values()
  );

  // Sort by newest first (assuming events come in chronological order)
  const sorted = [...deduped].reverse();

  if (sorted.length === 0) {
    return (
      <div className="py-12 text-center">
        <p className="text-[#6B7280] text-sm">No events recorded</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {sorted.slice(0, 20).map((event, index) => {
        const agentName = getAgentName(event.agent);
        const agentColor = getAgentHex(event.agent);

        return (
          <div
            key={`${event.agent}-${index}`}
            className="px-4 py-3 border border-[#1F2023] flex gap-3"
          >
            {/* Agent indicator dot */}
            <div
              className="w-2 h-2 rounded-full flex-shrink-0 mt-1"
              style={{ backgroundColor: agentColor }}
            />

            {/* Event content */}
            <div className="flex-1 min-w-0">
              <p className="text-xs text-[#9CA3AF] mb-1">
                <span style={{ color: agentColor }}>@{agentName}</span>
                {event.type && (
                  <>
                    {' '}
                    <span className="text-[#6B7280]">·</span>{' '}
                    <span className="text-[#9CA3AF]">{event.type}</span>
                  </>
                )}
              </p>
              <p className="text-sm text-[#F3F4F6] break-words">
                {event.impact || 'No impact recorded'}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
