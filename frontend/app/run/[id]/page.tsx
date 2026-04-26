'use client';

import { useEffect, useState, useRef } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import { useStore } from '@/lib/store';
import { useRunStream } from '@/lib/ws';
import { apiStep } from '@/lib/api';
import { StepResponse, ActionsPayload } from '@/lib/types';
import { AGENT_IDS } from '@/lib/agents';
import TopStrip from '@/components/TopStrip';
import SocietyScoreHero from '@/components/SocietyScoreHero';
import StateVitals from '@/components/StateVitals';
import CoalitionGraph from '@/components/CoalitionGraph';
import TabbedPanel from '@/components/TabbedPanel';
import AgentInspectorDrawer from './AgentInspectorDrawer';

export default function ActiveRun() {
  const params = useParams();
  const searchParams = useSearchParams();
  const runId = params.id as string;
  const inspectAgent = searchParams.get('inspect');

  const stepResponse = useStore((state) => state.stepResponse);
  const wsConnected = useStore((state) => state.wsConnected);
  const setStepResponse = useStore((state) => state.setStepResponse);
  const setCurrentRunId = useStore((state) => state.setCurrentRunId);
  const setWsConnected = useStore((state) => state.setWsConnected);
  const setLoading = useStore((state) => state.setLoading);

  const [speed, setSpeed] = useState(500);
  const [isPlaying, setIsPlaying] = useState(false);
  const [previousState, setPreviousState] = useState<StepResponse['state'] | null>(null);
  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    setCurrentRunId(runId);
    setLoading(false);
  }, [runId, setCurrentRunId, setLoading]);

  useRunStream({
    runId,
    onMessage: (data: StepResponse) => {
      setPreviousState(stepResponse?.state ?? null);
      setStepResponse(data);
    },
    onConnected: () => setWsConnected(true),
    onDisconnected: () => setWsConnected(false),
    onError: (error: string) => console.error('WebSocket error:', error),
  });

  useEffect(() => {
    if (isPlaying && !stepResponse?.done) {
      playIntervalRef.current = setInterval(async () => {
        try {
          const actionsPayload: ActionsPayload = {
            actions_dict: Object.fromEntries(
              AGENT_IDS.map((id) => [id, {}])
            ) as ActionsPayload['actions_dict'],
          };
          const newResponse = await apiStep(actionsPayload);
          setPreviousState(stepResponse?.state ?? null);
          setStepResponse(newResponse);
          if (newResponse.done) setIsPlaying(false);
        } catch (error) {
          console.error('Step failed:', error);
          setIsPlaying(false);
        }
      }, speed);
    }
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [isPlaying, speed, stepResponse, setStepResponse]);

  useEffect(() => {
    return () => {
      setIsPlaying(false);
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, []);

  const isCollapsed =
    stepResponse != null &&
    (stepResponse.state.stability < 0.2 || stepResponse.state.gdp < 0.3);
  const isDone = stepResponse?.done ?? false;
  const controlsDisabled = isCollapsed || isDone;

  const handleStep = async () => {
    if (controlsDisabled || !stepResponse) return;
    try {
      const actionsPayload: ActionsPayload = {
        actions_dict: Object.fromEntries(
          AGENT_IDS.map((id) => [id, {}])
        ) as ActionsPayload['actions_dict'],
      };
      const newResponse = await apiStep(actionsPayload);
      setPreviousState(stepResponse.state);
      setStepResponse(newResponse);
    } catch (error) {
      console.error('Step failed:', error);
    }
  };

  const handlePlayPause = () => {
    if (controlsDisabled) return;
    setIsPlaying(!isPlaying);
  };

  return (
    <div className="main-content">
      <TopStrip
        state={stepResponse?.state ?? null}
        wsConnected={wsConnected}
        done={isDone}
        collapsed={isCollapsed}
        headline={stepResponse?.headline}
      />

      <SocietyScoreHero metrics={stepResponse?.metrics ?? null} />

      <StateVitals
        state={stepResponse?.state ?? null}
        previousState={previousState}
      />

      <CoalitionGraph coalitionGraph={stepResponse?.coalition_graph ?? null} />

      <TabbedPanel stepResponse={stepResponse} />

      {/* Controls */}
      <div className="py-8 border-t border-[#1F2023] flex items-center justify-between gap-6">
        <div className="flex items-center gap-3">
          <button
            onClick={handleStep}
            disabled={controlsDisabled}
            className="px-5 py-2 bg-[#7DD3FC] text-[#0A0A0A] text-xs font-semibold uppercase tracking-[0.08em] hover:bg-[#BAE6FD] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Step
          </button>
          <button
            onClick={handlePlayPause}
            disabled={controlsDisabled}
            className="px-5 py-2 border border-[#1F2023] text-xs font-semibold uppercase tracking-[0.08em] text-[#9CA3AF] hover:border-[#7DD3FC] hover:text-[#7DD3FC] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          {isDone && (
            <button
              onClick={() => (window.location.href = '/history')}
              className="px-5 py-2 border border-[#34D399]/30 text-xs font-semibold uppercase tracking-[0.08em] text-[#34D399] hover:border-[#34D399] transition-colors"
            >
              View History
            </button>
          )}
        </div>

        {!controlsDisabled && (
          <div className="flex items-center gap-3">
            <span className="text-xs text-[#6B7280]">Speed: {speed}ms</span>
            <input
              type="range"
              min="50"
              max="1000"
              step="50"
              value={speed}
              onChange={(e) => setSpeed(parseInt(e.target.value))}
              disabled={isPlaying}
              className="w-28 accent-[#7DD3FC] h-1"
            />
          </div>
        )}
      </div>

      {inspectAgent && <AgentInspectorDrawer agentId={inspectAgent} />}
    </div>
  );
}
