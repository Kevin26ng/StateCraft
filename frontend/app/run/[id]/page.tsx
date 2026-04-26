'use client';

import { useEffect, useState, useRef } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import { useStore } from '@/lib/store';
import { useRunStream } from '@/lib/ws';
import { apiGetMetrics, apiStep } from '@/lib/api';
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

  // Store state
  const stepResponse = useStore((state) => state.stepResponse);
  const wsConnected = useStore((state) => state.wsConnected);
  const setStepResponse = useStore((state) => state.setStepResponse);
  const setCurrentRunId = useStore((state) => state.setCurrentRunId);
  const setWsConnected = useStore((state) => state.setWsConnected);
  const setLoading = useStore((state) => state.setLoading);

  // Local state
  const [speed, setSpeed] = useState(500);
  const [isPlaying, setIsPlaying] = useState(false);
  const [previousState, setPreviousState] = useState<any>(null);
  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize run
  useEffect(() => {
    setCurrentRunId(runId);

    const initializeRun = async () => {
      try {
        setLoading(true);
        const metrics = await apiGetMetrics(runId);
        setStepResponse(metrics);
        setPreviousState(metrics.state);
      } catch (error) {
        console.error('Failed to initialize run:', error);
      } finally {
        setLoading(false);
      }
    };

    initializeRun();
  }, [runId, setCurrentRunId, setStepResponse, setLoading]);

  // WebSocket connection
  useRunStream({
    runId,
    onMessage: (data: StepResponse) => {
      setPreviousState(stepResponse?.state);
      setStepResponse(data);
    },
    onConnected: () => {
      setWsConnected(true);
    },
    onDisconnected: () => {
      setWsConnected(false);
    },
    onError: (error: string) => {
      console.error('WebSocket error:', error);
    },
  });

  // Play/pause logic
  useEffect(() => {
    if (isPlaying && !stepResponse?.done) {
      playIntervalRef.current = setInterval(async () => {
        try {
          const actionsPayload: ActionsPayload = {
            actions_dict: Object.fromEntries(AGENT_IDS.map((id) => [id, {}])),
          };
          const newResponse = await apiStep(runId, actionsPayload);
          setPreviousState(stepResponse?.state);
          setStepResponse(newResponse);

          if (newResponse.done) {
            setIsPlaying(false);
          }
        } catch (error) {
          console.error('Step failed:', error);
          setIsPlaying(false);
        }
      }, speed);
    }

    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, [isPlaying, speed, runId, stepResponse, setStepResponse]);

  // Stop playing on unmount
  useEffect(() => {
    return () => {
      setIsPlaying(false);
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, []);

  // Check for collapsed or completed states
  const isCollapsed =
    stepResponse && (stepResponse.state.stability < 0.2 || stepResponse.state.gdp < 0.3);
  const isDone = stepResponse?.done ?? false;

  // Disable step/play if collapsed or done
  const controlsDisabled = isCollapsed || isDone;

  const handleStep = async () => {
    if (controlsDisabled || !stepResponse) return;

    try {
      const actionsPayload: ActionsPayload = {
        actions_dict: Object.fromEntries(AGENT_IDS.map((id) => [id, {}])),
      };
      const newResponse = await apiStep(runId, actionsPayload);
      setPreviousState(stepResponse?.state);
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
        collapsed={isCollapsed ?? false}
      />

      <SocietyScoreHero metrics={stepResponse?.metrics ?? null} />

      <StateVitals
        state={stepResponse?.state ?? null}
        previousState={previousState}
      />

      <CoalitionGraph coalitionGraph={stepResponse?.coalition_graph ?? null} />

      <TabbedPanel stepResponse={stepResponse} />

      {/* Control panel */}
      <div className="px-12 py-8 border-t border-[#1F2023] flex items-center justify-between gap-6">
        <div className="flex items-center gap-4">
          <button
            onClick={handleStep}
            disabled={controlsDisabled}
            className="px-4 py-2 bg-[#7DD3FC] text-[#0A0A0A] text-sm font-medium hover:bg-[#bfdbfe] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Step
          </button>

          <button
            onClick={handlePlayPause}
            disabled={controlsDisabled}
            className="px-4 py-2 bg-[#7DD3FC] text-[#0A0A0A] text-sm font-medium hover:bg-[#bfdbfe] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>

          {isDone && (
            <button
              onClick={() => (window.location.href = '/history')}
              className="px-4 py-2 bg-[#7DD3FC] text-[#0A0A0A] text-sm font-medium hover:bg-[#bfdbfe] transition-colors"
            >
              View in History
            </button>
          )}
        </div>

        {!controlsDisabled && (
          <div className="flex items-center gap-4">
            <label className="text-xs text-[#9CA3AF]">
              Speed: {speed}ms
            </label>
            <input
              type="range"
              min="50"
              max="1000"
              step="50"
              value={speed}
              onChange={(e) => setSpeed(parseInt(e.target.value))}
              disabled={isPlaying}
              className="w-32"
            />
          </div>
        )}
      </div>

      {/* Agent Inspector Drawer */}
      {inspectAgent && <AgentInspectorDrawer agentId={inspectAgent} />}
    </div>
  );
}
