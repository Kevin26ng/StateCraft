import axios, { AxiosInstance } from 'axios';
import {
  ActionsPayload,
  EpisodeMetrics,
  HistoryRun,
  ResetConfig,
  ResetResponse,
  Scenario,
  EpisodeMode,
  StepResponse,
} from './types';

const API_BASE_URL = 'http://localhost:5000';

const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export async function apiReset(config: ResetConfig): Promise<ResetResponse> {
  const response = await api.post<ResetResponse>('/reset', {
    scenario: config.scenario || 'pandemic',
    episode_mode: config.episode_mode || 'TRAINING',
    num_agents: config.num_agents || 6,
  });
  return response.data;
}

export async function apiGetMetrics(): Promise<EpisodeMetrics> {
  const response = await api.get<EpisodeMetrics>('/metrics');
  return response.data;
}

export async function apiStep(actions: ActionsPayload): Promise<StepResponse> {
  const response = await api.post<StepResponse>('/step', actions);
  return response.data;
}

export async function apiGetHistory(): Promise<{ runs: HistoryRun[] }> {
  const response = await api.get<{ episodes: EpisodeMetrics[] }>('/history');
  const episodes = response.data?.episodes ?? [];
  const runs: HistoryRun[] = episodes.map((ep, i) => ({
    id: `run-${i}`,
    scenario: 'pandemic' as Scenario,
    episode_mode: 'TRAINING' as EpisodeMode,
    turns_survived: ep.turns_survived,
    difficulty_tier: ep.difficulty_tier,
    society_score: ep.society_score,
    date: new Date().toISOString(),
    status: 'completed',
  }));
  return { runs };
}

export function wsConnect(): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return new WebSocket(`${protocol}//localhost:5000/ws/stream`);
}

export default api;
