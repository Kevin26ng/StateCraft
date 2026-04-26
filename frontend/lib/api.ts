import axios, { AxiosInstance } from 'axios';
import {
  ActionsPayload,
  HistoryResponse,
  ResetConfig,
  ResetResponse,
  StepResponse,
} from './types';

const API_BASE_URL = 'http://localhost:5000';

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

/**
 * Reset a simulation run
 */
export async function apiReset(config: ResetConfig): Promise<ResetResponse> {
  const response = await api.post<ResetResponse>('/reset', {
    scenario: config.scenario || 'pandemic',
    episode_mode: config.episode_mode || 'TRAINING',
    num_agents: config.num_agents || 6,
  });
  return response.data;
}

/**
 * Get metrics for a specific run (seed initial state)
 */
export async function apiGetMetrics(runId: string): Promise<StepResponse> {
  const response = await api.get<StepResponse>(`/metrics/${runId}`);
  return response.data;
}

/**
 * Step forward in the simulation
 */
export async function apiStep(runId: string, actions: ActionsPayload): Promise<StepResponse> {
  const response = await api.post<StepResponse>(`/step/${runId}`, actions);
  return response.data;
}

/**
 * Get historical runs
 */
export async function apiGetHistory(): Promise<HistoryResponse> {
  const response = await api.get<HistoryResponse>('/history');
  return response.data;
}

/**
 * Open a WebSocket connection to stream run data
 * Returns the WebSocket instance
 */
export function wsConnect(runId: string): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//localhost:5000/ws/stream/${runId}`;
  return new WebSocket(wsUrl);
}

export default api;
