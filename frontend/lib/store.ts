import { create } from 'zustand';
import { StepResponse, StoreState } from './types';
import { clampTrust } from './format';

export const useStore = create<StoreState>((set) => ({
  // Initial state
  currentRunId: null,
  stepResponse: null,
  wsConnected: false,
  loading: false,
  error: null,

  // Actions
  setCurrentRunId: (id: string) => {
    set({ currentRunId: id });
  },

  setStepResponse: (response: StepResponse) => {
    // Clamp trust matrix values to [0, 1] at store boundary
    const clampedTrustMatrix = response.trust_matrix.map((row) =>
      row.map((val) => clampTrust(val))
    );

    set({
      stepResponse: {
        ...response,
        trust_matrix: clampedTrustMatrix,
      },
    });
  },

  setWsConnected: (connected: boolean) => {
    set({ wsConnected: connected });
  },

  setLoading: (loading: boolean) => {
    set({ loading });
  },

  setError: (error: string | null) => {
    set({ error });
  },

  reset: () => {
    set({
      currentRunId: null,
      stepResponse: null,
      wsConnected: false,
      loading: false,
      error: null,
    });
  },
}));

// Selectors
export const selectCurrentRunId = (state: StoreState) => state.currentRunId;
export const selectStepResponse = (state: StoreState) => state.stepResponse;
export const selectWsConnected = (state: StoreState) => state.wsConnected;
export const selectLoading = (state: StoreState) => state.loading;
export const selectError = (state: StoreState) => state.error;

export const selectWorldState = (state: StoreState) => state.stepResponse?.state;
export const selectTrustMatrix = (state: StoreState) => state.stepResponse?.trust_matrix;
export const selectCoalitionGraph = (state: StoreState) =>
  state.stepResponse?.coalition_graph;
export const selectEvents = (state: StoreState) => state.stepResponse?.events;
export const selectMessages = (state: StoreState) => state.stepResponse?.messages;
export const selectMetrics = (state: StoreState) => state.stepResponse?.metrics;
export const selectHeadline = (state: StoreState) => state.stepResponse?.headline;
export const selectDone = (state: StoreState) => state.stepResponse?.done ?? false;
export const selectAuditorReport = (state: StoreState) =>
  state.stepResponse?.auditor_report;
