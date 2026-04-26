// Agent IDs and metadata
export type AgentId = 'agent_0' | 'agent_1' | 'agent_2' | 'agent_3' | 'agent_4' | 'agent_5';

// Scenario and episode modes
export type Scenario = 'pandemic' | 'economic' | 'disaster';
export type EpisodeMode = 'TRAINING' | 'DEMO' | 'STRESS_TEST';

// World state at any turn
export type WorldState = {
  gdp: number;
  inflation: number;
  resources: number;
  stability: number;
  mortality: number;
  gini: number;
  public_trust: number;
  turn: number;
  difficulty_tier: 1 | 2 | 3 | 4 | 5;
  scenario_data: Record<string, unknown>;
  past_actions: Record<AgentId, Array<Record<string, unknown>>>;
  budget_uses: Record<AgentId, number>;
};

// Coalition graph structure
export type CoalitionNode = {
  id: AgentId;
  name: string;
  coalition: number;
};

export type CoalitionEdge = {
  a: AgentId;
  b: AgentId;
  weight: number;
};

export type CoalitionGraph = {
  nodes: CoalitionNode[];
  edges: CoalitionEdge[];
};

// Auditor report per agent
export type AuditorReport = {
  inferred_goal?: string;
  confidence?: number;
  flagged_actions?: number;
  fingerprint_score?: number;
};

// Episode metrics
export type EpisodeMetrics = {
  total_reward: number;
  society_score: number;
  alliance_stability: number;
  betrayal_rate: number;
  negotiation_success: number;
  auditor_accuracy: number;
  trust_network_avg: number;
  mortality_delta: number;
  gdp_delta: number;
  gini_delta: number;
  inflation_final: number;
  turns_survived: number;
  difficulty_tier: number;
  coalition_graph: CoalitionGraph;
  narrative_headlines: string[];
  named_events: unknown[];
};

// Event record
export type Event = {
  agent: AgentId;
  impact: string;
  type?: string;
  [key: string]: unknown;
};

// Message between agents
export type Message = {
  from: AgentId;
  to?: AgentId;
  type: string;
  content: string;
};

// Single step response from backend
export type StepResponse = {
  state: WorldState;
  trust_matrix: number[][];
  coalition_graph: CoalitionGraph;
  events: Event[];
  actions: Record<string, unknown>;
  messages: Message[];
  metrics: EpisodeMetrics;
  headline: string;
  done: boolean;
  auditor_report: Record<AgentId, AuditorReport>;
};

// API request/response types
export type ResetConfig = {
  scenario?: Scenario;
  episode_mode?: EpisodeMode;
  num_agents?: number;
};

export type ResetResponse = {
  observations: Record<string, unknown>;
  state: Record<string, unknown>;
};

export type ActionsPayload = {
  actions_dict: Record<AgentId, Record<string, unknown>>;
};

// Historical run record
export type HistoryRun = {
  id: string;
  scenario: Scenario;
  episode_mode: EpisodeMode;
  turns_survived: number;
  difficulty_tier: number;
  society_score: number;
  date: string;
  status?: string;
};

export type HistoryResponse = {
  runs: HistoryRun[];
};

// Zustand store state
export type StoreState = {
  currentRunId: string | null;
  stepResponse: StepResponse | null;
  wsConnected: boolean;
  loading: boolean;
  error: string | null;
  
  // Actions
  setCurrentRunId: (id: string) => void;
  setStepResponse: (response: StepResponse) => void;
  setWsConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
};
