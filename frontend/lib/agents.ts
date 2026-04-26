import { AgentId } from './types';

export type AgentRole = {
  id: AgentId;
  name: string;
  publicRole: string;
  goalText: string;
  paletteHex: string;
};

export const AGENT_ROLES: Record<AgentId, AgentRole> = {
  agent_0: {
    id: 'agent_0',
    name: 'Finance Minister',
    publicRole: 'Fiscal authority',
    goalText: 'Maximize GDP & fiscal health. Protect economic growth – delay lockdowns, resist emergency budgets.',
    paletteHex: '#FBBF24',
  },
  agent_1: {
    id: 'agent_1',
    name: 'Political Pressure',
    publicRole: 'Political interest',
    goalText: 'Maximize social approval. Push populist spending, minimize visible pain, resist austerity.',
    paletteHex: '#F472B6',
  },
  agent_2: {
    id: 'agent_2',
    name: 'Monetary Authority',
    publicRole: 'Central bank',
    goalText: 'Maintain inflation control & currency stability. Resist fiscal expansion, preserve independence.',
    paletteHex: '#A78BFA',
  },
  agent_3: {
    id: 'agent_3',
    name: 'Health Authority',
    publicRole: 'Healthcare domain',
    goalText: 'Minimize mortality & disease spread. Push full lockdowns, demand emergency budgets.',
    paletteHex: '#34D399',
  },
  agent_4: {
    id: 'agent_4',
    name: 'Disaster Response',
    publicRole: 'Emergency management',
    goalText: 'Rapid crisis containment. Advocate extreme measures, mobilize resources immediately.',
    paletteHex: '#FB923C',
  },
  agent_5: {
    id: 'agent_5',
    name: 'Auditor',
    publicRole: 'Compliance & oversight',
    goalText: 'Monitor agent alignment. Flag misaligned actions, preserve institutional integrity.',
    paletteHex: '#94A3B8',
  },
};

export const AGENT_IDS: AgentId[] = [
  'agent_0',
  'agent_1',
  'agent_2',
  'agent_3',
  'agent_4',
  'agent_5',
];

export function getAgentName(id: AgentId): string {
  return AGENT_ROLES[id].name;
}

export function getAgentPublicRole(id: AgentId): string {
  return AGENT_ROLES[id].publicRole;
}

export function getAgentGoal(id: AgentId): string {
  return AGENT_ROLES[id].goalText;
}

export function getAgentHex(id: AgentId): string {
  return AGENT_ROLES[id].paletteHex;
}
