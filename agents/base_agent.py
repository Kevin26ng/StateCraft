"""
Base Agent — agents/base_agent.py (Section 4.1)

Abstract base class for all agents in the Crisis Governance Simulator.
Agents can be rule-based, RL-trained, or LLM-driven.
"""

import numpy as np
from abc import ABC, abstractmethod
from .roles import get_role_config, AGENT_ROLES


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Section 4.1 interface:
      - act(observation) -> dict of discrete policy choices
      - negotiate(state) -> dict of messages (450 tokens negotiation item)
      - compute_reward(state, prev_state) -> hidden goal reward (30%)
      - load_memory(store) / save_memory(store, events)
    """

    def __init__(self, agent_id: str, role: str = None,
                 hidden_goal_config: dict = None):
        self.agent_id = agent_id
        self.role = role or agent_id
        self.hidden_goal = hidden_goal_config or {}  # dict: type, weight, threshold
        self.memory = []  # loaded from cross-episode store at reset
        self.personality = {}  # risk_tolerance, honesty, ambition, cooperativeness

        # Load from role config
        role_config = get_role_config(agent_id)
        if role_config:
            self.role = role_config.get('name', agent_id)
            self.personality = {
                'base': role_config.get('personality', 'neutral'),
            }

    @abstractmethod
    def act(self, observation: dict) -> dict:
        """
        Return action dict with discrete policy choices.

        Must include: lockdown_level, interest_rate, emergency_budget,
                      resource_priority, foreign_policy, crisis_response

        Args:
            observation: dict with public_state, trust_row, coalition_map, agent_id

        Returns:
            dict of domain -> action
        """
        raise NotImplementedError

    def negotiate(self, state: dict, round_num: int = 1) -> list:
        """
        Return message dicts (450 tokens negotiation item).

        Format: { sender, target, type, content }

        Args:
            state: agent's observation of the world
            round_num: which negotiation round (1, 2, or 3)

        Returns:
            list of message dicts
        """
        return []  # Default: no messages

    def hidden_goal_reward(self, state: dict, prev_state: dict) -> float:
        """
        Compute this agent's hidden goal reward (30% of total).
        Override in subclass.
        """
        return 0.0

    def observe_result(self, reward: float, next_observation: dict,
                       done: bool) -> None:
        """Process the result of the previous action (for learning agents)."""
        pass

    def load_memory(self, store) -> None:
        """Load cross-episode memory from JSON/Redis store."""
        if store:
            self.memory = store.get(self.agent_id)

    def save_memory(self, store, events: list) -> None:
        """Append key events to persistent store."""
        if store:
            for event in events:
                store.append(self.agent_id, event)


class RandomAgent(BaseAgent):
    """Random agent — uniform random actions. Used as baseline."""

    def __init__(self, agent_id: str, seed=None):
        super().__init__(agent_id)
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict) -> dict:
        return {
            'lockdown_level': self.rng.choice(
                ['none', 'advisory', 'partial', 'full', 'emergency']),
            'emergency_budget': self.rng.choice(
                ['0', '5', '15', '30', '50']),
            'resource_priority': self.rng.choice(
                ['health', 'infrastructure', 'military', 'services']),
            'foreign_policy': self.rng.choice(
                ['isolate', 'neutral', 'engage', 'alliance']),
            'crisis_response': self.rng.choice(
                ['monitor', 'contain', 'escalate', 'emergency']),
            'interest_rate': self.rng.choice(
                ['-0.5', '-0.25', '0', '+0.25', '+0.5', '+1', '+2']),
        }

    def negotiate(self, state: dict, round_num: int = 1) -> list:
        if self.rng.random() > 0.7:
            target = f'agent_{self.rng.integers(0, 6)}'
            if target != self.agent_id:
                return [{
                    'target': target,
                    'type': self.rng.choice(['support', 'reject', 'inform']),
                    'content': f'{self.role} random message',
                }]
        return []
