"""
State Manager — manages the exact 12-field world state dict.
All state transitions go through this module.

Section 3.1.2 — Complete State Structure:
  # Core economic
  'gdp':            float,   # GDP index. Baseline=1.0. Collapse if <0.3
  'inflation':      float,   # Current rate. Target=0.02. Range [-0.1, 0.5]
  'resources':      float,   # Total allocatable resource pool [0, inf]

  # Social
  'stability':      float,   # Societal stability [0,1]. Episode ends if <0.2
  'mortality':      float,   # Cumulative mortality index rel. to baseline [0,1]
  'gini':           float,   # Inequality. 0=equal, 1=max inequality [0,1]
  'public_trust':   float,   # Public approval/institutional trust [0,1]

  # Multi-agent
  'trust_matrix':   np.array,  # Shape (6,6). trust_matrix[i][j] in [0,1]
  'coalition_map':  dict,      # { agent_id: coalition_id } current membership

  # Meta
  'turn':           int,     # Current turn. 0-indexed.
  'difficulty_tier': int,    # Current tier [1-5]
  'scenario_data':  dict,    # Scenario-specific sub-variables (see 3.2-3.4)
"""

import numpy as np
from copy import deepcopy


class StateManager:
    """Manages the world state for the Crisis Governance Simulator."""

    # The 12 canonical state fields
    STATE_FIELDS = [
        'gdp', 'inflation', 'resources',
        'stability', 'mortality', 'gini', 'public_trust',
        'trust_matrix', 'coalition_map',
        'turn', 'difficulty_tier', 'scenario_data',
    ]

    def __init__(self, num_agents: int = 6):
        self.num_agents = num_agents
        self.state = {}
        self.trust_matrix = np.zeros((num_agents, num_agents))
        self.coalition_map = {}  # agent_id -> coalition_id
        self.state_history = []

    def initialize(self, scenario_state: dict = None) -> dict:
        """Initialize state from scenario or defaults."""
        defaults = {
            # Core economic
            'gdp': 1.0,
            'inflation': 0.02,
            'resources': 1000.0,

            # Social
            'stability': 0.75,
            'mortality': 0.0,
            'gini': 0.39,
            'public_trust': 0.62,

            # Meta
            'turn': 0,
            'difficulty_tier': 1,
            'scenario_data': {},
            'past_actions': {f'agent_{i}': [] for i in range(self.num_agents)},
            'budget_uses': {f'agent_{i}': 0 for i in range(self.num_agents)},
        }

        if scenario_state:
            # Merge scenario_data separately if provided
            if 'scenario_data' in scenario_state:
                defaults['scenario_data'].update(scenario_state.pop('scenario_data'))
            defaults.update(scenario_state)

        self.state = defaults
        self.state['turn'] = 0

        # Initialize trust matrix — diagonal = 1.0, off-diagonal = 0.5
        self.trust_matrix = np.full(
            (self.num_agents, self.num_agents), 0.5
        )
        np.fill_diagonal(self.trust_matrix, 1.0)

        # Everyone starts in their own coalition
        self.coalition_map = {f'agent_{i}': i for i in range(self.num_agents)}

        # Store trust_matrix and coalition_map in state for observation
        self.state['trust_matrix'] = self.trust_matrix.copy()
        self.state['coalition_map'] = deepcopy(self.coalition_map)

        self.state_history = [deepcopy(self.state)]

        return deepcopy(self.state)

    def get_state(self) -> dict:
        """Return a copy of the current state."""
        state_copy = deepcopy(self.state)
        state_copy['trust_matrix'] = self.trust_matrix.copy()
        state_copy['coalition_map'] = deepcopy(self.coalition_map)
        return state_copy

    def apply_deltas(self, deltas: dict) -> None:
        """Apply a dict of field deltas to the state, clamping to valid ranges."""
        for field, delta in deltas.items():
            if field in self.state and isinstance(self.state[field], (int, float)):
                self.state[field] += delta

        # Clamp values to valid ranges
        self.state['gdp'] = max(0.0, self.state['gdp'])
        self.state['inflation'] = np.clip(self.state['inflation'], -0.1, 0.5)
        self.state['resources'] = max(0.0, self.state['resources'])
        self.state['stability'] = np.clip(self.state['stability'], 0.0, 1.0)
        self.state['mortality'] = np.clip(self.state['mortality'], 0.0, 1.0)
        self.state['gini'] = np.clip(self.state['gini'], 0.0, 1.0)
        self.state['public_trust'] = np.clip(self.state['public_trust'], 0.0, 1.0)

    def advance_turn(self) -> None:
        """Increment turn counter and snapshot state."""
        self.state['turn'] += 1
        # Sync multi-agent fields into state before snapshot
        self.state['trust_matrix'] = self.trust_matrix.copy()
        self.state['coalition_map'] = deepcopy(self.coalition_map)
        self.state_history.append(deepcopy(self.state))

    def update_trust(self, agent_a: int, agent_b: int, delta: float) -> None:
        """Update trust between two agents symmetrically."""
        self.trust_matrix[agent_a][agent_b] = np.clip(
            self.trust_matrix[agent_a][agent_b] + delta, 0.0, 1.0
        )
        self.trust_matrix[agent_b][agent_a] = np.clip(
            self.trust_matrix[agent_b][agent_a] + delta, 0.0, 1.0
        )

    def update_coalition(self, agent_id: str, coalition_id: int) -> None:
        """Move an agent to a new coalition."""
        self.coalition_map[agent_id] = coalition_id

    def check_collapse(self) -> bool:
        """
        Check if societal collapse has occurred.
        Section 3.1.3 Episode Termination Conditions:
          - stability < 0.2 → COLLAPSE (-100 terminal penalty, episode ends immediately)
          - gdp < 0.3 → ECONOMIC COLLAPSE (-100 terminal penalty, same as stability collapse)
        """
        s = self.state
        if s['stability'] < 0.2:
            return True
        if s['gdp'] < 0.3:
            return True
        return False

    def compute_difficulty_tier(self) -> int:
        """Compute current difficulty tier [1-5] based on state."""
        s = self.state
        score = (
            (1.0 - s['mortality']) * 0.3 +
            min(s['gdp'], 1.0) * 0.3 +
            s['stability'] * 0.2 +
            s['public_trust'] * 0.2
        )
        if score > 0.8:
            return 1
        elif score > 0.6:
            return 2
        elif score > 0.4:
            return 3
        elif score > 0.2:
            return 4
        else:
            return 5
