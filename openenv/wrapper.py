"""
OpenEnv Wrapper — openenv/wrapper.py (Task 1)

Wraps CrisisEnv in an OpenEnv-compatible Environment contract.
This is the judging baseline — without it the project fails the core rubric.

Adapts the existing CrisisEnv (agent_0..agent_5, string actions)
to a flat numpy observation + MultiDiscrete action space for PPO.
"""

import sys
import os
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.crisis_env import CrisisEnv
from core.aggregation import aggregate_actions
from core.rewards import RewardSystem
from openenv.tasks import get_all_tasks
from openenv.grader import CrisisGrader


# ── Agent ID mapping ──────────────────────────────────────────────────────────
# Canonical agent_id → descriptive name (for external API)
AGENT_IDS = ['agent_0', 'agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']
AGENT_NAMES = {
    'agent_0': 'finance_minister',
    'agent_1': 'political_pressure_agent',
    'agent_2': 'monetary_authority',
    'agent_3': 'public_health_authority',
    'agent_4': 'disaster_response_agent',
    'agent_5': 'auditor',
}
N_AGENTS = 6
OBS_DIM = 32


class ResetResult:
    """Minimal OpenEnv ResetResult compatible object."""
    def __init__(self, observations, info=None):
        self.observations = observations
        self.info = info or {}


class StepResult:
    """Minimal OpenEnv StepResult compatible object."""
    def __init__(self, observations, reward, done, truncated=False, info=None):
        self.observations = observations
        self.reward = reward
        self.done = done
        self.truncated = truncated
        self.info = info or {}


class CrisisGovernanceEnv:
    """
    OpenEnv-compatible wrapper around CrisisEnv.
    Exposes reset/step/observation/action spaces in the exact shape OpenEnv expects.

    Observation space: (n_agents, 32) float32 — one 32-dim vector per agent
    Action space: MultiDiscrete([5, 5, 5, 4, 4]) per agent

    The wrapper handles:
      - Converting per-agent partial obs dicts → flat numpy arrays
      - Converting flat action arrays → per-agent action dicts
      - Aggregating per-agent actions into final policy
      - Computing rewards via RewardSystem
    """

    metadata = {"render_modes": ["human", "json"]}

    def __init__(self, config=None):
        self._config = config or {}
        self._env = CrisisEnv(config=self._config)
        self._grader = CrisisGrader()
        self._tasks = get_all_tasks()
        self._reward_system = RewardSystem()

        # Store last observations and actions for reference
        self._last_obs_dict = {}
        self._last_actions = {}
        self._prev_state = {}

    @property
    def state(self):
        """Access underlying env state for metrics computation."""
        return self._env.state_manager.state

    @property
    def scenario(self):
        return self._env.scenario

    def reset(self, seed=None, options=None) -> ResetResult:
        """Reset the environment and return initial observations."""
        reset_config = options if options else None

        # Handle config dict passed as options
        if isinstance(reset_config, dict):
            scenario = reset_config.get('scenario', self._config.get('scenario', 'pandemic'))
            self._env.scenario_name = scenario

        obs_dict = self._env.reset(config=reset_config)
        self._last_obs_dict = obs_dict
        self._prev_state = {}

        flat_obs = self._flatten_observations(obs_dict)
        info = {
            "state": self._env.state_manager.state.copy(),
            "scenario": self._env.scenario_name,
        }
        return ResetResult(observations=flat_obs, info=info)

    def step(self, actions) -> StepResult:
        """
        Step the environment with the given actions.

        Args:
            actions: numpy array of shape (n_agents, 5) or (5,) for broadcast
                     Each row is [lockdown_idx, interest_idx, budget_idx, priority_idx, crisis_idx]

        Returns:
            StepResult with flat observations, scalar reward, done flag, info
        """
        actions = np.asarray(actions)

        # Convert flat actions → per-agent action dicts
        actions_dict = self._unflatten_actions(actions)
        self._last_actions = actions_dict

        # Save prev state for reward computation
        from copy import deepcopy
        self._prev_state = deepcopy(self._env.state_manager.state)

        # Aggregate per-agent actions into final policy action
        final_action = aggregate_actions(actions_dict)

        # Step the environment
        obs_dict, _, done, info = self._env.step(final_action, raw_agent_actions=actions_dict)
        self._last_obs_dict = obs_dict

        flat_obs = self._flatten_observations(obs_dict)

        # Compute per-agent rewards using RewardSystem
        current_state = self._env.state_manager.state
        rewards_dict = {}
        for agent_id in AGENT_IDS:
            rewards_dict[agent_id] = self._reward_system.compute_and_clip_rewards(
                state=current_state,
                prev_state=self._prev_state,
                agent_id=agent_id,
                done=done,
                agents=None,  # no agent objects in PPO mode
                actions_dict=actions_dict,
                final_action=final_action,
            )
            rewards_dict[agent_id] = float(np.clip(rewards_dict[agent_id], -10, 10))

        # Compute grader score for this step
        grade = self._grader.grade_step(current_state, info)
        info["grade"] = grade
        info["rewards_dict"] = rewards_dict
        info["actions_dict"] = actions_dict

        # Aggregate reward across agents (mean of clipped rewards)
        agg_reward = float(np.mean(list(rewards_dict.values())))

        return StepResult(
            observations=flat_obs,
            reward=agg_reward,
            done=done,
            truncated=False,
            info=info,
        )

    def get_tasks(self):
        return self._tasks

    def get_grader(self):
        return self._grader

    def _flatten_observations(self, obs_dict: dict) -> np.ndarray:
        """
        Convert per-agent partial observation dicts to flat numpy arrays.
        Shape: (n_agents, 32) — each row is one agent's observation.

        Layout (32 dims):
          [0-8]   9 state fields: gdp, inflation, resources, stability,
                  mortality, gini, public_trust, turn, difficulty_tier
          [9-14]  6 trust_matrix row values (agent's trust of each other agent)
          [15-20] 6 coalition encoding (1.0 if same coalition, 0.0 otherwise)
          [21]    1 memory flag (always 0.0 in PPO mode)
          [22-31] 10 padding zeros (reserved for causal horizon injection)
        """
        state = self._env.state_manager.state
        trust_matrix = self._env.state_manager.trust_matrix
        coalition_map = self._env.state_manager.coalition_map

        obs_keys = [
            'gdp', 'inflation', 'resources', 'stability',
            'mortality', 'gini', 'public_trust', 'turn',
            'difficulty_tier'
        ]

        flat = []
        for idx, agent_id in enumerate(AGENT_IDS):
            agent_obs = obs_dict.get(agent_id, {})
            public_state = agent_obs.get('public_state', {})

            # 9 state fields (use 0.0 for fields this agent can't see)
            row = [float(public_state.get(k, 0.0)) for k in obs_keys]

            # 6 trust row values
            trust_row = trust_matrix[idx].tolist()
            row += trust_row[:N_AGENTS]

            # 6 coalition encoding (1.0 if same coalition as this agent)
            my_coalition = coalition_map.get(agent_id, idx)
            coalition_enc = [
                1.0 if coalition_map.get(other_id, j) == my_coalition else 0.0
                for j, other_id in enumerate(AGENT_IDS)
            ]
            row += coalition_enc

            # 1 memory flag (always 0.0 in PPO mode — no agent objects)
            row += [0.0]

            # Ensure exactly 32 dims
            row = row[:OBS_DIM]
            row += [0.0] * (OBS_DIM - len(row))

            flat.append(row)

        return np.array(flat, dtype=np.float32)

    def _unflatten_actions(self, actions) -> dict:
        """
        Convert flat action array back to per-agent action dicts.

        actions shape: (n_agents, 5) or (5,) for single-agent mode.

        Action domains (matching existing CrisisEnv string values):
          [0] lockdown_level: ['none', 'advisory', 'partial', 'full', 'emergency']
          [1] interest_rate:  ['-0.5', '0', '+0.25', '+0.5', '+1']
          [2] emergency_budget: ['0', '5', '15', '30', '50']
          [3] resource_priority: ['health', 'infrastructure', 'military', 'services']
          [4] crisis_response: ['monitor', 'contain', 'escalate', 'emergency']
        """
        ACTION_MAPS = {
            'lockdown_level': ['none', 'advisory', 'partial', 'full', 'emergency'],
            'interest_rate': ['-0.5', '0', '+0.25', '+0.5', '+1'],
            'emergency_budget': ['0', '5', '15', '30', '50'],
            'resource_priority': ['health', 'infrastructure', 'military', 'services'],
            'crisis_response': ['monitor', 'contain', 'escalate', 'emergency'],
        }

        actions = np.asarray(actions)
        result = {}

        for i, agent_id in enumerate(AGENT_IDS):
            if actions.ndim == 2:
                a = actions[i]
            else:
                a = actions  # broadcast single action to all agents

            result[agent_id] = {
                'lockdown_level': ACTION_MAPS['lockdown_level'][int(a[0]) % 5],
                'interest_rate': ACTION_MAPS['interest_rate'][int(a[1]) % 5],
                'emergency_budget': ACTION_MAPS['emergency_budget'][int(a[2]) % 5],
                'resource_priority': ACTION_MAPS['resource_priority'][int(a[3]) % 4],
                'crisis_response': ACTION_MAPS['crisis_response'][int(a[4]) % 4],
            }

        return result
