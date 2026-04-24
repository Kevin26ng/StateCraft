"""
CrisisEnv — Main simulation environment (Section 3.1).

Orchestrates agents, state, dynamics, and scenarios per episode.
Uses the new core/ systems (trust, negotiation, aggregation, rewards).
"""

import os
import yaml
import numpy as np
from copy import deepcopy

from .state import StateManager
from .scenarios import ScenarioLoader
from core.step_logic import apply_outcome_noise, apply_joint_synergies

ACTION_LIMITS = {
    'lockdown_level': {
        'max_consecutive_at_level': {'full': 10, 'emergency': 5},
        'cooldown_after_max': 3,
    },
    'emergency_budget': {
        'max_per_episode': 3,
    },
    'crisis_response': {
        'max_consecutive_at_level': {'emergency': 5},
    }
}


class CrisisEnv:
    """
    Crisis Governance Simulator Environment.

    Section 3.1.1 — Class Interface:
      - reset(config) -> observations_dict
      - step(actions_dict) -> (obs, rewards, done, info)
      - get_observation(agent_id) -> partial observation dict

    Manages 6 agents governing through a multi-domain crisis.
    Each step:
      1. Agents submit actions per domain
      2. World dynamics compute state transitions
      3. Scenario-specific update equations run
      4. Crisis events are injected
      5. State is updated
      6. Turn advances
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = self._load_default_config()

        self.config = config
        self.scenario_name = config.get('scenario', 'pandemic')
        self.num_agents = config.get('num_agents', 6)
        self.episode_mode = config.get('episode_mode', 'TRAINING')
        self.max_steps = config.get('max_steps', {}).get(self.episode_mode, 30)
        self.demo_mode = config.get('demo_mode', False)

        self.state_manager = StateManager(num_agents=self.num_agents)
        self.scenario_loader = ScenarioLoader()

        self.state = {}
        self.scenario = None
        self.scenario_module = None
        self.episode = 0
        self.done = False

        # Logs for metrics computation
        self.coalition_history = []
        self.agreement_log = []
        self.defection_log = []
        self.negotiation_log = []
        self.state_history_ref = []

    def _load_default_config(self) -> dict:
        """Load config from config/config.yaml."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'config.yaml'
        )
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        # Fallback defaults
        return {
            'scenario': 'pandemic',
            'num_agents': 6,
            'episode_mode': 'TRAINING',
            'max_steps': {'TRAINING': 30, 'DEMO': 200, 'STRESS_TEST': 500},
            'demo_mode': False,
            'reward_clip': 10.0,
            'promotion_threshold': 0.70,
            'promotion_window': 20,
            'initial_tier': 1,
        }

    def _load_scenario_module(self, scenario_name: str):
        """Dynamically load the scenario-specific module."""
        if scenario_name == 'pandemic':
            from env import pandemic as mod
        elif scenario_name == 'economic':
            from env import economic as mod
        elif scenario_name == 'disaster':
            from env import disaster as mod
        else:
            mod = None
        self.scenario_module = mod

    def reset(self, config=None) -> dict:
        """
        Initialize or reset the environment for a new episode.

        Args:
            config: Optional ResetConfig to override scenario/settings.

        Returns:
            observations: dict of per-agent observations.
        """
        if config:
            if hasattr(config, 'scenario') and config.scenario:
                self.scenario_name = config.scenario
            if hasattr(config, 'episode_mode') and config.episode_mode:
                self.episode_mode = config.episode_mode
                self.max_steps = self.config.get('max_steps', {}).get(
                    self.episode_mode, 30
                )

        self.episode += 1
        self.done = False

        # Load scenario module
        self._load_scenario_module(self.scenario_name)

        # Load scenario data
        self.scenario = self.scenario_loader.load_scenario(self.scenario_name)

        # Get scenario-specific initial state
        if self.scenario_module and hasattr(self.scenario_module, 'get_initial_state'):
            initial = self.scenario_module.get_initial_state()
        else:
            initial = self.scenario.get('initial_state', {})

        # Initialize state
        self.state = self.state_manager.initialize(initial)

        # Reset logs
        self.coalition_history = [deepcopy(self.state_manager.coalition_map)]
        self.agreement_log = []
        self.defection_log = []
        self.negotiation_log = []
        self.state_history_ref = self.state_manager.state_history

        # Build per-agent observations
        observations = self._build_observations()
        return observations

    def load_historical_scenario(self, scenario_id: str) -> None:
        """Load a historical scenario for validation runs."""
        historical = self.scenario_loader.load_historical_scenario(scenario_id)
        self.scenario = {
            'initial_state': historical.get('initial_state', {}),
            'crisis_events': {},
        }
        # Convert timeline to crisis_events dict
        for event in historical.get('crisis_timeline', []):
            turn = event.pop('turn')
            event.pop('event', None)
            self.scenario['crisis_events'][turn] = {
                k.replace('_delta', ''): v
                for k, v in event.items()
            }

        self.state = self.state_manager.initialize(
            self.scenario.get('initial_state', {})
        )
        self.done = False
        self.coalition_history = [deepcopy(self.state_manager.coalition_map)]
        self.state_history_ref = self.state_manager.state_history

    def enforce_and_track_actions(self, actions_dict: dict) -> dict:
        """Enforces limits (Point 7C) and updates past_actions (Point 5)."""
        clamped = {}
        for agent_id, action in actions_dict.items():
            if not isinstance(action, dict):
                clamped[agent_id] = action
                continue
            clamped_act = deepcopy(action)
            history = self.state_manager.state.get('past_actions', {}).get(agent_id, [])
            budget_uses = self.state_manager.state.get('budget_uses', {}).get(agent_id, 0)
            
            # Limit lockdown
            for level, max_turns in ACTION_LIMITS['lockdown_level']['max_consecutive_at_level'].items():
                if clamped_act.get('lockdown_level') == level:
                    # check last max_turns
                    consecutive = sum(1 for h in history[-max_turns:] if h.get('lockdown_level') == level)
                    if consecutive >= max_turns:
                        clamped_act['lockdown_level'] = 'partial' if level == 'full' else 'full'
            
            # Limit budget
            if clamped_act.get('emergency_budget') not in ['0', None, 'none', 0]:
                if budget_uses >= ACTION_LIMITS['emergency_budget']['max_per_episode']:
                    clamped_act['emergency_budget'] = '0'
                else:
                    self.state_manager.state['budget_uses'][agent_id] = budget_uses + 1
                    
            # Limit crisis response
            for level, max_turns in ACTION_LIMITS['crisis_response']['max_consecutive_at_level'].items():
                if clamped_act.get('crisis_response') == level:
                    consecutive = sum(1 for h in history[-max_turns:] if h.get('crisis_response') == level)
                    if consecutive >= max_turns:
                        clamped_act['crisis_response'] = 'escalate'

            clamped[agent_id] = clamped_act
            
            # Update past actions
            self.state_manager.state.setdefault('past_actions', {})
            self.state_manager.state['past_actions'].setdefault(agent_id, []).append(clamped_act)
            if len(self.state_manager.state['past_actions'][agent_id]) > 5:
                self.state_manager.state['past_actions'][agent_id] = self.state_manager.state['past_actions'][agent_id][-5:]
                
        return clamped

    def _apply_budget_depletion(self, final_action: dict):
        """Depletes resources based on budget actions (Point 7A)."""
        budget_fractions = {'0': 0.0, '5': 0.05, '15': 0.15, '30': 0.30, '50': 0.50}
        budget_fraction = budget_fractions.get(str(final_action.get('emergency_budget', '0')), 0.0)
        state = self.state_manager.state
        state['resources'] = max(0.0, state['resources'] - (budget_fraction * state['resources']))
        if state['resources'] < 100.0:
            state['fiscal_deficit_flag'] = True

    def step(self, actions_dict: dict, raw_agent_actions: dict = None) -> tuple:
        """
        Advance the simulation by one turn.

        Args:
            actions_dict: dict of final aggregated actions per domain
                          (already resolved via core/aggregation.py)
            raw_agent_actions: dict of per-agent actions before aggregation
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        turn = self.state_manager.state['turn']

        # 1. Apply scenario-specific update equations
        scenario_deltas = {}
        sd_updates = {}
        if self.scenario_module and hasattr(self.scenario_module, 'update'):
            scenario_deltas, sd_updates = self.scenario_module.update(
                self.state_manager.state, actions_dict
            )

        # 2. Apply crisis events for this turn
        crisis_deltas = self.scenario_loader.get_crisis_events(
            self.scenario, turn + 1
        )
        for field, delta in crisis_deltas.items():
            scenario_deltas[field] = scenario_deltas.get(field, 0) + delta

        # 3. Apply all deltas to state
        self.state_manager.apply_deltas(scenario_deltas)

        # 3.5 Apply Synergies, Noise, and Budget Depletion
        self._apply_budget_depletion(actions_dict)
        
        if raw_agent_actions:
            # rewards dict dummy since we process rewards later
            self.state_manager.state, _ = apply_joint_synergies(
                self.state_manager.state, raw_agent_actions, {}
            )
            
        self.state_manager.state = apply_outcome_noise(self.state_manager.state)

        # 4. Update scenario_data sub-dict
        if sd_updates:
            current_sd = self.state_manager.state.get('scenario_data', {})
            current_sd.update(sd_updates)
            self.state_manager.state['scenario_data'] = current_sd

        # 5. Update difficulty tier
        self.state_manager.state['difficulty_tier'] = \
            self.state_manager.compute_difficulty_tier()

        # 6. Advance turn
        self.state_manager.advance_turn()

        # 7. Record coalition snapshot
        self.coalition_history.append(
            deepcopy(self.state_manager.coalition_map)
        )

        # 8. Check termination
        collapsed = self.state_manager.check_collapse()
        max_reached = self.state_manager.state['turn'] >= self.max_steps
        self.done = collapsed or max_reached

        # 9. Get updated state
        self.state = self.state_manager.get_state()

        # 10. Build observations
        observations = self._build_observations()

        info = {
            'final_action': actions_dict,
            'messages': [],
            'headline': '',
            'collapsed': collapsed,
        }

        # Return empty rewards — actual reward computation is done by core/rewards.py
        rewards = {f'agent_{i}': 0.0 for i in range(self.num_agents)}

        return observations, rewards, self.done, info

    def _build_observations(self) -> dict:
        """
        Build per-agent observations (partial observability).
        Section 3.1.4 — Observation Filtering.
        """
        observations = {}
        state = self.state_manager.state

        # Observation filters per agent role
        observation_filters = {
            'agent_0': {  # Finance: gdp, inflation, resources, coalition_map, trust_matrix (own row)
                'can_see': ['gdp', 'inflation', 'resources', 'stability'],
                'cannot_see': ['mortality', 'gini', 'public_trust'],
            },
            'agent_1': {  # Political: public_trust, coalition_map, stability, approval ratings
                'can_see': ['public_trust', 'stability', 'gini'],
                'cannot_see': ['inflation', 'mortality'],
            },
            'agent_2': {  # Central Bank: inflation, gdp, resources, interest_rate
                'can_see': ['inflation', 'gdp', 'resources'],
                'cannot_see': ['mortality', 'public_trust', 'gini'],
            },
            'agent_3': {  # Health: mortality, gini, public_trust, stability, pandemic scenario_data
                'can_see': ['mortality', 'gini', 'public_trust', 'stability'],
                'cannot_see': ['gdp', 'inflation'],
            },
            'agent_4': {  # Military: stability, resources, disaster scenario_data
                'can_see': ['stability', 'resources'],
                'cannot_see': ['inflation', 'gini'],
            },
            'agent_5': {  # Auditor: ALL state variables (full observation)
                'can_see': 'all',
                'cannot_see': [],
            },
        }

        for i in range(self.num_agents):
            agent_id = f'agent_{i}'
            filters = observation_filters.get(agent_id, {'can_see': 'all', 'cannot_see': []})

            # Build public state view
            if filters['can_see'] == 'all':
                public_state = {
                    'gdp': state['gdp'],
                    'inflation': state['inflation'],
                    'resources': state['resources'],
                    'stability': state['stability'],
                    'mortality': state['mortality'],
                    'gini': state['gini'],
                    'public_trust': state['public_trust'],
                    'turn': state['turn'],
                    'difficulty_tier': state['difficulty_tier'],
                }
            else:
                public_state = {
                    'turn': state['turn'],
                    'difficulty_tier': state['difficulty_tier'],
                }
                for field in filters['can_see']:
                    if field in state:
                        public_state[field] = state[field]

            obs = {
                'public_state': public_state,
                'trust_row': self.state_manager.trust_matrix[i].tolist(),
                'coalition_map': deepcopy(self.state_manager.coalition_map),
                'agent_id': agent_id,
            }

                # Auditor gets full scenario_data
            if agent_id == 'agent_5':
                obs['scenario_data'] = deepcopy(
                    state.get('scenario_data', {})
                )

            # Add past actions for Point 5
            obs['my_past_actions'] = state.get('past_actions', {}).get(agent_id, [])

            observations[agent_id] = obs

        return observations
