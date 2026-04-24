"""
Step Logic — core/step_logic.py (Section 6.2)

Master per-turn step: observe → negotiate → act → aggregate → step env → reward → log.
This is the canonical turn execution order from the README.
"""

import numpy as np
from copy import deepcopy

from core.negotiation import NegotiationSystem
from core.trust import TrustSystem
from core.aggregation import aggregate_actions
from core.rewards import RewardSystem


NOISE_CONFIG = {
    'gdp': 0.005,
    'mortality': 0.002,
    'stability': 0.003,
    'inflation': 0.001,
}

def apply_outcome_noise(state: dict, seed_offset: int = 0) -> dict:
    """Small Gaussian noise on all outcome equations prevents deterministic exploitation."""
    for key, sigma in NOISE_CONFIG.items():
        if key in state:
            noise = np.random.normal(0, sigma)
            # GDP max is 3.0, others max at 1.0 (approximated based on docs)
            state[key] = float(np.clip(state[key] + noise, 0.0, 3.0 if key == 'gdp' else 1.0))
    return state

def apply_joint_synergies(state: dict, actions_dict: dict, rewards_dict: dict) -> tuple:
    """
    JOINT ACTION SYNERGY MECHANICS
    Forces agents to learn joint strategy by defining explicit combination outcomes.
    """
    lockdown_scores = {'none': 0, 'advisory': 1, 'partial': 2, 'full': 3, 'emergency': 4}
    crisis_scores = {'monitor': 0, 'contain': 1, 'escalate': 2, 'emergency': 3}
    budget_fractions = {'0': 0.0, '5': 0.05, '15': 0.15, '30': 0.30, '50': 0.50}

    # agent_3 is health, agent_0 is finance, agent_4 is military, agent_2 is central bank
    lockdown = lockdown_scores.get(actions_dict.get('agent_3', {}).get('lockdown_level', 'none'), 0)
    stimulus = budget_fractions.get(str(actions_dict.get('agent_0', {}).get('emergency_budget', '0')), 0.0)
    crisis_r = crisis_scores.get(actions_dict.get('agent_4', {}).get('crisis_response', 'monitor'), 0)
    
    interest_rate_str = actions_dict.get('agent_2', {}).get('interest_rate', '0')
    interest = 3 if interest_rate_str in ['+0.5', '+1', '+2'] else 0

    # Synergy 1: Health lockdown + Finance stimulus
    if lockdown >= 3 and stimulus < 0.05:
        state['gdp'] -= 0.05 # Crash without buffer
    elif lockdown >= 3 and stimulus >= 0.15:
        rewards_dict['agent_3'] = rewards_dict.get('agent_3', 0.0) + 2.0
        rewards_dict['agent_0'] = rewards_dict.get('agent_0', 0.0) + 2.0

    # Synergy 2: Disaster response + Resource allocation
    if crisis_r >= 2 and actions_dict.get('agent_3', {}).get('resource_priority', '') == 'health':
        state['mortality'] -= 0.02

    # Synergy 3: High interest rate + High stimulus
    if interest >= 3 and stimulus >= 0.15:
        state['inflation'] += 0.02
        state['gdp'] -= 0.02

    return state, rewards_dict

class StepLogic:
    """
    Orchestrates the per-turn step logic.
    Section 6.2 — Main Training Loop body.
    """

    def __init__(self, env, agents: dict, trust_system: TrustSystem,
                 reward_system: RewardSystem, event_logger=None,
                 narrative_system=None):
        self.env = env
        self.agents = agents
        self.trust_system = trust_system
        self.reward_system = reward_system
        self.negotiation_system = NegotiationSystem(trust_system)
        self.event_logger = event_logger
        self.narrative_system = narrative_system

    def execute_turn(self, observations: dict) -> tuple:
        """
        Execute one complete turn.

        Section 6.2 step order:
          1. Each agent observes (partial)
          2. Negotiate (3 rounds)
          3. Act (discrete policy choices)
          4. Aggregate actions
          5. Step environment
          6. Reward clipping
          7. Log events

        Args:
            observations: dict of agent_id -> observation

        Returns:
            tuple: (observations, rewards, done, info)
        """
        AGENTS = list(self.agents.keys())

        # 1. Each agent observes (partial)
        # observations are already passed in from env

        # 2. Negotiate (3 rounds)
        self.negotiation_system.reset_turn()
        all_messages = []
        for round_num in range(1, 4):
            messages = self.negotiation_system.negotiate_round(
                self.agents, observations, round_num
            )
            self.negotiation_system.update_from_messages(messages)
            all_messages.extend(messages)

        # Resolve pending trades
        current_turn = observations.get(
            AGENTS[0], {}
        ).get('public_state', {}).get('turn', 0)
        self.trust_system.resolve_trades(current_turn)

        # 3. Act (discrete policy choices)
        actions = {}
        for agent_id in AGENTS:
            agent = self.agents[agent_id]
            actions[agent_id] = agent.act(observations.get(agent_id, {}))

        # 4. Aggregate actions
        final_action = aggregate_actions(actions)

        # 5. Step environment
        prev_state = deepcopy(self.env.state_manager.state)
        obs, rewards_raw, done, info = self.env.step(final_action)

        # Sync trust system state into env
        self.env.state_manager.trust_matrix = self.trust_system.get_trust_matrix()
        self.env.state_manager.coalition_map = self.trust_system.get_coalition_map()

        # 6. Reward clipping
        current_state = self.env.state_manager.state
        rewards = {}
        for agent_id in AGENTS:
            rewards[agent_id] = self.reward_system.compute_and_clip_rewards(
                state=current_state,
                prev_state=prev_state,
                agent_id=agent_id,
                done=done,
                agents=self.agents,
            )
        # Clip all to [-10, 10]
        rewards = {
            a: np.clip(rewards[a], -10, 10) for a in AGENTS
        }

        # 7. Log events
        if self.event_logger:
            self.event_logger.log_turn(
                obs, actions, all_messages, rewards
            )
        if self.narrative_system:
            headline = self.narrative_system.generate(
                current_state,
                self.event_logger.get_turn_events() if self.event_logger else [],
                current_state.get('turn', 0),
            )
            info['headline'] = headline

        info['final_action'] = final_action
        info['messages'] = all_messages
        info['actions_per_agent'] = actions

        return obs, rewards, done, info
