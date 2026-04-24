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
