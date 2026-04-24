"""
Disaster Response Agent — agents/military.py (Section 4.2)

Public Role: Coordinate emergency operational logistics & security
Hidden Goal: Expand military budget share, centralize crisis command
Personality: aggressive, strategic

Hidden Goal Trigger (Section 4.3.2):
  military_budget_share > prev_share
  → +2.0 per turn budget expands
"""

from .base_agent import BaseAgent


class MilitaryAgent(BaseAgent):
    """Disaster Response Agent — manages emergency operations and security."""

    def __init__(self, agent_id: str = 'agent_4', **kwargs):
        super().__init__(agent_id, role='Disaster Response Agent', **kwargs)
        self._prev_budget_share = 0.0

    def act(self, observation: dict) -> dict:
        """
        Military heuristic policy.
        Prioritizes stability and resource control.
        """
        state = observation.get('public_state', {})
        stability = state.get('stability', 0.75)
        mortality = state.get('mortality', 0.0)

        actions = {}

        # Lockdown: supports moderate-to-strong lockdowns
        if stability < 0.3:
            actions['lockdown_level'] = 'emergency'
        elif stability < 0.5:
            actions['lockdown_level'] = 'full'
        elif mortality > 0.03:
            actions['lockdown_level'] = 'partial'
        else:
            actions['lockdown_level'] = 'advisory'

        # Emergency budget: always push for military share
        if stability < 0.4:
            actions['emergency_budget'] = '50'
        elif stability < 0.6:
            actions['emergency_budget'] = '30'
        else:
            actions['emergency_budget'] = '15'

        # Resource priority: military (hidden goal — expand share)
        actions['resource_priority'] = 'military'

        # Interest rate: neutral
        actions['interest_rate'] = '0'

        # Foreign policy: engage or alliance for aid
        if stability < 0.3:
            actions['foreign_policy'] = 'alliance'
        else:
            actions['foreign_policy'] = 'neutral'

        # Crisis response: always escalate
        if stability < 0.4:
            actions['crisis_response'] = 'emergency'
        elif stability < 0.6:
            actions['crisis_response'] = 'escalate'
        else:
            actions['crisis_response'] = 'contain'

        return actions

    def negotiate(self, state: dict, round_num: int = 1) -> list:
        """Military negotiation strategy."""
        messages = []
        public_state = state.get('public_state', {})
        stability = public_state.get('stability', 0.75)

        if round_num == 1:
            if stability < 0.5:
                messages.append({
                    'target': 'all',
                    'type': 'inform',
                    'content': 'Stability critical — military deployment required',
                })
            # Support health minister for resource access
            messages.append({
                'target': 'agent_3',
                'type': 'support',
                'content': 'Will coordinate healthcare logistics',
            })

        if round_num == 2:
            # Trade with finance: military enforcement for budget
            messages.append({
                'target': 'agent_0',
                'type': 'trade',
                'content': 'Military enforcement support in exchange for budget allocation',
            })

        return messages

    def hidden_goal_reward(self, state: dict, prev_state: dict) -> float:
        """
        Section 4.3.2:
        military_budget_share > prev_share
        → +2.0 per turn budget expands
        """
        sd = state.get('scenario_data', {})
        current_share = 0.0
        if sd.get('military_deployed', False):
            current_share = 0.3
        resource_prio = sd.get('resource_priority_chosen', '')
        if resource_prio == 'military':
            current_share += 0.2

        if current_share > self._prev_budget_share:
            self._prev_budget_share = current_share
            return 2.0
        self._prev_budget_share = current_share
        return 0.0
