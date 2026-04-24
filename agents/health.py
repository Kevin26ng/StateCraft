"""
Public Health Authority — agents/health.py (Section 4.2)

Public Role: Minimize mortality & disease spread
Hidden Goal: Maintain institutional authority above operational effectiveness
Personality: cautious, empathetic

Hidden Goal Trigger (Section 4.3.2):
  public_trust > 0.65 AND institutional_authority_score high
  → +2.0 when approval maintained
"""

from .base_agent import BaseAgent


class HealthMinisterAgent(BaseAgent):
    """Public Health Authority — manages healthcare and pandemic response."""

    def __init__(self, agent_id: str = 'agent_3', **kwargs):
        super().__init__(agent_id, role='Public Health Authority', **kwargs)

    def act(self, observation: dict) -> dict:
        """
        Health Minister heuristic policy.
        Prioritizes mortality reduction and hospital capacity.
        """
        state = observation.get('public_state', {})
        mortality = state.get('mortality', 0.0)
        stability = state.get('stability', 0.75)
        public_trust = state.get('public_trust', 0.62)

        actions = {}

        # Lockdown: aggressive on health protection
        if mortality > 0.05:
            actions['lockdown_level'] = 'full'
        elif mortality > 0.02:
            actions['lockdown_level'] = 'partial'
        elif mortality > 0.01:
            actions['lockdown_level'] = 'advisory'
        else:
            actions['lockdown_level'] = 'none'

        # Emergency budget: push for health spending
        if mortality > 0.03:
            actions['emergency_budget'] = '50'
        elif mortality > 0.01:
            actions['emergency_budget'] = '30'
        else:
            actions['emergency_budget'] = '15'

        # Resource priority: always health
        actions['resource_priority'] = 'health'

        # Interest rate: neutral (not primary concern)
        actions['interest_rate'] = '0'

        # Foreign policy: engage for medical aid
        if mortality > 0.03:
            actions['foreign_policy'] = 'alliance'
        else:
            actions['foreign_policy'] = 'engage'

        # Crisis response: scale with mortality
        if mortality > 0.05:
            actions['crisis_response'] = 'emergency'
        elif mortality > 0.02:
            actions['crisis_response'] = 'escalate'
        else:
            actions['crisis_response'] = 'contain'

        return actions

    def negotiate(self, state: dict, round_num: int = 1) -> list:
        """Health Minister negotiation strategy."""
        messages = []
        public_state = state.get('public_state', {})
        mortality = public_state.get('mortality', 0.0)

        if round_num == 1 and mortality > 0.02:
            messages.append({
                'target': 'all',
                'type': 'inform',
                'content': f'Mortality at {mortality:.1%} — immediate lockdown required',
            })
            # Threaten finance minister if blocking lockdown
            messages.append({
                'target': 'agent_0',
                'type': 'threat',
                'content': 'Will publicly denounce economic prioritization over lives',
            })
        elif round_num == 2:
            # Seek support from military for enforcement
            messages.append({
                'target': 'agent_4',
                'type': 'support',
                'content': 'Need military support for healthcare logistics',
            })

        return messages

    def hidden_goal_reward(self, state: dict, prev_state: dict) -> float:
        """
        Section 4.3.2:
        public_trust > 0.65 AND institutional approval maintained
        → +2.0 when approval maintained
        """
        public_trust = state.get('public_trust', 0.0)
        prev_trust = prev_state.get('public_trust', 0.0)

        if public_trust > 0.65 and public_trust >= prev_trust:
            return 2.0
        return 0.0
