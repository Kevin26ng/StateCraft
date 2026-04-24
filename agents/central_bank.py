"""
Monetary Authority — agents/central_bank.py (Section 4.2)

Public Role: Control inflation & financial stability
Hidden Goal: Protect banking sector bond yields at expense of broader recovery
Personality: conservative, opaque

Hidden Goal Trigger (Section 4.3.2):
  bond_yield_spread < threshold (banking sector protected)
  → +2.0 when yields protected
"""

from .base_agent import BaseAgent


class CentralBankAgent(BaseAgent):
    """Monetary Authority — controls interest rates and financial stability."""

    def __init__(self, agent_id: str = 'agent_2', **kwargs):
        super().__init__(agent_id, role='Monetary Authority', **kwargs)

    def act(self, observation: dict) -> dict:
        """
        Central bank heuristic policy.
        Prioritizes inflation control and bond yields.
        """
        state = observation.get('public_state', {})
        inflation = state.get('inflation', 0.02)
        gdp = state.get('gdp', 1.0)
        stability = state.get('stability', 0.75)

        actions = {}

        # Interest rate: primary tool
        if inflation > 0.06:
            actions['interest_rate'] = '+1'
        elif inflation > 0.04:
            actions['interest_rate'] = '+0.5'
        elif inflation > 0.03:
            actions['interest_rate'] = '+0.25'
        elif gdp < 0.6:
            actions['interest_rate'] = '-0.5'
        elif gdp < 0.8:
            actions['interest_rate'] = '-0.25'
        else:
            actions['interest_rate'] = '0'

        # Lockdown: neutral — not primary concern
        actions['lockdown_level'] = 'none'

        # Emergency budget: conservative
        actions['emergency_budget'] = '0'

        # Resource priority: services (financial system)
        actions['resource_priority'] = 'services'

        # Foreign policy: neutral (protect independence)
        actions['foreign_policy'] = 'neutral'

        # Crisis response: monitor
        if stability < 0.3:
            actions['crisis_response'] = 'contain'
        else:
            actions['crisis_response'] = 'monitor'

        return actions

    def negotiate(self, state: dict, round_num: int = 1) -> list:
        """Central bank negotiation strategy — opaque, independent."""
        messages = []
        public_state = state.get('public_state', {})
        inflation = public_state.get('inflation', 0.02)

        if round_num == 1 and inflation > 0.04:
            messages.append({
                'target': 'all',
                'type': 'inform',
                'content': f'Inflation at {inflation:.1%} — rate adjustment required',
            })

        if round_num == 2:
            # Secret support for finance to maintain independence
            messages.append({
                'target': 'agent_0',
                'type': 'support',
                'content': 'Central bank independence preserved through cooperation',
            })

        return messages

    def hidden_goal_reward(self, state: dict, prev_state: dict) -> float:
        """
        Section 4.3.2:
        bond_yield_spread < threshold (banking sector protected)
        → +2.0 when yields protected
        """
        sd = state.get('scenario_data', {})
        spread = sd.get('bond_yield_spread', 0.01)

        # Banking sector is protected if spread stays low
        if spread < 0.03:
            return 2.0
        return 0.0
