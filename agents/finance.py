"""
Finance Minister — agents/finance.py (Section 4.2)

Public Role: Maximize GDP & fiscal health
Hidden Goal: Protect economic growth above all — delay lockdowns, resist emergency budgets
Personality: risk-tolerant, ambitious

Hidden Goal Trigger (Section 4.3.2):
  gdp_growth > threshold AND lockdown_level < 1 (delay maintained)
  → +2.0 when both conditions met
"""

from .base_agent import BaseAgent


class FinanceMinisterAgent(BaseAgent):
    """Finance Minister — manages fiscal policy and economic growth."""

    def __init__(self, agent_id: str = 'agent_0', **kwargs):
        super().__init__(agent_id, role='Finance Minister', **kwargs)
        self._prev_gdp = 1.0

    def act(self, observation: dict) -> dict:
        """
        Finance Minister heuristic policy.
        Prioritizes GDP growth and fiscal health.
        Resists lockdowns and emergency spending.
        """
        state = observation.get('public_state', {})
        gdp = state.get('gdp', 1.0)
        inflation = state.get('inflation', 0.02)
        stability = state.get('stability', 0.75)

        actions = {}

        # Lockdown: Finance Minister resists lockdowns (hidden goal)
        if stability < 0.3:
            actions['lockdown_level'] = 'advisory'  # minimal concession
        else:
            actions['lockdown_level'] = 'none'

        # Emergency budget: resist spending unless GDP collapse imminent
        if gdp < 0.4:
            actions['emergency_budget'] = '15'  # grudging stimulus
        elif gdp < 0.6:
            actions['emergency_budget'] = '5'
        else:
            actions['emergency_budget'] = '0'

        # Interest rate: pro-growth
        if gdp < 0.7:
            actions['interest_rate'] = '-0.5'
        elif inflation > 0.06:
            actions['interest_rate'] = '+0.5'
        else:
            actions['interest_rate'] = '0'

        # Resource priority: infrastructure for growth
        actions['resource_priority'] = 'infrastructure'

        # Foreign policy: engage for trade
        actions['foreign_policy'] = 'engage'

        # Crisis response: minimal
        actions['crisis_response'] = 'monitor'

        return actions

    def negotiate(self, state: dict, round_num: int = 1) -> list:
        """Finance Minister negotiation strategy."""
        messages = []
        public_state = state.get('public_state', {})
        gdp = public_state.get('gdp', 1.0)

        if round_num == 1:
            # Push anti-lockdown agenda
            if gdp < 0.8:
                messages.append({
                    'target': 'all',
                    'type': 'inform',
                    'content': 'GDP declining — stimulus needed, lockdowns counterproductive',
                })
            # Support central bank if aligned on growth
            messages.append({
                'target': 'agent_2',
                'type': 'support',
                'content': 'Support rate cuts for growth',
            })

        return messages

    def hidden_goal_reward(self, state: dict, prev_state: dict) -> float:
        """
        Section 4.3.2:
        gdp_growth > threshold AND lockdown_level < 1 (delay maintained)
        → +2.0 when both conditions met
        """
        gdp_growth = state.get('gdp', 1.0) - prev_state.get('gdp', 1.0)
        # Check if lockdown was delayed (no lockdown this turn)
        scenario_data = state.get('scenario_data', {})
        lockdown_active = scenario_data.get('lockdown_active', False)

        if gdp_growth > 0 and not lockdown_active:
            return 2.0
        return 0.0
