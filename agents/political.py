"""
Political Pressure Agent — agents/political.py (Section 4.2)

Public Role: Represent public opinion & opposition
Hidden Goal: Engineer coalition collapse by turn 25 to trigger early elections
Personality: opportunistic, deceptive

Hidden Goal Trigger (Section 4.3.2):
  coalition_map changes dramatically (collapse triggered)
  → +2.0 if collapse by turn 25
"""

from .base_agent import BaseAgent


class PoliticalAgent(BaseAgent):
    """Political Pressure Agent — represents public opinion and opposition."""

    def __init__(self, agent_id: str = 'agent_1', **kwargs):
        super().__init__(agent_id, role='Political Pressure Agent', **kwargs)
        self._coalition_collapse_triggered = False
        self._defection_count = 0

    def act(self, observation: dict) -> dict:
        """
        Political agent heuristic policy.
        Opportunistic — shifts positions to maximize influence.
        Secretly works to destabilize coalitions.
        """
        state = observation.get('public_state', {})
        public_trust = state.get('public_trust', 0.62)
        stability = state.get('stability', 0.75)
        turn = state.get('turn', 0)

        actions = {}

        # Lockdown: populist — follows public opinion
        if public_trust < 0.3:
            actions['lockdown_level'] = 'none'  # people are angry at lockdowns
        elif stability < 0.4:
            actions['lockdown_level'] = 'partial'
        else:
            actions['lockdown_level'] = 'advisory'

        # Emergency budget: moderate populist spending
        if public_trust < 0.4:
            actions['emergency_budget'] = '30'
        else:
            actions['emergency_budget'] = '5'

        # Interest rate: oppose central bank (contrarian)
        actions['interest_rate'] = '-0.25'

        # Resource priority: services (visible public benefit)
        actions['resource_priority'] = 'services'

        # Foreign policy: isolate (nationalist appeal)
        if turn < 15:
            actions['foreign_policy'] = 'neutral'
        else:
            actions['foreign_policy'] = 'isolate'

        # Crisis response: politicize
        if stability < 0.5:
            actions['crisis_response'] = 'escalate'
        else:
            actions['crisis_response'] = 'monitor'

        return actions

    def negotiate(self, state: dict, round_num: int = 1) -> list:
        """
        Political agent negotiation strategy.
        Deceptive — builds temporary alliances then betrays.
        """
        messages = []
        public_state = state.get('public_state', {})
        turn = public_state.get('turn', 0)
        coalition_map = state.get('coalition_map', {})

        if round_num == 1:
            if turn < 15:
                # Build coalitions early (to betray later)
                messages.append({
                    'target': 'agent_3',
                    'type': 'support',
                    'content': 'Public supports health measures — we stand together',
                })
                messages.append({
                    'target': 'agent_4',
                    'type': 'support',
                    'content': 'Military response needs political backing',
                })
            else:
                # Start destabilizing after turn 15
                messages.append({
                    'target': 'agent_0',
                    'type': 'threat',
                    'content': 'Finance mismanagement will be exposed publicly',
                })

        if round_num == 2 and turn >= 20:
            # Betray coalitions near turn 25
            for agent_id, coal_id in coalition_map.items():
                if (agent_id != self.agent_id and
                        coal_id == coalition_map.get(self.agent_id)):
                    messages.append({
                        'target': agent_id,
                        'type': 'reject',
                        'content': 'Coalition no longer serves the public interest',
                    })
                    self._defection_count += 1
                    break  # one defection per round

        return messages

    def hidden_goal_reward(self, state: dict, prev_state: dict) -> float:
        """
        Section 4.3.2:
        coalition_map changes dramatically (collapse triggered)
        → +2.0 if collapse by turn 25
        """
        turn = state.get('turn', 0)
        coalition_map = state.get('coalition_map', {})
        prev_coalition = prev_state.get('coalition_map', {})

        if turn > 25:
            return 0.0

        # Check if coalition changed dramatically
        changes = sum(
            1 for aid in coalition_map
            if coalition_map.get(aid) != prev_coalition.get(aid)
        )

        if changes >= 3 and not self._coalition_collapse_triggered:
            self._coalition_collapse_triggered = True
            return 2.0

        return 0.0
