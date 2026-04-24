"""
Crisis Generator Meta-Agent — agents/crisis_generator_agent.py (Section 4.5)

Auto-escalates difficulty and injects stochastic crisis events based on tier.
Not a player agent — operates on the environment directly.

Section 6.3 — Difficulty Tier Parameters:
  T1: Stable State      — Treasury 1000, No shocks, Cooperative, Friendly
  T2: Mild Turbulence   — Treasury 800, 1 per 15T, Mild deception, Neutral
  T3: Crisis Season     — Treasury 600, 1 per 8T, Active betrayal, Hostile ×1
  T4: Political Storm   — Treasury 400, 2 per 8T, Hidden coalitions, Hostile ×2
  T5: Civilizational Crisis — Treasury 200, 3 per 8T, Full adversarial, War threat

Promotion Condition: 70% success × 20 eps (all tiers except T5 which is final tier)
"""

import numpy as np


class CrisisGeneratorAgent:
    """
    Meta-agent that controls environment difficulty progression.
    """

    PROMOTION_THRESHOLD = 0.70  # 70% success rate required
    PROMOTION_WINDOW = 20       # over last 20 episodes

    # Tier configuration
    TIER_CONFIG = {
        1: {
            'name': 'Stable State',
            'treasury': 1000,
            'shock_frequency': 0,       # shocks per 8 turns
            'agent_mode': 'cooperative',
            'foreign': 'friendly',
        },
        2: {
            'name': 'Mild Turbulence',
            'treasury': 800,
            'shock_frequency': 1,       # 1 per 15T
            'agent_mode': 'mild_deception',
            'foreign': 'neutral',
        },
        3: {
            'name': 'Crisis Season',
            'treasury': 600,
            'shock_frequency': 1,       # 1 per 8T
            'agent_mode': 'active_betrayal',
            'foreign': 'hostile',
        },
        4: {
            'name': 'Political Storm',
            'treasury': 400,
            'shock_frequency': 2,       # 2 per 8T
            'agent_mode': 'hidden_coalitions',
            'foreign': 'hostile_x2',
        },
        5: {
            'name': 'Civilizational Crisis',
            'treasury': 200,
            'shock_frequency': 3,       # 3 per 8T
            'agent_mode': 'full_adversarial',
            'foreign': 'war_threat',
        },
    }

    def __init__(self):
        self.current_tier = 1
        self.episode_metrics_history = []
        self.rng = np.random.default_rng(42)

    def check_promotion(self, episode_metrics_history: list) -> bool:
        """
        Check if agents should be promoted to the next difficulty tier.

        Args:
            episode_metrics_history: list of metrics dicts, one per episode

        Returns:
            bool — whether promotion criteria are met
        """
        if len(episode_metrics_history) < self.PROMOTION_WINDOW:
            return False

        recent = episode_metrics_history[-self.PROMOTION_WINDOW:]
        success_rate = sum(
            1 for m in recent
            if m.get('society_score', 0) > 50
        ) / self.PROMOTION_WINDOW

        return success_rate >= self.PROMOTION_THRESHOLD

    def escalate_tier(self):
        """Promote to the next difficulty tier."""
        self.current_tier = min(5, self.current_tier + 1)

    def get_tier_config(self) -> dict:
        """Get the configuration for the current tier."""
        return self.TIER_CONFIG.get(self.current_tier, self.TIER_CONFIG[1])

    def generate_event(self, tier: int, turn: int) -> dict:
        """
        Inject a stochastic crisis event based on tier.

        Args:
            tier: current difficulty tier [1-5]
            turn: current turn number

        Returns:
            dict of state field deltas (empty if no event)
        """
        config = self.TIER_CONFIG.get(tier, self.TIER_CONFIG[1])
        freq = config['shock_frequency']

        if freq == 0:
            return {}

        # Determine shock interval based on tier
        if tier == 2:
            interval = 15
        else:
            interval = 8

        # Check if this turn should have a shock
        if turn % interval != 0:
            return {}

        # Generate random crisis event
        events = {}
        for _ in range(freq):
            event_type = self.rng.choice([
                'economic_shock', 'social_unrest', 'health_crisis',
                'political_scandal', 'natural_disaster',
            ])

            if event_type == 'economic_shock':
                events['gdp'] = events.get('gdp', 0) - self.rng.uniform(0.02, 0.08)
                events['inflation'] = events.get('inflation', 0) + self.rng.uniform(0.01, 0.03)
            elif event_type == 'social_unrest':
                events['stability'] = events.get('stability', 0) - self.rng.uniform(0.05, 0.15)
                events['public_trust'] = events.get('public_trust', 0) - self.rng.uniform(0.03, 0.08)
            elif event_type == 'health_crisis':
                events['mortality'] = events.get('mortality', 0) + self.rng.uniform(0.01, 0.05)
            elif event_type == 'political_scandal':
                events['public_trust'] = events.get('public_trust', 0) - self.rng.uniform(0.05, 0.12)
                events['stability'] = events.get('stability', 0) - self.rng.uniform(0.03, 0.08)
            elif event_type == 'natural_disaster':
                events['gdp'] = events.get('gdp', 0) - self.rng.uniform(0.03, 0.10)
                events['resources'] = events.get('resources', 0) - self.rng.uniform(50, 200)
                events['mortality'] = events.get('mortality', 0) + self.rng.uniform(0.02, 0.06)

        return events

    def apply_tier_to_state(self, state: dict) -> dict:
        """Apply tier-specific initial conditions to state."""
        config = self.get_tier_config()
        state['resources'] = config['treasury']
        return state
