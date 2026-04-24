"""
Action Aggregation — core/aggregation.py (Section 5.3)

Final policy = weighted average of all agents' discrete choices.
Ties broken by: health > central_bank > finance > political > military.
Auditor votes carry base weight.
"""

from collections import Counter

# Section 5.3 — Agent weights for policy aggregation
AGENT_WEIGHTS = {
    'health':       1.5,
    'central_bank': 1.4,
    'finance':      1.3,
    'political':    1.2,
    'military':     1.2,
    'auditor':      1.0,   # Auditor votes carry base weight
}

# Map agent_id to weight key
AGENT_ID_TO_WEIGHT_KEY = {
    'agent_0': 'finance',
    'agent_1': 'political',
    'agent_2': 'central_bank',
    'agent_3': 'health',
    'agent_4': 'military',
    'agent_5': 'auditor',
}

# Action domains that get aggregated
ACTION_DOMAINS = [
    'lockdown_level', 'emergency_budget', 'resource_priority',
    'foreign_policy', 'crisis_response', 'interest_rate',
]


def _weighted_vote(votes: list, agent_weights: dict) -> str:
    """
    Compute weighted vote across agents.

    Args:
        votes: list of (agent_id, action) tuples
        agent_weights: mapping of weight_key -> weight

    Returns:
        str — winning action
    """
    if not votes:
        return 'none'

    weighted_counts = {}
    for agent_id, action in votes:
        weight_key = AGENT_ID_TO_WEIGHT_KEY.get(agent_id, 'auditor')
        weight = agent_weights.get(weight_key, 1.0)
        weighted_counts[action] = weighted_counts.get(action, 0.0) + weight

    if not weighted_counts:
        return 'none'

    # Return action with highest weighted count
    return max(weighted_counts, key=weighted_counts.get)


def aggregate_actions(actions_dict: dict) -> dict:
    """
    Aggregate all agents' discrete choices into final policy actions.
    Section 5.3 — For discrete choices: weighted vote among non-abstaining agents.

    Args:
        actions_dict: { agent_id: { domain: action } }

    Returns:
        dict: { domain: final_action }
    """
    final_action = {}

    for domain in ACTION_DOMAINS:
        votes = []
        for agent_id, agent_actions in actions_dict.items():
            if isinstance(agent_actions, dict) and domain in agent_actions:
                action = agent_actions[domain]
                if action is not None:
                    votes.append((agent_id, action))

        if votes:
            final_action[domain] = _weighted_vote(votes, AGENT_WEIGHTS)
        # If no votes for this domain, don't include it

    return final_action
