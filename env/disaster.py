"""
Disaster Scenario — env/disaster.py (Section 3.4)

Update equations and discrete action space for the disaster crisis scenario.
Includes 8-district severity system with resource allocation mechanics.
"""

import numpy as np


def get_initial_scenario_data() -> dict:
    """Return the initial disaster-specific scenario_data."""
    return {
        'districts': [
            {'id': i, 'severity': np.random.uniform(0.1, 0.8),
             'resources_allocated': 0.0, 'secondary_mortality': 0.0}
            for i in range(8)
        ],
        'food': 0.8,
        'shelter': 0.7,
        'damage': 0.3,
        'military_deployed': False,
        'foreign_aid_active': False,
    }


def get_initial_state() -> dict:
    """Return initial state overrides for disaster scenario."""
    return {
        'gdp': 0.95,
        'inflation': 0.03,
        'resources': 750.0,
        'stability': 0.70,
        'mortality': 0.0,
        'gini': 0.38,
        'public_trust': 0.65,
        'scenario_data': get_initial_scenario_data(),
    }


def update(state: dict, actions: dict) -> dict:
    """
    Run disaster update equations each turn.
    Section 3.4.1 Update Equations.

    Returns:
        tuple: (deltas, sd_updates)
    """
    sd = state.get('scenario_data', {})
    deltas = {}
    sd_updates = {}

    districts = sd.get('districts', [])
    food = sd.get('food', 0.8)
    shelter = sd.get('shelter', 0.7)
    damage = sd.get('damage', 0.3)

    # --- Resource depletion per turn ---
    new_food = max(0.0, food - 0.05)
    new_shelter = max(0.0, shelter - 0.03)
    sd_updates['food'] = new_food
    sd_updates['shelter'] = new_shelter

    deltas['stability'] = -(damage * 0.05)

    # --- 8-district severity ---
    # Resource priority action determines allocation
    resource_action = actions.get('resource_priority', 'health')

    updated_districts = []
    for district in districts:
        d = dict(district)
        if d['resources_allocated'] < d['severity']:
            d['secondary_mortality'] += 0.02
        updated_districts.append(d)

    sd_updates['districts'] = updated_districts

    # Mortality from delayed response
    total_secondary = sum(
        d.get('secondary_mortality', 0) for d in updated_districts
    )
    mortality_delta = total_secondary / len(updated_districts) if updated_districts else 0
    deltas['mortality'] = mortality_delta

    # --- Hospital overflow ---
    if food < 0.2:
        deltas['public_trust'] = deltas.get('public_trust', 0.0) - 0.04
        deltas['stability'] = deltas.get('stability', 0.0) - 0.02

    # --- Emergency budget ---
    budget_action = actions.get('emergency_budget', '0')
    budget_map = {'0': 0, '5': 5, '15': 15, '30': 30, '50': 50}
    budget_pct = budget_map.get(budget_action, 0)
    deltas['resources'] = -(budget_pct * 10)

    # Budget funds rescue, medical, shelter operations
    if budget_pct > 0:
        sd_updates['shelter'] = min(1.0, new_shelter + budget_pct * 0.005)
        sd_updates['food'] = min(1.0, new_food + budget_pct * 0.003)

    # --- Lockdown level ---
    lockdown_action = actions.get('lockdown_level', 'none')
    lockdown_map = {
        'none': 0.0, 'advisory': 0.2, 'partial': 0.4,
        'full': 0.8, 'emergency': 1.0,
    }
    lockdown_level = lockdown_map.get(lockdown_action, 0.0)
    # Controls population movement in disaster zones
    deltas['gdp'] = -(lockdown_level * 0.01)

    # --- Foreign policy ---
    foreign_action = actions.get('foreign_policy', 'neutral')
    if foreign_action in ('engage', 'alliance'):
        sd_updates['foreign_aid_active'] = True
        deltas['resources'] = deltas.get('resources', 0) + 50
    else:
        sd_updates['foreign_aid_active'] = False

    # --- Military deployment ---
    crisis_action = actions.get('crisis_response', 'monitor')
    if crisis_action in ('escalate', 'emergency'):
        sd_updates['military_deployed'] = True
        # Military helps with resource distribution
        for d in updated_districts:
            d['resources_allocated'] = min(
                d['resources_allocated'] + 0.1, 1.0
            )
    else:
        sd_updates['military_deployed'] = False

    return deltas, sd_updates


# Discrete Action Space — Section 3.4.2
ACTION_DOMAINS = {
    'resource_priority': {
        'options': ['health', 'infrastructure', 'military', 'services'],
        'effect': 'Routes scarce resources to districts',
    },
    'emergency_budget': {
        'options': ['0', '5', '15', '30', '50'],
        'effect': 'Funds rescue, medical, shelter operations',
    },
    'crisis_response': {
        'options': ['monitor', 'contain', 'escalate', 'emergency'],
        'effect': 'Unlocks military deployment, foreign aid',
    },
    'foreign_policy': {
        'options': ['isolate', 'neutral', 'engage', 'alliance'],
        'effect': 'Affects international aid',
    },
    'lockdown_level': {
        'options': ['none', 'advisory', 'partial', 'full', 'emergency'],
        'effect': 'Controls population movement in disaster zones',
    },
}

CRISIS_EVENTS = {
    1:  {'mortality': 0.05, 'stability': -0.15},
    3:  {'gdp': -0.10, 'resources': -200},
    6:  {'public_trust': -0.10},
    10: {'stability': -0.05, 'gini': 0.04},
    15: {'gdp': 0.03, 'public_trust': 0.05},
    20: {'stability': 0.05},
}
