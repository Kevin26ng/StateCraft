"""
Economic Scenario — env/economic.py (Section 3.3)

Update equations and discrete action space for the economic crisis scenario.
"""

import numpy as np


def get_initial_scenario_data() -> dict:
    """Return the initial economic-specific scenario_data."""
    return {
        'interest_rate': 0.02,
        'bond_yield_spread': 0.01,
        'unemployment': 0.05,
        'banking_sector_health': 0.8,
        'consumer_confidence': 0.6,
        'trade_balance': 0.0,
        'interest_rate_lag_buffer': [],  # last 10 turns of interest rates
    }


def get_initial_state() -> dict:
    """Return initial state overrides for economic scenario."""
    return {
        'gdp': 0.85,
        'inflation': 0.08,
        'resources': 600.0,
        'stability': 0.60,
        'mortality': 0.01,
        'gini': 0.45,
        'public_trust': 0.50,
        'scenario_data': get_initial_scenario_data(),
    }


def update(state: dict, actions: dict) -> dict:
    """
    Run economic update equations each turn.
    Section 3.3.1 Update Equations.

    Returns:
        tuple: (deltas, sd_updates)
    """
    sd = state.get('scenario_data', {})
    deltas = {}
    sd_updates = {}

    interest_rate = sd.get('interest_rate', 0.02)
    unemployment = sd.get('unemployment', 0.05)
    gdp = state.get('gdp', 1.0)
    inflation = state.get('inflation', 0.02)
    lag_buffer = list(sd.get('interest_rate_lag_buffer', []))

    # --- Interest rate action ---
    rate_action = actions.get('interest_rate', '0')
    rate_map = {
        '-0.5': -0.005, '-0.25': -0.0025, '0': 0.0,
        '+0.5': 0.005, '+0.25': 0.0025, '+1': 0.01, '+2': 0.02,
    }
    rate_delta = rate_map.get(rate_action, 0.0)
    new_rate = np.clip(interest_rate + rate_delta, -0.01, 0.15)
    sd_updates['interest_rate'] = new_rate

    # Interest rate effect (lagged — takes 10 turns to propagate)
    lag_buffer.append(new_rate)
    if len(lag_buffer) > 10:
        lag_buffer.pop(0)
    sd_updates['interest_rate_lag_buffer'] = lag_buffer

    # Inflation pressure from lagged interest rate
    if len(lag_buffer) >= 10:
        lagged_rate = lag_buffer[0]
    else:
        lagged_rate = interest_rate
    inflation_pressure = 0.01 * (1 - (lagged_rate * 0.5))
    deltas['inflation'] = inflation_pressure  # lagged by 10T

    # --- Emergency budget action ---
    budget_action = actions.get('emergency_budget', '0')
    budget_map = {'0': 0, '5': 5, '15': 15, '30': 30, '50': 50}
    budget_pct = budget_map.get(budget_action, 0)

    # GDP update
    stimulus = budget_pct / 100.0
    gdp_delta = (stimulus * 0.3) - (inflation * 0.2)
    deltas['gdp'] = np.clip(gdp_delta, -0.06, 3.0)

    # Unemployment proxy
    new_unemployment = max(0.0, unemployment + 0.05 + ((1.0 - gdp) * 0.3) - stimulus * 0.5)
    sd_updates['unemployment'] = min(1.0, new_unemployment)

    # Public trust hit from high unemployment
    if new_unemployment > 0.10:
        deltas['public_trust'] = -0.02

    # Gini update (inequality worsens with recession)
    if gdp < 0.5:
        deltas['gini'] = 0.01

    # Resources consumed by stimulus
    deltas['resources'] = -(budget_pct * 10)

    return deltas, sd_updates


# Discrete Action Space — Section 3.3.2
ACTION_DOMAINS = {
    'interest_rate': {
        'options': ['-0.5', '-0.25', '0', '+0.25', '+0.5', '+1', '+2'],
        'effect': 'Inflation control with 10T delay',
    },
    'emergency_budget': {
        'options': ['0', '5', '15', '30', '50'],
        'effect': 'Stimulus. Adds to gdp next turn. Increases inflation.',
    },
    'resource_priority': {
        'options': ['health', 'infrastructure', 'military', 'services'],
        'effect': 'Shifts welfare vs growth investment',
    },
    'foreign_policy': {
        'options': ['isolate', 'neutral', 'engage', 'alliance'],
        'effect': 'Trade relationships affect GDP',
    },
    'crisis_response': {
        'options': ['monitor', 'contain', 'escalate', 'emergency'],
        'effect': 'Unlock fiscal emergency powers',
    },
}

CRISIS_EVENTS = {
    2:  {'gdp': -0.05},
    5:  {'inflation': 0.04, 'public_trust': -0.10},
    8:  {'resources': -150, 'stability': -0.08},
    12: {'gini': 0.05, 'stability': -0.06},
    18: {'gdp': -0.08},
    22: {'public_trust': -0.10, 'stability': -0.10},
}
