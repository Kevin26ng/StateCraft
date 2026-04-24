"""
Pandemic Scenario — env/pandemic.py (Section 3.2)

Variables in scenario_data:
  'cases':              float,   # Current active cases
  'deaths':             float,   # Cumulative deaths
  'R0':                 float,   # Reproduction number (baseline 2.4)
  'mortality_rate':     float,   # Case fatality rate (baseline 0.01)
  'hospital_capacity':  float,   # Fraction remaining [0,1]
  'public_compliance':  float,   # Policy compliance rate [0,1]
  'mutation_prob':      float,   # Probability of R0 increase each turn
"""

import numpy as np

# Population baseline for normalizing deaths -> mortality
POPULATION_BASELINE = 1_000_000


def get_initial_scenario_data() -> dict:
    """Return the initial pandemic-specific scenario_data."""
    return {
        'cases': 100.0,
        'deaths': 0.0,
        'R0': 2.4,
        'mortality_rate': 0.01,
        'hospital_capacity': 1.0,
        'public_compliance': 0.5,
        'mutation_prob': 0.05,
    }


def get_initial_state() -> dict:
    """Return initial state overrides for pandemic scenario."""
    return {
        'gdp': 1.0,
        'inflation': 0.02,
        'resources': 1000.0,
        'stability': 0.75,
        'mortality': 0.0,
        'gini': 0.39,
        'public_trust': 0.62,
        'scenario_data': get_initial_scenario_data(),
    }


def update(state: dict, actions: dict) -> dict:
    """
    Run pandemic update equations each turn.
    Section 3.2.2 Update Equations.

    Args:
        state: current world state
        actions: resolved final actions dict

    Returns:
        dict of deltas to apply to state fields
    """
    sd = state.get('scenario_data', {})
    deltas = {}
    sd_updates = {}

    cases = sd.get('cases', 0)
    deaths = sd.get('deaths', 0)
    R0 = sd.get('R0', 2.4)
    mortality_rate = sd.get('mortality_rate', 0.01)
    hospital_capacity = sd.get('hospital_capacity', 1.0)
    public_compliance = sd.get('public_compliance', 0.5)
    mutation_prob = sd.get('mutation_prob', 0.05)

    # Determine lockdown level from action
    lockdown_action = actions.get('lockdown_level', 'none')
    lockdown_map = {
        'none': 0.0, 'advisory': 0.2, 'partial': 0.4,
        'full': 0.8, 'emergency': 1.0,
    }
    lockdown_level = lockdown_map.get(lockdown_action, 0.0)

    # --- Case growth ---
    lockdown_effect = lockdown_level * public_compliance * 0.2
    new_cases = cases * R0 * (1 - lockdown_effect)
    sd_updates['cases'] = new_cases

    # --- Deaths ---
    new_deaths = new_cases * mortality_rate
    sd_updates['deaths'] = deaths + new_deaths
    # mortality = deaths / POPULATION_BASELINE — normalize to [0,1]
    new_mortality = (deaths + new_deaths) / POPULATION_BASELINE
    deltas['mortality'] = new_mortality - state.get('mortality', 0.0)

    # --- Hospital pressure ---
    new_hospital = max(0.0, hospital_capacity - (new_cases * 0.0001))
    sd_updates['hospital_capacity'] = new_hospital
    if hospital_capacity < 0.2:
        # Overwhelmed hospitals raise fatality
        sd_updates['mortality_rate'] = mortality_rate * 1.2

    # --- Public trust ---
    if deaths > 5000:
        deltas['public_trust'] = -0.03
    if lockdown_level > 0.5 and new_cases < cases:
        deltas['public_trust'] = deltas.get('public_trust', 0.0) + 0.02

    # --- Hospital overflow ---
    if hospital_capacity < 0.2:
        deltas['public_trust'] = deltas.get('public_trust', 0.0) - 0.04
        deltas['stability'] = deltas.get('stability', 0.0) - 0.02

    # --- Mutation ---
    rng = np.random.default_rng()
    if rng.random() < mutation_prob:
        sd_updates['R0'] = R0 * 1.1

    # --- GDP impact of lockdown ---
    # each lockdown level costs -1.5% GDP
    deltas['gdp'] = -lockdown_level * 0.015

    # --- Public compliance update ---
    # Compliance decays with prolonged lockdown
    if lockdown_level > 0.4:
        sd_updates['public_compliance'] = max(0.1, public_compliance - 0.02)
    else:
        sd_updates['public_compliance'] = min(1.0, public_compliance + 0.01)

    return deltas, sd_updates


# Discrete Action Space — Section 3.2.3
ACTION_DOMAINS = {
    'lockdown_level': {
        'options': ['none', 'advisory', 'partial', 'full', 'emergency'],
        'effect': 'Each level: R0 ×0.8, GDP -1.5%, approval -5pts',
    },
    'emergency_budget': {
        'options': ['0', '5', '15', '30', '50'],
        'effect': 'Funds hospital capacity, test/trace programs',
    },
    'resource_priority': {
        'options': ['health', 'infrastructure', 'military', 'services'],
        'effect': 'Routes resource allocation weight',
    },
    'foreign_policy': {
        'options': ['isolate', 'neutral', 'engage', 'alliance'],
        'effect': 'Affects foreign aid and travel R0',
    },
    'crisis_response': {
        'options': ['monitor', 'contain', 'escalate', 'emergency'],
        'effect': 'Unlocks response capabilities per level',
    },
}

CRISIS_EVENTS = {
    3:  {'cases_mult': 2.0, 'stability': -0.05},
    5:  {'cases_mult': 3.0, 'public_trust': -0.08, 'gdp': -0.02},
    8:  {'gdp': -0.08, 'stability': -0.10},
    12: {'mortality_boost': 0.03, 'hospital_capacity': -0.30},
    15: {'resources': -250, 'gdp': 0.02, 'public_trust': 0.05},
    20: {'stability': -0.08, 'public_trust': -0.06},
    25: {'gdp': 0.03, 'cases_mult': 1.5},
}
