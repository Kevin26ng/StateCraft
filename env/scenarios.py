"""
Scenario Loader — loads and manages crisis scenarios.
Supports pandemic, economic, and disaster scenarios.
Now uses the modular scenario files (env/pandemic.py, env/economic.py, env/disaster.py).
"""

import os
import yaml
from copy import deepcopy


class ScenarioLoader:
    """Loads crisis scenarios from modular scenario files and YAML configs."""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config'
            )
        self.config_dir = config_dir
        self.historical_dir = os.path.join(config_dir, 'historical_scenarios')

    def load_scenario(self, scenario_name: str) -> dict:
        """
        Load a scenario by name.
        Uses the modular scenario modules for initial state and crisis events.
        """
        if scenario_name == 'pandemic':
            from env.pandemic import get_initial_state, CRISIS_EVENTS
            return {
                'initial_state': get_initial_state(),
                'crisis_events': deepcopy(CRISIS_EVENTS),
            }
        elif scenario_name == 'economic':
            from env.economic import get_initial_state, CRISIS_EVENTS
            return {
                'initial_state': get_initial_state(),
                'crisis_events': deepcopy(CRISIS_EVENTS),
            }
        elif scenario_name == 'disaster':
            from env.disaster import get_initial_state, CRISIS_EVENTS
            return {
                'initial_state': get_initial_state(),
                'crisis_events': deepcopy(CRISIS_EVENTS),
            }
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")

    def load_historical_scenario(self, scenario_id: str) -> dict:
        """Load a historical scenario from YAML for validation."""
        yaml_path = os.path.join(self.historical_dir, f'{scenario_id}.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            return data
        raise FileNotFoundError(f"Historical scenario not found: {yaml_path}")

    def get_crisis_events(self, scenario: dict, turn: int) -> dict:
        """Get crisis events scheduled for a specific turn."""
        events = scenario.get('crisis_events', {})
        return events.get(turn, {})

    def list_scenarios(self) -> list:
        """List all available scenario names."""
        return ['pandemic', 'economic', 'disaster']
