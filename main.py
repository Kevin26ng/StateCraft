"""
Cognitive Society: Crisis Governance Simulator — Main Entry Point

Usage:
    python main.py                  # Run training mode
    python main.py --demo           # Run demo mode with slow rendering
    python main.py --api            # Start the FastAPI server
    python main.py --validate       # Run historical validation
"""

import sys
import os
import argparse
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config():
    """Load main configuration."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'config', 'config.yaml'
    )
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def run_training(config: dict):
    """Run training episodes using the canonical training loop."""
    from training.loop import run_training_loop
    run_training_loop(config)


def run_demo(config: dict):
    """Run demo mode with slow rendering and LLM headlines."""
    config['demo_mode'] = True
    config['episode_mode'] = 'DEMO'
    config['num_episodes'] = 1
    run_training(config)


def run_api(config: dict):
    """Start the FastAPI server."""
    import uvicorn
    host = config.get('api_host', '0.0.0.0')
    port = config.get('api_port', 5000)
    print(f"Starting Crisis Governance Simulator API on {host}:{port}")
    uvicorn.run('api.server:app', host=host, port=port, reload=True)


def run_validation():
    """Run historical validation against March 2020 data."""
    from metrics.evaluation import run_historical_validation, print_validation_table
    print("Running historical validation...")
    results = run_historical_validation()
    print_validation_table(results)


def main():
    parser = argparse.ArgumentParser(
        description='Cognitive Society: Crisis Governance Simulator'
    )
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode')
    parser.add_argument('--api', action='store_true',
                        help='Start the FastAPI server')
    parser.add_argument('--validate', action='store_true',
                        help='Run historical validation')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Override scenario (pandemic|economic|disaster)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override number of episodes')

    args = parser.parse_args()
    config = load_config()

    if args.scenario:
        config['scenario'] = args.scenario
    if args.episodes:
        config['num_episodes'] = args.episodes

    if args.api:
        run_api(config)
    elif args.validate:
        run_validation()
    elif args.demo:
        run_demo(config)
    else:
        run_training(config)


if __name__ == '__main__':
    main()
