"""
Training Loop — training/loop.py (Section 6)

Main training loop supporting 3 episode modes.
Uses core/step_logic.py for the canonical turn execution order.
"""

import sys
import os
import time
import numpy as np
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.crisis_env import CrisisEnv
from agents.finance import FinanceMinisterAgent
from agents.health import HealthMinisterAgent
from agents.military import MilitaryAgent
from agents.central_bank import CentralBankAgent
from agents.political import PoliticalAgent
from agents.auditor import AuditorAgent
from agents.crisis_generator_agent import CrisisGeneratorAgent
from core.trust import TrustSystem
from core.negotiation import NegotiationSystem
from core.aggregation import aggregate_actions
from core.rewards import RewardSystem
from metrics.tracker import MetricsTracker
from logs.event_logger import EventLogger
from logs.narrative import NarrativeSystem
from memory.store import MemoryStore


# Section 6.1 — Three Episode Modes
EPISODE_MODE = {'TRAINING', 'DEMO', 'STRESS_TEST'}
MAX_STEPS = {'TRAINING': 30, 'DEMO': 200, 'STRESS_TEST': 500}
NUM_EPISODES = 500


def create_agents(memory_store=None) -> dict:
    """Create the 6 canonical agents."""
    agents = {
        'agent_0': FinanceMinisterAgent('agent_0'),
        'agent_1': PoliticalAgent('agent_1'),
        'agent_2': CentralBankAgent('agent_2'),
        'agent_3': HealthMinisterAgent('agent_3'),
        'agent_4': MilitaryAgent('agent_4'),
        'agent_5': AuditorAgent('agent_5'),
    }
    # Load cross-episode memory
    if memory_store:
        for agent in agents.values():
            agent.load_memory(memory_store)
    return agents


def run_training_loop(config: dict = None):
    """
    Section 6.2 — Main Training Loop.

    for episode in range(NUM_EPISODES):
        obs = env.reset()                     # loads cross-episode memory
        done = False
        episode_rewards = { agent: 0 for agent in AGENTS }

        while not done:
            # 1. Each agent observes (partial)
            # 2. Negotiate (3 rounds)
            # 3. Act (discrete policy choices)
            # 4. Aggregate actions
            # 5. Step environment
            # 6. Reward clipping
            # 7. Log events

        # End of episode
        metrics = tracker.compute_episode_metrics(env)
        memory_store.save_episode_summary(agents, episode, metrics)
        crisis_generator.check_and_promote(metrics_history)
    """
    if config is None:
        config = {}

    mode = config.get('episode_mode', 'TRAINING')
    max_steps = MAX_STEPS.get(mode, 30)
    num_episodes = config.get('num_episodes', NUM_EPISODES)
    scenario = config.get('scenario', 'pandemic')

    # Initialize systems
    env = CrisisEnv(config)
    tracker = MetricsTracker()
    event_logger = EventLogger()
    narrative = NarrativeSystem()
    memory_store = MemoryStore(
        backend=config.get('memory_backend', 'json'),
        path=config.get('memory_path', './data/memory.json'),
    )
    trust_system = TrustSystem(n_agents=6)
    negotiation_system = NegotiationSystem(trust_system)
    reward_system = RewardSystem()
    crisis_generator = CrisisGeneratorAgent()

    # Create agents
    AGENTS = create_agents(memory_store)
    AGENT_IDS = list(AGENTS.keys())

    print("=" * 70)
    print("Cognitive Society: Crisis Governance Simulator")
    print(f"Mode: {mode}")
    print(f"Scenario: {scenario}")
    print(f"Episodes: {num_episodes}")
    print("=" * 70)

    metrics_history = []

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        event_logger.clear_turn_events()
        done = False
        episode_rewards = {agent_id: 0.0 for agent_id in AGENT_IDS}

        # Apply tier-specific initial conditions
        crisis_generator.apply_tier_to_state(env.state_manager.state)

        # Reset trust system for new episode
        trust_system._init_defaults()

        for step_num in range(max_steps):
            if done:
                break

            prev_state = deepcopy(env.state_manager.state)

            # 1. Each agent observes (partial)
            observations = obs

            # 2. Negotiate (3 rounds)
            negotiation_system.reset_turn()
            all_messages = []
            for round_num in range(1, 4):
                messages = negotiation_system.negotiate_round(
                    AGENTS, observations, round_num
                )
                negotiation_system.update_from_messages(messages)
                all_messages.extend(messages)

            # Resolve pending trades
            trust_system.resolve_trades(env.state_manager.state.get('turn', 0))

            # 3. Act (discrete policy choices)
            actions = {}
            for agent_id in AGENT_IDS:
                agent = AGENTS[agent_id]
                actions[agent_id] = agent.act(observations.get(agent_id, {}))

            # 4. Aggregate actions
            final_action = aggregate_actions(actions)

            # 5. Step environment
            obs, rewards_raw, done, info = env.step(final_action)

            # Inject crisis generator events
            crisis_event = crisis_generator.generate_event(
                crisis_generator.current_tier,
                env.state_manager.state.get('turn', 0),
            )
            if crisis_event:
                env.state_manager.apply_deltas(crisis_event)

            # Sync trust system state into env
            env.state_manager.trust_matrix = trust_system.get_trust_matrix()
            env.state_manager.coalition_map = trust_system.get_coalition_map()

            # 6. Reward clipping
            current_state = env.state_manager.state
            rewards = {}
            for agent_id in AGENT_IDS:
                rewards[agent_id] = reward_system.compute_and_clip_rewards(
                    state=current_state,
                    prev_state=prev_state,
                    agent_id=agent_id,
                    done=done,
                    agents=AGENTS,
                )
            rewards = {
                a: float(np.clip(rewards[a], -10, 10)) for a in AGENT_IDS
            }

            # 7. Log events
            narrative_headline = narrative.generate(
                current_state,
                event_logger.get_turn_events(),
                current_state.get('turn', 0),
            )

            # Accumulate episode rewards
            for a in AGENT_IDS:
                episode_rewards[a] += rewards[a]

            # Update auditor history
            auditor = AGENTS.get('agent_5')
            if hasattr(auditor, 'update_history'):
                for agent_id in AGENT_IDS:
                    if agent_id != 'agent_5':
                        auditor.update_history(agent_id, {
                            'actions': actions.get(agent_id, {}),
                            'messages': [m for m in all_messages
                                         if m.get('sender') == agent_id],
                        })

            # Demo mode: slow rendering
            if config.get('demo_mode', False):
                time.sleep(0.5)
                print(f"  Turn {step_num+1}: {narrative_headline}")

            event_logger.clear_turn_events()

        # --- End of episode ---
        tracker.accumulate_reward(sum(episode_rewards.values()))
        metrics = tracker.compute_episode_metrics(env)
        tracker.record_episode(metrics)
        metrics_history.append(metrics)

        # Run auditor audit at end of episode
        if hasattr(AGENTS.get('agent_5'), 'run_audit'):
            AGENTS['agent_5'].run_audit()

        # Store key events in memory
        for agent_id in AGENT_IDS:
            memory_store.append(agent_id, {
                'episode': episode,
                'summary': (
                    f'Score: {metrics["society_score"]:.0f}/100, '
                    f'GDP: {env.state["gdp"]:.2f}, '
                    f'Mortality: {env.state["mortality"]:.2%}'
                ),
            })

        # Check tier promotion
        if crisis_generator.check_promotion(metrics_history):
            old_tier = crisis_generator.current_tier
            crisis_generator.escalate_tier()
            print(f"  *** TIER PROMOTION: {old_tier} -> {crisis_generator.current_tier} ***")

        # Progress output
        if episode % 50 == 0 or episode <= 3:
            print(
                f"Ep {episode:>4d} | "
                f"Score: {metrics['society_score']:5.1f} | "
                f"Stability: {metrics['alliance_stability']:5.1f} | "
                f"Trust: {metrics['trust_network_avg']:.2f} | "
                f"Turns: {metrics['turns_survived']:>3d} | "
                f"Tier: {crisis_generator.current_tier}"
            )

    print("\n" + "=" * 70)
    print("Training complete.")
    print(f"Final society score: {metrics['society_score']:.1f}/100")
    print(f"Final tier: {crisis_generator.current_tier}")
    print("=" * 70)

    return metrics_history


if __name__ == '__main__':
    run_training_loop()
