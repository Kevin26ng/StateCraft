
"""
Fix 2: Curriculum Learning with Agent Scaling
=============================================
Prevents non-stationarity from killing training convergence.
 
Problem: When 6 agents all learn simultaneously, each agent's environment is
non-stationary (because the other 5 agents are also changing their policies).
Agent A learns a good strategy, then Agent B adapts, and A's strategy becomes bad.
Training oscillates forever and never converges.
 
Solution: Start with 2 agents + 4 frozen (random) agents. Once the 2 converge,
unfreeze the next 2. Then the last 2. By the time all 6 are learning, the first
4 have stable-enough policies that the environment is mostly stationary.
 
This is NOT a new training loop — it's a wrapper around the existing one that
controls WHICH agents are "learning" (adapting heuristics) vs "frozen" (using
a fixed baseline policy).
 
Usage:
    from training.curriculum import CurriculumScheduler, run_curriculum_training
    
    # Option 1: Use the full curriculum training loop
    run_curriculum_training(config)
    
    # Option 2: Use the scheduler manually in your own loop
    scheduler = CurriculumScheduler()
    active_agents = scheduler.get_active_agents(episode=50)
"""
 
import numpy as np
from copy import deepcopy
 
 
# ─────────────────────────────────────────────────────────────
# 1. CURRICULUM PHASES
# ─────────────────────────────────────────────────────────────
# Each phase defines which agents are "active" (learning/adapting)
# and which are "frozen" (using a static baseline policy).
#
# Phase order is deliberate:
#   Phase 1: Finance + Health (core policy tension: GDP vs mortality)
#   Phase 2: + Central Bank + Military (supporting roles that react to policy)
#   Phase 3: + Political + Auditor (meta-game: trust, betrayal, oversight)
#
# This mirrors how a human would build the system: get the core
# policy debate working first, then add complexity.
 
CURRICULUM_PHASES = [
    {
        'name': 'core_policy',
        'active_agents': ['agent_0', 'agent_3'],  # Finance + Health
        'frozen_agents': ['agent_1', 'agent_2', 'agent_4', 'agent_5'],
        'min_episodes': 80,
        'promotion_threshold': 0.6,  # 60% of episodes must show improvement
        'description': 'Finance vs Health learn the core GDP/mortality tradeoff',
    },
    {
        'name': 'supporting_roles',
        'active_agents': ['agent_0', 'agent_2', 'agent_3', 'agent_4'],  # + Central Bank + Military
        'frozen_agents': ['agent_1', 'agent_5'],
        'min_episodes': 100,
        'promotion_threshold': 0.6,
        'description': 'Central Bank and Military learn to support core policy',
    },
    {
        'name': 'full_system',
        'active_agents': ['agent_0', 'agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5'],
        'frozen_agents': [],
        'min_episodes': 150,
        'promotion_threshold': 0.65,
        'description': 'All 6 agents learn together (Political + Auditor join)',
    },
]
 
 
# ─────────────────────────────────────────────────────────────
# 2. FROZEN AGENT POLICY
# ─────────────────────────────────────────────────────────────
# When an agent is "frozen," it uses this safe baseline policy
# instead of its normal heuristic. This gives active agents a
# stable, predictable environment to learn against.
 
FROZEN_POLICY = {
    'lockdown_level': 'advisory',    # Mild — doesn't crash GDP or ignore health
    'emergency_budget': '5',          # Small spend — doesn't bankrupt or neglect
    'interest_rate': '0',             # Hold steady
    'resource_priority': 'services',  # Neutral priority
    'foreign_policy': 'neutral',      # Don't rock the boat
    'crisis_response': 'contain',     # Moderate response
}
 
 
class FrozenAgentWrapper:
    """
    Wraps an existing agent and overrides act() to return the frozen policy.
    All other methods (negotiate, hidden_goal_reward, etc.) still work normally.
    
    When unfrozen, it returns the original agent's behavior transparently.
    """
    
    def __init__(self, agent):
        self._agent = agent
        self.frozen = True
    
    def act(self, observation: dict) -> dict:
        if self.frozen:
            return deepcopy(FROZEN_POLICY)
        return self._agent.act(observation)
    
    def unfreeze(self):
        self.frozen = False
    
    def freeze(self):
        self.frozen = True
    
    # Forward everything else to the real agent
    def __getattr__(self, name):
        return getattr(self._agent, name)
 
 
# ─────────────────────────────────────────────────────────────
# 3. CURRICULUM SCHEDULER
# ─────────────────────────────────────────────────────────────
 
class CurriculumScheduler:
    """
    Manages curriculum phase progression.
    
    Tracks which phase we're in, whether to promote to next phase,
    and which agents should be active vs frozen.
    
    The scheduler looks at a rolling window of episode rewards.
    If the active agents show consistent improvement over the window,
    it promotes to the next phase.
    """
    
    def __init__(self, phases=None):
        self.phases = phases or CURRICULUM_PHASES
        self.current_phase_idx = 0
        self.phase_start_episode = 0
        self.reward_history = []  # List of per-episode total rewards
        self.phase_history = []   # Log of phase transitions
        
    @property
    def current_phase(self) -> dict:
        return self.phases[min(self.current_phase_idx, len(self.phases) - 1)]
    
    @property
    def phase_name(self) -> str:
        return self.current_phase['name']
    
    def get_active_agents(self) -> list:
        """Return list of agent_ids that should be actively learning."""
        return self.current_phase['active_agents']
    
    def get_frozen_agents(self) -> list:
        """Return list of agent_ids that should use frozen policy."""
        return self.current_phase['frozen_agents']
    
    def is_final_phase(self) -> bool:
        return self.current_phase_idx >= len(self.phases) - 1
    
    def record_episode(self, episode: int, total_reward: float,
                       per_agent_rewards: dict = None):
        """Record episode result and check for phase promotion."""
        self.reward_history.append({
            'episode': episode,
            'total_reward': total_reward,
            'per_agent': per_agent_rewards or {},
            'phase': self.current_phase_idx,
        })
    
    def should_promote(self, episode: int) -> bool:
        """
        Check if we should advance to the next curriculum phase.
        
        Conditions:
        1. Spent at least min_episodes in current phase
        2. Reward trend is positive over last 20 episodes
        3. At least promotion_threshold of last 20 episodes improved vs previous 20
        """
        phase = self.current_phase
        episodes_in_phase = episode - self.phase_start_episode
        
        # Must spend minimum time in phase
        if episodes_in_phase < phase['min_episodes']:
            return False
        
        # Already at final phase
        if self.is_final_phase():
            return False
        
        # Check improvement over rolling window
        window = 20
        if len(self.reward_history) < window * 2:
            return False
        
        recent = [r['total_reward'] for r in self.reward_history[-window:]]
        previous = [r['total_reward'] for r in self.reward_history[-window*2:-window]]
        
        recent_mean = np.mean(recent)
        previous_mean = np.mean(previous)
        
        # Positive trend?
        if recent_mean <= previous_mean:
            return False
        
        # What fraction of recent episodes beat the previous average?
        beats = sum(1 for r in recent if r > previous_mean)
        improvement_ratio = beats / window
        
        return improvement_ratio >= phase['promotion_threshold']
    
    def promote(self, episode: int):
        """Advance to the next curriculum phase."""
        old_phase = self.current_phase['name']
        self.current_phase_idx = min(
            self.current_phase_idx + 1, len(self.phases) - 1
        )
        new_phase = self.current_phase['name']
        
        self.phase_start_episode = episode
        self.phase_history.append({
            'episode': episode,
            'from_phase': old_phase,
            'to_phase': new_phase,
        })
        
        return old_phase, new_phase
    
    def get_status(self) -> dict:
        """Return current curriculum status for logging."""
        return {
            'phase': self.phase_name,
            'phase_idx': self.current_phase_idx,
            'active_agents': self.get_active_agents(),
            'frozen_agents': self.get_frozen_agents(),
            'episodes_in_phase': len(self.reward_history) - self.phase_start_episode,
            'is_final': self.is_final_phase(),
            'transitions': self.phase_history,
        }
 
 
# ─────────────────────────────────────────────────────────────
# 4. INTEGRATION: Apply curriculum to existing training loop
# ─────────────────────────────────────────────────────────────
 
def apply_curriculum_to_agents(agents: dict, scheduler: CurriculumScheduler) -> dict:
    """
    Wrap agents with FrozenAgentWrapper based on current curriculum phase.
    
    Call this at the start of each episode to update which agents are active.
    
    Args:
        agents: dict of agent_id -> BaseAgent instances
        scheduler: CurriculumScheduler instance
    
    Returns:
        dict of agent_id -> FrozenAgentWrapper instances
    """
    active_ids = set(scheduler.get_active_agents())
    wrapped = {}
    
    for agent_id, agent in agents.items():
        if isinstance(agent, FrozenAgentWrapper):
            # Already wrapped — just update frozen status
            if agent_id in active_ids:
                agent.unfreeze()
            else:
                agent.freeze()
            wrapped[agent_id] = agent
        else:
            # Wrap for the first time
            wrapper = FrozenAgentWrapper(agent)
            if agent_id in active_ids:
                wrapper.unfreeze()
            else:
                wrapper.freeze()
            wrapped[agent_id] = wrapper
    
    return wrapped
 
 
def run_curriculum_training(config: dict = None):
    """
    Drop-in replacement for run_training_loop() that adds curriculum scheduling.
    
    This wraps the existing training loop from training/loop.py.
    The only changes:
    1. Agents are wrapped with FrozenAgentWrapper
    2. CurriculumScheduler controls which agents are active per phase
    3. Phase promotions are logged
    
    Everything else (env, trust, negotiation, rewards, metrics) stays identical.
    """
    import sys
    import os
    import time
    from copy import deepcopy
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from env.crisis_env import CrisisEnv
    from training.loop import create_agents, check_metric_constraints, MAX_STEPS, NUM_EPISODES
    from core.trust import TrustSystem
    from core.negotiation import NegotiationSystem
    from core.aggregation import aggregate_actions
    from core.rewards import RewardSystem
    from core.active_rewards import ActiveRewardWrapper
    from metrics.tracker import MetricsTracker
    from logs.event_logger import EventLogger
    from logs.narrative import NarrativeSystem
    from memory.store import MemoryStore
    from agents.crisis_generator_agent import CrisisGeneratorAgent
    
    if config is None:
        config = {}
    
    mode = config.get('episode_mode', 'TRAINING')
    max_steps = MAX_STEPS.get(mode, 30)
    num_episodes = config.get('num_episodes', NUM_EPISODES)
    
    # Initialize systems (same as original)
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
    
    # Use ActiveRewardWrapper instead of raw RewardSystem
    base_reward_system = RewardSystem()
    reward_system = ActiveRewardWrapper(base_reward_system)
    
    crisis_generator = CrisisGeneratorAgent()
    
    # Create agents and wrap with curriculum
    base_agents = create_agents(memory_store)
    scheduler = CurriculumScheduler()
    AGENTS = apply_curriculum_to_agents(base_agents, scheduler)
    AGENT_IDS = list(AGENTS.keys())
    
    print("=" * 70)
    print("StateCraft — Curriculum Training")
    print(f"Mode: {mode} | Episodes: {num_episodes}")
    print(f"Phase 1: {scheduler.current_phase['description']}")
    print(f"Active: {scheduler.get_active_agents()}")
    print(f"Frozen: {scheduler.get_frozen_agents()}")
    print("=" * 70)
    
    metrics_history = []
    baseline_metrics = {}
    
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        event_logger.clear_turn_events()
        done = False
        episode_rewards = {agent_id: 0.0 for agent_id in AGENT_IDS}
        
        # Reset active reward tracking for new episode
        reward_system.reset_episode()
        
        # Apply curriculum: update which agents are active/frozen
        AGENTS = apply_curriculum_to_agents(AGENTS, scheduler)
        
        # Apply tier conditions
        crisis_generator.apply_tier_to_state(env.state_manager.state)
        trust_system._init_defaults()
        
        for step_num in range(max_steps):
            if done:
                break
            
            prev_state = deepcopy(env.state_manager.state)
            observations = obs
            
            # Negotiate (3 rounds) — same as original
            negotiation_system.reset_turn()
            all_messages = []
            for round_num in range(1, 4):
                messages = negotiation_system.negotiate_round(
                    AGENTS, observations, round_num
                )
                negotiation_system.update_from_messages(messages)
                all_messages.extend(messages)
            trust_system.resolve_trades(env.state_manager.state.get('turn', 0))
            
            # Act — frozen agents return FROZEN_POLICY, active agents use heuristics
            actions = {}
            for agent_id in AGENT_IDS:
                actions[agent_id] = AGENTS[agent_id].act(observations.get(agent_id, {}))
            
            actions = env.enforce_and_track_actions(actions)
            final_action = aggregate_actions(actions)
            
            # Step environment
            obs, rewards_raw, done, info = env.step(final_action, raw_agent_actions=actions)
            
            # Crisis events
            crisis_event = crisis_generator.generate_event(
                crisis_generator.current_tier,
                env.state_manager.state.get('turn', 0),
            )
            if crisis_event:
                env.state_manager.apply_deltas(crisis_event)
            
            # Sync trust
            env.state_manager.trust_matrix = trust_system.get_trust_matrix()
            env.state_manager.coalition_map = trust_system.get_coalition_map()
            
            # Compute rewards with active-action bonuses
            current_state = env.state_manager.state
            rewards = {}
            for agent_id in AGENT_IDS:
                rewards[agent_id] = reward_system.compute_and_clip_rewards(
                    state=current_state,
                    prev_state=prev_state,
                    agent_id=agent_id,
                    done=done,
                    agents=AGENTS,
                    actions_dict=actions,
                    final_action=final_action
                )
            rewards = {a: float(np.clip(rewards[a], -10, 10)) for a in AGENT_IDS}
            
            # Accumulate
            for a in AGENT_IDS:
                episode_rewards[a] += rewards[a]
            
            event_logger.clear_turn_events()
        
        # --- End of episode ---
        total_ep_reward = sum(episode_rewards.values())
        tracker.accumulate_reward(total_ep_reward)
        metrics = tracker.compute_episode_metrics(env)
        tracker.record_episode(metrics)
        metrics_history.append(metrics)
        
        # Record for curriculum
        scheduler.record_episode(episode, total_ep_reward, episode_rewards)
        
        # Check phase promotion
        if scheduler.should_promote(episode):
            old_phase, new_phase = scheduler.promote(episode)
            AGENTS = apply_curriculum_to_agents(AGENTS, scheduler)
            print(f"\n{'='*70}")
            print(f"CURRICULUM PROMOTION at episode {episode}!")
            print(f"  {old_phase} -> {new_phase}")
            print(f"  Now active: {scheduler.get_active_agents()}")
            print(f"  Still frozen: {scheduler.get_frozen_agents()}")
            print(f"  {scheduler.current_phase['description']}")
            print(f"{'='*70}\n")
        
        # Periodic logging
        if episode % 10 == 0:
            recent_rewards = [r['total_reward'] for r in scheduler.reward_history[-10:]]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            print(f"Episode {episode:4d} | Phase: {scheduler.phase_name:20s} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Active: {len(scheduler.get_active_agents())}/6")
        
        # Save memory
        for agent_id in AGENT_IDS:
            agent = AGENTS[agent_id]
            if hasattr(agent, 'save_memory'):
                agent.save_memory(memory_store, [{
                    'episode': episode,
                    'reward': episode_rewards[agent_id],
                    'phase': scheduler.phase_name,
                }])
    
    # Final summary
    print("\n" + "=" * 70)
    print("CURRICULUM TRAINING COMPLETE")
    print(f"Total episodes: {num_episodes}")
    print(f"Phase transitions: {len(scheduler.phase_history)}")
    for t in scheduler.phase_history:
        print(f"  Episode {t['episode']}: {t['from_phase']} -> {t['to_phase']}")
    print(f"Final phase: {scheduler.phase_name}")
    print("=" * 70)
    
    return scheduler, metrics_history
 
 
# ─────────────────────────────────────────────────────────────
# 5. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────
 
if __name__ == '__main__':
    run_curriculum_training({
        'episode_mode': 'TRAINING',
        'num_episodes': 500,
        'scenario': 'pandemic',
    })
