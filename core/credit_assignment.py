
"""
Fix 3: Counterfactual Credit Assignment
========================================
Prevents freeloading — agents can't ride others' success for free reward.
 
Problem: Standard multi-agent RL gives each agent the team's total reward.
If 5 agents do great work and 1 does nothing, the freeloader gets the same
reward as everyone else. Over time, more agents learn to freeload.
Eventually nobody does anything (connects to the boring equilibrium problem).
 
Solution: Counterfactual baseline.
 
For each agent, ask: "What would have happened if THIS agent had done nothing,
but everyone else did the same thing?"
 
agent_reward = actual_outcome - counterfactual_outcome
 
If the agent's actions HELPED → positive difference → positive reward
If the agent's actions HURT → negative difference → negative reward  
If the agent did nothing useful → zero difference → zero reward
 
This is the "difference reward" from multi-agent RL literature (Wolpert & Tumer 2002),
adapted for StateCraft's specific state dynamics.
 
Usage:
    from core.credit_assignment import CreditAssigner
    
    assigner = CreditAssigner(env)
    
    # After each step, compute counterfactual rewards:
    credit_rewards = assigner.compute_credits(
        state=current_state,
        prev_state=prev_state, 
        actions=actions_dict,
        final_action=final_action,
        base_rewards=rewards
    )
    
    # credit_rewards[agent_id] = how much THIS agent specifically contributed
"""
 
import numpy as np
from copy import deepcopy
 
 
# ─────────────────────────────────────────────────────────────
# 1. COUNTERFACTUAL ACTION GENERATOR
# ─────────────────────────────────────────────────────────────
# For each agent, generate the "what if they did nothing" scenario.
 
# The default "do nothing" action for counterfactual computation
DEFAULT_ACTION = {
    'lockdown_level': 'none',
    'emergency_budget': '0',
    'interest_rate': '0',
    'resource_priority': 'services',
    'foreign_policy': 'neutral',
    'crisis_response': 'monitor',
}
 
 
def build_counterfactual_actions(actions: dict, agent_id: str) -> dict:
    """
    Create a counterfactual action set where agent_id does nothing
    but all other agents act normally.
    
    Args:
        actions: dict of agent_id -> action_dict (what everyone actually did)
        agent_id: the agent to "remove" (replace with default action)
    
    Returns:
        dict of agent_id -> action_dict with agent_id replaced by DEFAULT_ACTION
    """
    counterfactual = {}
    for aid, action in actions.items():
        if aid == agent_id:
            counterfactual[aid] = deepcopy(DEFAULT_ACTION)
        else:
            counterfactual[aid] = deepcopy(action)
    return counterfactual
 
 
# ─────────────────────────────────────────────────────────────
# 2. STATE QUALITY SCORER
# ─────────────────────────────────────────────────────────────
# A single scalar that captures "how good is this state?"
# Used to compare actual outcome vs counterfactual outcome.
 
def compute_state_quality(state: dict) -> float:
    """
    Compute a single quality score for a world state.
    
    Higher = better. Captures the overall health of the simulated society.
    Weighted to match StateCraft's priorities:
    - GDP close to 1.0 is good
    - Low mortality is good
    - High stability is good
    - Inflation near 2% is good
    - High public trust is good
    
    Returns: float score (typically in range [-5, 10])
    """
    gdp = state.get('gdp', 1.0)
    mortality = state.get('mortality', 0.0)
    stability = state.get('stability', 0.75)
    inflation = state.get('inflation', 0.02)
    public_trust = state.get('public_trust', 0.5)
    
    score = (
        gdp * 2.0                          # GDP at 1.0 → +2.0
        - mortality * 10.0                   # Mortality is very bad
        + stability * 2.0                    # Stability is very good
        - abs(inflation - 0.02) * 3.0        # Inflation deviation is bad
        + public_trust * 1.0                 # Public trust is good
    )
    
    # Collapse penalty
    if stability < 0.2 or gdp < 0.3:
        score -= 20.0
    
    return score
 
 
def compute_state_delta(state: dict, prev_state: dict) -> float:
    """
    Compute the CHANGE in state quality from prev_state to state.
    Positive = things got better. Negative = things got worse.
    """
    return compute_state_quality(state) - compute_state_quality(prev_state)
 
 
# ─────────────────────────────────────────────────────────────
# 3. LIGHTWEIGHT COUNTERFACTUAL SIMULATOR
# ─────────────────────────────────────────────────────────────
# Instead of re-running the full environment (expensive), we estimate
# what would have happened using the action effect model from
# env/dynamics.py. This is an approximation but it's fast enough
# to run 6 times per turn (once per agent).
 
# These are the approximate per-action effects extracted from StateCraft's
# dynamics.py and step_logic.py. Not perfect, but good enough for
# credit assignment.
 
ACTION_EFFECTS = {
    'lockdown_level': {
        'none':      {'gdp': +0.005, 'mortality': +0.005, 'stability': +0.005},
        'advisory':  {'gdp': +0.002, 'mortality': +0.001, 'stability': +0.003},
        'partial':   {'gdp': -0.010, 'mortality': -0.008, 'stability': -0.005},
        'full':      {'gdp': -0.030, 'mortality': -0.020, 'stability': -0.015},
        'emergency': {'gdp': -0.050, 'mortality': -0.030, 'stability': -0.025},
    },
    'emergency_budget': {
        '0':  {'gdp': 0.000, 'stability': 0.000},
        '5':  {'gdp': +0.005, 'stability': +0.005},
        '15': {'gdp': +0.010, 'stability': +0.010},
        '30': {'gdp': +0.015, 'stability': +0.015},
        '50': {'gdp': +0.020, 'stability': +0.020},
    },
    'interest_rate': {
        '-0.5':  {'gdp': +0.010, 'inflation': +0.005},
        '-0.25': {'gdp': +0.005, 'inflation': +0.003},
        '0':     {'gdp': 0.000, 'inflation': 0.000},
        '+0.25': {'gdp': -0.003, 'inflation': -0.003},
        '+0.5':  {'gdp': -0.005, 'inflation': -0.005},
        '+1':    {'gdp': -0.010, 'inflation': -0.010},
        '+2':    {'gdp': -0.020, 'inflation': -0.020},
    },
    'crisis_response': {
        'monitor':   {'stability': -0.005, 'mortality': +0.003},
        'contain':   {'stability': +0.005, 'mortality': -0.005},
        'escalate':  {'stability': +0.010, 'mortality': -0.010},
        'emergency': {'stability': +0.020, 'mortality': -0.020},
    },
}
 
 
def estimate_counterfactual_state(prev_state: dict, actual_actions: dict,
                                   counterfactual_actions: dict) -> dict:
    """
    Estimate what the state WOULD have been if counterfactual_actions were
    taken instead of actual_actions.
    
    Method: 
    1. Start from actual current state
    2. Subtract the effect of the agent's actual action
    3. Add the effect of the counterfactual (default) action
    
    This is a linear approximation — good enough for credit assignment,
    not a full simulation.
    """
    # Find which agent changed (should be exactly one)
    changed_agent = None
    for aid in actual_actions:
        if actual_actions[aid] != counterfactual_actions[aid]:
            changed_agent = aid
            break
    
    if changed_agent is None:
        return deepcopy(prev_state)
    
    # Get actual and counterfactual actions for the changed agent
    actual_act = actual_actions[changed_agent]
    cf_act = counterfactual_actions[changed_agent]
    
    # Compute delta: what's the difference in effect?
    cf_state = deepcopy(prev_state)
    
    for domain, effects_map in ACTION_EFFECTS.items():
        actual_val = str(actual_act.get(domain, ''))
        cf_val = str(cf_act.get(domain, ''))
        
        actual_effects = effects_map.get(actual_val, {})
        cf_effects = effects_map.get(cf_val, {})
        
        # Apply the difference
        for metric, actual_effect in actual_effects.items():
            cf_effect = cf_effects.get(metric, 0.0)
            diff = cf_effect - actual_effect  # What changes if we swap actions
            if metric in cf_state:
                cf_state[metric] = cf_state[metric] + diff
    
    # Clip to valid ranges
    for key in ['gdp', 'stability', 'public_trust']:
        if key in cf_state:
            cf_state[key] = max(0.0, min(3.0 if key == 'gdp' else 1.0, cf_state[key]))
    for key in ['mortality', 'inflation']:
        if key in cf_state:
            cf_state[key] = max(0.0, min(1.0, cf_state[key]))
    
    return cf_state
 
 
# ─────────────────────────────────────────────────────────────
# 4. CREDIT ASSIGNER
# ─────────────────────────────────────────────────────────────
 
class CreditAssigner:
    """
    Computes counterfactual credit for each agent.
    
    For each agent:
        credit = actual_state_quality - counterfactual_state_quality
    
    Where counterfactual = "what if this agent did nothing?"
    
    Positive credit = this agent's actions made things BETTER
    Negative credit = this agent's actions made things WORSE
    Zero credit = this agent had no meaningful impact (freeloader)
    
    Usage:
        assigner = CreditAssigner()
        
        # Each turn:
        credits = assigner.compute_credits(
            state=current_state,
            prev_state=prev_state,
            actions=actions_dict,
        )
        # credits = {'agent_0': 0.5, 'agent_1': -0.2, 'agent_2': 0.0, ...}
        
        # Use credits to adjust rewards:
        for agent_id in agents:
            adjusted_reward = base_reward[agent_id] + credit_weight * credits[agent_id]
    """
    
    def __init__(self, credit_weight: float = 0.5):
        """
        Args:
            credit_weight: How much to weight counterfactual credit vs base reward.
                           0.0 = ignore credits (pure shared reward)
                           1.0 = full credit assignment
                           0.5 = balanced (recommended)
        """
        self.credit_weight = credit_weight
        self.credit_history = {}  # agent_id -> list of credits per turn
    
    def compute_credits(self, state: dict, prev_state: dict,
                        actions: dict) -> dict:
        """
        Compute counterfactual credit for each agent.
        
        Args:
            state: current state after all actions applied
            prev_state: state before actions were applied  
            actions: dict of agent_id -> action_dict
        
        Returns:
            dict of agent_id -> float credit score
        """
        actual_quality = compute_state_quality(state)
        credits = {}
        
        for agent_id in actions:
            # Build counterfactual: this agent does nothing, everyone else acts normally
            cf_actions = build_counterfactual_actions(actions, agent_id)
            
            # Estimate what state would have been
            cf_state = estimate_counterfactual_state(prev_state, actions, cf_actions)
            cf_quality = compute_state_quality(cf_state)
            
            # Credit = actual - counterfactual
            # Positive = this agent helped
            # Negative = this agent hurt
            credit = actual_quality - cf_quality
            credits[agent_id] = credit
            
            # Track history
            if agent_id not in self.credit_history:
                self.credit_history[agent_id] = []
            self.credit_history[agent_id].append(credit)
        
        return credits
    
    def adjust_rewards(self, base_rewards: dict, credits: dict) -> dict:
        """
        Adjust base rewards using counterfactual credits.
        
        Formula: adjusted = base + credit_weight * credit
        
        Args:
            base_rewards: dict of agent_id -> base reward (from RewardSystem)
            credits: dict of agent_id -> credit (from compute_credits)
        
        Returns:
            dict of agent_id -> adjusted reward
        """
        adjusted = {}
        for agent_id in base_rewards:
            base = base_rewards[agent_id]
            credit = credits.get(agent_id, 0.0)
            adjusted[agent_id] = base + self.credit_weight * credit
        return adjusted
    
    def get_freeloaders(self, threshold: float = 0.1, window: int = 10) -> list:
        """
        Identify agents whose recent credits are near zero (freeloading).
        
        Args:
            threshold: credits below this magnitude are "freeloading"
            window: how many recent turns to check
        
        Returns:
            list of agent_ids that are freeloading
        """
        freeloaders = []
        for agent_id, history in self.credit_history.items():
            if len(history) < window:
                continue
            recent = history[-window:]
            avg_credit = np.mean(np.abs(recent))
            if avg_credit < threshold:
                freeloaders.append(agent_id)
        return freeloaders
    
    def get_top_contributors(self, window: int = 10) -> list:
        """
        Rank agents by average positive contribution over recent window.
        
        Returns:
            list of (agent_id, avg_credit) sorted by contribution (highest first)
        """
        contributions = []
        for agent_id, history in self.credit_history.items():
            recent = history[-window:] if len(history) >= window else history
            avg_credit = np.mean(recent) if recent else 0.0
            contributions.append((agent_id, avg_credit))
        
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions
    
    def reset_episode(self):
        """Reset credit tracking for a new episode."""
        self.credit_history = {}
    
    def get_episode_summary(self) -> dict:
        """
        Get per-agent credit summary for the episode.
        
        Returns:
            dict of agent_id -> {total, mean, min, max, abs_mean}
        """
        summary = {}
        for agent_id, history in self.credit_history.items():
            if not history:
                summary[agent_id] = {
                    'total': 0.0, 'mean': 0.0,
                    'min': 0.0, 'max': 0.0, 'abs_mean': 0.0,
                }
                continue
            arr = np.array(history)
            summary[agent_id] = {
                'total': float(np.sum(arr)),
                'mean': float(np.mean(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'abs_mean': float(np.mean(np.abs(arr))),
            }
        return summary
 
 
# ─────────────────────────────────────────────────────────────
# 5. INTEGRATION HELPER
# ─────────────────────────────────────────────────────────────
# Shows exactly how to integrate CreditAssigner into the training loop.
 
def integrate_credit_assignment(training_loop_rewards: dict,
                                 assigner: CreditAssigner,
                                 state: dict, prev_state: dict,
                                 actions: dict,
                                 clip_min: float = -10.0,
                                 clip_max: float = 10.0) -> dict:
    """
    One-liner integration for the training loop.
    
    Call this right after computing rewards in the training loop:
    
        # In training/loop.py, after line 226:
        rewards = integrate_credit_assignment(
            rewards, assigner, current_state, prev_state, actions
        )
    
    Args:
        training_loop_rewards: dict of agent_id -> reward (from RewardSystem)
        assigner: CreditAssigner instance
        state: current state
        prev_state: previous state
        actions: dict of agent_id -> action_dict
    
    Returns:
        dict of agent_id -> adjusted reward (clipped)
    """
    credits = assigner.compute_credits(state, prev_state, actions)
    adjusted = assigner.adjust_rewards(training_loop_rewards, credits)
    
    # Clip to PPO range
    return {
        agent_id: float(np.clip(r, clip_min, clip_max))
        for agent_id, r in adjusted.items()
    }
