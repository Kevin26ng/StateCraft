
"""
Fix 1: Per-Agent Role Rewards + Active-Action Bonuses
=====================================================
Prevents boring equilibrium where all agents converge to "do nothing."
 
Problem: Agents discover that voting 'none' on everything avoids policy cost,
triggers agreement bonus (+1.5), and still earns passive rewards (survival +0.5,
coalition stability +0.8, GDP baseline +2.0). Optimal strategy = universal inaction.
 
Solution: Three mechanisms that make DOING SOMETHING always better than doing nothing.
 
Drop-in replacement for the reward computation. Integrates with existing
RewardSystem by wrapping compute_and_clip_rewards().
 
Usage:
    from core.active_rewards import ActiveRewardWrapper
    reward_system = ActiveRewardWrapper(base_reward_system)
    # Then use reward_system.compute_and_clip_rewards() as before
"""
 
import numpy as np
from collections import Counter
 
 
# ─────────────────────────────────────────────────────────────
# 1. INACTION DETECTOR
# ─────────────────────────────────────────────────────────────
# Maps each action domain to its "do nothing" value
INACTION_VALUES = {
    'lockdown_level': 'none',
    'emergency_budget': '0',
    'crisis_response': 'monitor',
    'interest_rate': '0',
    'foreign_policy': 'neutral',
    'resource_priority': None,  # No default "do nothing" — any choice is active
}
 
def compute_action_activity_score(action: dict) -> float:
    """
    Score how "active" an agent's action is. 0.0 = total inaction, 1.0 = max activity.
    
    Counts how many action domains differ from the "do nothing" default.
    This is NOT about whether the action is GOOD — just whether the agent
    is actually DOING something vs. defaulting to inaction on everything.
    """
    if not action:
        return 0.0
    
    active_count = 0
    total_domains = 0
    
    for domain, inaction_val in INACTION_VALUES.items():
        if inaction_val is None:
            continue  # Skip domains without a clear inaction default
        total_domains += 1
        agent_choice = action.get(domain)
        if agent_choice is not None and str(agent_choice) != str(inaction_val):
            active_count += 1
    
    if total_domains == 0:
        return 0.0
    return active_count / total_domains
 
 
def is_inaction(action: dict) -> bool:
    """True if the agent is doing literally nothing across all domains."""
    return compute_action_activity_score(action) == 0.0
 
 
# ─────────────────────────────────────────────────────────────
# 2. PER-ROLE ACTIVE-ACTION BONUSES
# ─────────────────────────────────────────────────────────────
# Each role has specific actions that COUNT as "doing their job."
# If they're doing their job, they get a small bonus.
# If they're NOT doing their job during a crisis, they get penalized.
 
def compute_role_activity_bonus(agent_id: str, action: dict, state: dict) -> float:
    """
    Role-specific bonus for agents who take actions aligned with their role
    during situations that DEMAND action.
    
    Returns: float bonus/penalty in [-1.0, +1.0]
    
    The key insight: this doesn't reward GOOD actions, it rewards RELEVANT actions.
    A Finance Minister who sets interest rates during an inflation crisis is being
    relevant — even if the rate choice is wrong. A Finance Minister who votes 'none'
    on everything during hyperinflation is negligent.
    """
    from core.rewards import AGENT_ID_TO_ROLE
    
    role = AGENT_ID_TO_ROLE.get(agent_id, 'auditor')
    bonus = 0.0
    
    gdp = state.get('gdp', 1.0)
    mortality = state.get('mortality', 0.0)
    stability = state.get('stability', 0.75)
    inflation = state.get('inflation', 0.02)
    
    if role == 'finance':
        # Finance MUST act when GDP is dropping or inflation is high
        crisis_intensity = max(0, 1.0 - gdp) + max(0, inflation - 0.04)
        if crisis_intensity > 0.2:
            # There's an economic problem — did Finance do anything?
            if action.get('emergency_budget', '0') != '0':
                bonus += 0.5  # Spending during crisis = good
            if action.get('interest_rate', '0') != '0':
                bonus += 0.3  # Adjusting rates = good
            if action.get('emergency_budget', '0') == '0' and action.get('interest_rate', '0') == '0':
                bonus -= 0.5  # Doing nothing during economic crisis = bad
    
    elif role == 'health':
        # Health MUST act when mortality is rising
        if mortality > 0.03:
            if action.get('lockdown_level', 'none') not in ['none', 'advisory']:
                bonus += 0.5  # Imposing lockdown during health crisis = good
            if action.get('resource_priority') == 'health':
                bonus += 0.3  # Prioritizing health resources = good
            if action.get('lockdown_level', 'none') == 'none':
                bonus -= 0.5  # No lockdown during mortality spike = bad
    
    elif role == 'military':
        # Military MUST act when stability is low
        if stability < 0.5:
            if action.get('crisis_response', 'monitor') not in ['monitor']:
                bonus += 0.5  # Active crisis response = good
            if action.get('crisis_response', 'monitor') == 'monitor':
                bonus -= 0.4  # Just monitoring during instability = bad
    
    elif role == 'central_bank':
        # Central bank MUST act when inflation is out of control
        if abs(inflation - 0.02) > 0.03:
            if action.get('interest_rate', '0') != '0':
                bonus += 0.5  # Adjusting rates during inflation = good
            else:
                bonus -= 0.4  # Holding rates during inflation crisis = bad
    
    elif role == 'political':
        # Political agent should be actively engaging/negotiating
        if stability < 0.5 or gdp < 0.6:
            if action.get('foreign_policy', 'neutral') != 'neutral':
                bonus += 0.3  # Taking a diplomatic stance = good
            if action.get('lockdown_level', 'none') != 'none':
                bonus += 0.2  # Engaging on lockdown policy = good
    
    elif role == 'auditor':
        # Auditor has no direct policy actions — skip activity bonus
        pass
    
    return float(np.clip(bonus, -1.0, 1.0))
 
 
# ─────────────────────────────────────────────────────────────
# 3. AGREEMENT BONUS FIX — Must agree on DOING something
# ─────────────────────────────────────────────────────────────
 
def compute_active_agreement_bonus(actions_dict: dict) -> float:
    """
    Replaces the original compute_agreement_bonus from core/rewards.py.
    
    Original problem: All agents voting 'none' has variance=0, earning +1.5 free.
    Fix: Agreement bonus ONLY triggers if agents agree on an ACTIVE action.
    
    Still rewards consensus — but consensus on "do nothing" gets you nothing.
    """
    if not actions_dict:
        return 0.0
    
    lockdown_scores = {'none': 0, 'advisory': 1, 'partial': 2, 'full': 3, 'emergency': 4}
    lockdown_votes = [
        lockdown_scores.get(a.get('lockdown_level', 'none'), 0)
        for a in actions_dict.values()
    ]
    if not lockdown_votes:
        return 0.0
    
    action_variance = np.var(lockdown_votes)
    
    # Find most common vote
    vote_counts = Counter(lockdown_votes)
    most_common_vote = vote_counts.most_common(1)[0][0]
    
    # Agreement exists AND it's not "do nothing"
    if action_variance < 0.5 and most_common_vote > 0:
        return 1.5
    
    return 0.0
 
 
# ─────────────────────────────────────────────────────────────
# 4. DIVERSITY BONUS — Prevent all agents converging to same action
# ─────────────────────────────────────────────────────────────
 
def compute_action_diversity_bonus(actions_dict: dict) -> dict:
    """
    Small bonus when agents take DIFFERENT actions from each other.
    Prevents degenerate equilibrium where everyone copies the same strategy.
    
    Returns: dict of agent_id -> bonus
    """
    if not actions_dict or len(actions_dict) < 2:
        return {a: 0.0 for a in actions_dict} if actions_dict else {}
    
    # Count unique lockdown votes
    lockdown_votes = [
        a.get('lockdown_level', 'none') for a in actions_dict.values()
    ]
    unique_votes = len(set(lockdown_votes))
    total_agents = len(lockdown_votes)
    
    # Diversity ratio: 1.0 if everyone is different, 0.0 if all the same
    diversity = (unique_votes - 1) / max(total_agents - 1, 1)
    
    # Small bonus for diversity (0 to 0.3)
    bonus_per_agent = diversity * 0.3
    
    return {agent_id: bonus_per_agent for agent_id in actions_dict}
 
 
# ─────────────────────────────────────────────────────────────
# 5. WRAPPER — Drop-in replacement for RewardSystem
# ─────────────────────────────────────────────────────────────
 
class ActiveRewardWrapper:
    """
    Wraps the existing RewardSystem and adds:
    1. Active-action bonuses per role
    2. Inaction penalty
    3. Fixed agreement bonus (must agree on DOING something)
    4. Action diversity bonus
    
    Usage:
        from core.rewards import RewardSystem
        from core.active_rewards import ActiveRewardWrapper
        
        base = RewardSystem()
        reward_system = ActiveRewardWrapper(base)
        
        # Use exactly like RewardSystem:
        reward = reward_system.compute_and_clip_rewards(
            state=state, prev_state=prev_state, agent_id='agent_0',
            done=done, agents=agents, actions_dict=actions, final_action=final_action
        )
    """
    
    def __init__(self, base_reward_system):
        self.base = base_reward_system
        # Track per-agent inaction streaks
        self.inaction_streaks = {}  # agent_id -> consecutive inaction turns
    
    def compute_and_clip_rewards(self, state: dict, prev_state: dict,
                                  agent_id: str, done: bool,
                                  agents: dict = None,
                                  actions_dict: dict = None,
                                  final_action: dict = None) -> float:
        """
        Compute reward with active-action bonuses layered on top of base rewards.
        """
        # 1. Get base reward from existing system
        base_reward = self.base.compute_and_clip_rewards(
            state=state, prev_state=prev_state, agent_id=agent_id,
            done=done, agents=agents, actions_dict=actions_dict,
            final_action=final_action
        )
        
        # If episode ended in collapse, don't add bonuses
        if done and (state.get('stability', 1) < 0.2 or state.get('gdp', 1) < 0.3):
            return base_reward
        
        total_bonus = 0.0
        
        # 2. Per-agent action (get this agent's individual action)
        agent_action = actions_dict.get(agent_id, {}) if actions_dict else {}
        
        # 3. Role-specific activity bonus
        role_bonus = compute_role_activity_bonus(agent_id, agent_action, state)
        total_bonus += role_bonus
        
        # 4. Inaction streak penalty (escalating)
        if is_inaction(agent_action):
            self.inaction_streaks[agent_id] = self.inaction_streaks.get(agent_id, 0) + 1
            streak = self.inaction_streaks[agent_id]
            # Penalty grows: -0.2, -0.4, -0.6, -0.8, -1.0 (capped)
            inaction_penalty = -min(streak * 0.2, 1.0)
            total_bonus += inaction_penalty
        else:
            self.inaction_streaks[agent_id] = 0
            # Small reward for being active at all
            activity = compute_action_activity_score(agent_action)
            total_bonus += activity * 0.2  # Up to +0.2 for full activity
        
        # 5. Diversity bonus
        if actions_dict:
            diversity_bonuses = compute_action_diversity_bonus(actions_dict)
            total_bonus += diversity_bonuses.get(agent_id, 0.0)
        
        # 6. Replace agreement bonus with active version
        # (Subtract original agreement bonus, add fixed version)
        if actions_dict:
            from core.rewards import compute_agreement_bonus
            original_agreement = compute_agreement_bonus(actions_dict)
            fixed_agreement = compute_active_agreement_bonus(actions_dict)
            # The original agreement bonus was added in compute_global_reward * 0.5
            # We correct the difference
            agreement_correction = (fixed_agreement - original_agreement) * 0.5
            total_bonus += agreement_correction
        
        # Combine and clip
        final_reward = base_reward + total_bonus
        return float(np.clip(final_reward, self.base.clip_min, self.base.clip_max))
    
    def reset_episode(self):
        """Call at the start of each episode to reset tracking."""
        self.inaction_streaks = {}
    
    # Forward all other attributes to the base system
    def __getattr__(self, name):
        return getattr(self.base, name)
