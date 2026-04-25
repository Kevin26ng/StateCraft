"""
CausalReasoningScore — causal/score.py (Task 11)

Unified causal understanding score per agent per episode.
Plot alongside reward during training.
If reward climbs but causal stays flat = reward hacking.
If both climb = genuine learning.
"""

import numpy as np
from causal.planner import CausalHorizonPlanner


class CausalReasoningScore:
    """
    Components (weighted):
      prediction_accuracy (0.35): Did causal claims come true?
      horizon_depth_score (0.25): How far ahead were you reasoning?
      generalization (0.25): Do strategies transfer across scenarios?
      consistency (0.15): Did stated reasoning match actions?
    """

    def __init__(self, planner: CausalHorizonPlanner):
        self.planner = planner
        self.episode_scores = []

    def compute_episode_score(self, agent_id, episode, episode_chains,
                              exploit_log=None, scenario_history=None):
        agent_chains = [c for c in episode_chains
                        if c.get("agent_id") == agent_id
                        and c.get("causal_accuracy") is not None]

        # 1. Prediction accuracy
        prediction_accuracy = (np.mean([c["causal_accuracy"] for c in agent_chains])
                               if agent_chains else 0.0)

        # 2. Horizon depth score
        if agent_chains:
            max_delay = max(self.planner.CAUSAL_DELAYS.values())
            depths = [self.planner.CAUSAL_DELAYS.get((c["trigger"], c["outcome"]), 1)
                      for c in agent_chains]
            horizon_score = min(1.0, np.mean(depths) / max_delay)
        else:
            horizon_score = 0.0

        # 3. Generalization
        if scenario_history and len(set(scenario_history)) > 1:
            generalization = min(1.0, len(set(scenario_history)) / 3.0)
        else:
            generalization = 0.3

        # 4. Consistency
        if exploit_log:
            mismatches = [e for e in exploit_log
                          if e.get("agent") == agent_id
                          and e.get("type") == "statement_action_mismatch"]
            consistency_score = max(0.0, 1.0 - len(mismatches) * 0.12)
        else:
            consistency_score = 1.0

        causal_score = (prediction_accuracy * 0.35 + horizon_score * 0.25 +
                        generalization * 0.25 + consistency_score * 0.15)

        self.episode_scores.append({
            "agent_id": agent_id, "episode": episode,
            "causal_score": float(causal_score),
            "breakdown": {
                "prediction_accuracy": float(prediction_accuracy),
                "horizon_depth_score": float(horizon_score),
                "generalization": float(generalization),
                "consistency": float(consistency_score),
            }
        })
        return causal_score

    def get_mean_causal_score(self, episode):
        ep_scores = [s["causal_score"] for s in self.episode_scores
                     if s["episode"] == episode]
        return float(np.mean(ep_scores)) if ep_scores else 0.0

    def get_training_curve_data(self):
        episodes = sorted(set(s["episode"] for s in self.episode_scores))
        return {"episodes": episodes,
                "causal_scores": [self.get_mean_causal_score(ep) for ep in episodes]}
