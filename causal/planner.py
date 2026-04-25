"""
CausalHorizonPlanner — causal/planner.py (Task 10)

Tracks causal chains started by agent actions.
Agents see their pending consequences in their observation.
Key mechanism for long-horizon planning.
"""

import copy


class CausalHorizonPlanner:
    CAUSAL_DELAYS = {
        ("lockdown", "mortality_reduction"): 14,
        ("rate_hike", "inflation_reduction"): 12,
        ("stimulus", "gdp_growth"): 8,
        ("coalition_break", "stability_drop"): 3,
        ("rate_hike", "unemployment_rise"): 10,
        ("disaster_relief", "stability_gain"): 5,
        ("emergency_budget", "hospital_relief"): 4,
        ("full_lockdown", "compliance_drop"): 6,
    }

    PREDICTION_TABLE = {
        "mortality_reduction": lambda mag: -0.08 * mag,
        "inflation_reduction": lambda mag: -0.05 * mag,
        "gdp_growth": lambda mag: +0.04 * mag,
        "stability_drop": lambda mag: -0.15 * mag,
        "unemployment_rise": lambda mag: +0.06 * mag,
        "stability_gain": lambda mag: +0.10 * mag,
        "hospital_relief": lambda mag: +0.12 * mag,
        "compliance_drop": lambda mag: -0.08 * mag,
    }

    def __init__(self):
        self.pending_chains = []
        self.resolved_chains = []

    def register_action(self, turn, agent_id, action_dict):
        """Register all causal chains this action set in motion."""
        if not isinstance(action_dict, dict):
            return []
        new_chains = []
        for (trigger, outcome), delay in self.CAUSAL_DELAYS.items():
            if self._action_matches_trigger(action_dict, trigger):
                magnitude = self._estimate_magnitude(action_dict, trigger)
                chain = {
                    "id": f"{agent_id}_{turn}_{trigger}",
                    "agent_id": agent_id, "trigger": trigger,
                    "outcome": outcome, "started_turn": turn,
                    "resolves_turn": turn + delay,
                    "magnitude": magnitude,
                    "predicted_delta": self.PREDICTION_TABLE[outcome](magnitude),
                    "actual_delta": None, "causal_accuracy": None,
                }
                self.pending_chains.append(chain)
                new_chains.append(chain)
        return new_chains

    def resolve_chains(self, current_turn, world_state_delta):
        """Resolve chains whose delay has elapsed. Compute causal accuracy."""
        just_resolved, still_pending = [], []
        for chain in self.pending_chains:
            if current_turn >= chain["resolves_turn"]:
                actual = world_state_delta.get(chain["outcome"], 0.0)
                chain["actual_delta"] = actual
                chain["causal_accuracy"] = self._accuracy(chain["predicted_delta"], actual)
                self.resolved_chains.append(chain)
                just_resolved.append(chain)
            else:
                still_pending.append(chain)
        self.pending_chains = still_pending
        return just_resolved

    def get_agent_horizon_view(self, agent_id, current_turn):
        my_chains = [c for c in self.pending_chains if c["agent_id"] == agent_id]
        if not my_chains:
            return {"chains_in_flight": 0, "pending_consequences": [], "longest_horizon": 0}
        return {
            "chains_in_flight": len(my_chains),
            "pending_consequences": [
                {"outcome": c["outcome"],
                 "expected_turn": c["resolves_turn"],
                 "turns_remaining": c["resolves_turn"] - current_turn,
                 "predicted_magnitude": c["predicted_delta"],
                 "trigger": c["trigger"]}
                for c in sorted(my_chains, key=lambda c: c["resolves_turn"])
            ],
            "longest_horizon": max(c["resolves_turn"] - current_turn for c in my_chains),
        }

    def get_horizon_observation_vector(self, agent_id, current_turn, vector_dim=8):
        """Compact fixed-length vector for PPO observation injection."""
        view = self.get_agent_horizon_view(agent_id, current_turn)
        chains = view["pending_consequences"]
        vec = [0.0] * vector_dim
        for i, c in enumerate(chains[:4]):
            if i * 2 + 1 < vector_dim:
                vec[i * 2] = c["turns_remaining"] / 14.0
                vec[i * 2 + 1] = c["predicted_magnitude"]
        return vec

    def _action_matches_trigger(self, action, trigger):
        lockdown_scores = {'none':0,'advisory':1,'partial':2,'full':3,'emergency':4}
        lock_val = lockdown_scores.get(action.get("lockdown_level", "none"), 0)
        ir = action.get("interest_rate", "0")
        ir_val = float(str(ir).replace('+','')) if ir else 0.0
        budget = action.get("emergency_budget", "0")
        budget_val = int(budget) if str(budget).isdigit() else 0
        crisis = action.get("crisis_response", "monitor")

        mappings = {
            "lockdown": lock_val >= 2,
            "rate_hike": ir_val > 0.0,
            "stimulus": budget_val >= 15,
            "coalition_break": False,
            "disaster_relief": crisis in ("escalate", "emergency"),
            "emergency_budget": budget_val >= 30,
            "full_lockdown": lock_val >= 4,
        }
        return mappings.get(trigger, False)

    def _estimate_magnitude(self, action, trigger):
        lockdown_scores = {'none':0,'advisory':1,'partial':2,'full':3,'emergency':4}
        lock_val = lockdown_scores.get(action.get("lockdown_level", "none"), 0)
        ir = action.get("interest_rate", "0")
        ir_val = abs(float(str(ir).replace('+',''))) if ir else 0.0
        budget = action.get("emergency_budget", "0")
        budget_val = int(budget) if str(budget).isdigit() else 0

        if trigger == "lockdown": return lock_val / 4.0
        if trigger == "rate_hike": return ir_val / 2.0
        if trigger in ("stimulus", "emergency_budget"): return budget_val / 50.0
        if trigger == "disaster_relief":
            return 0.7 if action.get("crisis_response") == "emergency" else 0.4
        if trigger == "full_lockdown": return lock_val / 4.0
        return 0.5

    def _accuracy(self, predicted, actual):
        if predicted == 0:
            return 1.0 if abs(actual) < 0.01 else 0.0
        error = abs(predicted - actual) / (abs(predicted) + 1e-8)
        return float(max(0.0, 1.0 - error))

    def reset(self):
        """Call at start of each episode. Keeps resolved_chains for scoring."""
        self.pending_chains = []
