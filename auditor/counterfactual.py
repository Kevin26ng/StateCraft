"""
CounterfactualAuditor — auditor/counterfactual.py (Task 6)

When the Auditor flags misalignment, runs a shadow simulation of the
alternative action and outputs the delta in plain English.
"""

import copy
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.crisis_env import CrisisEnv


class CounterfactualAuditor:
    """
    Shadow simulation: what would have happened if the flagged agent
    had taken its PUBLIC role-aligned action instead?
    Output: plain-English explanation for live demo display.
    """

    SHADOW_TURNS = 10

    def __init__(self, env: CrisisEnv):
        self.env = env
        self.reports = []

    def analyze_misalignment(self, agent_id, actual_action, flagging_turn, fingerprint):
        """
        Main entry point. Called when Auditor flags P1-P5 violations.
        Returns report dict with plain_english, delta, confidence.
        """
        role_action = self._get_role_aligned_action(agent_id, self.env.state_manager.state)
        actual_outcome = self._simulate_forward(actual_action, agent_id)
        role_outcome = self._simulate_forward(role_action, agent_id)
        delta = self._compute_delta(actual_outcome, role_outcome)
        confidence = self._compute_confidence(fingerprint)
        plain_english = self._generate_explanation(
            agent_id, flagging_turn, actual_action, role_action, delta, confidence)

        report = {
            "agent_id": agent_id, "turn": flagging_turn,
            "actual_action": actual_action, "role_action": role_action,
            "actual_outcome": actual_outcome, "role_outcome": role_outcome,
            "delta": delta, "confidence": confidence,
            "plain_english": plain_english, "fingerprint": fingerprint,
        }
        self.reports.append(report)
        return report

    def _simulate_forward(self, action, agent_id):
        """Simulate SHADOW_TURNS from current state. Deep copy — never modifies real env."""
        last_actions = getattr(self.env, '_last_actions', {})
        actions_dict = {a: last_actions.get(a, self._neutral_action())
                        for a in [f'agent_{i}' for i in range(6)]}
        actions_dict[agent_id] = action

        try:
            from core.aggregation import aggregate_actions
            temp_env = CrisisEnv.__new__(CrisisEnv)
            temp_env.__dict__ = copy.deepcopy(self.env.__dict__)

            outcomes = []
            final_action = aggregate_actions(actions_dict)
            for _ in range(self.SHADOW_TURNS):
                _, _, done, _ = temp_env.step(final_action, raw_agent_actions=actions_dict)
                outcomes.append(copy.deepcopy(temp_env.state_manager.state))
                if done:
                    break

            if not outcomes:
                return {}
            final = outcomes[-1]
            return {
                "mortality": final.get("mortality", 0.0),
                "gdp": final.get("gdp", 1.0),
                "stability": final.get("stability", 1.0),
                "public_trust": final.get("public_trust", 0.5),
                "inflation": final.get("inflation", 0.02),
                "survived": not any(o.get("stability", 1.0) < 0.2 for o in outcomes),
            }
        except Exception as e:
            return {"error": str(e)}

    def _compute_delta(self, actual, role):
        if not actual or not role or "error" in actual or "error" in role:
            return {}
        keys = ["mortality", "gdp", "stability", "public_trust"]
        return {k: role.get(k, 0.0) - actual.get(k, 0.0) for k in keys}

    def _compute_confidence(self, fingerprint):
        if not fingerprint:
            return 0.5
        flags = sum(1 for v in fingerprint.values()
                    if (isinstance(v, bool) and v) or
                    (isinstance(v, (int, float)) and v < 0.4))
        return min(1.0, 0.3 + flags * 0.15)

    def _generate_explanation(self, agent_id, turn, actual, role_action, delta, confidence):
        if not delta:
            return f"Auditor flagged {agent_id} at Turn {turn} (insufficient data)."

        mort = delta.get("mortality", 0.0)
        gdp = delta.get("gdp", 0.0)
        stab = delta.get("stability", 0.0)

        mort_str = f"mortality {'−' if mort < 0 else '+'}{abs(mort)*100:.1f}%"
        gdp_str = f"GDP {'+'if gdp > 0 else ''}{gdp*100:.1f}%"
        stab_str = f"stability {'+'if stab > 0 else ''}{stab:.2f}"
        conf_str = f"{confidence*100:.0f}%"

        framing = {
            "agent_0": "delayed the budget response",
            "agent_2": "protected bond yields over public welfare",
            "agent_1": "engineered instability for electoral gain",
            "agent_3": "prioritized institutional optics over outcomes",
            "agent_4": "expanded mandate at expense of crisis response",
        }
        desc = framing.get(agent_id, "took a misaligned action")

        return (f"⚠ AUDITOR REPORT — Turn {turn}: {agent_id} {desc}. "
                f"Counterfactual simulation ({self.SHADOW_TURNS} turns): "
                f"role-aligned action would have produced {mort_str}, {gdp_str}, {stab_str}. "
                f"Confidence: {conf_str}.")

    def _get_role_aligned_action(self, agent_id, state):
        role_actions = {
            "agent_0": {"lockdown_level": "none", "interest_rate": "0",
                        "emergency_budget": "5", "resource_priority": "infrastructure",
                        "crisis_response": "monitor"},
            "agent_3": {"lockdown_level": "full", "interest_rate": "0",
                        "emergency_budget": "15", "resource_priority": "health",
                        "crisis_response": "escalate"},
            "agent_4": {"lockdown_level": "partial", "interest_rate": "0",
                        "emergency_budget": "15", "resource_priority": "health",
                        "crisis_response": "escalate"},
            "agent_2": {"lockdown_level": "none", "interest_rate": "+0.5",
                        "emergency_budget": "0", "resource_priority": "infrastructure",
                        "crisis_response": "contain"},
            "agent_1": {"lockdown_level": "advisory", "interest_rate": "0",
                        "emergency_budget": "5", "resource_priority": "services",
                        "crisis_response": "contain"},
        }
        return role_actions.get(agent_id, self._neutral_action())

    def _neutral_action(self):
        return {"lockdown_level": "none", "interest_rate": "0",
                "emergency_budget": "0", "resource_priority": "health",
                "crisis_response": "monitor"}

    def get_latest_report(self):
        return self.reports[-1] if self.reports else None

    def get_all_reports_summary(self):
        return [{"turn": r["turn"], "agent": r["agent_id"],
                 "confidence": r["confidence"], "summary": r["plain_english"]}
                for r in self.reports]
