"""
RewardHackingDefender — defense/reward_defender.py (Task 13)

Three independent verification layers that make it computationally
hard to hack reward. An agent must fool ALL THREE simultaneously.
"""

import numpy as np
from collections import Counter


class RewardHackingDefender:
    def __init__(self, rewards_config=None):
        self.exploit_log = []
        self.rewards_config = rewards_config or {}

    # ── Layer 1: Behavioral Consistency ────────────────────────────────────
    def check_action_statement_consistency(self, agent_id, message, action, turn):
        if not message or not action:
            return 1.0
        content = message.get("content", "").lower() if isinstance(message, dict) else str(message).lower()

        expected_lock = self._infer_lockdown_from_text(content)
        lockdown_scores = {'none':0,'advisory':1,'partial':2,'full':3,'emergency':4}
        actual_lock = lockdown_scores.get(action.get('lockdown_level','none'), 0)

        expected_budget = self._infer_budget_from_text(content)
        budget_str = action.get('emergency_budget', '0')
        actual_budget = int(budget_str) if str(budget_str).isdigit() else 0

        consistency = 1.0
        if expected_lock is not None:
            consistency = min(consistency, 1.0 - abs(expected_lock - actual_lock) / 4.0)
        if expected_budget is not None:
            consistency = min(consistency, 1.0 - abs(expected_budget - actual_budget) / 50.0)

        if consistency < 0.4:
            self.exploit_log.append({
                "turn": turn, "agent": agent_id,
                "type": "statement_action_mismatch", "severity": "medium",
                "detail": f"Said '{content[:40]}' but lockdown={actual_lock}, budget={actual_budget}"
            })
        return consistency

    def _infer_lockdown_from_text(self, text):
        if any(w in text for w in ["full lockdown", "close", "shut down", "emergency"]):
            return 3
        if any(w in text for w in ["partial", "restrict", "limit"]):
            return 2
        if any(w in text for w in ["open", "no lockdown", "keep business"]):
            return 0
        return None

    def _infer_budget_from_text(self, text):
        if any(w in text for w in ["major stimulus", "large budget", "emergency spending"]):
            return 40
        if any(w in text for w in ["cut spending", "no budget", "austerity"]):
            return 0
        return None

    # ── Layer 2: Causal Claim Verification ─────────────────────────────────
    def verify_causal_claims(self, resolved_chains, current_turn):
        penalties = {}
        for chain in resolved_chains:
            if chain.get("causal_accuracy", 1.0) < 0.25:
                agent_id = chain["agent_id"]
                penalties[agent_id] = penalties.get(agent_id, 0.0) - 0.8
                self.exploit_log.append({
                    "turn": current_turn, "agent": agent_id,
                    "type": "false_causal_claim", "severity": "high",
                    "detail": (f"Chain {chain['trigger']}→{chain['outcome']}: "
                               f"predicted {chain['predicted_delta']:.3f}, "
                               f"got {chain['actual_delta']:.3f}")
                })
        return penalties

    # ── Layer 3: Independent Reward Recomputation ──────────────────────────
    def verify_reward_computation(self, agent_id, claimed_reward, state, prev_state, turn):
        verified = self._recompute_independently(agent_id, state, prev_state)
        discrepancy = abs(verified - claimed_reward)
        if discrepancy > 1.5:
            self.exploit_log.append({
                "turn": turn, "agent": agent_id,
                "type": "reward_inflation", "severity": "critical",
                "detail": f"Claimed {claimed_reward:.3f}, verified {verified:.3f}"
            })
            return verified
        return claimed_reward

    def _recompute_independently(self, agent_id, state, prev_state):
        ROLE_REWARDS = {
            "agent_0": lambda s, p: s.get("gdp", 1.0) * 2.0,
            "agent_3": lambda s, p: -(s.get("mortality", 0) - p.get("mortality", 0)) * 3.0,
            "agent_4": lambda s, p: s.get("stability", 1.0) * 1.8,
            "agent_2": lambda s, p: -abs(s.get("inflation", 0.02) - 0.02) * 2.0,
            "agent_1": lambda s, p: s.get("public_trust", 0.5) * 2.0,
            "agent_5": lambda s, p: 0.0,
        }
        fn = ROLE_REWARDS.get(agent_id, lambda s, p: 0.0)
        return float(np.clip(fn(state, prev_state), -10.0, 10.0))

    # ── Reporting ──────────────────────────────────────────────────────────
    def get_exploit_report(self):
        if not self.exploit_log:
            return {"summary": "No reward hacking detected.", "total": 0, "critical": 0, "high": 0}
        critical = [e for e in self.exploit_log if e["severity"] == "critical"]
        high = [e for e in self.exploit_log if e["severity"] == "high"]
        offender = Counter(e["agent"] for e in self.exploit_log).most_common(1)
        return {
            "total": len(self.exploit_log), "critical": len(critical), "high": len(high),
            "most_active_agent": offender[0][0] if offender else None,
            "most_recent": self.exploit_log[-1],
            "summary": (f"{len(self.exploit_log)} exploit attempts. "
                        f"{len(critical)} critical. {len(high)} false causal claims.")
        }

    def get_dashboard_data(self):
        report = self.get_exploit_report()
        return {"exploit_count": report["total"], "critical_count": report.get("critical", 0),
                "summary": report["summary"], "recent_log": self.exploit_log[-10:]}
