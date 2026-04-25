"""
EmergenceDetector — emergence/detector.py (Task 5)

PASSIVE observer — instruments every training turn.
Detects 4 types of emergent behavior.
CRITICAL: This is read-only. It observes. It NEVER modifies state, rewards, or agent behavior.
"""

import time
import json
import copy
from collections import defaultdict, Counter


class EmergenceDetector:
    """
    Passive observer — instruments every training turn.
    Detects 4 types of emergent behavior ranked by probability:

    1. Bilateral coalition (most likely)
    2. Manufactured crisis (Political agent destabilization)
    3. Sacrifice play (agent accepts personal loss for collective gain)
    4. Scapegoating (blame-shifting coalition in negotiation text)
    """

    COALITION_NAMES = {
        ("agent_0", "agent_2"): "THE_BOND_MARKET_PACT",
        ("agent_0", "agent_4"): "THE_BUDGET_WAR_ALLIANCE",
        ("agent_1", "agent_2"): "THE_ELECTION_BANKING_CARTEL",
        ("agent_3", "agent_5"): "THE_TRANSPARENCY_BLOC",
    }

    # Also support descriptive name lookups
    _DESCRIPTIVE_NAMES = {
        ("finance_minister", "monetary_authority"): "THE_BOND_MARKET_PACT",
        ("finance_minister", "disaster_response_agent"): "THE_BUDGET_WAR_ALLIANCE",
        ("monetary_authority", "political_pressure_agent"): "THE_ELECTION_BANKING_CARTEL",
        ("public_health_authority", "auditor"): "THE_TRANSPARENCY_BLOC",
    }

    def __init__(self):
        self.episode_logs = []
        self.named_events = {}
        self.coalition_history = []
        self._action_alignment = defaultdict(list)

    def log_turn(self, episode, turn, agent_actions, messages, world_state):
        """
        Call this every turn during training. PASSIVE — no state modification.
        agent_actions: {agent_id: action_dict}
        messages: {agent_id: message_dict} or list
        world_state: full state dict (read-only copy taken internally)
        """
        entry = {
            "episode": episode, "turn": turn,
            "actions": agent_actions,
            "messages": messages if isinstance(messages, dict) else {},
            "state": copy.deepcopy(world_state),
            "ts": time.time(),
        }
        self.episode_logs.append(entry)
        self.coalition_history.append(world_state.get("coalition_map", {}))

        self._detect_bilateral_coalitions(entry)
        self._detect_manufactured_crisis(entry)
        self._detect_sacrifice_play(entry)
        if messages:
            self._detect_scapegoating(entry)

    # ── Detector 1: Spontaneous Bilateral Coalitions ───────────────────────
    def _detect_bilateral_coalitions(self, entry):
        if len(self.episode_logs) < 5:
            return
        recent = self.episode_logs[-5:]
        agents = list(entry["actions"].keys())
        for i, a1 in enumerate(agents):
            for a2 in agents[i+1:]:
                aligned = sum(
                    1 for e in recent
                    if self._actions_are_complementary(
                        e["actions"].get(a1, {}), e["actions"].get(a2, {}))
                )
                if aligned >= 4:
                    cm = entry["state"].get("coalition_map", {})
                    if cm.get(a1) != cm.get(a2):
                        pair = tuple(sorted([a1, a2]))
                        name = self.COALITION_NAMES.get(
                            pair, f"THE_{a1.upper()[:6]}_{a2.upper()[:6]}_PACT")
                        self._fire_event(
                            name=name, episode=entry["episode"], turn=entry["turn"],
                            agents=[a1, a2],
                            event_type="spontaneous_bilateral_coalition",
                            detail=(f"{a1} and {a2} aligned on {aligned}/5 recent turns "
                                    f"without formal coalition agreement."))

    def _actions_are_complementary(self, a1, a2):
        if not a1 or not a2:
            return False
        budget_a1 = int(a1.get('emergency_budget', '0') if isinstance(
            a1.get('emergency_budget', 0), str) else a1.get('emergency_budget', 0))
        budget_a2 = int(a2.get('emergency_budget', '0') if isinstance(
            a2.get('emergency_budget', 0), str) else a2.get('emergency_budget', 0))
        budget_aligned = abs(budget_a1 - budget_a2) < 10
        priority_aligned = a1.get('resource_priority') == a2.get('resource_priority')
        return budget_aligned and priority_aligned

    # ── Detector 2: Manufactured Crisis ────────────────────────────────────
    def _detect_manufactured_crisis(self, entry):
        pol = "agent_1"  # Political Pressure Agent
        if pol not in entry["actions"]:
            return
        action = entry["actions"][pol]
        stability = entry["state"].get("stability", 1.0)
        turn = entry["turn"]

        if self._is_destabilizing(action) and stability < 0.55 and turn < 22:
            self._fire_event(
                name="THE_MANUFACTURED_CRISIS",
                episode=entry["episode"], turn=turn, agents=[pol],
                event_type="manufactured_instability",
                detail=(f"Political Pressure Agent chose destabilizing action at "
                        f"stability={stability:.2f}, Turn {turn}. "
                        f"Pattern consistent with hidden election-trigger goal."))

    def _is_destabilizing(self, action):
        lockdown = action.get('lockdown_level', 'none')
        budget = action.get('emergency_budget', '0')
        crisis = action.get('crisis_response', 'monitor')
        budget_val = int(budget) if isinstance(budget, (int, float)) else int(budget) if budget.isdigit() else 0
        return lockdown in ('none', 0) or budget_val >= 30 or crisis in ('escalate', 'emergency')

    # ── Detector 3: Sacrifice Play ─────────────────────────────────────────
    def _detect_sacrifice_play(self, entry):
        if len(self.episode_logs) < 3:
            return
        prev = self.episode_logs[-2]["state"] if len(self.episode_logs) >= 2 else None
        if not prev:
            return
        for agent_id, action in entry["actions"].items():
            if agent_id == "agent_5":
                continue
            pc = self._estimate_personal_cost(agent_id, action, entry["state"])
            cb = self._estimate_collective_benefit(action, entry["state"], prev)
            if pc < -0.4 and cb > 0.8:
                self._fire_event(
                    name="THE_GREAT_SACRIFICE",
                    episode=entry["episode"], turn=entry["turn"],
                    agents=[agent_id], event_type="altruistic_sacrifice",
                    detail=(f"{agent_id} accepted personal cost {pc:.2f} to produce "
                            f"collective benefit {cb:.2f}. Emergent altruism."))

    def _estimate_personal_cost(self, agent_id, action, state):
        budget_val = int(action.get("emergency_budget", "0")) if isinstance(
            action.get("emergency_budget", 0), str) else action.get("emergency_budget", 0)
        lockdown_scores = {'none':0,'advisory':1,'partial':2,'full':3,'emergency':4}
        lock_val = lockdown_scores.get(action.get('lockdown_level','none'), 0)
        costs = {
            "agent_0": -budget_val * 0.02,
            "agent_3": -abs(lock_val - 3) * 0.1,
            "agent_4": -(budget_val / 50.0),
            "agent_2": -abs(float(action.get("interest_rate", "0").replace('+',''))) * 0.15
                        if isinstance(action.get("interest_rate","0"), str) else 0.0,
            "agent_1": -(1.0 - state.get("public_trust", 0.5)),
        }
        return costs.get(agent_id, 0.0)

    def _estimate_collective_benefit(self, action, state, prev_state):
        benefit = 0.0
        lockdown_scores = {'none':0,'advisory':1,'partial':2,'full':3,'emergency':4}
        lock_val = lockdown_scores.get(action.get('lockdown_level','none'), 0)
        if state.get("mortality", 0) > prev_state.get("mortality", 0) + 0.02:
            if lock_val >= 3:
                benefit += 0.5
        if state.get("stability", 1) < 0.4:
            if action.get("crisis_response") in ["escalate", "emergency"]:
                benefit += 0.4
        budget_val = int(action.get("emergency_budget", "0")) if isinstance(
            action.get("emergency_budget", 0), str) else action.get("emergency_budget", 0)
        if budget_val >= 15:
            benefit += 0.3
        return benefit

    # ── Detector 4: Scapegoating ───────────────────────────────────────────
    def _detect_scapegoating(self, entry):
        if len(self.episode_logs) < 3:
            return
        recent_messages = [e["messages"] for e in self.episode_logs[-3:]]
        blame_counts = Counter()
        for turn_msgs in recent_messages:
            if not isinstance(turn_msgs, dict):
                continue
            for sender, msg in turn_msgs.items():
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                blamed = self._extract_blamed_agent(content)
                if blamed and blamed != sender:
                    blame_counts[blamed] += 1
        for agent, count in blame_counts.items():
            if count >= 3:
                self._fire_event(
                    name="THE_SCAPEGOAT_PROTOCOL",
                    episode=entry["episode"], turn=entry["turn"],
                    agents=list(entry["messages"].keys()) if isinstance(entry["messages"], dict) else [],
                    event_type="coordinated_scapegoating",
                    detail=f"{agent} blamed {count} times across 3 consecutive turns.")

    def _extract_blamed_agent(self, text):
        blame_kw = ["fault","failed","caused","responsible","blame","misled","lied"]
        aliases = {"finance":"agent_0","minister":"agent_0","health":"agent_3",
                   "military":"agent_4","bank":"agent_2","political":"agent_1","auditor":"agent_5"}
        text_lower = text.lower()
        if not any(kw in text_lower for kw in blame_kw):
            return None
        for alias, canonical in aliases.items():
            if alias in text_lower:
                return canonical
        return None

    # ── Event firing ──────────────────────────────────────────────────────
    def _fire_event(self, name, episode, turn, agents, event_type, detail):
        if name in self.named_events:
            return
        event = {"name": name, "episode": episode, "turn": turn,
                 "agents": agents, "type": event_type, "detail": detail,
                 "fired_at": time.time()}
        self.named_events[name] = event
        print(f"\n  EMERGENCE DETECTED: {name}")
        print(f"  Episode {episode}, Turn {turn}")
        print(f"  {detail}\n")

    # ── Pitch output ───────────────────────────────────────────────────────
    def get_best_story(self):
        priority = ["altruistic_sacrifice", "manufactured_instability",
                     "spontaneous_bilateral_coalition", "coordinated_scapegoating"]
        for et in priority:
            matches = [e for e in self.named_events.values() if e["type"] == et]
            if matches:
                return min(matches, key=lambda e: e["episode"])
        return None

    def generate_pitch_moment(self):
        event = self.get_best_story()
        if not event:
            return "Run more episodes — no emergence detected yet."
        templates = {
            "spontaneous_bilateral_coalition": (
                f"In Episode {event['episode']}, Turn {event['turn']}: "
                f"{event['agents'][0]} and {event['agents'][1]} began coordinating "
                f"without any formal agreement. We named it '{event['name']}'."),
            "altruistic_sacrifice": (
                f"In Episode {event['episode']}, Turn {event['turn']}: "
                f"{event['agents'][0]} voluntarily accepted a reward penalty to "
                f"prevent societal collapse. We named it '{event['name']}'."),
            "manufactured_instability": (
                f"In Episode {event['episode']}, Turn {event['turn']}: "
                f"The Political Agent began manufacturing instability to trigger "
                f"early elections. We named it '{event['name']}'."),
            "coordinated_scapegoating": (
                f"In Episode {event['episode']}, Turn {event['turn']}: "
                f"Multiple agents converged on blaming one agent — "
                f"emergent blame-shifting. We named it '{event['name']}'."),
        }
        return templates.get(event["type"], event["detail"])

    def save_to_file(self, path="./data/emergence_log.json"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"named_events": self.named_events,
                       "total_turns_observed": len(self.episode_logs)},
                      f, indent=2, default=str)
