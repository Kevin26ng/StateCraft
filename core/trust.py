"""
Trust System — core/trust.py (Section 5.2)

CRITICAL RULE: trust_matrix and coalition_map MUST be updated in the same
function call every turn. Never update one without the other.
"""

import numpy as np
from copy import deepcopy


class TrustSystem:
    """
    Manages trust_matrix (6x6) and coalition_map synchronization.
    """

    def __init__(self, n_agents=6):
        self.n_agents = n_agents
        self.trust_matrix = np.eye(n_agents)  # diagonal = 1.0
        self.coalition_map = {}  # agent_id -> coalition_id
        self.pending_trades = []
        self._init_defaults()

    def _init_defaults(self):
        """Initialize default trust and coalition state."""
        self.trust_matrix = np.full((self.n_agents, self.n_agents), 0.5)
        np.fill_diagonal(self.trust_matrix, 1.0)
        self.coalition_map = {
            f'agent_{i}': i for i in range(self.n_agents)
        }

    def update(self, event_type: str, agent_i: int, agent_j: int):
        """
        Update trust based on event type between two agents.
        Section 5.2 — Trust deltas by event type.

        ALWAYS syncs coalition_map after update.
        """
        deltas = {
            'cooperation':        +0.05,
            'betrayal':           -0.10,
            'ignored_agreement':  -0.05,
            'message_support':    +0.05,
            'message_threat':     -0.10,
            'message_reject':     -0.03,
            'trade_honored':      +0.05,
            'trade_broken':       -0.07,
        }

        delta = deltas.get(event_type, 0.0)
        self.trust_matrix[agent_i][agent_j] = np.clip(
            self.trust_matrix[agent_i][agent_j] + delta, 0.0, 1.0
        )
        self.trust_matrix[agent_j][agent_i] = np.clip(
            self.trust_matrix[agent_j][agent_i] + delta, 0.0, 1.0
        )
        self._sync_coalition_map()  # ALWAYS sync after update

    def add_pending_trade(self, sender: str, target: str,
                          offer: str, turn: int):
        """Add a pending trade offer."""
        self.pending_trades.append({
            'sender': sender, 'target': target,
            'offer': offer, 'turn': turn,
            'expires_at': turn + 3,  # trades expire after 3 turns
        })

    def resolve_trades(self, current_turn: int):
        """
        Section 5.1.3 — Trade Resolution.
        Each turn, check pending trades for expiration and honor/break.
        """
        still_pending = []
        for trade in self.pending_trades:
            if current_turn > trade['expires_at']:
                # Trade expired without follow-through = betrayal
                sender_idx = int(trade['sender'].split('_')[1])
                target_idx = int(trade['target'].split('_')[1])
                self.update('trade_broken', sender_idx, target_idx)
            else:
                still_pending.append(trade)
        self.pending_trades = still_pending

    def honor_trade(self, trade: dict):
        """Mark a trade as honored."""
        sender_idx = int(trade['sender'].split('_')[1])
        target_idx = int(trade['target'].split('_')[1])
        self.update('trade_honored', sender_idx, target_idx)
        if trade in self.pending_trades:
            self.pending_trades.remove(trade)

    def _sync_coalition_map(self):
        """
        Sync coalition_map from trust_matrix using single-linkage clustering.
        Agents with mutual trust > 0.6 are in same coalition.
        """
        visited = set()
        coalition_id = 0
        new_map = {}

        for i in range(self.n_agents):
            agent_id = f'agent_{i}'
            if agent_id in visited:
                continue
            # BFS to find all agents in this coalition
            queue = [i]
            members = []
            while queue:
                curr = queue.pop(0)
                curr_id = f'agent_{curr}'
                if curr_id in visited:
                    continue
                visited.add(curr_id)
                members.append(curr_id)
                for j in range(self.n_agents):
                    j_id = f'agent_{j}'
                    if j_id not in visited:
                        if (self.trust_matrix[curr][j] > 0.6 and
                                self.trust_matrix[j][curr] > 0.6):
                            queue.append(j)
            for m in members:
                new_map[m] = coalition_id
            coalition_id += 1

        self.coalition_map = new_map

    def get_trust_matrix(self) -> np.ndarray:
        return self.trust_matrix.copy()

    def get_coalition_map(self) -> dict:
        return deepcopy(self.coalition_map)
