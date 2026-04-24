"""
Negotiation System — core/negotiation.py (Section 5.1)

Message passing, trust updates from message types.
Implements the 3-round negotiation protocol.
"""

from copy import deepcopy


# Section 5.1.1 — Message Format
# message = {
#   'sender':  str,   # agent canonical name
#   'target':  str,   # agent canonical name OR 'all'
#   'type':    str,   # 'support'|'threat'|'trade'|'reject'|'inform'
#   'content': str,   # free-text <50 tokens
#   'turn':    int,
# }


class NegotiationSystem:
    """
    Implements the negotiation protocol.

    Section 5.1 — Each turn has up to 3 rounds of negotiation.
    Messages update trust_matrix based on type (Section 5.1.2).
    Trade messages create pending_trades with conditional follow-through.
    """

    def __init__(self, trust_system=None):
        self.trust_system = trust_system
        self.round_messages = {1: [], 2: [], 3: []}
        self.current_round = 0

    def reset_turn(self):
        """Reset negotiation state for a new turn."""
        self.round_messages = {1: [], 2: [], 3: []}
        self.current_round = 0

    def negotiate_round(self, agents, observations, round_num: int) -> list:
        """
        Run one negotiation round.

        Args:
            agents: dict of agent_id -> agent instance
            observations: dict of agent_id -> observation
            round_num: which round (1, 2, or 3)

        Returns:
            list of messages from this round
        """
        self.current_round = round_num
        messages = []

        for agent_id, agent in agents.items():
            if hasattr(agent, 'negotiate'):
                agent_msgs = agent.negotiate(observations.get(agent_id, {}),
                                             round_num)
                for msg in agent_msgs:
                    formatted = {
                        'sender': agent_id,
                        'target': msg.get('target', 'all'),
                        'type': msg.get('type', 'inform'),
                        'content': msg.get('content', ''),
                        'turn': observations.get(agent_id, {}).get(
                            'public_state', {}).get('turn', 0),
                    }
                    messages.append(formatted)

        self.round_messages[round_num] = messages
        return messages

    def update_from_messages(self, messages: list):
        """
        Section 5.1.2 — Trust Effects from Message Types.

        Process messages and update trust_matrix accordingly.
        """
        if not self.trust_system:
            return

        for msg in messages:
            sender = msg.get('sender', '')
            target = msg.get('target', '')

            if target == 'all' or not target:
                continue  # broadcast messages don't directly affect trust

            try:
                sender_idx = int(sender.split('_')[1])
                target_idx = int(target.split('_')[1])
            except (ValueError, IndexError):
                continue

            msg_type = msg.get('type', 'inform')

            if msg_type == 'support':
                self.trust_system.update('message_support',
                                         sender_idx, target_idx)
            elif msg_type == 'threat':
                self.trust_system.update('message_threat',
                                         sender_idx, target_idx)
            elif msg_type == 'trade':
                # No immediate effect — conditional on follow-through
                self.trust_system.add_pending_trade(
                    sender, target,
                    msg.get('content', ''),
                    msg.get('turn', 0)
                )
            elif msg_type == 'reject':
                self.trust_system.update('message_reject',
                                         sender_idx, target_idx)
            elif msg_type == 'inform':
                pass  # neutral — used by Auditor for flagging

    def get_all_messages(self) -> list:
        """Get all messages from all rounds."""
        all_msgs = []
        for round_num in sorted(self.round_messages.keys()):
            all_msgs.extend(self.round_messages[round_num])
        return all_msgs

    def get_final_round_messages(self) -> list:
        """Get messages from the last round."""
        return self.round_messages.get(self.current_round, [])
