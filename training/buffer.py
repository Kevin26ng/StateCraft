"""
PPO Replay Buffer — training/buffer.py

Stub for PPO experience replay buffer.
To be fully implemented when integrating Stable-Baselines3 or HF TRL.
"""

import numpy as np
from collections import deque


class PPOBuffer:
    """
    Experience replay buffer for PPO training.

    Stores transitions: (obs, action, reward, next_obs, done, log_prob, value)
    """

    def __init__(self, capacity: int = 10000, num_agents: int = 6):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffers = {
            f'agent_{i}': deque(maxlen=capacity)
            for i in range(num_agents)
        }

    def store(self, agent_id: str, transition: dict) -> None:
        """
        Store a transition for an agent.

        Args:
            agent_id: which agent
            transition: dict with keys:
                obs, action, reward, next_obs, done, log_prob, value
        """
        if agent_id in self.buffers:
            self.buffers[agent_id].append(transition)

    def sample_batch(self, agent_id: str, batch_size: int = 64) -> list:
        """
        Sample a batch of transitions for an agent.

        Args:
            agent_id: which agent
            batch_size: number of transitions to sample

        Returns:
            list of transition dicts
        """
        buffer = self.buffers.get(agent_id, [])
        if len(buffer) < batch_size:
            return list(buffer)

        indices = np.random.choice(len(buffer), batch_size, replace=False)
        return [buffer[i] for i in indices]

    def get_all(self, agent_id: str) -> list:
        """Get all stored transitions for an agent."""
        return list(self.buffers.get(agent_id, []))

    def clear(self, agent_id: str = None) -> None:
        """Clear buffer for a specific agent or all agents."""
        if agent_id:
            if agent_id in self.buffers:
                self.buffers[agent_id].clear()
        else:
            for buf in self.buffers.values():
                buf.clear()

    def size(self, agent_id: str = None) -> int:
        """Get current buffer size."""
        if agent_id:
            return len(self.buffers.get(agent_id, []))
        return sum(len(b) for b in self.buffers.values())

    def is_ready(self, agent_id: str, min_size: int = 64) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size(agent_id) >= min_size
