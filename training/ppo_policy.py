"""
PPO Policy Network — training/ppo_policy.py (Task 3)

Real torch.nn.Module policy + value head for Crisis Governance Simulator.
Replaces heuristic agents with a shared Actor-Critic network.

Architecture:
  - Role embedding: nn.Embedding(6, 8)
  - Input: obs(32) + role_emb(8) = 40 dims
  - Trunk: Linear(40,256) → LayerNorm → ReLU → Linear(256,256) → LayerNorm → ReLU
  - Policy heads: 5 separate Linear(256, n) for MultiDiscrete([5,5,5,4,4])
  - Value head: Linear(256, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

N_AGENTS = 6
OBS_DIM = 32  # matches wrapper._flatten_observations output
ACTION_DIMS = [5, 5, 5, 4, 4]  # MultiDiscrete action space
ROLE_EMBED_DIM = 8

# Maps canonical agent_id (agent_0..5) to role index for embedding
# Order matches existing AGENT_ID_TO_ROLE in core/rewards.py:
#   agent_0=finance, agent_1=political, agent_2=central_bank,
#   agent_3=health, agent_4=military, agent_5=auditor
AGENT_ROLE_IDS = {
    'finance_minister': 0,        # agent_0
    'political_pressure_agent': 1, # agent_1
    'monetary_authority': 2,       # agent_2
    'public_health_authority': 3,  # agent_3
    'disaster_response_agent': 4,  # agent_4
    'auditor': 5,                  # agent_5
}

# Mapping from agent_id to role index (for use in training loop)
AGENT_ID_TO_ROLE_IDX = {
    'agent_0': 0,
    'agent_1': 1,
    'agent_2': 2,
    'agent_3': 3,
    'agent_4': 4,
    'agent_5': 5,
}


class CrisisActorCritic(nn.Module):
    """
    Shared Actor-Critic network for all 6 agents.
    Role embeddings allow agents to share weights while maintaining
    role-specific behavior — critical for multi-agent PPO.
    """

    def __init__(self, obs_dim=OBS_DIM, role_embed_dim=ROLE_EMBED_DIM,
                 action_dims=None, hidden_dim=256):
        super().__init__()

        if action_dims is None:
            action_dims = ACTION_DIMS

        self.action_dims = action_dims
        self.role_embedding = nn.Embedding(N_AGENTS, role_embed_dim)

        input_dim = obs_dim + role_embed_dim

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Separate policy head per action domain
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_actions)
            for n_actions in action_dims
        ])

        # Single value head (shared across all agents)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, role_ids: torch.Tensor):
        """
        obs: (batch, obs_dim) float32
        role_ids: (batch,) int64 — index into role embeddings

        Returns: action_logits list, value
        """
        role_emb = self.role_embedding(role_ids)    # (batch, role_embed_dim)
        x = torch.cat([obs, role_emb], dim=-1)      # (batch, obs_dim + role_embed_dim)
        features = self.trunk(x)                     # (batch, hidden_dim)

        action_logits = [head(features) for head in self.policy_heads]
        value = self.value_head(features).squeeze(-1)

        return action_logits, value

    def get_action_and_value(self, obs: torch.Tensor, role_ids: torch.Tensor,
                             action=None):
        """
        Sample actions from the policy, compute log-probs and entropy.
        Used during rollout collection and PPO update.

        Args:
            obs: (batch, obs_dim) float32
            role_ids: (batch,) int64
            action: optional (batch, n_action_domains) — if provided, compute
                    log-probs for these actions instead of sampling new ones

        Returns:
            actions: (batch, n_action_domains) int64
            log_probs: (batch,) float32 — SUM of log-probs across action domains
            entropy: (batch,) float32 — mean entropy across action domains
            value: (batch,) float32
        """
        action_logits, value = self.forward(obs, role_ids)

        distributions = [Categorical(logits=logits) for logits in action_logits]

        if action is None:
            # Sample new actions
            actions = torch.stack([d.sample() for d in distributions], dim=-1)
        else:
            actions = action  # use provided actions (for PPO update)

        # SUM of log-probs across action domains (joint log-probability)
        log_probs = torch.stack(
            [d.log_prob(actions[:, i]) for i, d in enumerate(distributions)],
            dim=-1
        ).sum(dim=-1)

        # Mean entropy across action domains
        entropy = torch.stack(
            [d.entropy() for d in distributions], dim=-1
        ).mean(dim=-1)

        return actions, log_probs, entropy, value
