"""
Scenario Generalization Evaluator — eval/generalization.py (Task 14)

Train on Pandemic only, evaluate zero-shot on Economic and Disaster.
Pure evaluation script — no additional training required.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np
from openenv.wrapper import CrisisGovernanceEnv
from training.ppo_policy import CrisisActorCritic, AGENT_ID_TO_ROLE_IDX

N_EVAL_EPISODES = 20
SCENARIOS = ["pandemic", "economic", "disaster"]


def evaluate_policy(policy, scenario, n_episodes=N_EVAL_EPISODES, seed=0):
    torch.manual_seed(seed)
    env = CrisisGovernanceEnv(config={"scenario": scenario})
    role_ids = torch.LongTensor(list(AGENT_ID_TO_ROLE_IDX.values()))

    all_scores, all_rewards, all_turns = [], [], []
    for ep in range(n_episodes):
        obs = torch.FloatTensor(env.reset().observations)
        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                actions, _, _, _ = policy.get_action_and_value(obs, role_ids)
            step = env.step(actions.numpy())
            ep_reward += step.reward
            obs = torch.FloatTensor(step.observations)
            done = step.done

        from metrics.tracker import MetricsTracker
        metrics = MetricsTracker().compute_episode_metrics(env._env)
        all_scores.append(metrics.get("society_score", 0.0))
        all_rewards.append(ep_reward)
        all_turns.append(metrics.get("turns_survived", 0))

    return {"scenario": scenario, "mean_reward": float(np.mean(all_rewards)),
            "mean_score": float(np.mean(all_scores)), "std_score": float(np.std(all_scores)),
            "mean_turns": float(np.mean(all_turns)), "n_episodes": n_episodes}


def run_generalization_test(checkpoint_path, train_scenario="pandemic"):
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Train the policy first: python -m training.ppo_trainer")
        return None

    policy = CrisisActorCritic()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"Loaded checkpoint from episode {ckpt['episode']}")

    results = {}
    for scenario in SCENARIOS:
        print(f"  Evaluating on {scenario}...")
        results[scenario] = evaluate_policy(policy, scenario)

    train_score = results[train_scenario]["mean_score"]
    transfer_results = {}
    for scenario, res in results.items():
        if scenario != train_scenario:
            gap = res["mean_score"] - train_score
            transfer_results[scenario] = {**res, "transfer_gap": gap,
                                           "transfer_positive": gap > -5.0}

    output = {"train_scenario": train_scenario,
              "train_performance": results[train_scenario],
              "transfer_results": transfer_results}

    os.makedirs("./checkpoints", exist_ok=True)
    with open("./checkpoints/generalization_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*60)
    print("SCENARIO GENERALIZATION RESULTS")
    print("="*60)
    t = results[train_scenario]
    print(f"{'TRAINING':20s} | score={t['mean_score']:.1f} ± {t['std_score']:.1f} | turns={t['mean_turns']:.0f}")
    for s, r in transfer_results.items():
        flag = "✓" if r["transfer_positive"] else "✗"
        print(f"{s.upper():20s} | score={r['mean_score']:.1f} ± {r['std_score']:.1f} | "
              f"gap={r['transfer_gap']:+.1f} | {flag}")
    print("="*60)
    return output


if __name__ == "__main__":
    run_generalization_test("./checkpoints/policy_final.pt")
