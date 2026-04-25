"""
Cloud/Colab training launcher for StateCraft.

Features:
- Runs either standard loop or curriculum loop
- Saves metrics + run metadata to an output folder
- Optionally uploads artifacts to Hugging Face Hub (dataset/model repo)

Security:
- Reads token from env var (default: HF_TOKEN)
- Never stores tokens in repo files
"""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Optional


def detect_runtime() -> dict:
    return {
        "is_colab": "COLAB_RELEASE_TAG" in os.environ,
        "has_gpu": bool(os.environ.get("COLAB_GPU")),
        "has_tpu": bool(os.environ.get("COLAB_TPU_ADDR")),
        "python": os.sys.version.split()[0],
    }


def make_output_dir(base_dir: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_training(config: dict, use_curriculum: bool):
    if use_curriculum:
        from training.curriculum import run_curriculum_training

        scheduler, metrics_history = run_curriculum_training(config)
        curriculum = {
            "final_phase": scheduler.phase_name,
            "phase_transitions": scheduler.phase_history,
        }
        return metrics_history, curriculum

    from training.loop import run_training_loop

    metrics_history = run_training_loop(config)
    return metrics_history, None


def save_artifacts(output_dir: str, config: dict, runtime: dict,
                   metrics_history: list, curriculum: Optional[dict]):
    metrics_path = os.path.join(output_dir, "metrics_history.json")
    config_path = os.path.join(output_dir, "run_config.json")
    summary_path = os.path.join(output_dir, "summary.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    summary = {
        "runtime": runtime,
        "episodes": len(metrics_history),
        "final_metrics": metrics_history[-1] if metrics_history else {},
    }
    if curriculum:
        summary["curriculum"] = curriculum

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def maybe_upload_to_hub(output_dir: str, repo_id: str, repo_type: str,
                        private: bool, token_env: str):
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(
            f"Missing token in env var '{token_env}'. "
            "Set it before using --push-to-hub."
        )

    api = HfApi(token=token)
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
        token=token,
    )

    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=output_dir,
        commit_message="Upload StateCraft training artifacts",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StateCraft in Colab/VS Code")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--scenario", type=str, default="pandemic")
    parser.add_argument("--mode", type=str, default="TRAINING")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use training.curriculum.run_curriculum_training")
    parser.add_argument("--output-dir", type=str, default="./outputs")

    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo-id", type=str, default="")
    parser.add_argument("--hf-repo-type", type=str, default="dataset",
                        choices=["dataset", "model", "space"])
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-token-env", type=str, default="HF_TOKEN")
    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        "episode_mode": args.mode,
        "num_episodes": args.episodes,
        "scenario": args.scenario,
    }

    runtime = detect_runtime()
    print("Runtime:", runtime)

    output_dir = make_output_dir(args.output_dir)
    print("Output dir:", output_dir)

    metrics_history, curriculum = run_training(config, args.curriculum)
    save_artifacts(output_dir, config, runtime, metrics_history, curriculum)

    if args.push_to_hub:
        if not args.hf_repo_id:
            raise ValueError("--hf-repo-id is required when --push-to-hub is used")
        maybe_upload_to_hub(
            output_dir=output_dir,
            repo_id=args.hf_repo_id,
            repo_type=args.hf_repo_type,
            private=args.hf_private,
            token_env=args.hf_token_env,
        )
        print(f"Uploaded artifacts to Hugging Face: {args.hf_repo_type}/{args.hf_repo_id}")

    print("Cloud training run completed.")


if __name__ == "__main__":
    main()
