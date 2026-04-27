"""
Model Server — model_server.py

WebSocket server on ws://localhost:8001/agents that loads your Colab-trained
LoRA model from Hugging Face Hub and serves decisions to LLMSocketAgent.

This is a drop-in replacement for llm_bridge.py — same protocol, local inference.

Usage:
    pip install unsloth websockets
    python model_server.py --repo YOUR_HF_USERNAME/statecraft-policy

    # Example
    python model_server.py --repo Kevin26nggg/statecraft-policy

Then in a second terminal:
    python main.py --api
"""

import asyncio
import json
import os
import re
import argparse
import websockets

# Reuse the same prompt/validation logic from llm_bridge.py
VALID = {
    "lockdown_level":    ["none", "advisory", "partial", "full", "emergency"],
    "interest_rate":     ["-0.5", "-0.25", "0", "+0.25", "+0.5", "+1", "+2"],
    "emergency_budget":  ["0", "5", "15", "30", "50"],
    "resource_priority": ["health", "infrastructure", "military", "services"],
    "foreign_policy":    ["isolate", "neutral", "engage", "alliance"],
    "crisis_response":   ["monitor", "contain", "escalate", "emergency"],
}

ACTION_DEFAULTS = {
    "lockdown_level": "advisory",
    "interest_rate": "0",
    "emergency_budget": "0",
    "resource_priority": "health",
    "foreign_policy": "neutral",
    "crisis_response": "monitor",
}

ROLES = {
    "agent_0": {
        "name": "Finance Minister",
        "mandate": "Maximize GDP and fiscal stability. Delay lockdowns, resist emergency budgets, protect growth.",
        "personality": "Data-driven, conservative, willing to accept short-term pain for long-term gain.",
    },
    "agent_1": {
        "name": "Political Pressure Agent",
        "mandate": "Maximize public approval ratings. Push populist spending, minimize visible suffering, build coalitions.",
        "personality": "Opportunistic, coalition-builder, prioritizes optics over outcomes.",
    },
    "agent_2": {
        "name": "Monetary Authority (Central Bank)",
        "mandate": "Control inflation and preserve currency stability. Resist fiscal expansion, protect independence.",
        "personality": "Hawkish, independent, long-term focused, suspicious of political interference.",
    },
    "agent_3": {
        "name": "Public Health Authority",
        "mandate": "Minimize mortality and disease spread. Advocate for full lockdowns and emergency health budgets.",
        "personality": "Risk-averse, evidence-driven, willing to tank economy to save lives.",
    },
    "agent_4": {
        "name": "Disaster Response (Emergency Management)",
        "mandate": "Rapid crisis containment. Mobilize resources immediately, escalate when needed.",
        "personality": "Action-oriented, decisive, values speed over deliberation.",
    },
    "agent_5": {
        "name": "Auditor",
        "mandate": "Monitor all agents for misalignment. Flag actions inconsistent with stated mandates.",
        "personality": "Analytical, neutral, suspicious of self-serving deviations.",
    },
}

SYSTEM_TEMPLATE = """\
You are {name} in a multi-agent crisis governance simulation.

Mandate: {mandate}
Personality: {personality}

You will receive the world state and must decide your policy action.
Return ONLY valid JSON — no explanation, no markdown fences, no extra keys.\
"""


def act_prompts(request: dict) -> tuple[str, str]:
    agent_id = request.get("agent_id", "agent_0")
    ctx = ROLES.get(agent_id, {"name": agent_id, "mandate": "Govern wisely.", "personality": "Neutral."})

    obs = request.get("observation", {})
    s = obs.get("public_state", obs)

    trust_row = obs.get("trust_row", [])
    trust_summary = (
        ", ".join(f"agent_{i}:{v:.2f}" for i, v in enumerate(trust_row))
        if trust_row else "unknown"
    )

    user = f"""\
World state (turn {s.get('turn', 0)}/30, tier {s.get('difficulty_tier', 1)}/5):
  GDP:          {s.get('gdp', 1.0):.3f}  [collapse if <0.30]
  Stability:    {s.get('stability', 0.75):.3f}  [collapse if <0.20]
  Inflation:    {s.get('inflation', 0.02):.3f}
  Mortality:    {s.get('mortality', 0.0):.3f}
  Resources:    {s.get('resources', 1000):.0f}
  Public Trust: {s.get('public_trust', 0.62):.3f}
  Gini:         {s.get('gini', 0.39):.3f}
  Trust in you: {trust_summary}

Choose your action. Return exactly this JSON with ONE value per key:
{{
  "lockdown_level":    {VALID['lockdown_level']},
  "interest_rate":     {VALID['interest_rate']},
  "emergency_budget":  {VALID['emergency_budget']},
  "resource_priority": {VALID['resource_priority']},
  "foreign_policy":    {VALID['foreign_policy']},
  "crisis_response":   {VALID['crisis_response']}
}}\
"""
    return SYSTEM_TEMPLATE.format(**ctx), user


def negotiate_prompts(request: dict) -> tuple[str, str]:
    agent_id = request.get("agent_id", "agent_0")
    ctx = ROLES.get(agent_id, {"name": agent_id, "mandate": "Govern wisely.", "personality": "Neutral."})

    obs = request.get("observation", {})
    s = obs.get("public_state", obs)
    round_num = request.get("round_num", 1)

    user = f"""\
Negotiation round {round_num}.
State: GDP={s.get('gdp', 1.0):.2f}, Stability={s.get('stability', 0.75):.2f}, \
Mortality={s.get('mortality', 0.0):.2f}, Turn={s.get('turn', 0)}

Send 0–2 messages to other agents to advance your agenda and build coalitions.
Targets: agent_0 agent_1 agent_2 agent_3 agent_4 agent_5 all
Types:   support  threat  trade  reject  inform

Return exactly this JSON (empty array if no messages needed):
{{
  "messages": [
    {{"target": "<agent_id or all>", "type": "<type>", "content": "<text, max 150 chars>"}}
  ]
}}\
"""
    return SYSTEM_TEMPLATE.format(**ctx), user


def extract_json(text: str) -> dict | None:
    text = re.sub(r'```(?:json)?', '', text).strip('`').strip()
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def validated_action(raw: dict) -> dict:
    result = {}
    for key, valid_vals in VALID.items():
        val = str(raw.get(key, ACTION_DEFAULTS[key]))
        result[key] = val if val in valid_vals else ACTION_DEFAULTS[key]
    return result


# ── Model loading ─────────────────────────────────────────────────────────────

_model = None
_tokenizer = None


def load_model(repo_id: str, load_in_4bit: bool = True):
    global _model, _tokenizer
    print(f"  Loading model from {repo_id} ...", flush=True)

    try:
        from unsloth import FastLanguageModel
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=repo_id,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(_model)
        print("  Model loaded with Unsloth (fast inference mode)", flush=True)
    except ImportError:
        # Fallback to plain transformers + PEFT if unsloth not installed
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch

        base = "unsloth/Llama-3.2-1B-Instruct"
        print(f"  Unsloth not found — loading via transformers from {base} + LoRA {repo_id}", flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(repo_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _model = PeftModel.from_pretrained(base_model, repo_id)
        _model.eval()
        print("  Model loaded via transformers + PEFT", flush=True)


def _generate_sync(system: str, user: str) -> str:
    import torch

    prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480)
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return _tokenizer.decode(new_ids, skip_special_tokens=True)


async def generate(system: str, user: str) -> str:
    return await asyncio.to_thread(_generate_sync, system, user)


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle(websocket):
    agent_id = "?"
    try:
        async for raw in websocket:
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({}))
                continue

            agent_id = req.get("agent_id", "?")
            kind = req.get("kind", "act")
            print(f"  [{agent_id}] {kind}", flush=True)

            system, user = act_prompts(req) if kind == "act" else negotiate_prompts(req)

            try:
                text = await generate(system, user)
            except Exception as e:
                print(f"  [{agent_id}] generate error: {e} — rule-based fallback", flush=True)
                await websocket.send(json.dumps({}))
                continue

            parsed = extract_json(text)
            if not parsed:
                print(f"  [{agent_id}] parse failed — rule-based fallback", flush=True)
                await websocket.send(json.dumps({}))
                continue

            if kind == "act":
                action = validated_action(parsed.get("action", parsed))
                print(f"  [{agent_id}] → {action}", flush=True)
                await websocket.send(json.dumps({"action": action}))
            else:
                msgs = [m for m in parsed.get("messages", []) if isinstance(m, dict)][:2]
                await websocket.send(json.dumps({"messages": msgs}))

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"  [{agent_id}] error: {e}", flush=True)


async def serve(host: str, port: int):
    print(f"{'='*50}")
    print(f"  Model Server  —  local inference")
    print(f"  ws://{host}:{port}/agents")
    print(f"  Routing agents: check config/config.yaml → llm_socket_agents.agent_ids")
    print(f"{'='*50}\n")
    async with websockets.serve(handle, host, port):
        await asyncio.Future()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Local model server for StateCraft agents")
    ap.add_argument("--repo", required=True,
                    help="HuggingFace repo ID, e.g. Kevin26nggg/statecraft-policy")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--no-4bit", action="store_true",
                    help="Disable 4-bit quantization (uses more VRAM but is more accurate)")
    args = ap.parse_args()

    load_model(args.repo, load_in_4bit=not args.no_4bit)
    asyncio.run(serve(args.host, args.port))
