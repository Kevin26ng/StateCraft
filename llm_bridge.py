"""
LLM Bridge Server — llm_bridge.py

WebSocket server on ws://localhost:8001/agents that receives agent decision
requests from LLMSocketAgent and routes them to Claude or OpenAI.

The training loop calls this automatically for any agent listed under
llm_socket_agents.agent_ids in config/config.yaml.

Usage:
    # Claude (Haiku — fast & cheap)
    ANTHROPIC_API_KEY=sk-ant-... python llm_bridge.py

    # OpenAI (GPT-4o-mini)
    OPENAI_API_KEY=sk-...        python llm_bridge.py --provider openai

    # Specific agents or all 6
    python llm_bridge.py --agents agent_0,agent_1,agent_5

Then in a second terminal:
    python main.py               # training loop routes agent_1 & agent_5 here
    python main.py --api         # API server (for the web UI)
"""

import asyncio
import json
import os
import re
import argparse
import websockets

# ── Valid discrete action values ──────────────────────────────────────────────
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

# ── Per-role context injected into every prompt ───────────────────────────────
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


# ── Prompt builders ───────────────────────────────────────────────────────────

def act_prompts(request: dict) -> tuple[str, str]:
    agent_id = request.get("agent_id", "agent_0")
    ctx = ROLES.get(agent_id, {"name": agent_id, "mandate": "Govern wisely.", "personality": "Neutral."})

    obs = request.get("observation", {})
    s = obs.get("public_state", obs)  # unwrap if nested

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


# ── Response parsing & validation ─────────────────────────────────────────────

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


# ── LLM callers (run in thread to avoid blocking event loop) ─────────────────

def _claude_sync(system: str, user: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def _openai_sync(system: str, user: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


async def call_llm(system: str, user: str, provider: str) -> str:
    fn = _claude_sync if provider == "claude" else _openai_sync
    return await asyncio.to_thread(fn, system, user)


# ── WebSocket handler ─────────────────────────────────────────────────────────

PROVIDER = "claude"


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
                text = await call_llm(system, user, PROVIDER)
            except Exception as e:
                print(f"  [{agent_id}] LLM error: {e} — rule-based fallback", flush=True)
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
    print(f"  LLM Bridge  —  provider: {PROVIDER.upper()}")
    print(f"  ws://{host}:{port}/agents")
    print(f"  Routing agents: check config/config.yaml → llm_socket_agents.agent_ids")
    print(f"{'='*50}\n")
    async with websockets.serve(handle, host, port):
        await asyncio.Future()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM Bridge for StateCraft agents")
    ap.add_argument("--provider", choices=["claude", "openai"], default="claude",
                    help="LLM provider (default: claude)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    PROVIDER = args.provider

    if args.provider == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY=sk-ant-...")
        exit(1)
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY=sk-...")
        exit(1)

    asyncio.run(serve(args.host, args.port))
