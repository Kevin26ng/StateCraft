"""
API Layer — api/server.py (Section 9)

FastAPI endpoints and WebSocket for the Crisis Governance Simulator.
"""

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import StepResponse, ActionsPayload, ResetConfig
from env.crisis_env import CrisisEnv
from core.trust import TrustSystem
from core.aggregation import aggregate_actions
from core.rewards import RewardSystem
from metrics.tracker import MetricsTracker
from logs.event_logger import EventLogger
from logs.narrative import NarrativeSystem
from agents.auditor import AuditorAgent

# ─────────────────────────────────────────────────────────────────────
# 9.1 FastAPI Endpoints
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(title='Crisis Governance Simulator API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
env = CrisisEnv()
tracker = MetricsTracker()
event_logger = EventLogger()
narrative = NarrativeSystem()
trust_system = TrustSystem()
reward_system = RewardSystem()
auditor = AuditorAgent()
metrics_history = []


# POST /reset — initialize or reset environment
@app.post('/reset')
async def reset(config: ResetConfig = None):
    obs = env.reset(config)
    event_logger.clear_turn_events()
    trust_system._init_defaults()
    return {'observations': obs, 'state': env.state}


# POST /step — advance one turn
@app.post('/step')
async def step(actions: ActionsPayload):
    event_logger.clear_turn_events()

    # Enforce action limits & tracking
    raw_actions = actions.actions_dict
    raw_actions = env.enforce_and_track_actions(raw_actions)

    # Aggregate actions
    final_action = aggregate_actions(raw_actions)

    obs, rewards, done, info = env.step(final_action, raw_agent_actions=raw_actions)
    metrics = tracker.compute_episode_metrics(env)
    events = event_logger.get_turn_events()

    # Generate headline
    headline = narrative.generate(env.state, events, env.state.get('turn', 0))

    # Sync trust system
    env.state_manager.trust_matrix = trust_system.get_trust_matrix()
    env.state_manager.coalition_map = trust_system.get_coalition_map()

    response = StepResponse(
        state=env.state,
        trust_matrix=env.state_manager.trust_matrix.tolist(),
        coalition_graph=tracker.get_coalition_graph(env),
        events=events,
        actions=final_action,
        messages=info.get('messages', []),
        metrics=metrics,
        headline=headline,
        done=done,
        auditor_report=auditor.fingerprint_cache,
    )

    if done:
        tracker.record_episode(metrics)
        metrics_history.append(metrics)

    return response


# GET /metrics — Current episode metrics
@app.get('/metrics')
async def get_metrics():
    return tracker.compute_episode_metrics(env)


# GET /history — all episode metrics
@app.get('/history')
async def get_history():
    return {'episodes': metrics_history}


# ─────────────────────────────────────────────────────────────────────
# 9.2 WebSocket — Real-Time Streaming
# ─────────────────────────────────────────────────────────────────────

@app.websocket('/ws/stream')
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Push state update after every env.step()
            payload = {
                'turn':         env.state_manager.state.get('turn', 0),
                'state':        env.state_manager.state,
                'trust_matrix': env.state_manager.trust_matrix.tolist(),
                'events':       event_logger.get_turn_events(),
                'headline':     narrative.last_headline,
                'metrics':      tracker.get_current_metrics(),
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.0)  # yield — demo mode uses sleep(0.5)
    except WebSocketDisconnect:
        pass
