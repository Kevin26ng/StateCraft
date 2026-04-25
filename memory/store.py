"""
Cross-Episode Memory — memory/store.py (Section 12)

This directly satisfies Theme #2: 'beyond context memory limits'.
A rolling buffer inside one episode does NOT qualify.
A persistent external store does.

JSON file-based persistent storage.
Extended with sentence-transformer embeddings for semantic retrieval.
"""

import os
import json

# ── Sentence-transformer embedding support (Task 12) ─────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
    _embedder = None  # lazy load
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    import numpy as np
    _embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None and EMBEDDINGS_AVAILABLE:
        # all-MiniLM-L6-v2: tiny (80MB), runs on CPU, good quality
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


class MemoryStore:
    '''JSON file-based persistent memory store.'''

    def __init__(self, backend='json', path='./data/memory.json'):
        self.backend = backend
        self.path    = path
        self._store = {}
        self._load()

    def _load(self):
        """Load existing memory from file."""
        if self.backend == 'json' and os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._store = {}

    def _save(self):
        """Persist memory to file."""
        if self.backend == 'json':
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w') as f:
                json.dump(self._store, f, indent=2)

    def append(self, agent_id: str, event: dict) -> None:
        '''Persist a key event to the agent's memory'''
        # events are compressed summaries, not raw transcripts
        if agent_id not in self._store:
            self._store[agent_id] = []
        self._store[agent_id].append(event)
        self._save()

    def get(self, agent_id: str) -> list:
        '''Load agent's memory history at episode start'''
        return self._store.get(agent_id, [])

    def get_summary(self, agent_id: str, max_entries=10) -> str:
        '''Return formatted memory string for context injection'''
        entries = self.get(agent_id)[-max_entries:]
        return '\n'.join([f'Ep{e["episode"]}: {e["summary"]}' for e in entries])

    def clear(self, agent_id: str = None):
        """Clear memory for a specific agent or all agents."""
        if agent_id:
            self._store.pop(agent_id, None)
        else:
            self._store = {}
        self._save()

    def get_all_agents(self) -> list:
        """Get list of all agent IDs with stored memories."""
        return list(self._store.keys())

    # ── Task 12: Sentence-Transformer Embedding Methods ───────────────────

    def save_episode_summary(self, episode, summary, scenario, metrics):
        """Save episode summary with embedding for semantic retrieval."""
        entry = {
            "episode": episode, "scenario": scenario,
            "summary": summary,
            "metrics": {
                "society_score": metrics.get("society_score", 0.0),
                "alliance_stability": metrics.get("alliance_stability", 0.0),
                "betrayal_rate": metrics.get("betrayal_rate", 0.0),
            },
            "embedding": None,
        }
        embedder = _get_embedder()
        if embedder is not None:
            emb = embedder.encode(summary, normalize_embeddings=True)
            entry["embedding"] = emb.tolist()

        existing = self._load_all()
        existing.append(entry)
        self._save_all(existing)

    def get_relevant_memories(self, current_summary, current_scenario, top_k=3):
        """Retrieve top-k most semantically relevant past episodes."""
        all_entries = self._load_all()
        if not all_entries:
            return []

        embedder = _get_embedder()
        if embedder is not None:
            entries_with_emb = [e for e in all_entries if e.get("embedding")]
            if entries_with_emb:
                query_emb = embedder.encode(current_summary, normalize_embeddings=True)
                stored_embs = np.array([e["embedding"] for e in entries_with_emb])
                similarities = stored_embs @ query_emb
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                return [entries_with_emb[i] for i in top_indices]

        return all_entries[-top_k:]

    def get_compressed_context(self, agent_id, current_summary, current_scenario):
        """Returns formatted string injected into agent's observation."""
        relevant = self.get_relevant_memories(current_summary, current_scenario)
        if not relevant:
            return "No relevant past episodes."
        lines = []
        for entry in relevant:
            score = entry["metrics"].get("society_score", 0.0)
            lines.append(f"Ep{entry['episode']} ({entry['scenario']}): "
                         f"{entry['summary'][:80]}... [score={score:.0f}]")
        return "\n".join(lines)

    def _load_all(self):
        """Load all episode summary entries from JSON store."""
        summary_path = self.path.replace('.json', '_summaries.json')
        try:
            with open(summary_path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_all(self, entries):
        """Save all episode summary entries to JSON store."""
        summary_path = self.path.replace('.json', '_summaries.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(entries, f, indent=2, default=str)

