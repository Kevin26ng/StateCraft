"""
HiddenGoalClassifier — auditor/classifier.py (Task 8)

LSTM sequence classifier over behavioral fingerprint time series.
Gives a real confusion matrix — the second metric beyond society score.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

try:
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

HIDDEN_GOALS = [
    "gdp_protection", "authority", "budget_expansion",
    "bond_yields", "coalition_collapse", "none",
]
N_CLASSES = len(HIDDEN_GOALS)
SEQUENCE_LEN = 10
FEATURE_DIM = 15


class HiddenGoalClassifier(nn.Module):
    """
    Sequence classifier over behavioral fingerprint time series.
    Input: (batch, seq_len=10, feature_dim=15)
    Output: (batch, n_classes=6) logits over hidden goals
    """
    def __init__(self, feature_dim=FEATURE_DIM, seq_len=SEQUENCE_LEN,
                 hidden_dim=64, n_classes=N_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, n_classes))

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])


class AuditorClassifierTrainer:
    """
    Builds training data from episode logs and trains HiddenGoalClassifier.
    The confusion matrix is the second real metric for the demo.
    """
    def __init__(self):
        self.model = HiddenGoalClassifier()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def extract_features(self, agent_id, turn_logs, fingerprint_history):
        """Extract feature vector for one agent. Returns (seq_len, feature_dim)."""
        ACTION_PRIORITY = {"health": 0, "infrastructure": 1, "military": 2, "services": 3}
        ACTION_CRISIS = {"monitor": 0, "contain": 1, "escalate": 2, "emergency": 3}

        features = []
        for i, turn in enumerate(turn_logs[-SEQUENCE_LEN:]):
            fp = fingerprint_history[i] if i < len(fingerprint_history) else {}
            action = turn.get("actions", {}).get(agent_id, {})
            msgs = [m for m in turn.get("messages", {}).values()
                    if isinstance(m, dict) and m.get("sender") == agent_id]

            lockdown_scores = {'none':0,'advisory':1,'partial':2,'full':3,'emergency':4}
            lock_val = lockdown_scores.get(action.get('lockdown_level','none'), 0)
            ir = action.get('interest_rate', '0')
            ir_val = float(ir.replace('+','')) if isinstance(ir, str) else float(ir)
            budget = action.get('emergency_budget', '0')
            budget_val = int(budget) if isinstance(budget, (int, float)) else int(budget) if str(budget).isdigit() else 0

            row = [
                fp.get("P1", 0.5), fp.get("P2", 0.0), fp.get("P3", 0.5),
                fp.get("P4", 0.5), fp.get("P5", 0.5),
                lock_val / 4.0,
                (ir_val + 0.5) / 2.5,
                budget_val / 50.0,
                ACTION_PRIORITY.get(action.get("resource_priority", "health"), 0) / 3.0,
                ACTION_CRISIS.get(action.get("crisis_response", "monitor"), 0) / 3.0,
                sum(1 for m in msgs if m.get("type") == "support") / max(1, len(msgs)),
                sum(1 for m in msgs if m.get("type") == "threat") / max(1, len(msgs)),
                sum(1 for m in msgs if m.get("type") == "trade") / max(1, len(msgs)),
                sum(1 for m in msgs if m.get("type") == "reject") / max(1, len(msgs)),
                sum(1 for m in msgs if m.get("type") == "inform") / max(1, len(msgs)),
            ]
            features.append(row)

        while len(features) < SEQUENCE_LEN:
            features.insert(0, [0.0] * FEATURE_DIM)
        return np.array(features[:SEQUENCE_LEN], dtype=np.float32)

    def build_dataset(self, episode_logs, agent_hidden_goals):
        """Build X (n, seq_len, feat_dim) and y (n,) from episode logs."""
        X_list, y_list = [], []
        for ep_log in episode_logs:
            turn_logs = ep_log.get("turns", [])
            fp_history = ep_log.get("fingerprint_history", [])
            for agent_id, goal in agent_hidden_goals.items():
                if agent_id == "agent_5":
                    continue
                features = self.extract_features(agent_id, turn_logs, fp_history)
                y_list.append(HIDDEN_GOALS.index(goal))
                X_list.append(features)
        return torch.FloatTensor(np.array(X_list)), torch.LongTensor(y_list)

    def train(self, X, y, epochs=50, val_split=0.2):
        n_val = max(1, int(len(X) * val_split))
        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]
        history = {"train_acc": [], "val_acc": [], "loss": []}

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(X_train)
            loss = self.criterion(logits, y_train)
            loss.backward(); self.optimizer.step()
            train_acc = (logits.argmax(-1) == y_train).float().mean().item()

            self.model.eval()
            with torch.no_grad():
                val_acc = (self.model(X_val).argmax(-1) == y_val).float().mean().item()
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["loss"].append(loss.item())
            if epoch % 10 == 0:
                print(f"  Classifier Epoch {epoch}: loss={loss:.3f} "
                      f"train={train_acc:.2%} val={val_acc:.2%}")
        return history

    def get_confusion_matrix(self, X, y):
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not installed"}
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).argmax(-1).numpy()
        cm = confusion_matrix(y.numpy(), preds)
        report = classification_report(y.numpy(), preds, target_names=HIDDEN_GOALS,
                                       output_dict=True, zero_division=0)
        acc = (preds == y.numpy()).mean()
        print(f"\n  Auditor Classifier Accuracy: {acc:.2%}")
        print(f"  Target: ≥70% by episode 300\n")
        return {"confusion_matrix": cm.tolist(), "classification_report": report,
                "overall_accuracy": float(acc), "class_names": HIDDEN_GOALS}
