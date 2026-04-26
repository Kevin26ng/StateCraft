# StateCraft: Inside Our Crisis Governance Simulator

Have you ever wondered how complex political decisions are made during a crisis? How do different government departments balance public welfare against their own institutional goals? 

We built **StateCraft**, a multi-agent reinforcement learning (MARL) environment, to simulate exactly this. By placing AI agents in high-stakes crisis scenarios (like pandemics, economic crashes, and natural disasters), we can observe how they negotiate, form coalitions, and sometimes, inevitably, clash.

![StateCraft Simulator UI](screen.png)

---

## 🌍 The OpenEnv RL Environment

StateCraft isn't just a basic sandbox; it's built on the **OpenEnv** standard. It simulates the complex, high-stakes dynamics of crisis governance. The environment forces our AI agents to make difficult trade-offs under pressure. 

Every action has a cost. For instance, imposing a lockdown might cost `-0.02` in resources, while budget allocations scale linearly. Furthermore, the environment features **Joint Action Synergies**—meaning if the Health Authority imposes a lockdown while the Finance Minister pushes a stimulus package, the synergy might save the economy, but if their actions are mismatched, it could lead to a disastrous crash.

To keep things realistic and prevent our agents from simply "memorizing" the perfect path, we inject Gaussian noise into all deterministic scenario equations. You can't just min-max your way out of a crisis!

![Environment / Dashboard](<screen 1.png>)

---

## 🎭 Meet the 6 Agents

We trained six distinct AI agents simultaneously using a **Shared PPO Actor-Critic** network. By using role-embeddings, the single network learns specialized behaviors efficiently. 

But here is where it gets interesting: **Hidden Goals**.

Each agent has a public duty (what they are supposed to do) and a hidden goal (what they are secretly trained to achieve). This dual-incentive structure is the engine of our simulation's political tension.

| Role | Public Duty | The Hidden Goal |
|------|-------------|-----------------|
| 🏛️ **Finance Minister** | Maximize GDP & fiscal health | Protect economic growth above all—delay lockdowns, resist emergency budgets. |
| 📢 **Political Pressure** | Represent public opinion | Engineer coalition collapse by turn 25 to trigger early elections. |
| 🏦 **Monetary Authority**| Control inflation | Protect banking sector bond yields at the expense of broader recovery. |
| 🏥 **Health Authority** | Minimize mortality | Maintain institutional authority above operational effectiveness. |
| 🚒 **Disaster Response** | Coordinate emergency logistics | Expand military budget share, centralize crisis command. |
| 👁️ **The Auditor** | Monitor & flag misalignment | *No hidden goal* — purely acts to infer and catch other agents misbehaving. |

![Agents Overview](<screen 1-1.png>)

---

## 🧠 Advanced Mechanics & Emergent Behavior

What happens when you let these agents loose? We built a passive observer algorithm to automatically identify spontaneous societal behaviors, such as bilateral coalitions forming, manufactured crises, and coordinated scapegoating. 

Agents also possess **Semantic Memory**. They compress event summaries across episodes and use them to inform future decisions, bypassing standard context limits. 

![Memory/Emergent Behavior Screenshot Placeholder](Paste your metrics or emergent behavior screenshot here)

### The Auditor's Counterfactuals
Perhaps the coolest feature is our independent **Auditor agent**. The Auditor monitors the simulation, flags misaligned behavior, and runs *shadow simulations*. It then generates plain-English explanations of *what would have happened* if the offending agent had acted purely in the public's interest. 

### The Reward Stack
To enforce these complex trade-offs, StateCraft uses a comprehensive 13-signal reward stack, clipped to `[-10, 10]` per turn to stabilize PPO training. An agent's base reward is a weighted blend: **70% Public Role Performance** and **30% Hidden Goal Completion**. 

This includes everything from a basic `survival_bonus` to a `societal_collapse` terminal penalty of `-100`. 

![Reward Metrics Screenshot Placeholder](Paste your training metrics or reward chart screenshot here)

---

## 📈 Training Outcomes

Over 2,000 GRPO (Group Relative Policy Optimisation) episodes, our policy progresses through three distinct phases: noisy exploration (ep. 1-500), rapid learning (ep. 501-1750), and convergence (ep. 1751-2000). The final checkpoint is saved as a LoRA adapter to seed the auditor rollouts.

![Training Phases Placeholder](Paste your training phases screenshot here)

![alt text](<Screenshot 2026-04-26 162716.png>)

Ultimately, the simulation achieves a strong Best Reward of 8.91 and an impressive Auditor Accuracy of 87.3%, successfully demonstrating the agents' learning curves.

![Training Report Placeholder](Paste your training report screenshot here)



![alt text](<Screenshot 2026-04-26 162724.png>)

---

StateCraft represents a new frontier in modeling political and crisis dynamics using reinforcement learning. By giving agents realistic constraints and conflicting incentives, we've created a simulator that doesn't just solve problems—it negotiates them.
