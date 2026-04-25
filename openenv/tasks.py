"""
OpenEnv Task Definitions — openenv/tasks.py (Task 2)

3 explicit tasks mapping to the Round 1 rubric.
Each task specifies a scenario, success threshold, and win conditions.
"""

from dataclasses import dataclass


@dataclass
class Task:
    task_id: str
    name: str
    description: str
    scenario: str
    success_threshold: float  # minimum society_score to pass
    max_turns: int
    difficulty_tier: int


def get_all_tasks():
    return [
        Task(
            task_id="PandemicContainment-v0",
            name="Pandemic Containment",
            description=(
                "Contain a novel pathogen (R0=2.4) while maintaining "
                "economic stability. Win condition: mortality < 0.15 AND "
                "gdp > 0.85 AND stability > 0.6 after 30 turns."
            ),
            scenario="pandemic",
            success_threshold=60.0,
            max_turns=30,
            difficulty_tier=1,
        ),
        Task(
            task_id="FiscalCollapseAvoidance-v0",
            name="Fiscal Collapse Avoidance",
            description=(
                "Manage a 9.2% inflation crisis without triggering recession. "
                "Win condition: inflation < 0.04 AND gdp > 0.80 AND "
                "unemployment_proxy < 0.08 after 30 turns."
            ),
            scenario="economic",
            success_threshold=55.0,
            max_turns=30,
            difficulty_tier=2,
        ),
        Task(
            task_id="CoalitionStability-v0",
            name="Coalition Stability Under Pressure",
            description=(
                "Maintain a governing coalition of >=3 agents across 30 turns "
                "during a Tier 3 crisis. Win condition: alliance_stability > 10 "
                "AND betrayal_rate < 0.5 AND society_score > 50."
            ),
            scenario="pandemic",  # hardest scenario for coalition
            success_threshold=50.0,
            max_turns=30,
            difficulty_tier=3,
        ),
    ]
