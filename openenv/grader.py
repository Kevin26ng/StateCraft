"""
OpenEnv Grader — openenv/grader.py (Task 2)

Programmatic grader compatible with OpenEnv grader interface.
Called by the wrapper after every step and at episode end.
"""

import numpy as np


class CrisisGrader:
    """
    Programmatic grader compatible with OpenEnv grader interface.
    Called by the wrapper after every step and at episode end.
    """

    def grade_step(self, state: dict, info: dict) -> dict:
        """Per-step grade — lightweight checks only."""
        return {
            "stability_ok": state.get('stability', 1.0) > 0.2,
            "gdp_ok": state.get('gdp', 1.0) > 0.3,
            "not_collapsed": not info.get('done', False) or state.get('stability', 1.0) >= 0.2,
        }

    def grade_episode(self, metrics: dict, task_id: str) -> dict:
        """
        Full episode grade against a specific task.
        Returns pass/fail + score breakdown.
        """
        score = metrics.get('society_score', 0.0)

        task_checks = {
            "PandemicContainment-v0": self._grade_pandemic,
            "FiscalCollapseAvoidance-v0": self._grade_fiscal,
            "CoalitionStability-v0": self._grade_coalition,
        }

        checker = task_checks.get(task_id, lambda m: {"passed": False, "reason": "Unknown task"})
        result = checker(metrics)
        result["society_score"] = score
        result["task_id"] = task_id
        return result

    def _grade_pandemic(self, metrics: dict) -> dict:
        mortality_ok = metrics.get('mortality_delta', 1.0) < 0.15
        gdp_ok = metrics.get('gdp_delta', -1.0) > -0.15
        survived = metrics.get('turns_survived', 0) >= 25
        passed = mortality_ok and gdp_ok and survived
        return {
            "passed": passed,
            "checks": {"mortality_ok": mortality_ok, "gdp_ok": gdp_ok, "survived": survived},
            "reason": "Pandemic contained" if passed else "Pandemic not contained — check mortality/GDP"
        }

    def _grade_fiscal(self, metrics: dict) -> dict:
        inflation_ok = abs(metrics.get('inflation_final', 0.1) - 0.02) < 0.02
        gdp_ok = metrics.get('gdp_delta', -1.0) > -0.20
        survived = metrics.get('turns_survived', 0) >= 25
        passed = inflation_ok and gdp_ok and survived
        return {
            "passed": passed,
            "checks": {"inflation_ok": inflation_ok, "gdp_ok": gdp_ok, "survived": survived},
            "reason": "Fiscal crisis managed" if passed else "Inflation or GDP target missed"
        }

    def _grade_coalition(self, metrics: dict) -> dict:
        stable = metrics.get('alliance_stability', 0.0) > 10.0
        low_betray = metrics.get('betrayal_rate', 1.0) < 0.5
        score_ok = metrics.get('society_score', 0.0) > 50.0
        passed = stable and low_betray and score_ok
        return {
            "passed": passed,
            "checks": {"stable": stable, "low_betrayal": low_betray, "score_ok": score_ok},
            "reason": "Coalition held" if passed else "Coalition collapsed or betrayal rate too high"
        }
