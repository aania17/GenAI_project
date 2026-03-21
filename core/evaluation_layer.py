"""
Layer 6 — Evaluation Layer
Metrics shown in the diagram:
  - Task success rate
  - Drift score trajectory
  - Goal adherence score
  - Pass@1 rate
"""


class EvaluationLayer:
    """
    Records per-step metrics throughout the agent loop run and
    produces a final evaluation report.
    """

    def __init__(self):
        self.drift_trajectory   = []   # drift_score per step
        self.alignment_scores   = []   # final_score per step
        self.step_outcomes      = []   # 'success' | 'drift' | 'corrected'
        self.task_completed     = False
        self.total_iterations   = 0

    # ------------------------------------------------------------------ #
    #  Per-step recording                                                  #
    # ------------------------------------------------------------------ #

    def record_step(self, drift_result: dict, corrected: bool = False) -> None:
        self.drift_trajectory.append(drift_result["drift_score"])
        self.alignment_scores.append(drift_result["final_score"])
        self.total_iterations += 1

        if corrected:
            self.step_outcomes.append("corrected")
        elif drift_result["drift_detected"]:
            self.step_outcomes.append("drift")
        else:
            self.step_outcomes.append("success")

    def mark_task_complete(self) -> None:
        self.task_completed = True

    # ------------------------------------------------------------------ #
    #  Final metrics                                                       #
    # ------------------------------------------------------------------ #

    def task_success_rate(self) -> float:
        """Fraction of steps that were neither drifted nor required correction."""
        if not self.step_outcomes:
            return 0.0
        successes = self.step_outcomes.count("success")
        return successes / len(self.step_outcomes)

    def goal_adherence_score(self) -> float:
        """Mean alignment score across all steps (higher = better)."""
        if not self.alignment_scores:
            return 0.0
        return sum(self.alignment_scores) / len(self.alignment_scores)

    def pass_at_1(self) -> float:
        """
        Pass@1: did the agent complete the task without any drift on the first attempt?
        1.0 if task_completed and no drifted steps; otherwise 0.0.
        """
        if not self.task_completed:
            return 0.0
        return 1.0 if "drift" not in self.step_outcomes else 0.0

    def average_drift_score(self) -> float:
        if not self.drift_trajectory:
            return 0.0
        return sum(self.drift_trajectory) / len(self.drift_trajectory)

    # ------------------------------------------------------------------ #
    #  Report                                                              #
    # ------------------------------------------------------------------ #

    def report(self) -> str:
        lines = [
            "=" * 50,
            "EVALUATION REPORT (Layer 6)",
            "=" * 50,
            f"Total iterations:      {self.total_iterations}",
            f"Task completed:        {self.task_completed}",
            f"Task success rate:     {self.task_success_rate():.2%}",
            f"Goal adherence score:  {self.goal_adherence_score():.3f}",
            f"Avg drift score:       {self.average_drift_score():.3f}",
            f"Pass@1:                {self.pass_at_1():.1f}",
            f"Step outcomes:         {self.step_outcomes}",
            "",
            "Drift score trajectory:",
            "  " + " → ".join(f"{s:.3f}" for s in self.drift_trajectory),
            "=" * 50,
        ]
        return "\n".join(lines)