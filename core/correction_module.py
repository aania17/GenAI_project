"""
Correction Module
Triggered when Drift(t) > τ.

Two interventions:
  1. Goal Reminder     — short redirect back to the goal
  2. Plan Regeneration — concrete next subtask step
"""

from prompts.prompt_templates import replan_prompt


class CorrectionModule:

    TASK_SIGNALS = [
        "search", "find", "read", "summarize", "categorize", "review",
        "paper", "source", "report", "research", "academic", "analyze",
        "extract", "synthesize", "gather", "collect", "identify",
    ]

    def apply_correction(self, goal_data: dict, step: str, context: dict, llm=None) -> str:
        print("\n⚠️  DRIFT DETECTED — APPLYING CORRECTION")

        goal_text = goal_data["goal_text"]
        subtasks  = goal_data.get("subtasks", [])
        steps_done = [s for s in context.get("steps", []) if len(s) < 120 and not s.startswith("Goal:")]

        if self._has_task_signal(step):
            print("   Strategy: Goal Reminder")
            return self._goal_reminder(goal_text, subtasks, steps_done)
        else:
            print("   Strategy: Plan Regeneration")
            return self._plan_regeneration(goal_text, subtasks, steps_done, llm)

    def _goal_reminder(self, goal_text: str, subtasks: list, steps_done: list) -> str:
        """
        Return the NEXT concrete subtask action — short and clean.
        This is what gets stored as the corrected step, so it must look
        like a real action, not a long reminder paragraph.
        """
        next_subtask = (
            subtasks[len(steps_done)] if len(steps_done) < len(subtasks)
            else subtasks[-1] if subtasks
            else "continue research"
        )

        action_map = {
            "find papers":        "Search academic databases for relevant papers",
            "search databases":   "Search Web of Science and Scopus for papers",
            "read abstracts":     "Read and extract key points from paper abstracts",
            "summarize":          "Summarize key findings from the collected papers",
            "extract findings":   "Extract and list the main research findings",
            "categorize":         "Group papers by theme and research area",
            "synthesize":         "Synthesize findings into a coherent summary",
            "generate report":    "Write a structured literature review report",
            "report":             "Write a final structured report",
            "gather information": "Collect academic sources on the topic",
            "process results":    "Analyze and organize research results",
            "understand task":    "Define the research scope and methodology",
        }
        return action_map.get(next_subtask.lower(), f"Continue: {next_subtask} for goal: {goal_text[:60]}")

    def _plan_regeneration(self, goal_text: str, subtasks: list, steps_done: list, llm=None) -> str:
        """
        Use LLM to generate a corrected step, or fall back to the subtask map.
        """
        if llm:
            prompt   = replan_prompt(goal_text, steps_done[-1] if steps_done else "")
            response = llm.generate(prompt, max_length=60).strip()
            # Only use it if it looks like a real action (not a single letter or echo)
            if len(response) > 10 and not response.lower().startswith("goal:"):
                return response

        # Fallback — return the first uncompleted subtask as a concrete action
        return self._goal_reminder(goal_text, subtasks, steps_done)

    def _has_task_signal(self, step: str) -> bool:
        step_lower = step.lower()
        return any(signal in step_lower for signal in self.TASK_SIGNALS)