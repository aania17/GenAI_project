"""
Layer 1 — Input Layer
"""

class GoalProcessor:
    CONSTRAINT_MAP = {
        "academic":   "academic sources only",
        "summary":    "structured summary required",
        "report":     "report generation required",
        "peer":       "peer-reviewed sources",
        "recent":     "recent publications preferred",
        "categorize": "categorization of findings required",
    }

    SUBTASK_TEMPLATES = {
        "survey":   ["find papers", "summarize", "categorize", "generate report"],
        "research": ["search databases", "read abstracts", "extract findings", "synthesize"],
        "default":  ["understand task", "gather information", "process results", "report"],
    }

    def extract_goal(self, user_input: str, environment_state: str = "", tool_outputs: list = None) -> dict:
        constraints = self._extract_constraints(user_input)
        subtasks    = self._extract_subtasks(user_input)
        return {
            "goal_text":         user_input.strip(),
            "constraints":       constraints,
            "subtasks":          subtasks,
            "environment_state": environment_state,
            "tool_outputs":      tool_outputs or [],
        }

    def _extract_constraints(self, text: str) -> list[str]:
        text_lower = text.lower()
        return [label for keyword, label in self.CONSTRAINT_MAP.items() if keyword in text_lower]

    def _extract_subtasks(self, text: str) -> list[str]:
        text_lower = text.lower()
        for key, tasks in self.SUBTASK_TEMPLATES.items():
            if key in text_lower:
                return tasks
        return self.SUBTASK_TEMPLATES["default"]