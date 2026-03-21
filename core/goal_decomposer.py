"""
Layer 4 — Goal Decomposer
"""

class GoalDecomposer:
    def decompose(self, goal_data: dict, llm=None) -> list[str]:
        subtasks = goal_data.get("subtasks", [])
        if subtasks:
            return subtasks
        if llm:
            return self._llm_decompose(goal_data["goal_text"], llm)
        return ["understand task", "gather information", "process results", "generate report"]

    def _llm_decompose(self, goal_text: str, llm) -> list[str]:
        prompt = f"Break this goal into 4 ordered subtasks.\nGoal: {goal_text}\nList exactly 4 subtasks, one per line, no numbering."
        response = llm.generate(prompt, max_length=100)
        tasks = [line.strip() for line in response.split("\n") if line.strip()]
        return tasks[:4] if tasks else ["search", "summarize", "categorize", "report"]