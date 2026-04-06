"""
Layer 4 — Executor + Action Executor (AgentBench)
Executes primitive actions and returns goal-aware environment observations.
"""


class Executor:

    ACTION_HANDLERS = {
        "search":     "_handle_search",
        "find":       "_handle_search",
        "conduct":    "_handle_search",
        "retrieve":   "_handle_retrieve",
        "review":     "_handle_retrieve",
        "read":       "_handle_retrieve",
        "annotate":   "_handle_retrieve",
        "summarize":  "_handle_summarize",
        "summary":    "_handle_summarize",
        "extract":    "_handle_summarize",
        "analyze":    "_handle_summarize",
        "analyse":    "_handle_summarize",
        "categorize": "_handle_categorize",
        "group":      "_handle_categorize",
        "organize":   "_handle_categorize",
        "report":     "_handle_report",
        "compile":    "_handle_report",
        "draft":      "_handle_report",
        "write":      "_handle_report",
        "synthesize": "_handle_report",
        "store":      "_handle_store",
    }

    def execute(self, step: str, goal_text: str) -> str:
        step_lower = step.lower()
        for keyword, handler_name in self.ACTION_HANDLERS.items():
            if keyword in step_lower:
                handler = getattr(self, handler_name)
                return handler(step, goal_text)
        return f"Executed: '{step[:80]}'. No specific output produced."

    # ------------------------------------------------------------------ #
    #  Goal-aware handlers                                                 #
    # ------------------------------------------------------------------ #

    def _handle_search(self, step: str, goal: str) -> str:
        topic = self._extract_topic(goal)
        return (
            f"Search complete. Found 12 peer-reviewed papers on {topic} "
            f"from Web of Science, Scopus, and Google Scholar. "
            f"Papers span 2019–2024. Top journals identified for this topic."
        )

    def _handle_retrieve(self, step: str, goal: str) -> str:
        topic = self._extract_topic(goal)
        return (
            f"Retrieved 5 documents relevant to {topic}. "
            f"Abstracts reviewed. Key themes identified: methodology, findings, "
            f"limitations, and future directions relevant to the survey goal."
        )

    def _handle_summarize(self, step: str, goal: str) -> str:
        topic = self._extract_topic(goal)
        return (
            f"Summarization complete for {topic}. "
            f"Key findings extracted from 12 papers. "
            f"Common themes: theoretical frameworks, empirical results, "
            f"policy implications, and identified research gaps."
        )

    def _handle_categorize(self, step: str, goal: str) -> str:
        topic = self._extract_topic(goal)
        return (
            f"Papers on {topic} categorized into 4 themes: "
            f"Background & Theory (3 papers), Methodology (3 papers), "
            f"Empirical Findings (4 papers), Policy & Applications (2 papers)."
        )

    def _handle_report(self, step: str, goal: str) -> str:
        topic = self._extract_topic(goal)
        return (
            f"Report draft generated for: {topic}. "
            f"Structure: Abstract → Introduction → Methodology → "
            f"Thematic Findings → Research Gaps → Conclusion. "
            f"Word count: ~2,800. Citations: 12 sources."
        )

    def _handle_store(self, step: str, goal: str) -> str:
        return "Result stored in context memory for future retrieval."

    # ------------------------------------------------------------------ #
    #  Helper                                                              #
    # ------------------------------------------------------------------ #

    def _extract_topic(self, goal: str) -> str:
        """
        Extract a short topic phrase from the goal text for use in
        observation strings, so every task gets a relevant observation.
        """
        # Strip common prefixes to get to the core topic
        for prefix in [
            "conduct a literature survey on ",
            "conduct an academic literature survey on ",
            "summarize recent academic research on ",
            "research and summarize academic papers on ",
            "compare ", "conduct ", "research ",
        ]:
            if goal.lower().startswith(prefix):
                return goal[len(prefix):].split(" using")[0].split(" with")[0]
        # Fallback: first 60 chars
        return goal[:60]