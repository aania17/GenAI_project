"""
Drift Detection Module
Continuously evaluates alignment using two metrics from the diagram:

  Metric 1 — Embedding Drift Score
    Drift(t) = 1 − cosine_similarity(G, R_t)
    where G = goal embedding, R_t = current reasoning step embedding

  Metric 2 — LLM Alignment Judge
    Prompt: rate alignment 1–5, then derive a normalised score
    Catches subtle semantic drift not detectable by embeddings alone

Final: weighted combination → Drift(t) > τ → flag drift
"""


class DriftDetector:
    """
    Implements the two-metric drift detection shown in the right panel
    of the architecture diagram.
    """

    # Task-relevant keyword heuristic (bonus score)
    TASK_KEYWORDS = [
        "renewable", "energy", "paper", "research", "survey",
        "summarize", "categorize", "report", "academic",
        "literature", "source", "review", "solar", "wind",
    ]

    def __init__(self, embedder, llm=None, threshold: float = 0.5):
        """
        Args:
            embedder:  EmbeddingEngine instance
            llm:       LLMEngine instance (enables Metric 2; optional)
            threshold: Drift(t) threshold τ — flag drift when exceeded
        """
        self.embedder  = embedder
        self.llm       = llm
        self.threshold = threshold

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def check_drift(self, goal_text: str, goal_embedding, step_text: str) -> dict:
        """
        Run both metrics and return a full result dict.
        """
        # Metric 1 — Embedding Drift Score
        m1 = self._metric1_embedding(goal_embedding, step_text)

        # Metric 2 — LLM Alignment Judge (if LLM is available)
        m2 = self._metric2_llm_judge(goal_text, step_text) if self.llm else None

        # Keyword heuristic bonus
        keyword_score = self._keyword_score(step_text)

        # Combined alignment score
        if m2 is not None:
            # Blend: 50% embedding sim, 20% keyword, 30% LLM judge
            final_score = 0.5 * m1["similarity"] + 0.2 * keyword_score + 0.3 * m2["normalised"]
        else:
            # Without LLM: 70% embedding sim, 30% keyword
            final_score = 0.7 * m1["similarity"] + 0.3 * keyword_score

        # Drift score = 1 − alignment (per architecture spec)
        drift_score   = 1.0 - final_score
        drift_detected = drift_score > self.threshold

        return {
            "step":            step_text,
            "similarity":      m1["similarity"],
            "keyword_score":   keyword_score,
            "llm_score":       m2["raw"] if m2 else None,
            "llm_normalised":  m2["normalised"] if m2 else None,
            "final_score":     final_score,
            "drift_score":     drift_score,
            "drift_detected":  drift_detected,
        }

    # ------------------------------------------------------------------ #
    #  Metric 1 — Embedding Drift Score                                   #
    # ------------------------------------------------------------------ #

    def _metric1_embedding(self, goal_embedding, step_text: str) -> dict:
        step_embedding = self.embedder.embed(step_text)
        similarity     = self.embedder.cosine_similarity(goal_embedding, step_embedding)
        return {"similarity": similarity, "drift": 1.0 - similarity}

    # ------------------------------------------------------------------ #
    #  Metric 2 — LLM Alignment Judge                                     #
    # ------------------------------------------------------------------ #

    def _metric2_llm_judge(self, goal_text: str, step_text: str) -> dict:
        """
        Prompt the LLM to rate alignment on a 1–5 scale.
        Normalise to 0.0–1.0.
        """
        prompt = (
            f"Original goal: {goal_text}\n"
            f"Current reasoning step: {step_text}\n\n"
            f"Rate how well this step aligns with the original goal.\n"
            f"Reply with a single integer from 1 (completely unrelated) to 5 (perfectly aligned).\n"
            f"Rating:"
        )
        response = self.llm.generate(prompt, max_length=10).strip()

        # Parse the first digit found in the response
        rating = self._parse_rating(response)
        normalised = (rating - 1) / 4.0   # map 1–5 → 0.0–1.0

        return {"raw": rating, "normalised": normalised}

    def _parse_rating(self, text: str) -> int:
        for char in text:
            if char.isdigit():
                val = int(char)
                if 1 <= val <= 5:
                    return val
        return 3   # neutral fallback

    # ------------------------------------------------------------------ #
    #  Keyword heuristic                                                   #
    # ------------------------------------------------------------------ #

    def _keyword_score(self, text: str) -> float:
        text_lower = text.lower()
        hits = sum(1 for kw in self.TASK_KEYWORDS if kw in text_lower)
        return hits / len(self.TASK_KEYWORDS)