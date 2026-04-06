"""
Layer 4 — Reasoning Engine (Llama 3.2 version)

With Llama 3.2 we no longer need heavy prompt sanitisation.
The model follows instructions cleanly.
"""

from prompts.prompt_templates import (
    structured_reasoning_prompt,
    goal_anchoring_prompt,
    reflection_prompt,
)


class ReasoningEngine:

    ANCHOR_EVERY_N_STEPS = 3

    def __init__(self, llm):
        self.llm = llm

    def generate_step(self, goal_data: dict, context: dict) -> str:
        goal_text    = goal_data["goal_text"]
        constraints  = goal_data.get("constraints", [])
        steps_so_far = context.get("steps", [])

        # Filter out correction reminder text — keep only real action steps
        real_steps = [s for s in steps_so_far
                      if len(s) < 200 and not s.startswith("ORIGINAL GOAL")]

        context_clean = dict(context)
        context_clean["steps"]    = real_steps
        context_clean["subtasks"] = goal_data.get("subtasks", [])

        if real_steps and len(real_steps) % self.ANCHOR_EVERY_N_STEPS == 0:
            prompt = goal_anchoring_prompt(goal_text, constraints, real_steps)
        else:
            prompt = structured_reasoning_prompt(goal_text, context_clean)

        response = self.llm.generate(prompt, max_length=80)
        return self._clean(response)

    def reflect(self, goal_text: str, step: str) -> str:
        """YES/NO alignment check — Llama 3.2 handles this reliably."""
        prompt = reflection_prompt(goal_text, step)
        result = self.llm.generate(prompt, max_length=10).strip().lower()
        return "yes" if "yes" in result else "no"

    def _clean(self, text: str) -> str:
        """
        Light cleanup for Llama 3.2 output.
        The model follows instructions well so this is minimal.
        """
        result = text.strip()

        # Remove any accidental prompt echo (rare with Llama but possible)
        for prefix in ["GOAL:", "STEP:", "ONE SENTENCE:", "ACTION:"]:
            if result.upper().startswith(prefix):
                result = result[len(prefix):].strip()

        # Take first sentence/line only
        for delimiter in ["\n", ". ", "! ", "? "]:
            if delimiter in result:
                candidate = result.split(delimiter)[0].strip()
                if len(candidate) > 15:   # don't take a fragment
                    result = candidate
                    break

        # Fallback if something went wrong
        if len(result) < 5:
            return "Search academic databases for relevant papers on the topic"

        return result