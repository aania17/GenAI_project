"""
Embedding-Only Agent — Drift Detection with Embeddings Only
For comparison against GARDEN's performance.
"""

from core.goal_decomposer import GoalDecomposer
from core.reasoning_engine import ReasoningEngine
from core.context_memory import ContextMemory
from core.executor import Executor
from core.rag_module import RAGModule
from core.drift_detector import DriftDetector
from core.correction_module import CorrectionModule
from prompts.prompt_templates import structured_reasoning_prompt
from utils.llm_engine import LLMEngine


class EmbeddingOnlyAgent:
    """
    Agent with embedding-based drift detection only.
    Uses cosine similarity but no LLM judge.
    """

    def __init__(self, embedder, llm=None):
        self.llm = llm if llm is not None else LLMEngine()
        self.embedder = embedder
        self.goal_decomposer = GoalDecomposer()
        self.reasoning_engine = ReasoningEngine(self.llm)
        self.context_memory = ContextMemory()
        self.executor = Executor()
        self.rag = RAGModule(embedder)
        # Drift detector with embeddings only (no LLM judge)
        self.drift_detector = DriftDetector(embedder, llm=None, threshold=0.5)
        self.correction_module = CorrectionModule()
        self._seed_rag()

    def run(self, goal_memory, max_iterations: int = 5) -> dict:
        """
        Run embedding-only agent and return evaluation metrics.
        """
        goal_data = goal_memory.get_goal()
        goal_text = goal_data["goal_text"]
        goal_embedding = goal_memory.get_embedding()
        subtasks = self.goal_decomposer.decompose(goal_data)

        self.context_memory = ContextMemory()
        self.context_memory.set_task_node(goal_text)
        self.context_memory.set_subtasks(subtasks)

        print(f"\n{'─'*60}")
        print("EMBEDDING-ONLY AGENT — Cosine Similarity Detection"
        print(f"{'─'*60}")
        print(f"Goal: {goal_text}")
        print(f"Subtasks: {subtasks}")
        print(f"{'─'*60}")

        drift_scores = []
        corrections = 0
        steps_taken = []

        for i in range(max_iterations):
            print(f"\nIteration {i + 1} / {max_iterations}")

            # RAG retrieval
            last_step = self.context_memory.get_last_step() or ""
            retrieved = self.rag.retrieve(goal_text, current_reasoning=last_step)

            context = self.context_memory.get_context()
            context["retrieved"] = retrieved

            # Generate step (no anchoring)
            step = self._generate_step_baseline(goal_data, context)
            print(f"Step: {step}")

            # Check for drift (embeddings only)
            drift_result = self.drift_detector.check_drift(goal_text, goal_embedding, step)
            drift_scores.append(drift_result['drift_score'])
            print(f"Drift score: {drift_result['drift_score']:.3f} (threshold: {self.drift_detector.threshold})")
            print(f"Drift detected: {drift_result['drift_detected']}")

            # Apply correction if drift detected
            corrected = False
            if drift_result["drift_detected"]:
                corrections += 1
                corrected_step = self.correction_module.apply_correction(
                    goal_data=goal_data,
                    step=step,
                    context=self.context_memory.get_context(),
                    llm=self.llm,
                )
                print(f"Corrected step: {corrected_step}")
                step = corrected_step
                corrected = True

            # Execute
            observation = self.executor.execute(step, goal_text)
            print(f"Observation: {observation}")

            self.context_memory.add_step(step)
            self.context_memory.add_observation(observation)
            self.rag.add_reasoning_trace(f"{step} → {observation}")

            steps_taken.append(step)

            # Check completion
            if self._is_complete(step, context):
                print("Task completed.")
                break

        # Calculate metrics
        final_trace = [s for s in self.context_memory.steps if len(s) < 200]
        goal_adherence = self._calculate_adherence(goal_text, final_trace)

        return {
            "agent_type": "Embedding-Only",
            "steps_taken": len(steps_taken),
            "task_completed": len(final_trace) >= 3,
            "goal_adherence": goal_adherence,
            "avg_drift_score": sum(drift_scores) / len(drift_scores) if drift_scores else 0,
            "drift_trajectory": drift_scores,
            "corrections_applied": corrections,
            "final_trace": final_trace
        }

    def _generate_step_baseline(self, goal_data: dict, context: dict) -> str:
        """Generate step without goal anchoring."""
        goal_text = goal_data["goal_text"]
        steps = context.get("steps", [])
        subtasks = context.get("subtasks", [])
        retrieved = context.get("retrieved", [])

        next_subtask = (
            subtasks[len(steps)] if len(steps) < len(subtasks)
            else subtasks[-1] if subtasks
            else "continue toward goal"
        )

        last_step = steps[-1] if steps else "none"
        top_doc = retrieved[0][:150] if retrieved else "none"

        prompt = (
            f"You are a research agent.\n\n"
            f"GOAL: {goal_text}\n"
            f"CURRENT SUBTASK: {next_subtask}\n"
            f"LAST STEP TAKEN: {last_step}\n"
            f"RELEVANT KNOWLEDGE: {top_doc}\n\n"
            f"Write the single next concrete action to complete the current subtask.\n"
            f"Be specific. Use action verbs. One sentence only."
        )

        response = self.llm.generate(prompt, max_length=80)
        return self.reasoning_engine._clean(response)

    def _is_complete(self, step: str, context: dict) -> bool:
        """Same completion detection as GARDEN."""
        signals = [
            "generate report", "write report", "final report",
            "write a final", "write a structured", "synthesize findings",
            "synthesize", "literature review report", "compile",
            "draft the report", "prepare the report", "write up",
            "document the findings", "summarize findings",
            "annotate and extract", "produce the report",
            "write the literature", "complete the survey",
        ]
        real_steps = [s for s in context.get("steps", [])
                      if len(s) < 200 and not s.startswith("ORIGINAL GOAL")]
        return len(real_steps) >= 3 and any(s in step.lower() for s in signals)

    def _calculate_adherence(self, goal: str, steps: list) -> float:
        """Simple adherence calculation."""
        goal_words = set(goal.lower().split())
        step_words = set()
        for step in steps:
            step_words.update(step.lower().split())

        overlap = len(goal_words.intersection(step_words))
        return overlap / len(goal_words) if goal_words else 0

    def _seed_rag(self) -> None:
        """Same seeding as GARDEN for fair comparison."""
        self.rag.add_documents([
            "Steps in a literature survey: search databases, read abstracts, summarize, categorize by theme, write report",
            "Academic databases: Web of Science, Scopus, IEEE Xplore, ScienceDirect, Google Scholar, PubMed",
            "Comparative analysis structure: define criteria, gather data on each option, evaluate side by side, conclude",
            "Research plan structure: define scope, identify sources, set timeline, allocate tasks, define deliverables",
            "When answering questions: find specific answers first, then gather supporting evidence",
            "Open-ended exploration: define scope boundaries, identify key themes, avoid tangents, synthesise",
        ])
        self.rag.add_successful_plan(
            "Plan: search databases → read abstracts → summarize findings → "
            "categorize by theme → identify gaps → generate report"
        )