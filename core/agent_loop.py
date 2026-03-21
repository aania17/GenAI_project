"""
Agent Reasoning Loop — integrates all 6 layers.
"""

from core.goal_decomposer     import GoalDecomposer
from core.reasoning_engine    import ReasoningEngine
from core.context_memory      import ContextMemory
from core.executor            import Executor
from core.rag_module          import RAGModule
from core.drift_detector      import DriftDetector
from core.correction_module   import CorrectionModule
from core.backtracking_engine import BacktrackingEngine
from core.evaluation_layer    import EvaluationLayer
from utils.llm_engine         import LLMEngine


class AgentLoop:
    def __init__(self, embedder, llm=None, use_llm_judge: bool = True):
        # Accept a pre-loaded LLM or create one if not provided
        self.llm = llm if llm is not None else LLMEngine()

        self.goal_decomposer   = GoalDecomposer()
        self.reasoning_engine  = ReasoningEngine(self.llm)
        self.context_memory    = ContextMemory()
        self.executor          = Executor()
        self.backtracker       = BacktrackingEngine()

        llm_judge = self.llm if use_llm_judge else None
        self.drift_detector    = DriftDetector(embedder, llm=llm_judge, threshold=0.5)
        self.correction_module = CorrectionModule()

        self.rag = RAGModule(embedder)
        self._seed_rag()

        self.evaluator = EvaluationLayer()

    def run(self, goal_memory, max_iterations: int = 5) -> None:
        goal_data      = goal_memory.get_goal()
        goal_embedding = goal_memory.get_embedding()
        goal_text      = goal_data["goal_text"]

        # Reset per-run state
        self.context_memory = ContextMemory()
        self.backtracker    = BacktrackingEngine()
        self.evaluator      = EvaluationLayer()

        print("\n" + "=" * 60)
        print("  GARDEN — Goal-Anchored Retrieval-Driven Drift Evaluation")
        print("=" * 60)
        print(f"  Goal:        {goal_text}")
        print(f"  Constraints: {goal_data.get('constraints', [])}")

        subtasks = self.goal_decomposer.decompose(goal_data)
        self.context_memory.set_task_node(goal_text)
        self.context_memory.set_subtasks(subtasks)
        print(f"\n  Subtasks: {subtasks}")
        print("=" * 60)

        for i in range(max_iterations):
            print(f"\n{'─'*60}")
            print(f"  Iteration {i + 1} / {max_iterations}")
            print(f"{'─'*60}")

            self.backtracker.save_checkpoint(
                step_index       = i,
                subtask_index    = min(i, len(subtasks) - 1),
                context_snapshot = self.context_memory.get_context(),
            )

            # RAG retrieval
            last_step = self.context_memory.get_last_step() or ""
            retrieved = self.rag.retrieve(goal_text, current_reasoning=last_step)

            context = self.context_memory.get_context()
            context["retrieved"] = retrieved

            # Generate step
            step = self.reasoning_engine.generate_step(goal_data, context)
            print(f"\n  Step generated: {step}")

            # Reflection
            reflection = self.reasoning_engine.reflect(goal_text, step)
            print(f"  Reflection:     {reflection}")

            # Completion check
            if self._is_complete(step, context):
                self.context_memory.add_step(step)
                self.evaluator.mark_task_complete()
                print("\n  ✅ Task complete — stopping loop.")
                break

            # Execute
            observation = self.executor.execute(step, goal_text)
            print(f"  Observation:    {observation}")
            self.context_memory.add_step(step)
            self.context_memory.add_observation(observation)
            self.rag.add_reasoning_trace(f"{step} → {observation}")

            # Drift detection
            drift_result = self.drift_detector.check_drift(goal_text, goal_embedding, step)
            print(f"\n  Drift score:    {drift_result['drift_score']:.3f}  (threshold: {self.drift_detector.threshold})")
            print(f"  Alignment:      {drift_result['final_score']:.3f}")
            if drift_result.get("llm_score") is not None:
                print(f"  LLM judge:      {drift_result['llm_score']}/5")
            print(f"  Drift detected: {drift_result['drift_detected']}")

            corrected = False

            if drift_result["drift_detected"]:
                self.backtracker.record_drift()

                corrected_step = self.correction_module.apply_correction(
                    goal_data = goal_data,
                    step      = step,
                    context   = self.context_memory.get_context(),
                    llm       = self.llm,
                )
                print(f"  Corrected step: {corrected_step}")
                self.context_memory.replace_last_step(corrected_step)
                corrected = True

                if self.backtracker.should_backtrack():
                    checkpoint = self.backtracker.backtrack()
                    if checkpoint:
                        print(f"  Rolled back to checkpoint at step {checkpoint['step_index']}")
                    else:
                        new_subtasks = self.backtracker.force_replan(goal_data)
                        self.context_memory.set_subtasks(new_subtasks)
            else:
                self.backtracker.clear_drift()

            self.evaluator.record_step(drift_result, corrected=corrected)

        print("\n" + self.evaluator.report())
        print("\n  Final reasoning trace:")
        for idx, s in enumerate(self.context_memory.steps, 1):
            print(f"    {idx}. {s}")

    def _is_complete(self, step: str, context: dict) -> bool:
        """
        Task is complete when the step signals a final deliverable
        and at least 3 real steps have already been taken.
        Expanded to catch Llama 3.2's varied phrasing.
        """
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

    def _seed_rag(self) -> None:
        self.rag.add_documents([
            "Steps in a literature survey: search databases, read abstracts, summarize, categorize by theme, write report",
            "Renewable energy sources: solar photovoltaic, wind turbines, hydroelectric, geothermal, biomass",
            "Academic databases: Web of Science, Scopus, IEEE Xplore, ScienceDirect, Google Scholar",
            "Literature review structure: introduction, methodology, thematic findings, research gaps, conclusion",
            "AI in healthcare: machine learning for diagnosis, medical imaging, clinical decision support",
            "Carbon taxation: carbon pricing, emissions trading, policy effectiveness, economic impact",
            "Categorization themes: technology, policy, economics, grid integration, storage, adoption",
        ])
        self.rag.add_successful_plan(
            "Plan: search databases → read abstracts → summarize findings → "
            "categorize by theme → identify gaps → generate report"
        )