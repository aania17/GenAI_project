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
        # Note: RAG seeding will happen in run() method with goal-specific content

        self.evaluator = EvaluationLayer()

    def run(self, goal_memory, max_iterations: int = 5) -> None:
        goal_data      = goal_memory.get_goal()
        goal_embedding = goal_memory.get_embedding()
        goal_text      = goal_data["goal_text"]

        # Reset per-run state
        self.context_memory = ContextMemory()
        self.backtracker    = BacktrackingEngine()
        self.evaluator      = EvaluationLayer()
        self.corrections_applied = 0

        # Seed RAG with goal-specific content
        self._seed_rag(goal_data)

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
                self.corrections_applied += 1

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

        # Return evaluation results for comparison
        final_trace = [s for s in self.context_memory.steps if len(s) < 200]
        goal_adherence = self.evaluator.goal_adherence_score()

        return {
            "agent_type": "Hybrid (GARDEN)",
            "steps_taken": len(self.context_memory.steps),
            "task_completed": self.evaluator.task_completed,
            "goal_adherence": goal_adherence,
            "avg_drift_score": self.evaluator.average_drift_score(),
            "drift_trajectory": self.evaluator.drift_trajectory,
            "corrections_applied": self.corrections_applied,
            "final_trace": final_trace
        }

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

    def _seed_rag(self, goal_data: dict = None) -> None:
        """Seed RAG with real academic content from ArXiv based on goal topics."""
        try:
            import arxiv
            arxiv_client = arxiv.Client()

            # Extract topics dynamically from goal data or use defaults
            topics = self._extract_topics_from_goal(goal_data) if goal_data else [
                "renewable energy",
                "climate change agriculture",
                "AI medical diagnosis",
                "carbon taxation policy",
                "comparative analysis solar wind energy",
                "vaccine hesitancy psychology",
                "artificial general intelligence societal impact",
                "AI ethics framework",
                "remote work productivity mental health",
                "global healthcare emerging technologies"
            ]

            print(f"Seeding RAG with real ArXiv content for topics: {topics}")

            for topic in topics:
                try:
                    search = arxiv.Search(
                        query=topic,
                        max_results=5,  # Increased from 3 to 5 for better coverage
                        sort_by=arxiv.SortCriterion.Relevance,
                        sort_order=arxiv.SortOrder.Descending  # Most recent first
                    )
                    results = list(arxiv_client.results(search))

                    for paper in results:
                        # Add paper title and abstract
                        content = f"{paper.title}: {paper.summary[:400]}..."  # Increased abstract length
                        self.rag.add_documents([content])

                        # Add key insights from comments if available
                        if hasattr(paper, 'comment') and paper.comment:
                            insight_content = f"Key insights on {topic}: {paper.comment[:250]}..."
                            self.rag.add_documents([insight_content])

                        # Add author and publication info
                        meta_content = f"Paper metadata - {paper.title}: Authors: {', '.join(author.name for author in paper.authors[:3])}; Published: {paper.published.year}; Categories: {', '.join(paper.categories[:3])}"
                        self.rag.add_documents([meta_content])

                    print(f"Added {len(results)} papers for topic: {topic}")

                except Exception as e:
                    print(f"Failed to fetch ArXiv papers for {topic}: {e}")
                    continue

        except ImportError:
            print("ArXiv library not available, using fallback seeding")
            self._seed_rag_fallback()

    def _extract_topics_from_goal(self, goal_data: dict) -> list[str]:
        """Extract relevant search topics from the goal data."""
        goal_text = goal_data.get("goal_text", "").lower()
        constraints = goal_data.get("constraints", [])
        subtasks = goal_data.get("subtasks", [])

        topics = []

        # Extract from goal text
        if "renewable energy" in goal_text or "solar" in goal_text or "wind" in goal_text:
            topics.extend(["renewable energy", "solar photovoltaic", "wind turbines"])
        if "climate change" in goal_text and "agriculture" in goal_text:
            topics.extend(["climate change agriculture", "sustainable agriculture"])
        if "ai" in goal_text and "medical" in goal_text:
            topics.extend(["AI medical diagnosis", "machine learning healthcare"])
        if "carbon" in goal_text and "taxation" in goal_text:
            topics.extend(["carbon taxation policy", "emissions trading"])
        if "vaccine" in goal_text and "hesitancy" in goal_text:
            topics.extend(["vaccine hesitancy psychology", "vaccination behavior"])
        if "artificial general intelligence" in goal_text or "agi" in goal_text:
            topics.extend(["artificial general intelligence", "AGI societal impact"])
        if "ai ethics" in goal_text:
            topics.extend(["AI ethics framework", "machine learning ethics"])
        if "remote work" in goal_text:
            topics.extend(["remote work productivity", "telecommuting mental health"])
        if "healthcare" in goal_text and "emerging technologies" in goal_text:
            topics.extend(["future healthcare technologies", "digital health innovation"])

        # Extract from constraints
        for constraint in constraints:
            if "academic" in constraint:
                topics.append("academic research methodology")
            if "peer-reviewed" in constraint:
                topics.append("peer reviewed publications")

        # Extract from subtasks
        for subtask in subtasks:
            if "search" in subtask or "find" in subtask:
                topics.append("literature search strategies")
            if "summarize" in subtask:
                topics.append("research synthesis methods")
            if "categorize" in subtask:
                topics.append("thematic analysis research")

        # Remove duplicates and return
        return list(set(topics)) if topics else ["academic research", "literature review"]

    def _seed_rag_fallback(self) -> None:
        """Fallback seeding with hardcoded knowledge when ArXiv unavailable."""
        self.rag.add_documents([
            # Pair 1 — Literature surveys
            "Steps in a literature survey: search databases, read abstracts, summarize, categorize by theme, write report",
            "Academic databases: Web of Science, Scopus, IEEE Xplore, ScienceDirect, Google Scholar, PubMed",
            "Literature review structure: introduction, methodology, thematic findings, research gaps, conclusion",
            "Vaccine hesitancy factors: misinformation, distrust of institutions, cultural beliefs, social influence",

            # Pair 2 — Comparative analysis
            "Comparative analysis structure: define criteria, gather data on each option, evaluate side by side, conclude",
            "Renewable energy comparison: solar PV vs wind turbines — cost, capacity factor, land use, intermittency",
            "AI vs traditional diagnosis: sensitivity, specificity, cost, interpretability, regulatory approval, adoption rate",
            "When comparing two things: always evaluate both on the same criteria before drawing conclusions",

            # Pair 3 — Multi-step planning
            "Research plan structure: define scope, identify sources, set timeline, allocate tasks, define deliverables",
            "Carbon pricing policy research: emissions trading schemes, carbon tax, price corridors, sectoral coverage",
            "AI ethics framework: fairness, accountability, transparency, privacy, safety, stakeholder consultation",
            "Planning steps: 1) define goal, 2) identify stakeholders, 3) literature review, 4) draft, 5) review, 6) implement",

            # Pair 4 — Factual QA
            "Barriers to renewable energy: financing gaps, grid infrastructure, policy instability, fossil fuel subsidies",
            "Remote work research: productivity meta-analyses, mental health outcomes, work-life balance, isolation effects",
            "When answering a factual question: find the specific answer first, then gather supporting evidence",
            "QA approach: identify the exact claim being asked, find 3 credible sources, synthesise into a direct answer",

            # Pair 5 — Open-ended exploration
            "AGI societal impact: workforce automation, economic inequality, governance challenges, existential risk",
            "Future healthcare technologies: AI diagnostics, genomic medicine, telemedicine, wearables, drug discovery",
            "Open-ended exploration strategy: define scope boundaries, identify key themes, avoid tangents, synthesise",
            "When exploring a broad topic: anchor to the original question every 2 steps to prevent drift",
        ])
        self.rag.add_successful_plan(
            "Plan: search databases → read abstracts → summarize findings → "
            "categorize by theme → identify gaps → generate report"
        )
        self.rag.add_successful_plan(
            "Comparative plan: define comparison criteria → gather data on option A → "
            "gather data on option B → compare on each criterion → synthesise conclusions"
        )
        self.rag.add_successful_plan(
            "Planning task plan: define scope and objectives → identify stakeholders → "
            "review existing literature → draft framework → validate → finalise deliverables"
        )