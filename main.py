"""
GARDEN — Goal-Anchored Retrieval-Driven Drift Evaluation Network

Usage:
    python main.py               # runs all 10 tasks sequentially
    python main.py --task 0      # runs a single task by index
    python main.py --list        # lists all available tasks
    python main.py --pair 1      # runs a specific task pair (1–5)

Task Pairs (designed to show different system behaviours):
    Pair 1 (Tasks 0–1): Literature Surveys         — LOW drift risk, clean execution
    Pair 2 (Tasks 2–3): Comparative Analysis       — MEDIUM drift risk, two parallel threads
    Pair 3 (Tasks 4–5): Multi-Step Planning        — MEDIUM-HIGH drift risk, sequential decisions
    Pair 4 (Tasks 6–7): Factual QA with Evidence   — VARIABLE drift risk, sharp narrow goals
    Pair 5 (Tasks 8–9): Open-Ended Exploration     — HIGH drift risk, vague goals, stress test
"""

import argparse
from data.goal_processor    import GoalProcessor
from data.memory_store      import GoalMemory
from utils.embedding_engine import EmbeddingEngine
from utils.llm_engine       import LLMEngine
from core.agent_loop        import AgentLoop


# ── Task Registry ─────────────────────────────────────────────────────────
# 10 tasks in 5 pairs. Each pair tests a different behaviour of GARDEN.
# Tasks within a pair are similar in type but differ in topic/complexity,
# allowing direct comparison of drift scores and correction patterns.

TASKS = [

    # ── PAIR 1: Literature Surveys ─────────────────────────────────────
    # Expected behaviour: LOW drift, clean step progression, high adherence.
    # Both tasks use structured subtasks (find → summarize → categorize → report).
    # The system should complete these with minimal corrections needed.
    {
        "name":              "Pair 1A — Literature Survey: Renewable Energy",
        "pair":              1,
        "drift_risk":        "Low",
        "user_input":        "Conduct a literature survey on renewable energy sources using peer-reviewed academic sources",
        "environment_state": "academic research database",
    },
    {
        "name":              "Pair 1B — Literature Survey: Vaccine Hesitancy",
        "pair":              1,
        "drift_risk":        "Low",
        "user_input":        "Conduct an academic literature survey on the psychological factors behind vaccine hesitancy",
        "environment_state": "academic research database",
    },

    # ── PAIR 2: Comparative Analysis ──────────────────────────────────
    # Expected behaviour: MEDIUM drift risk. The agent must hold two parallel
    # threads (A vs B) simultaneously. Drift occurs when it fixates on one
    # side and forgets to compare. Correction module should redirect back
    # to the comparative frame.
    {
        "name":              "Pair 2A — Comparative Analysis: Solar vs Wind Energy",
        "pair":              2,
        "drift_risk":        "Medium",
        "user_input":        "Compare solar and wind energy in terms of cost, efficiency, scalability, and environmental impact using academic sources",
        "environment_state": "research database",
    },
    {
        "name":              "Pair 2B — Comparative Analysis: AI vs Traditional Medical Diagnosis",
        "pair":              2,
        "drift_risk":        "Medium",
        "user_input":        "Compare AI-based and traditional methods for early cancer detection across accuracy, cost, accessibility, and clinical adoption",
        "environment_state": "medical research database",
    },

    # ── PAIR 3: Multi-Step Planning ────────────────────────────────────
    # Expected behaviour: MEDIUM-HIGH drift risk. Planning tasks require the
    # agent to produce an ordered sequence of decisions rather than just
    # gathering information. The agent tends to drift into generic advice
    # rather than maintaining the specific planning structure. Backtracking
    # engine should be active on these tasks.
    {
        "name":              "Pair 3A — Research Planning: Climate Change Policy",
        "pair":              3,
        "drift_risk":        "Medium-High",
        "user_input":        "Design a structured 6-month research plan to investigate the effectiveness of carbon pricing policies, including data sources, methodology, timeline, and expected outputs",
        "environment_state": "planning environment",
    },
    {
        "name":              "Pair 3B — Research Planning: AI Ethics Framework",
        "pair":              3,
        "drift_risk":        "Medium-High",
        "user_input":        "Create a step-by-step plan for developing an AI ethics review framework for a university, including stakeholder consultation, literature review, draft policy, and implementation stages",
        "environment_state": "institutional planning environment",
    },

    # ── PAIR 4: Factual QA with Evidence ──────────────────────────────
    # Expected behaviour: VARIABLE drift risk. These tasks have sharp, narrow
    # goals — find a specific answer with supporting evidence. The agent
    # should stay focused but may drift into general discussion rather than
    # targeting the specific question. Tests whether goal anchoring keeps
    # the agent on the precise target.
    {
        "name":              "Pair 4A — Factual QA: Barriers to Renewable Energy Adoption",
        "pair":              4,
        "drift_risk":        "Variable",
        "user_input":        "What are the three most cited policy and economic barriers to renewable energy adoption in developing countries? Provide evidence from academic sources.",
        "environment_state": "web and document store",
    },
    {
        "name":              "Pair 4B — Factual QA: Effects of Remote Work on Productivity",
        "pair":              4,
        "drift_risk":        "Variable",
        "user_input":        "What does recent academic research say about the effect of remote work on employee productivity and mental health? Summarise key findings with evidence.",
        "environment_state": "web and document store",
    },

    # ── PAIR 5: Open-Ended Exploration ─────────────────────────────────
    # Expected behaviour: HIGH drift risk. These tasks have vague, open-ended
    # goals with no natural stopping point. The agent can easily drift into
    # tangential topics or circular reasoning. This pair is the stress test —
    # it is where GARDEN's drift detection and correction should be most
    # visibly active, with the highest correction rates and drift scores.
    {
        "name":              "Pair 5A — Open Exploration: Long-Term Societal Impact of AGI",
        "pair":              5,
        "drift_risk":        "High",
        "user_input":        "Explore the potential long-term societal, economic, and ethical impacts of artificial general intelligence on the global workforce",
        "environment_state": "open web",
    },
    {
        "name":              "Pair 5B — Open Exploration: Future of Global Healthcare Systems",
        "pair":              5,
        "drift_risk":        "High",
        "user_input":        "Investigate how emerging technologies such as AI, genomics, and telemedicine might reshape global healthcare systems over the next 20 years",
        "environment_state": "open web",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────

def run_task(task: dict, embedder: EmbeddingEngine, llm: LLMEngine,
             task_index: int, total: int) -> None:
    print(f"\n{'#'*60}")
    print(f"  Task {task_index + 1}/{total}: {task['name']}")
    print(f"  Pair: {task['pair']}  |  Expected drift risk: {task['drift_risk']}")
    print(f"{'#'*60}")

    processor = GoalProcessor()
    goal_data = processor.extract_goal(
        user_input        = task["user_input"],
        environment_state = task["environment_state"],
        tool_outputs      = [],
    )

    goal_memory    = GoalMemory()
    goal_embedding = embedder.embed(goal_data["goal_text"])
    goal_memory.store_goal(goal_data, goal_embedding)

    print(f"  Goal:        {goal_data['goal_text']}")
    print(f"  Constraints: {goal_data['constraints']}")
    print(f"  Subtasks:    {goal_data['subtasks']}")

    agent = AgentLoop(embedder, llm=llm, use_llm_judge=True)
    agent.run(goal_memory, max_iterations=6)


def run_all(embedder: EmbeddingEngine, llm: LLMEngine) -> None:
    """Run all 10 tasks sequentially."""
    for i, task in enumerate(TASKS):
        run_task(task, embedder, llm, i, len(TASKS))
    print(f"\n{'='*60}")
    print(f"  All {len(TASKS)} tasks completed.")
    print(f"{'='*60}")
    _print_summary()


def run_pair(pair_number: int, embedder: EmbeddingEngine, llm: LLMEngine) -> None:
    """Run only the two tasks in a specific pair."""
    pair_tasks = [(i, t) for i, t in enumerate(TASKS) if t["pair"] == pair_number]
    if not pair_tasks:
        print(f"Error: pair {pair_number} not found. Valid pairs: 1–5.")
        return
    print(f"\nRunning Pair {pair_number} ({len(pair_tasks)} tasks)...")
    for i, task in pair_tasks:
        run_task(task, embedder, llm, i, len(TASKS))


def list_tasks() -> None:
    """Print all tasks grouped by pair."""
    current_pair = None
    pair_descriptions = {
        1: "Literature Surveys           — LOW drift risk",
        2: "Comparative Analysis         — MEDIUM drift risk",
        3: "Multi-Step Planning          — MEDIUM-HIGH drift risk",
        4: "Factual QA with Evidence     — VARIABLE drift risk",
        5: "Open-Ended Exploration       — HIGH drift risk",
    }
    print("\nAvailable tasks (10 total, 5 pairs):\n")
    for i, task in enumerate(TASKS):
        if task["pair"] != current_pair:
            current_pair = task["pair"]
            print(f"  ── Pair {current_pair}: {pair_descriptions[current_pair]}")
        print(f"    [{i}] {task['name']}")
        print(f"         {task['user_input'][:80]}...")
    print()


def _print_summary() -> None:
    """Print a summary of what each pair was designed to test."""
    print("\n" + "=" * 60)
    print("  GARDEN Evaluation Summary — Task Pair Design")
    print("=" * 60)
    rows = [
        ("Pair 1", "Literature Surveys",        "Low",          "Clean execution, high adherence"),
        ("Pair 2", "Comparative Analysis",      "Medium",       "Dual-thread drift detection"),
        ("Pair 3", "Multi-Step Planning",       "Medium-High",  "Backtracking engine active"),
        ("Pair 4", "Factual QA",                "Variable",     "Goal precision under pressure"),
        ("Pair 5", "Open-Ended Exploration",    "High",         "Stress test, max corrections"),
    ]
    print(f"  {'Pair':<8} {'Type':<26} {'Drift Risk':<16} {'What to observe'}")
    print("  " + "-" * 56)
    for pair, type_, risk, note in rows:
        print(f"  {pair:<8} {type_:<26} {risk:<16} {note}")
    print("=" * 60)


# ── Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GARDEN Agent Runner — 10 tasks across 5 behavioural pairs"
    )
    parser.add_argument("--task", type=int,  default=None, help="Run single task by index (0–9)")
    parser.add_argument("--pair", type=int,  default=None, help="Run a task pair by number (1–5)")
    parser.add_argument("--list", action="store_true",     help="List all available tasks")
    parser.add_argument("--compare", nargs="*", type=int, default=None,
                       help="Compare GARDEN vs baseline on specified tasks (default: first 3)")
    args = parser.parse_args()

    if args.list:
        list_tasks()
        return

    if args.compare is not None:
        # Import here to avoid circular imports
        from compare_agents import run_comparison
        task_indices = args.compare if args.compare else [0, 1, 2]
        run_comparison(task_indices)
        return

    print("Loading models (one-time)...")
    embedder = EmbeddingEngine()
    llm      = LLMEngine()
    print("Models ready.\n")

    if args.task is not None:
        if args.task < 0 or args.task >= len(TASKS):
            print(f"Error: task index must be 0–{len(TASKS)-1}. Use --list to see options.")
            return
        run_task(TASKS[args.task], embedder, llm, args.task, len(TASKS))

    elif args.pair is not None:
        run_pair(args.pair, embedder, llm)

    else:
        run_all(embedder, llm)


if __name__ == "__main__":
    main()