"""
GARDEN — Goal-Anchored Retrieval-Driven Drift Evaluation Network

Usage:
    python main.py               # runs all tasks sequentially
    python main.py --task 0      # runs a single task by index
    python main.py --list        # lists all available tasks
"""

import argparse
from data.goal_processor    import GoalProcessor
from data.memory_store      import GoalMemory
from utils.embedding_engine import EmbeddingEngine
from utils.llm_engine       import LLMEngine
from core.agent_loop        import AgentLoop


TASKS = [
    {
        "name":              "Literature Survey — Renewable Energy",
        "user_input":        "Conduct a literature survey on renewable energy using academic sources",
        "environment_state": "research database",
    },
    {
        "name":              "Literature Survey — Climate Change",
        "user_input":        "Conduct an academic literature survey on the effects of climate change on agriculture",
        "environment_state": "research database",
    },
    {
        "name":              "Research Summary — AI in Healthcare",
        "user_input":        "Summarize recent academic research on AI applications in medical diagnosis",
        "environment_state": "research database",
    },
    {
        "name":              "Policy Research — Carbon Taxation",
        "user_input":        "Research and summarize academic papers on carbon taxation policy effectiveness",
        "environment_state": "policy database",
    },
    {
        "name":              "Comparative Survey — Solar vs Wind Energy",
        "user_input":        "Compare solar and wind energy in terms of cost, efficiency, and adoption using academic sources",
        "environment_state": "research database",
    },
]


def run_task(task: dict, embedder: EmbeddingEngine, llm: LLMEngine,
             task_index: int, total: int) -> None:
    print(f"\n{'#'*60}")
    print(f"  Task {task_index + 1}/{total}: {task['name']}")
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

    # Pass the pre-loaded LLM in — no reloading between tasks
    agent = AgentLoop(embedder, llm=llm, use_llm_judge=False)
    agent.run(goal_memory, max_iterations=5)


def run_all(embedder: EmbeddingEngine, llm: LLMEngine) -> None:
    for i, task in enumerate(TASKS):
        run_task(task, embedder, llm, i, len(TASKS))
    print(f"\n{'='*60}")
    print(f"  All {len(TASKS)} tasks completed.")
    print(f"{'='*60}")


def list_tasks() -> None:
    print("\nAvailable tasks:")
    for i, task in enumerate(TASKS):
        print(f"  [{i}] {task['name']}")
        print(f"       {task['user_input']}")


def main():
    parser = argparse.ArgumentParser(description="GARDEN Agent Runner")
    parser.add_argument("--task", type=int, default=None, help="Run single task by index")
    parser.add_argument("--list", action="store_true",    help="List all available tasks")
    args = parser.parse_args()

    if args.list:
        list_tasks()
        return

    # Load both models ONCE — shared across all tasks
    print("Loading models (one-time)...")
    embedder = EmbeddingEngine()
    llm      = LLMEngine()
    print("Models ready.\n")

    if args.task is not None:
        if args.task < 0 or args.task >= len(TASKS):
            print(f"Error: task index must be 0–{len(TASKS)-1}. Use --list to see options.")
            return
        run_task(TASKS[args.task], embedder, llm, args.task, len(TASKS))
    else:
        run_all(embedder, llm)


if __name__ == "__main__":
    main()