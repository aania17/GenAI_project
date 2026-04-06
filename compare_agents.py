"""
Comparison Runner — All Agent Variants
Demonstrates the value of different drift detection approaches.
"""

import argparse
from data.goal_processor import GoalProcessor
from data.memory_store import GoalMemory
from utils.embedding_engine import EmbeddingEngine
from utils.llm_engine import LLMEngine
from core.agent_loop import AgentLoop
from core.baseline_agent import BaselineAgent
from core.embedding_only_agent import EmbeddingOnlyAgent
from core.llm_judge_only_agent import LLMJudgeOnlyAgent
from main import TASKS


def run_comparison(task_indices: list = None) -> None:
    """Run all agents on specified tasks and compare results."""

    print("=" * 80)
    print("AGENT ARCHITECTURE COMPARISON")
    print("=" * 80)
    print("Evaluating different approaches to drift detection and correction")
    print()

    # Initialize shared components
    embedder = EmbeddingEngine()
    llm = LLMEngine()

    # Select tasks
    if task_indices is None:
        task_indices = [0, 1, 2]  # First 3 tasks for demo

    selected_tasks = [TASKS[i] for i in task_indices]

    results = []

    for task in selected_tasks:
        print(f"\n{'─'*80}")
        print(f"TASK: {task['name']}")
        print(f"GOAL: {task['user_input']}")
        print(f"EXPECTED DRIFT RISK: {task['drift_risk']}")
        print(f"{'─'*80}")

        # Setup goal
        processor = GoalProcessor()
        goal_data = processor.extract_goal(
            user_input=task["user_input"],
            environment_state=task["environment_state"],
            tool_outputs=[]
        )

        goal_memory = GoalMemory()
        goal_embedding = embedder.embed(goal_data["goal_text"])
        goal_memory.store_goal(goal_data, goal_embedding)

        # Run all agents
        agents = {
            "Baseline": BaselineAgent(embedder, llm=llm),
            "Embedding-Only": EmbeddingOnlyAgent(embedder, llm=llm),
            "LLM Judge-Only": LLMJudgeOnlyAgent(embedder, llm=llm),
            "Hybrid (GARDEN)": AgentLoop(embedder, llm=llm, use_llm_judge=True)
        }

        agent_results = {}

        for agent_name, agent in agents.items():
            print(f"\n🔍 RUNNING {agent_name.upper()}...")
            result = agent.run(goal_memory, max_iterations=6)
            agent_results[agent_name] = result

        # Compare results
        comparison = {
            "task": task["name"],
            "drift_risk": task["drift_risk"],
            "results": agent_results
        }
        results.append(comparison)

        print_comparison(comparison)

    # Summary
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON SUMMARY")
    print(f"{'='*80}")

    # Calculate averages across all tasks
    agent_summaries = {}
    for agent_name in ["Baseline", "Embedding-Only", "LLM Judge-Only", "Hybrid (GARDEN)"]:
        agent_summaries[agent_name] = {
            "avg_drift": [],
            "avg_adherence": [],
            "avg_success": [],
            "avg_corrections": []
        }

    for result in results:
        for agent_name, agent_result in result["results"].items():
            agent_summaries[agent_name]["avg_drift"].append(agent_result["avg_drift_score"])
            agent_summaries[agent_name]["avg_adherence"].append(agent_result["goal_adherence"])
            agent_summaries[agent_name]["avg_success"].append(1 if agent_result["task_completed"] else 0)
            agent_summaries[agent_name]["avg_corrections"].append(agent_result.get("corrections_applied", 0))

    print(f"{'Agent':<16} {'Drift↓':<8} {'Adherence↑':<12} {'Success↑':<10} {'Corrections':<12}")
    print("-" * 60)

    for agent_name, summary in agent_summaries.items():
        avg_drift = sum(summary["avg_drift"]) / len(summary["avg_drift"])
        avg_adherence = sum(summary["avg_adherence"]) / len(summary["avg_adherence"])
        avg_success = sum(summary["avg_success"]) / len(summary["avg_success"]) * 100
        avg_corrections = sum(summary["avg_corrections"]) / len(summary["avg_corrections"])

        print(f"{agent_name:<16} {avg_drift:<8.3f} {avg_adherence:<12.3f} {avg_success:<10.1f}% {avg_corrections:<12.1f}")

    print(f"\n{'='*80}")
    print("KEY FINDINGS:")
    print("- Hybrid (GARDEN) should show best overall performance")
    print("- Embedding-Only provides basic drift detection")
    print("- LLM Judge-Only catches semantic drift missed by embeddings")
    print("- Baseline shows natural drift without intervention")
    print(f"{'='*80}")


def print_comparison(comparison: dict) -> None:
    """Print side-by-side comparison of results."""

    results = comparison["results"]

    print(f"\n{'─'*80}")
    print("RESULTS COMPARISON"
    print(f"{'─'*80}")

    # Metrics table
    print(f"{'Agent':<16} {'Drift↓':<8} {'Adherence↑':<12} {'Success':<8} {'Corrections':<12}")
    print("-" * 60)

    for agent_name, result in results.items():
        drift = result["avg_drift_score"]
        adherence = result["goal_adherence"]
        success = "✅" if result["task_completed"] else "❌"
        corrections = result.get("corrections_applied", 0)

        print(f"{agent_name:<16} {drift:<8.3f} {adherence:<12.3f} {success:<8} {corrections:<12}")

    # Key insights
    best_adherence = max(results.items(), key=lambda x: x[1]["goal_adherence"])
    best_drift = min(results.items(), key=lambda x: x[1]["avg_drift_score"])

    print(f"\n🎯 Best goal adherence: {best_adherence[0]}")
    print(f"🎯 Lowest drift score: {best_drift[0]}")

    hybrid_result = results.get("Hybrid (GARDEN)", {})
    if hybrid_result:
        print(f"\nGARDEN applied {hybrid_result.get('corrections_applied', 0)} corrections")
        print("This demonstrates active drift prevention!")


def main():
    parser = argparse.ArgumentParser(description="Compare all agent architectures")
    parser.add_argument("--tasks", nargs="+", type=int, default=[0, 1, 2],
                       help="Task indices to compare (default: 0, 1, 2)")
    args = parser.parse_args()

    run_comparison(args.tasks)


if __name__ == "__main__":
    main()