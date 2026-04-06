"""
GARDEN: Goal-Aware Research Data Evaluation Network
Main entry point with real-world data integration.
"""

import argparse
import json
from core.agent_loop import AgentLoop
from core.evaluation_layer import EvaluationLayer


def run_single_goal(goal_text: str, agent_mode: str = "hybrid") -> Dict:
    """Run agent loop for a single goal."""
    agent = AgentLoop()
    result = agent.run(goal_text, agent_mode)
    return result


def compare_agent_modes(goal_text: str) -> Dict:
    """
    Compare performance across all four agent modes:
    - baseline: No drift detection, basic execution
    - embedding_only: Uses embedding-based drift detection
    - judge_only: Uses LLM judge for drift detection
    - hybrid: Combined embedding + judge approach
    """
    modes = ["baseline", "embedding_only", "judge_only", "hybrid"]
    comparison_results = {}
    
    for mode in modes:
        print(f"\n▶ Running {mode.upper()} mode...")
        agent = AgentLoop()
        result = agent.run(goal_text, agent_mode=mode)
        comparison_results[mode] = result
    
    # Analyze comparative results
    analysis = {
        "goal": goal_text,
        "results": comparison_results,
        "analysis": {
            "drift_detection_effectiveness": {},
            "correction_efficiency": {},
            "evaluation_scores": {}
        }
    }
    
    for mode in modes:
        result = comparison_results[mode]
        analysis["analysis"]["drift_detection_effectiveness"][mode] = result.get("drift_detected", False)
        analysis["analysis"]["correction_efficiency"][mode] = len(result.get("corrections_applied", []))
        analysis["analysis"]["evaluation_scores"][mode] = result.get("final_evaluation", {}).get("score", 0)
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="GARDEN: Real-World Data Integration System")
    parser.add_argument("--goal", type=str, default="Conduct a comprehensive survey on machine learning applications in healthcare",
                        help="Research goal for the agent")
    parser.add_argument("--mode", type=str, default="hybrid", 
                        choices=["baseline", "embedding_only", "judge_only", "hybrid"],
                        help="Agent mode for execution")
    parser.add_argument("--compare", type=int, default=0,
                        help="Compare all agent modes (1=yes, 0=no)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("GARDEN: Goal-Aware Research Data Evaluation Network")
    print("Real-World Data Integration Enabled")
    print("="*70)
    
    if args.compare:
        print(f"\n🔬 Conducting ablation study comparison...")
        results = compare_agent_modes(args.goal)
        print(json.dumps(results, indent=2))
    else:
        print(f"\n🎯 Goal: {args.goal}")
        print(f"📊 Mode: {args.mode}")
        print(f"\n▶ Executing agent loop...")
        
        results = run_single_goal(args.goal, args.mode)
        
        print("\n" + "="*70)
        print("EXECUTION RESULTS")
        print("="*70)
        print(json.dumps(results, indent=2))
    
    print("\n✅ Execution complete!")


if __name__ == "__main__":
    main()
