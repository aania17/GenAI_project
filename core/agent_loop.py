"""
Layer 1 — Agent Loop
Orchestrates the 6-layer GARDEN architecture.
Enhanced with dynamic RAG seeding based on goal content.
Real-world data integration for scientific validity.
"""

import json
from typing import Dict, List, Tuple


class AgentLoop:
    """Main orchestrator for the GARDEN 6-layer architecture."""

    def __init__(self):
        """Initialize all layers of the GARDEN system."""
        try:
            from core.goal_decomposer import GoalDecomposer
            from core.reasoning_engine import ReasoningEngine
            from core.executor import Executor
            from core.rag_module import RAGModule
            from core.drift_detector import DriftDetector
            from core.correction_module import CorrectionModule
            from core.evaluation_layer import EvaluationLayer
            
            self.goal_decomposer = GoalDecomposer()
            self.reasoning_engine = ReasoningEngine()
            self.executor = Executor()
            self.rag_module = RAGModule()
            self.drift_detector = DriftDetector()
            self.correction_module = CorrectionModule()
            self.evaluation_layer = EvaluationLayer()
            self.context_memory = {}
        except ImportError as e:
            print(f"Warning: Some modules not available: {e}")

    def run(self, goal_text: str, agent_mode: str = "hybrid") -> Dict:
        """
        Execute the GARDEN pipeline with real-world data integration.
        
        Args:
            goal_text: The user's research goal
            agent_mode: One of 'baseline', 'embedding_only', 'judge_only', 'hybrid'
        
        Returns:
            Dictionary containing execution results and evaluations
        """
        result = {
            "goal": goal_text,
            "agent_mode": agent_mode,
            "steps": [],
            "observations": [],
            "drift_detected": False,
            "corrections_applied": [],
            "final_evaluation": None
        }

        try:
            # Layer 1: Goal Decomposition
            subtasks = self.goal_decomposer.decompose(goal_text)
            result["steps"] = subtasks

            # Layer 2: Seed RAG with real-world data
            self._seed_rag(goal_text)

            # Layer 3-6: Process each subtask through reasoning, execution, drift detection, correction
            for idx, subtask in enumerate(subtasks):
                # Layer 3: Reasoning Engine Plans Action Steps
                action_plan = self.reasoning_engine.reason(subtask)
                
                # Layer 4: Executor Performs Actions
                observation = self.executor.execute(action_plan, goal_text)
                result["observations"].append({
                    "subtask": subtask,
                    "action": action_plan,
                    "observation": observation
                })

                # Layer 5: Drift Detection
                drift_detected = self.drift_detector.detect(observation, goal_text)
                if drift_detected:
                    result["drift_detected"] = True
                    # Layer 6: Correction Module
                    correction = self.correction_module.correct(observation, subtask)
                    result["corrections_applied"].append(correction)
                    observation = correction

            # Evaluation Layer: Assess Performance
            evaluation = self.evaluation_layer.evaluate(result, agent_mode)
            result["final_evaluation"] = evaluation

        except Exception as e:
            result["error"] = str(e)
            print(f"Error in agent loop: {e}")

        return result

    def _seed_rag(self, goal_data: str) -> None:
        """
        Seed the RAG module with real-world data based on goal content.
        Dynamically fetches ArXiv papers and web content for the specific goal.
        
        Args:
            goal_data: The research goal text to extract topics from
        """
        topics = self._extract_topics_from_goal(goal_data)
        
        for topic in topics:
            try:
                # Search ArXiv for academic papers
                papers = self.executor._search_arxiv(topic, max_results=5)
                for paper in papers:
                    # Add paper to RAG knowledge base
                    doc = {
                        "id": paper.entry_id,
                        "title": paper.title,
                        "content": paper.summary,
                        "source": "arxiv",
                        "topic": topic
                    }
                    self.rag_module.add_document(doc)
                    
            except Exception as e:
                print(f"Error seeding ArXiv for {topic}: {e}")
            
            try:
                # Search web for additional content
                web_results = self.executor._search_web(topic, max_results=3)
                for result in web_results:
                    doc = {
                        "id": result.get('href', ''),
                        "title": result.get('title', ''),
                        "content": result.get('body', ''),
                        "source": "web",
                        "topic": topic
                    }
                    self.rag_module.add_document(doc)
                    
            except Exception as e:
                print(f"Error seeding web for {topic}: {e}")

    def _extract_topics_from_goal(self, goal: str) -> List[str]:
        """
        Extract key research topics from the goal text.
        Returns list of search queries for RAG seeding.
        
        Args:
            goal: The research goal text
        
        Returns:
            List of topic strings to search for
        """
        # Normalize and clean goal text
        cleaned_goal = goal.lower()
        
        # Remove common prefixes
        prefixes = [
            "conduct a literature survey on ",
            "conduct an academic literature survey on ",
            "summarize recent academic research on ",
            "research and summarize academic papers on ",
            "conduct ", "research ", "analyze ", "investigate ", "study "
        ]
        
        for prefix in prefixes:
            if cleaned_goal.startswith(prefix):
                cleaned_goal = cleaned_goal[len(prefix):]
                break
        
        # Extract main topic (first 100 chars)
        topics = [cleaned_goal[:100]]
        
        # Add specific keyword-based topics for better retrieval
        keywords = [
            "machine learning", "deep learning", "artificial intelligence", 
            "neural network", "algorithm", "optimization", "classification",
            "healthcare", "medical", "diagnosis", "prediction"
        ]
        
        for keyword in keywords:
            if keyword in cleaned_goal:
                topics.append(keyword)
        
        # Return top 3 unique topics
        return list(set(topics))[:3]

    def compare_modes(self, goal_text: str) -> Dict:
        """
        Compare performance across all four agent modes.
        Used for ablation study to evaluate each component's contribution.
        
        Args:
            goal_text: The research goal to test
        
        Returns:
            Dictionary with comparison results for all modes
        """
        modes = ["baseline", "embedding_only", "judge_only", "hybrid"]
        comparison_results = {}
        
        for mode in modes:
            print(f"\n▶ Running {mode.upper()} mode...")
            result = self.run(goal_text, agent_mode=mode)
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
            analysis["analysis"]["evaluation_scores"][mode] = result.get("final_evaluation", {}).get("score", 0) if result.get("final_evaluation") else 0
        
        return analysis

    def get_context_memory(self) -> Dict:
        """Retrieve the current context memory state."""
        return self.context_memory

    def clear_context_memory(self) -> None:
        """Clear the context memory."""
        self.context_memory = {}
