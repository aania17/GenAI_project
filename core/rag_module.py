"""
Layer 5 — Retrieval-Augmented Generation (RAG)
Three components from the diagram:
  - Vector Store  (FAISS index)
  - Retriever     (top-k nearest neighbour search)
  - Context Injector (formats retrieved docs for prompt injection)
"""

import faiss
import numpy as np


class RAGModule:
    """
    Stores reasoning traces, task docs, and successful plans in a FAISS
    vector store, then retrieves the most relevant entries to ground each
    reasoning step.

    query = goal + current_reasoning   (as shown in the diagram)
    """

    def __init__(self, embedder, dim: int = 384):
        self.embedder  = embedder
        self.index     = faiss.IndexFlatL2(dim)
        self.documents = []   # raw text entries

    # ------------------------------------------------------------------ #
    #  Vector Store — add documents                                        #
    # ------------------------------------------------------------------ #

    def add_documents(self, docs: list[str]) -> None:
        """Add a batch of documents to the vector store."""
        for doc in docs:
            vec = self.embedder.embed(doc)
            self.index.add(np.array([vec], dtype="float32"))
            self.documents.append(doc)

    def add_reasoning_trace(self, trace: str) -> None:
        """Store a completed reasoning trace for future retrieval."""
        self.add_documents([f"[TRACE] {trace}"])

    def add_successful_plan(self, plan: str) -> None:
        """Store a plan that successfully completed a task."""
        self.add_documents([f"[PLAN] {plan}"])

    # ------------------------------------------------------------------ #
    #  Retriever                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(self, goal: str, current_reasoning: str = "", k: int = 3) -> list[str]:
        """
        Retrieve top-k relevant documents.
        query = goal + current_reasoning  (matches diagram)
        """
        if len(self.documents) == 0:
            return []

        query = f"{goal} {current_reasoning}".strip()
        query_vec = np.array([self.embedder.embed(query)], dtype="float32")

        k_actual = min(k, len(self.documents))
        _, indices = self.index.search(query_vec, k_actual)

        return [self.documents[i] for i in indices[0] if i < len(self.documents)]

    # ------------------------------------------------------------------ #
    #  Context Injector                                                    #
    # ------------------------------------------------------------------ #

    def inject_context(self, retrieved: list[str]) -> str:
        """
        Formats retrieved documents for injection into the reasoning prompt.
        Maps to the 'Context Injector' component in the diagram.
        """
        if not retrieved:
            return "No relevant context retrieved."
        lines = [f"  [{i+1}] {doc}" for i, doc in enumerate(retrieved)]
        return "Retrieved context:\n" + "\n".join(lines)