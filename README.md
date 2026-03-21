# GARDEN
## Goal-Anchored Retrieval-Driven Drift Evaluation Network

A research implementation of a long-horizon LLM agent that detects and corrects **context drift** — the phenomenon where an AI agent gradually stops following its original goal as reasoning steps accumulate.

---

## The Problem

When LLM agents run for many steps, they increasingly condition their outputs on recent context rather than the original instruction. The agent may shift its reasoning toward unintended topics, follow incorrect intermediate plans, or lose alignment with the original objective entirely. This is called **goal drift** or **context drift**, and it is a serious open problem in agentic AI systems.

GARDEN addresses this through three mechanisms: persistent goal memory, continuous drift monitoring, and automatic corrective planning.

---

## Architecture Overview

The system is organised into 6 layers that mirror the architecture diagram from the accompanying report.

```
Layer 1 — Input Layer
  User Goal + Task Context + Environment State + Tool Outputs
        │
        ▼
Layer 2 — Goal Memory Layer
  Persistent: Goal Text + Goal Embedding (G) + Constraints
        │
        ▼
Layer 3 — Prompt Engineering Layer
  Goal Anchoring Prompt | Reflection Prompt | Structured Reasoning Template
        │
        ▼
Layer 4 — Agent Workflow Layer  ◄─────────────────────────────┐
  Goal Decomposer → Context Memory → Executor                 │
       ↑                   │              │                   │
  Backtracking Engine ◄────┘    Environment Observation       │
                                          │                   │
                           ┌──────────────┘                   │
                           ▼                                  │
              Drift Detection Module                          │
              Metric 1: Drift(t) = 1 − cosine_sim(G, R_t)    │
              Metric 2: LLM Alignment Judge (1–5)             │
                           │                                  │
                    Drift(t) > τ ?                            │
                           │ YES                              │
                           ▼                                  │
              Correction Module                               │
              Goal Reminder | Plan Regeneration ──────────────┘
        │
        ▼
Layer 5 — Retrieval-Augmented Generation (RAG)
  Vector Store | Retriever | Context Injector
  query = goal + current_reasoning
        │
        ▼
Layer 6 — Evaluation Layer
  Task success rate | Drift score trajectory
  Goal adherence score | Pass@1
```

---

## Project Structure

```
genai_project/
├── main.py                        ← Entry point, task registry, CLI
├── requirements.txt
│
├── data/
│   ├── __init__.py
│   ├── goal_processor.py          ← Layer 1: extracts goal, constraints, subtasks
│   └── memory_store.py            ← Layer 2: persistent goal + embedding storage
│
├── prompts/
│   ├── __init__.py
│   └── prompt_templates.py        ← Layer 3: all prompt templates
│
├── core/
│   ├── __init__.py
│   ├── goal_decomposer.py         ← Layer 4: breaks goal into ordered subtasks
│   ├── reasoning_engine.py        ← Layer 4: generates structured reasoning steps
│   ├── context_memory.py          ← Layer 4: stores steps, observations, subtasks
│   ├── executor.py                ← Layer 4: executes actions, returns observations
│   ├── backtracking_engine.py     ← Layer 4: checkpoint + rollback on repeated drift
│   ├── rag_module.py              ← Layer 5: FAISS vector store + retriever + injector
│   ├── drift_detector.py          ← Drift Detection: embedding score + LLM judge
│   ├── correction_module.py       ← Correction: goal reminder or plan regeneration
│   ├── evaluation_layer.py        ← Layer 6: all evaluation metrics
│   ├── agent_loop.py              ← Master loop integrating all 6 layers
│   └── agentbench_runner.py       ← AgentBench evaluation harness
│
└── utils/
    ├── __init__.py
    ├── embedding_engine.py        ← Sentence-Transformers wrapper (all-MiniLM-L6-v2)
    └── llm_engine.py              ← Ollama/Llama 3.2 wrapper
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/download/windows) installed and running

### 1. Install Ollama

Download and install from [ollama.com/download](https://ollama.com/download/windows). After installation, pull the model:

```bash
ollama pull llama3.2
```

Verify it works:

```bash
ollama run llama3.2 "say hello"
```

Ollama runs as a background server at `http://localhost:11434`. Keep it running while using GARDEN.

### 2. Set up Python environment

```bash
cd genai_project
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run all 5 tasks

```bash
python main.py
```

### Run a single task by index

```bash
python main.py --task 0    # Renewable Energy survey
python main.py --task 1    # Climate Change survey
python main.py --task 2    # AI in Healthcare
python main.py --task 3    # Carbon Taxation
python main.py --task 4    # Solar vs Wind comparison
```

### List all available tasks

```bash
python main.py --list
```

### Expected output structure

```
Loading models (one-time)...
LLM ready: llama3.2 via Ollama ✅
Models ready.

############################################################
  Task 1/5: Literature Survey — Renewable Energy
############################################################
  Goal:        Conduct a literature survey on renewable energy using academic sources
  Constraints: ['academic sources only']
  Subtasks:    ['find papers', 'summarize', 'categorize', 'generate report']

============================================================
  Iteration 1 / 5
  Step generated: Search Scopus and Web of Science using keywords...
  Reflection:     yes
  Observation:    Search complete. Found 12 peer-reviewed papers...
  Drift score:    0.382  (threshold: 0.5)
  Alignment:      0.618
  Drift detected: False
...

EVALUATION REPORT (Layer 6)
==================================================
Task success rate:     80.00%
Goal adherence score:  0.589
Avg drift score:       0.411
Pass@1:                0.0
```

---

## Layer-by-Layer Technical Reference

### Layer 1 — Input Layer (`data/goal_processor.py`)

`GoalProcessor.extract_goal()` takes a natural language user instruction and produces a structured goal dictionary containing the goal text, extracted constraints (academic, peer-reviewed, recent, etc.), ordered subtasks, environment state, and tool outputs.

Subtask templates are matched by keyword: goals containing "survey" map to `[find papers, summarize, categorize, generate report]`, goals containing "research" map to `[search databases, read abstracts, extract findings, synthesize]`, and all others fall back to a default template.

### Layer 2 — Goal Memory Layer (`data/memory_store.py`)

`GoalMemory` stores the goal dictionary and its vector embedding persistently for the duration of a task run. It is passed directly into `AgentLoop.run()` so the original goal is always accessible — even after many reasoning steps have accumulated. This prevents the goal from being silently dropped from the context window.

### Layer 3 — Prompt Engineering Layer (`prompts/prompt_templates.py`)

Four prompt templates are implemented:

`structured_reasoning_prompt` is the main step-generation prompt. It injects the goal, the current subtask phase, the most recent step, and the top RAG-retrieved document. It instructs the LLM to produce one concrete action sentence.

`goal_anchoring_prompt` is injected every 3 steps to re-anchor the agent. It includes the original goal, constraints, and last completed step.

`reflection_prompt` asks the LLM a yes/no question about whether the current step aligns with the goal. Used for per-step self-evaluation.

`replan_prompt` is triggered by the Correction Module when a step is completely off-topic. It asks the LLM to produce a single corrected action from scratch.

### Layer 4 — Agent Workflow Layer

**`goal_decomposer.py`** — Decomposes the goal into an ordered subtask list. Uses the subtasks from `GoalProcessor` directly if available; falls back to LLM decomposition if not.

**`reasoning_engine.py`** — Calls the LLM with the appropriate prompt template to generate the next step. Filters correction reminder text out of the step history before building prompts. Applies light output cleaning to remove prompt echoes and isolate the first sentence.

**`context_memory.py`** — Maintains the agent's intermediate state: task node, subtasks, all reasoning steps, and environment observations. Supports replacing the last step (used after drift correction) and filtering steps for history building.

**`executor.py`** — Dispatches each step to an action handler based on 20 keyword matches (search, find, conduct, retrieve, review, read, annotate, summarize, extract, analyze, categorize, group, organize, report, compile, draft, write, synthesize, store). Each handler returns a goal-aware observation string that reflects the actual task topic rather than a hardcoded generic response.

**`backtracking_engine.py`** — Saves a checkpoint snapshot before every iteration. When drift is detected on 2 consecutive iterations (`MAX_CONSECUTIVE_DRIFT = 2`), it rolls back to the most recent checkpoint. If no checkpoint is available, it forces a full replan by resetting the subtask list.

### Drift Detection Module (`core/drift_detector.py`)

Two metrics are combined to produce a final drift score:

**Metric 1 — Embedding Drift Score**

```
Drift(t) = 1 − cosine_similarity(G, R_t)
```

Where `G` is the goal embedding (computed once at the start) and `R_t` is the embedding of the current reasoning step. Uses `all-MiniLM-L6-v2` (384-dimensional) via Sentence Transformers.

**Metric 2 — LLM Alignment Judge**

The LLM rates step-to-goal alignment on a 1–5 scale. The score is normalised to 0–1 and blended into the final score.

**Combined score (with LLM judge enabled):**
```
final_score = 0.5 × similarity + 0.2 × keyword_score + 0.3 × llm_normalised
```

**Combined score (without LLM judge):**
```
final_score = 0.7 × similarity + 0.3 × keyword_score
```

`drift_score = 1 − final_score`. Drift is flagged when `drift_score > τ` where `τ = 0.5`.

### Correction Module (`core/correction_module.py`)

When drift is detected, one of two strategies is chosen:

**Goal Reminder** — used when the drifted step still contains at least one task-relevant signal. Returns the next expected subtask as a short concrete action.

**Plan Regeneration** — used when the step is completely off-topic. Calls the LLM with `replan_prompt` to generate a corrected step from scratch. Falls back to the subtask action map if the LLM output is unusable.

### Layer 5 — RAG Module (`core/rag_module.py`)

A FAISS `IndexFlatL2` vector store stores documents as 384-dimensional embeddings. Supports three document types: general knowledge documents (seeded at startup), reasoning traces (added after each iteration), and successful plan templates.

Retrieval query: `goal_text + current_reasoning_step`. Returns top-3 nearest neighbours by L2 distance.

The RAG store is pre-seeded with 7 domain documents covering literature survey methodology, academic databases, renewable energy, AI in healthcare, carbon taxation, and categorization frameworks.

### Layer 6 — Evaluation Layer (`core/evaluation_layer.py`)

Four metrics are tracked per task run:

| Metric | Definition |
|---|---|
| Task success rate | Fraction of steps classified as `success` (not drifted, not corrected) |
| Goal adherence score | Mean `final_score` across all iterations (0–1, higher is better) |
| Avg drift score | Mean `drift_score` across all iterations (0–1, lower is better) |
| Pass@1 | 1.0 if task completed with zero drift events; 0.0 otherwise |

---

## Task Registry

Five tasks are registered in `main.py`. Add new tasks by appending to the `TASKS` list — no other code changes needed.

| Index | Name | Goal |
|---|---|---|
| 0 | Literature Survey — Renewable Energy | Literature survey on renewable energy using academic sources |
| 1 | Literature Survey — Climate Change | Survey on effects of climate change on agriculture |
| 2 | Research Summary — AI in Healthcare | Summary of AI applications in medical diagnosis |
| 3 | Policy Research — Carbon Taxation | Carbon taxation policy effectiveness research |
| 4 | Comparative Survey — Solar vs Wind | Comparison of solar and wind energy |

---

## Papers Referenced

1. Evaluating Goal Drift in Language Model Agents — Rauno Arike et al.
2. Drift No More? Context Equilibria in Multi-Turn LLM Interactions — Vardhan Dongre et al.
3. ReCAP: Recursive Context-Aware Reasoning and Planning — Zhenyu Zhang et al.
4. ReAct: Synergizing Reasoning and Acting in Language Models — Shunyu Yao et al.
5. AgentBench: Evaluating LLMs as Agents — Xiaohan Liu et al.

---

## Dependencies

| Package | Purpose |
|---|---|
| `requests` | Ollama API communication |
| `sentence-transformers` | Embedding model (all-MiniLM-L6-v2) |
| `faiss-cpu` | Vector store for RAG |
| `numpy` | Embedding arithmetic |
| `arxiv` | Optional: fetch real paper metadata |
| `duckduckgo-search` | Optional: real web search in executor |
| `tqdm` | Progress bars |

Ollama (external): provides Llama 3.2 locally at `http://localhost:11434`.

---

## Extending the Project

### Add a new task

Open `main.py` and append to the `TASKS` list:

```python
{
    "name":              "Your Task Name",
    "user_input":        "Your goal description here",
    "environment_state": "research database",
}
```

### Change the LLM model

```python
LLMEngine(model="llama3.1")   # Llama 3.1
LLMEngine(model="mistral")    # Mistral 7B
LLMEngine(model="gemma2")     # Gemma 2
```

Any model available via `ollama list` can be used.

### Adjust drift sensitivity

```python
# In agent_loop.py __init__:
self.drift_detector = DriftDetector(embedder, llm=llm_judge, threshold=0.4)  # stricter
self.drift_detector = DriftDetector(embedder, llm=llm_judge, threshold=0.6)  # more lenient
```

### Replace the mock executor with real tools

```python
# In executor.py, replace _handle_search:
from duckduckgo_search import DDGS

def _handle_search(self, step: str, goal: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(f"{goal} academic research", max_results=5))
    return "Found: " + " | ".join(r["title"] for r in results)
```

---

## Troubleshooting

**`❌ Ollama is not running`** — Open the Ollama desktop app from the Start menu, or run `ollama serve` in a terminal.

**`⚠️ Model 'llama3.2' not found`** — Run `ollama pull llama3.2` and wait for the ~2GB download.

**`ModuleNotFoundError`** — Activate your virtual environment (`venv\Scripts\activate`) and run `pip install -r requirements.txt`.

**Steps are repetitive** — Add more domain-specific documents to `_seed_rag()` in `agent_loop.py`, or increase `max_iterations`.
