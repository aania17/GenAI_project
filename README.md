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
├── main.py                        ← Entry point, 10-task registry, CLI
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

### Run all 10 tasks

```bash
python main.py
```

### Run a single task by index (0–9)

```bash
python main.py --task 0    # Pair 1A — Literature Survey: Renewable Energy
python main.py --task 1    # Pair 1B — Literature Survey: Vaccine Hesitancy
python main.py --task 2    # Pair 2A — Comparative Analysis: Solar vs Wind
python main.py --task 3    # Pair 2B — Comparative Analysis: AI vs Traditional Diagnosis
python main.py --task 4    # Pair 3A — Research Planning: Climate Policy
python main.py --task 5    # Pair 3B — Research Planning: AI Ethics Framework
python main.py --task 6    # Pair 4A — Factual QA: Barriers to Renewable Energy
python main.py --task 7    # Pair 4B — Factual QA: Remote Work and Productivity
python main.py --task 8    # Pair 5A — Open Exploration: Long-Term Impact of AGI
python main.py --task 9    # Pair 5B — Open Exploration: Future of Global Healthcare
```

### Run a full task pair (both tasks in a pair)

```bash
python main.py --pair 1    # Literature Surveys         — Low drift risk
python main.py --pair 2    # Comparative Analysis       — Medium drift risk
python main.py --pair 3    # Multi-Step Planning        — Medium-High drift risk
python main.py --pair 4    # Factual QA with Evidence   — Variable drift risk
python main.py --pair 5    # Open-Ended Exploration     — High drift risk
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
  Task 1/10: Pair 1A — Literature Survey: Renewable Energy
  Pair: 1  |  Expected drift risk: Low
############################################################
  Goal:        Conduct a literature survey on renewable energy sources using peer-reviewed academic sources
  Constraints: ['academic sources only', 'peer-reviewed sources']
  Subtasks:    ['find papers', 'summarize', 'categorize', 'generate report']

============================================================
  Iteration 1 / 6
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

## Task Registry — 10 Tasks in 5 Behavioural Pairs

The 10 tasks are organised into 5 pairs, each designed to elicit a different pattern of agent behaviour and demonstrate a different aspect of GARDEN's capabilities.

| Pair | Type | Drift Risk | Tasks | What to observe |
|---|---|---|---|---|
| 1 | Literature Surveys | Low | 0, 1 | Clean execution, high adherence, minimal corrections |
| 2 | Comparative Analysis | Medium | 2, 3 | Dual-thread drift — agent fixates on one side of comparison |
| 3 | Multi-Step Planning | Medium-High | 4, 5 | Sequential decision drift, backtracking engine active |
| 4 | Factual QA | Variable | 6, 7 | Goal broadening — agent drifts from specific answer to general essay |
| 5 | Open-Ended Exploration | High | 8, 9 | Stress test — maximum corrections, highest drift scores |

### Pair 1 — Literature Surveys (Low drift risk)

These tasks have clear, well-scoped goals and a natural sequential structure. The agent should progress cleanly through `find papers → summarize → categorize → generate report` with high alignment scores and few or no corrections needed. This pair establishes a performance baseline.

- **Task 0:** Conduct a literature survey on renewable energy sources using peer-reviewed academic sources
- **Task 1:** Conduct an academic literature survey on the psychological factors behind vaccine hesitancy

### Pair 2 — Comparative Analysis (Medium drift risk)

These tasks require the agent to simultaneously track two subjects and evaluate them on the same criteria. The natural failure mode is fixating on one side and forgetting the comparison frame. GARDEN's correction module should redirect back to the comparative structure when this happens.

- **Task 2:** Compare solar and wind energy in terms of cost, efficiency, scalability, and environmental impact
- **Task 3:** Compare AI-based and traditional methods for early cancer detection across accuracy, cost, accessibility, and clinical adoption

### Pair 3 — Multi-Step Planning (Medium-High drift risk)

Planning tasks require producing an ordered sequence of actionable decisions rather than just gathering information. The agent tends to drift into generic advice rather than maintaining the specific planning structure. The backtracking engine is most visibly active on these tasks.

- **Task 4:** Design a structured 6-month research plan to investigate carbon pricing policy effectiveness
- **Task 5:** Create a step-by-step plan for developing an AI ethics review framework for a university

### Pair 4 — Factual QA with Evidence (Variable drift risk)

These tasks have sharp, narrow goals — find a specific answer and back it with evidence. The drift pattern is different here: the agent does not go off-topic, it broadens — drifting from finding the specific answer into writing a general essay. Goal anchoring is what keeps it precise.

- **Task 6:** What are the three most cited policy and economic barriers to renewable energy adoption in developing countries?
- **Task 7:** What does recent academic research say about the effect of remote work on employee productivity and mental health?

### Pair 5 — Open-Ended Exploration (High drift risk)

These are the stress tests. The goals are deliberately vague with no natural stopping point. Drift is almost guaranteed, and the correction module will fire most frequently. This pair gives the most dramatic demonstration of GARDEN's value — without drift correction, these tasks would spiral completely off course.

- **Task 8:** Explore the potential long-term societal, economic, and ethical impacts of artificial general intelligence on the global workforce
- **Task 9:** Investigate how AI, genomics, and telemedicine might reshape global healthcare systems over the next 20 years

---

## Layer-by-Layer Technical Reference

### Layer 1 — Input Layer (`data/goal_processor.py`)

`GoalProcessor.extract_goal()` takes a natural language user instruction and produces a structured goal dictionary containing the goal text, extracted constraints, ordered subtasks, environment state, and tool outputs.

**Constraint detection** — keywords like `academic`, `peer-reviewed`, `recent`, `evidence`, `structured`, `compare`, and `plan` are mapped to constraint labels that guide the agent's behaviour.

**Subtask template matching** — goals are matched by keyword to one of 9 subtask templates:

| Keyword | Subtask template |
|---|---|
| `survey` | find papers → summarize → categorize → generate report |
| `research` | search databases → read abstracts → extract findings → synthesize |
| `compare` | define criteria → gather data on each option → compare → synthesize |
| `plan` / `design` | define scope → identify stakeholders → draft plan → review and finalize |
| `what` / `how` | identify question → search for evidence → extract findings → formulate answer |
| `explore` | define boundaries → gather on key themes → identify patterns → synthesize |
| `investigate` | define scope → gather information → analyze → synthesize and report |
| default | understand task → gather information → process results → report |

### Layer 2 — Goal Memory Layer (`data/memory_store.py`)

`GoalMemory` stores the goal dictionary and its vector embedding persistently for the duration of a task run. It is passed directly into `AgentLoop.run()` so the original goal is always accessible — even after many reasoning steps have accumulated. This prevents the goal from being silently dropped from the context window.

### Layer 3 — Prompt Engineering Layer (`prompts/prompt_templates.py`)

Four prompt templates are implemented, all optimised for Llama 3.2's instruction-following format:

`structured_reasoning_prompt` — main step-generation prompt. Injects goal, current subtask phase, most recent step, and top RAG-retrieved document. Instructs the LLM to produce one concrete action sentence.

`goal_anchoring_prompt` — injected every 3 steps to re-anchor the agent. Includes original goal, constraints, and last completed step.

`reflection_prompt` — yes/no alignment check after each step. Used for per-step self-evaluation.

`replan_prompt` — triggered by the Correction Module when a step is completely off-topic. Asks the LLM to produce a single corrected action from scratch.

### Layer 4 — Agent Workflow Layer

**`goal_decomposer.py`** — Decomposes the goal into an ordered subtask list using the templates from `GoalProcessor`. Falls back to LLM-based decomposition if no template matches.

**`reasoning_engine.py`** — Calls the LLM with the appropriate prompt template to generate the next step. Filters correction text out of step history before building prompts. Applies light output cleaning to remove echoes and isolate the first sentence.

**`context_memory.py`** — Maintains the agent's intermediate state: task node, subtasks, all reasoning steps, and environment observations. Supports replacing the last step after drift correction.

**`executor.py`** — Dispatches each step to an action handler based on 20 keyword matches. Each handler returns a goal-aware observation string that reflects the actual task topic. The `_extract_topic()` method strips common instruction prefixes from the goal to derive the core subject.

**`backtracking_engine.py`** — Saves a checkpoint before every iteration. When drift is detected on 2 consecutive iterations (`MAX_CONSECUTIVE_DRIFT = 2`), it rolls back to the most recent checkpoint. If no checkpoint is available, forces a full replan.

### Drift Detection Module (`core/drift_detector.py`)

Two metrics are combined to produce a final drift score:

**Metric 1 — Embedding Drift Score**

```
Drift(t) = 1 − cosine_similarity(G, R_t)
```

Where `G` is the goal embedding (computed once at the start) and `R_t` is the embedding of the current reasoning step. Uses `all-MiniLM-L6-v2` (384-dimensional) via Sentence Transformers.

**Metric 2 — LLM Alignment Judge**

The LLM rates step-to-goal alignment on a 1–5 scale. The rating is normalised to 0–1 and blended into the final score.

**Combined score (LLM judge enabled — default):**
```
final_score = 0.5 × embedding_similarity + 0.2 × keyword_score + 0.3 × llm_normalised
```

**Combined score (LLM judge disabled):**
```
final_score = 0.7 × embedding_similarity + 0.3 × keyword_score
```

`drift_score = 1 − final_score`. Drift is flagged when `drift_score > τ` where `τ = 0.5`.

### Correction Module (`core/correction_module.py`)

When drift is detected, one of two strategies is chosen based on whether the drifted step still contains task-relevant signals:

**Goal Reminder** — used when the step contains at least one signal from a 17-word task vocabulary (search, find, read, summarize, categorize, review, paper, source, report, research, academic, analyze, extract, synthesize, gather, collect, identify). Returns the next expected subtask as a short concrete action under 100 characters.

**Plan Regeneration** — used when the step is completely off-topic. Calls the LLM with `replan_prompt` to generate a corrected step from scratch. Falls back to the subtask action map if the LLM output is unusable (too short or echoes the prompt).

### Layer 5 — RAG Module (`core/rag_module.py`)

A FAISS `IndexFlatL2` vector store stores documents as 384-dimensional embeddings. Supports three document types: general knowledge documents (seeded at startup with 20 domain-specific entries), reasoning traces (added after each iteration as `step → observation`), and successful plan templates (3 templates covering survey, comparative, and planning task types).

Retrieval query: `goal_text + current_reasoning_step`. Returns top-3 nearest neighbours by L2 distance.

The RAG store is seeded with domain knowledge covering all 5 task pair types: literature survey methodology, comparative analysis structure, research planning, factual QA strategy, and open-ended exploration anchoring.

### Layer 6 — Evaluation Layer (`core/evaluation_layer.py`)

Four metrics are tracked per task run:

| Metric | Definition |
|---|---|
| Task success rate | Fraction of steps classified as `success` (not drifted, not corrected) |
| Goal adherence score | Mean `final_score` across all iterations (0–1, higher is better) |
| Avg drift score | Mean `drift_score` across all iterations (0–1, lower is better) |
| Pass@1 | 1.0 if task completed with zero drift events; 0.0 otherwise |

Each step is classified as `success`, `drift`, or `corrected`. The drift trajectory records the per-step drift score sequence across all iterations.

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
| `arxiv` | Optional: fetch real paper metadata for RAG seeding |
| `duckduckgo-search` | Optional: real web search in executor |
| `tqdm` | Progress bars |

Ollama (external, not pip): provides Llama 3.2 locally at `http://localhost:11434`.

---

## Extending the Project

### Add a new task

Open `main.py` and append to the `TASKS` list:

```python
{
    "name":              "Your Task Name",
    "pair":              6,              # assign to a new or existing pair
    "drift_risk":        "Medium",
    "user_input":        "Your goal description here",
    "environment_state": "research database",
}
```

No other code changes are needed.

### Change the LLM model

```python
LLMEngine(model="llama3.1")   # Llama 3.1 (8B — better quality, more RAM)
LLMEngine(model="mistral")    # Mistral 7B
LLMEngine(model="gemma2")     # Gemma 2
```

Any model available via `ollama list` can be used.

### Adjust drift sensitivity

```python
# In core/agent_loop.py __init__:
self.drift_detector = DriftDetector(embedder, llm=llm_judge, threshold=0.4)  # stricter
self.drift_detector = DriftDetector(embedder, llm=llm_judge, threshold=0.6)  # more lenient
```

### Replace the mock executor with real tools

```python
# In core/executor.py, replace _handle_search():
from duckduckgo_search import DDGS

def _handle_search(self, step: str, goal: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(f"{goal} academic research", max_results=5))
    return "Found: " + " | ".join(r["title"] for r in results)
```

### Connect to AgentBench

```python
# In core/agentbench_runner.py, replace _load_task():
from agentbench import TaskLoader

def _load_task(self, task_id: str) -> dict:
    task = TaskLoader.load(task_id)
    return {
        "goal":        task.instruction,
        "environment": task.environment_name,
        "tools":       task.available_tools,
    }
```

---

## Troubleshooting

**`❌ Ollama is not running`** — Open the Ollama desktop app from the Start menu, or run `ollama serve` in a terminal. The Ollama icon should appear in the system tray before running `python main.py`.

**`⚠️ Model 'llama3.2' not found`** — Run `ollama pull llama3.2` and wait for the ~2GB download to complete.

**`ModuleNotFoundError`** — Activate your virtual environment (`venv\Scripts\activate`) and run `pip install -r requirements.txt`.

**`TypeError: extract_goal() got unexpected keyword argument`** — Your local `goal_processor.py` is an old version. Replace it with the current one from the repository.

**Steps are repetitive across iterations** — Add more domain-specific documents to `_seed_rag()` in `core/agent_loop.py`, or increase `max_iterations` in `main.py`.

**Drift detected on every step** — Lower the threshold: `DriftDetector(..., threshold=0.6)`. Pair 5 (open-ended) tasks naturally produce higher drift scores and may need a more lenient threshold.
