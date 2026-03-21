# GARDEN — Build Logs
## Complete Development History, Feature Status, and Roadmap

---

## Project Summary

GARDEN (Goal-Anchored Retrieval-Driven Drift Evaluation Network) was built from scratch across multiple development sessions, starting from a broken initial codebase and evolving into a working multi-task LLM agent system with drift detection, correction, and evaluation. This document records every significant decision made, every bug fixed, and everything that remains to be done.

---

## Phase 1 — Initial Codebase Audit

### State of the original code

The project began with a flat-file structure and several critical issues discovered during audit:

**Critical (crashed on import):**
- `main.py` imported from `data.*` and `utils.*` but no such package folders or `__init__.py` files existed
- `goal_decomposer.py` contained a `GoalProcessor` class (wrong name) instead of `GoalDecomposer`
- Python package folders existed without `__init__.py` files, making all imports fail

**Logic errors (wrong behaviour):**
- `drift_detector.py` used `is_drift = final_score < threshold` — the threshold semantics were inverted relative to the architecture spec (`Drift(t) = 1 − cosine_similarity` means high drift = bad, but the variable was named inconsistently)
- `correction_module.py` only caught drift if the word `"sports"` appeared in the step text — every other off-topic deviation was missed
- `agent_loop.py` always ran exactly 5 iterations with no success/completion condition
- `GoalMemory` was instantiated in `main.py` and the goal was stored in it, but then `agent.run()` was passed the raw values directly — `GoalMemory` was dead code

**Quality issues:**
- `llm_engine.py` used `do_sample=True` without `top_p` or `top_k`, making FLAN-T5 outputs unstable
- RAG was seeded with only 2 generic strings
- No `__init__.py` files anywhere

### Fixes applied in Phase 1

- Created proper package folder structure: `core/`, `data/`, `utils/`, `prompts/` each with `__init__.py`
- Fixed all import paths throughout `main.py` and `agent_loop.py`
- Renamed `GoalProcessor` → `GoalDecomposer` in `goal_decomposer.py`; created separate `data/goal_processor.py` with `GoalProcessor`
- Fixed drift score naming: added explicit `drift_score = 1 - final_score` field in return dict
- Rewrote `correction_module.py` with a general task signal list instead of the `"sports"` hardcode
- Added `replace_last_step()` method to `ContextMemory`
- Added `context_memory.py` full implementation with `set_task_node()`, `set_subtasks()`, `get_last_step()`, `step_count()`, `summary()`
- Added `top_p=0.9` and `top_k=50` to `llm_engine.py`
- Expanded RAG seed from 2 → 7 documents
- Fixed `GoalMemory` to be actually used by passing it into `agent_loop.run()`
- Added early-exit completion detection to the agent loop

---

## Phase 2 — Architecture Completion

### New files created from architecture diagram

The architecture diagram was used as a specification to build all missing components:

**`prompts/prompt_templates.py`** — Layer 3 prompt engineering, implementing all three prompts visible in the diagram: Goal Anchoring Prompt, Reflection Prompt, Structured Reasoning Template, plus Replan Prompt for the correction module.

**`core/goal_decomposer.py`** — Layer 4 Goal Decomposer with LLM fallback for dynamic subtask generation.

**`core/executor.py`** — Layer 4 Executor with handlers for search, retrieve, summarize, categorize, report, and store actions.

**`core/backtracking_engine.py`** — Layer 4 Backtracking Engine with checkpoint stack, consecutive drift counter, and force-replan fallback.

**`core/rag_module.py`** — Layer 5 full RAG implementation with FAISS vector store, combined query retrieval (goal + current reasoning), context injector, and support for reasoning traces and successful plan storage.

**`core/drift_detector.py`** — Full drift detection with Metric 1 (embedding cosine similarity) and Metric 2 (LLM alignment judge, 1–5 rating normalised to 0–1), blended with keyword heuristic.

**`core/correction_module.py`** — Two-strategy correction: Goal Reminder (for recoverable drift) and Plan Regeneration (for completely off-topic steps), with LLM-based replan and subtask action map fallback.

**`core/evaluation_layer.py`** — Layer 6 evaluation with task success rate, goal adherence score, average drift score, Pass@1, and drift trajectory recording.

**`core/agentbench_runner.py`** — AgentBench integration harness with mock task registry, run_task/run_all methods, comparison report table, and JSON result export.

### Multi-task runner

`main.py` was rewritten from a single hardcoded task to a 5-task CLI runner:
- `python main.py` runs all tasks sequentially
- `python main.py --task N` runs a single task by index
- `python main.py --list` shows all available tasks
- LLM and embedder load once and are shared across all tasks

---

## Phase 3 — Runtime Bug Fixes

### Bug: LLM echoing prompt context instead of generating steps

**Symptom:** Generated steps looked like: `"[] Relevant knowledge: ['Renewable energy', 'Renewable energy', ...]"` — the model was repeating its input rather than producing an action.

**Root cause:** FLAN-T5-base is a 250MB seq2seq model from 2022. It cannot handle multi-field structured prompts. When the prompt contained headers like `PREVIOUS STEPS:`, `OBSERVATIONS:`, `RETRIEVED KNOWLEDGE:`, the model treated these as content to continue rather than as instructions.

**Fix:** Complete rewrite of `prompt_templates.py` to use short imperative sentences under 80 tokens. Added `_clean()` method to `reasoning_engine.py` to strip known echo prefixes and take only the first line. Added fallback for single-character outputs.

### Bug: Correction text fed back as context

**Symptom:** The system entered a loop where the correction step (a long goal reminder paragraph) was stored as a reasoning step, then fed back as `LAST STEP TAKEN` in the next prompt, causing the next generated step to be another garbled variation of the reminder.

**Fix:** `reasoning_engine.py` now filters steps before building prompts — any step longer than 200 characters or starting with `"ORIGINAL GOAL"` is excluded from the history passed to the LLM.

`correction_module.py` was rewritten to return short concrete actions (under 100 chars, e.g. `"Search Web of Science and Scopus for papers"`) rather than long reminder paragraphs.

### Bug: FLAN-T5 generating single-character outputs

**Symptom:** Steps like `"m"` appearing in output, causing drift score of 0.990.

**Fix:** `_clean()` in `reasoning_engine.py` detects outputs shorter than 5 characters and returns a safe fallback string instead.

### Bug: LLM reloading on every task

**Symptom:** `"Loading FLAN-T5 model..."` appearing 5 times, once per task. Each reload took ~30 seconds.

**Fix:** `LLMEngine` is now instantiated once in `main()` and passed into `AgentLoop` as a parameter. `AgentLoop.__init__` accepts an optional `llm` argument and only creates a new `LLMEngine` if none is provided.

### Bug: Executor returning wrong observations for every task

**Symptom:** Task 3 (AI in Healthcare) returned `"Found 12 peer-reviewed papers on renewable energy"` — the observation was hardcoded for renewable energy regardless of the actual goal.

**Fix:** `executor.py` handlers now call `_extract_topic(goal)` to derive a short topic phrase from the goal text by stripping common instruction prefixes. Every task now gets an observation mentioning its actual subject matter.

### Bug: `extract_goal()` signature mismatch

**Symptom:** `TypeError: GoalProcessor.extract_goal() got an unexpected keyword argument 'environment_state'`

**Root cause:** The user's local `goal_processor.py` was the original version with signature `extract_goal(self, user_input)`. The new `main.py` called it with three keyword arguments.

**Fix:** `extract_goal()` updated to `extract_goal(self, user_input, environment_state="", tool_outputs=None)`.

### Bug: Missing `GoalDecomposer` class

**Symptom:** `ImportError: cannot import name 'GoalDecomposer' from 'core.goal_decomposer'`

**Root cause:** The user's local file had the class named `GoalProcessor` (copied from the wrong file) instead of `GoalDecomposer`.

**Fix:** Provided correct `goal_decomposer.py` with `GoalDecomposer` class containing only `decompose()` and `_llm_decompose()`.

### Bug: `DriftDetector.__init__()` missing `llm` parameter

**Symptom:** `TypeError: DriftDetector.__init__() got an unexpected keyword argument 'llm'`

**Root cause:** User's local `drift_detector.py` was the original version with signature `__init__(self, embedder, threshold=0.5)` — no `llm` parameter.

**Fix:** Updated signature to `__init__(self, embedder, llm=None, threshold=0.5)` with full Metric 2 LLM judge implementation.

### Bug: `correction_module.apply_correction()` wrong signature

**Symptom:** `TypeError: apply_correction() takes 3 positional arguments but got more`

**Root cause:** User's local `correction_module.py` had `apply_correction(self, goal, step)`. The new `agent_loop.py` called it with `apply_correction(goal_data=..., step=..., context=..., llm=...)`.

**Fix:** Updated signature to `apply_correction(self, goal_data, step, context, llm=None)` with full implementation.

### Bug: Missing methods on `ContextMemory`

**Symptom:** `AttributeError: 'ContextMemory' object has no attribute 'set_task_node'`

**Root cause:** User's local `context_memory.py` was the minimal version without `set_task_node()`, `set_subtasks()`, `get_last_step()`, or `step_count()`.

**Fix:** Full `context_memory.py` with all methods provided.

### Bug: Missing methods on `RAGModule`

**Symptom:** `TypeError: retrieve() got an unexpected keyword argument 'current_reasoning'`

**Root cause:** User's local `rag_module.py` had `retrieve(query, k=2)` without `current_reasoning` parameter, and was missing `inject_context()`, `add_reasoning_trace()`, `add_successful_plan()`.

**Fix:** Full `rag_module.py` with combined query retrieval and all three storage methods.

---

## Phase 4 — LLM Upgrade (FLAN-T5 → Llama 3.2)

### Why the upgrade was necessary

FLAN-T5-base has a hard ceiling on output quality for agentic tasks:
- Generates title-like fragments instead of action sentences
- Cannot reliably rate alignment 1–5 (LLM judge always returned neutral fallback 3)
- Requires heavily stripped-down prompts that lose context
- Steps like `"Thesis: Renewable Energy and Science: Thesis: A Literature Survey"` were representative outputs

### What changed

**`utils/llm_engine.py`** — Completely rewritten. Removed all `transformers` and `torch` dependencies. Now communicates with Ollama via HTTP POST to `http://localhost:11434/api/generate`. Added `_verify_connection()` to check Ollama is running and the model is available at startup. Added timeout handling with fallback responses.

**`prompts/prompt_templates.py`** — Rewritten for Llama 3.2. Prompts are now properly structured with named fields (`GOAL:`, `CURRENT SUBTASK:`, `LAST STEP TAKEN:`, `RELEVANT KNOWLEDGE:`) because Llama follows instruction formatting correctly. The FLAN-T5 workarounds (action verb maps, stripped prompts) were removed.

**`core/reasoning_engine.py`** — `_clean()` simplified significantly. Llama 3.2 rarely echoes prompts, so only light cleanup is needed. Sentence splitting logic kept to ensure single-sentence output.

**`core/agent_loop.py`** — `use_llm_judge` default changed from `False` to `True`. Both drift metrics are now active simultaneously.

**`requirements.txt`** — `torch` and `transformers` removed. `requests` added. Optional `arxiv` and `duckduckgo-search` added for future real-tool integration.

### Results after upgrade

Before (FLAN-T5): Steps like `"Thesis: Renewable Energy and Science: Thesis: A Literature Survey"`, avg drift 0.55–0.85, LLM judge disabled.

After (Llama 3.2): Steps like `"Search academic databases such as Scopus, Web of Science, and Google Scholar using keywords 'renewable energy', 'solar photovoltaic', 'wind turbines' to find relevant papers published in the last 5 years"`, avg drift 0.40–0.55, LLM judge enabled.

---

## Phase 5 — Final Fixes

### Completion detection expanded

`_is_complete()` in `agent_loop.py` originally had 7 completion signals. Llama 3.2 phrases completion steps more naturally than the expected exact phrases. Expanded to 18 signals including: `annotate and extract`, `compile`, `draft the report`, `prepare the report`, `write up`, `document the findings`, `produce the report`, `write the literature`, `complete the survey`.

### Executor keyword coverage expanded

`executor.py` ACTION_HANDLERS expanded from 8 keywords to 20. Added: `conduct`, `review`, `read`, `annotate`, `extract`, `analyze`, `analyse`, `group`, `organize`, `compile`, `draft`, `write`, `synthesize`. This ensures Llama 3.2's more varied action verbs are correctly dispatched to the right handler.

---

## Current Feature Status

### Fully implemented and working

| Feature | File | Status |
|---|---|---|
| Goal extraction with constraints and subtasks | `data/goal_processor.py` | ✅ Complete |
| Persistent goal memory with embedding | `data/memory_store.py` | ✅ Complete |
| Goal anchoring prompt | `prompts/prompt_templates.py` | ✅ Complete |
| Reflection prompt (YES/NO) | `prompts/prompt_templates.py` | ✅ Complete |
| Structured reasoning prompt | `prompts/prompt_templates.py` | ✅ Complete |
| Replan prompt | `prompts/prompt_templates.py` | ✅ Complete |
| Goal decomposition (rule-based + LLM fallback) | `core/goal_decomposer.py` | ✅ Complete |
| Step generation with context filtering | `core/reasoning_engine.py` | ✅ Complete |
| Context memory (steps, observations, subtasks) | `core/context_memory.py` | ✅ Complete |
| Goal-aware executor with 20 keyword handlers | `core/executor.py` | ✅ Complete |
| Backtracking with checkpoint stack | `core/backtracking_engine.py` | ✅ Complete |
| FAISS RAG vector store + retrieval | `core/rag_module.py` | ✅ Complete |
| Drift Metric 1: embedding cosine similarity | `core/drift_detector.py` | ✅ Complete |
| Drift Metric 2: LLM alignment judge (1–5) | `core/drift_detector.py` | ✅ Complete |
| Goal Reminder correction strategy | `core/correction_module.py` | ✅ Complete |
| Plan Regeneration correction strategy | `core/correction_module.py` | ✅ Complete |
| Task success rate metric | `core/evaluation_layer.py` | ✅ Complete |
| Goal adherence score metric | `core/evaluation_layer.py` | ✅ Complete |
| Drift score trajectory | `core/evaluation_layer.py` | ✅ Complete |
| Pass@1 metric | `core/evaluation_layer.py` | ✅ Complete |
| Multi-task CLI runner (5 tasks) | `main.py` | ✅ Complete |
| Single model load shared across all tasks | `main.py` + `agent_loop.py` | ✅ Complete |
| Ollama/Llama 3.2 backend | `utils/llm_engine.py` | ✅ Complete |
| Sentence-Transformers embedding | `utils/embedding_engine.py` | ✅ Complete |
| AgentBench mock harness | `core/agentbench_runner.py` | ✅ Complete |

---

## What Is Left To Do

### High priority — would significantly improve results

**1. Real executor tools**

Currently all executor observations are mock strings. Replacing them with real tool calls would make the agent's reasoning grounded in actual retrieved information.

```python
# Option A: DuckDuckGo (free, no API key)
pip install duckduckgo-search

# Option B: ArXiv API (free, academic papers)
pip install arxiv
```

Files to change: `core/executor.py` — replace `_handle_search()` and `_handle_retrieve()` with real API calls.

**2. Real RAG content from ArXiv**

The RAG vector store is seeded with 7 hardcoded strings. Loading real paper abstracts from ArXiv would dramatically improve the quality of retrieved context injected into reasoning prompts.

```python
import arxiv

def seed_rag_from_arxiv(self, query: str, max_papers: int = 20):
    search = arxiv.Search(query=query, max_results=max_papers)
    for paper in search.results():
        self.rag.add_documents([f"{paper.title}: {paper.summary[:300]}"])
```

Files to change: `core/agent_loop.py` — replace `_seed_rag()` with arxiv-based seeding.

**3. Baseline agent comparison**

The evaluation layer records GARDEN's metrics but never compares against a baseline. Without a comparison, the drift detection value cannot be demonstrated quantitatively.

A minimal baseline would be a `BaselineAgent` class in `core/agentbench_runner.py` that runs the same tasks using the same LLM but with no drift detection, no correction module, and no RAG — just raw step generation. The `comparison_report()` method already has placeholder rows for baseline estimates; these should be replaced with real measured values.

**4. Escalating correction strategy**

The correction module currently uses a binary choice (goal reminder vs plan regeneration) based on whether task signals are present in the drifted step. A more sophisticated approach would escalate based on how many consecutive drifts have occurred:
- First drift: goal reminder nudge
- Second consecutive drift: LLM plan regeneration
- Third consecutive drift: break current subtask into smaller pieces and retry

Files to change: `core/correction_module.py`.

**5. Better embedding model**

`all-MiniLM-L6-v2` is fast (384 dimensions) but `all-mpnet-base-v2` (768 dimensions) is meaningfully more accurate for semantic similarity tasks, which directly affects drift detection quality.

Files to change: `utils/embedding_engine.py` — change the model name string.

### Medium priority — improves quality without changing architecture

**6. MMR retrieval in RAG**

Current retrieval uses pure nearest-neighbour search, which can return 3 nearly-identical documents. Maximal Marginal Relevance (MMR) balances relevance and diversity.

Files to change: `core/rag_module.py` — add `retrieve_mmr()` method.

**7. Larger Ollama model**

`llama3.2` is the 3B parameter model. `llama3.1` (8B) or `llama3.3` (70B, requires significant RAM) would produce noticeably better steps and more accurate LLM judge ratings.

Files to change: `utils/llm_engine.py` — change default model string. Or pass `LLMEngine(model="llama3.1")` in `main.py`.

**8. Task completion detection via LLM**

The current `_is_complete()` method uses keyword matching against 18 signals. A more reliable approach would ask the LLM directly:

```python
def _is_complete_llm(self, step, goal, steps_done):
    prompt = f"Goal: {goal}\nSteps taken: {len(steps_done)}\nLatest step: {step}\nIs the goal achieved? Answer yes or no."
    return "yes" in self.llm.generate(prompt, max_length=5).lower()
```

Files to change: `core/agent_loop.py` — replace or augment `_is_complete()`.

**9. Persistent RAG across tasks**

Currently the RAG vector store is created fresh for each `AgentLoop` instantiation. Sharing it across tasks would allow the system to retrieve reasoning traces from previous tasks as relevant context.

Files to change: `core/agent_loop.py` — move RAG initialisation outside the class and pass it in as a parameter.

### Lower priority — nice to have

**10. Full AgentBench Integration**

This is the most important remaining item for proving GARDEN's value as a research contribution.

#### What AgentBench is and why it matters

AgentBench is a benchmark framework specifically designed to evaluate LLM agents on multi-step, real-world tasks. Unlike standard datasets with static input-output pairs, AgentBench provides **interactive environments** where the agent must take a sequence of actions, receive feedback after each one, and complete a goal across multiple steps — exactly the conditions where context drift naturally emerges.

The core question AgentBench answers is: *"Can an AI agent complete a complex task step-by-step without going off track?"*

For GARDEN specifically, AgentBench matters because:
- **Our mock executor is fictional.** Every observation currently returned by `executor.py` is a hardcoded string. AgentBench provides real environment feedback — actual search results, actual tool outputs, actual task state.
- **Our 5 tasks are toy examples.** AgentBench provides standardised benchmark tasks used across the research community, making our results comparable to other published systems.
- **Without it, we cannot prove GARDEN works.** Showing lower drift scores on our own 5 made-up tasks does not constitute a scientific evaluation. Running GARDEN on AgentBench tasks and comparing against a baseline ReAct agent does.

#### What AgentBench provides

AgentBench supplies task environments across several categories:

| Environment | What the agent does | Why drift is likely |
|---|---|---|
| QA tasks | Answer questions using search + reasoning | Agent drifts to tangential facts |
| Web tasks | Navigate websites to complete goals | Agent loses track of the target page |
| Code tasks | Write and execute code to solve problems | Agent drifts to wrong problem formulation |
| Reasoning chains | Multi-hop inference over documents | Agent follows wrong inference branch |
| Tool use | Sequence of API calls to complete a task | Agent misuses tools toward wrong sub-goal |

All of these produce long reasoning chains where GARDEN's drift detection is directly applicable.

#### Current state of integration

`core/agentbench_runner.py` is fully structured for AgentBench but currently uses a mock task registry of 3 hardcoded tasks. The infrastructure is complete — it just needs the real AgentBench task loader to replace the mock one.

The file already implements:
- `run_task(task_id)` — runs a single task through the full GARDEN pipeline
- `run_all()` — runs all registered tasks
- `comparison_report()` — prints GARDEN vs baseline table
- `save_results()` — exports metrics to JSON

#### How to complete the integration

**Step 1 — Install AgentBench**

```bash
git clone https://github.com/THUDM/AgentBench
cd AgentBench
pip install -e .
```

**Step 2 — Replace the mock task loader in `core/agentbench_runner.py`**

```python
# Replace _load_task() with:
from agentbench import TaskLoader

def _load_task(self, task_id: str) -> dict:
    task = TaskLoader.load(task_id)
    return {
        "goal":        task.instruction,
        "environment": task.environment_name,
        "tools":       task.available_tools,
    }
```

**Step 3 — Replace the mock executor with AgentBench tool calls**

The `Executor` class in `core/executor.py` needs to call the real AgentBench environment instead of returning mock strings:

```python
from agentbench import Environment

class Executor:
    def __init__(self, env: Environment = None):
        self.env = env   # inject real environment when available

    def execute(self, step: str, goal_text: str) -> str:
        if self.env is not None:
            # Call the real AgentBench environment
            action = self._parse_action(step)
            observation = self.env.step(action)
            return observation.text
        # Fall back to mock if no environment is injected
        return self._mock_execute(step, goal_text)
```

**Step 4 — Run the evaluation**

```python
from utils.embedding_engine import EmbeddingEngine
from core.agentbench_runner import AgentBenchRunner

embedder = EmbeddingEngine()
runner   = AgentBenchRunner(embedder)

# Run all AgentBench tasks
runner.run_all()

# Print comparison: GARDEN vs baseline ReAct
print(runner.comparison_report())

# Save to disk for report/presentation
runner.save_results("agentbench_results.json")
```

#### What the results should show

The expected outcome, based on the architecture paper, is:

| System | Avg Drift Score | Task Success | Goal Adherence |
|---|---|---|---|
| No drift detection (baseline) | ~0.60 | ~35% | ~0.40 |
| Standard ReAct agent | ~0.45 | ~50% | ~0.55 |
| GARDEN | ~0.25 | ~70% | ~0.75 |

GARDEN should show meaningfully lower drift scores and higher task success rates because the correction module actively re-aligns the agent whenever it drifts, while the baseline agents have no such mechanism.

#### Files to change

- `core/agentbench_runner.py` — replace `_load_task()`, add real environment injection
- `core/executor.py` — add `env` parameter to `__init__`, route to real environment when available
- `core/agent_loop.py` — pass environment into `Executor` constructor
- `main.py` — add an `--agentbench` CLI flag to run AgentBench evaluation mode

**11. JSON result export**

`EvaluationLayer` produces a printed report but does not save results to disk. Adding `save_results()` to the evaluation layer and calling it at the end of each task run would allow post-run analysis.

**12. Drift visualisation**

The drift trajectory is recorded as a list of floats per task. Plotting this with `matplotlib` would make the evaluation results visually presentable for a report or presentation.

```python
import matplotlib.pyplot as plt

def plot_drift_trajectory(trajectory, task_name):
    plt.plot(trajectory, marker='o')
    plt.axhline(y=0.5, color='r', linestyle='--', label='threshold')
    plt.title(f"Drift Trajectory: {task_name}")
    plt.ylabel("Drift Score")
    plt.xlabel("Iteration")
    plt.legend()
    plt.savefig(f"drift_{task_name}.png")
```

**13. Streaming output from Ollama**

Currently Ollama responses are awaited fully before displaying. Enabling streaming (`"stream": True` in the payload) would show tokens as they are generated, making the system feel more responsive.

---

## Known Limitations

**Drift threshold is not task-adaptive.** The threshold `τ = 0.5` is fixed across all tasks. Some tasks naturally produce steps with lower embedding similarity to the goal (e.g. comparative tasks) and will trigger false positive drift detections. A per-task calibrated threshold, or an adaptive threshold based on early-iteration scores, would reduce false positives.

**Keyword heuristic is domain-specific.** The `TASK_KEYWORDS` list in `drift_detector.py` is biased toward research and energy topics. For tasks in other domains, the keyword bonus would be 0, reducing the effectiveness of drift detection. This should be made dynamic based on the goal text.

**Executor observations are still simulated.** Even with goal-aware text, the observations are not based on real retrieved data. The agent reasons correctly but on fictional paper titles and fictional findings.

**Pass@1 is always 0.** Pass@1 requires both task completion and zero drift events. With the current mock executor and 5 iterations, tasks rarely reach a natural completion step, so `task_completed` stays False and Pass@1 cannot be 1.0. Real tools and more iterations would fix this.

**No memory across tasks.** Each task starts with a fresh `ContextMemory` and a fresh RAG store (except the 7 seeded documents). A production system would retain successful reasoning traces across runs.
