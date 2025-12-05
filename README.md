# DSC180A-Q1Project

## Overview
Our project aims to explore how to evaluate LLM systems with a focus on how context policies shapes their performance. Our main research question is 
>How do context policies shape the reliability and efficiency of LLM agents that design, run, and interpret ML experiments?


## Task: ML Experimentation
Our goal for Q1 is to build a framework to systematically evaluate how different context policies (history management, summarization, retrieval, structured state, budget-aware mixing) affect LLM-based ML experimentation. We evaluate agents **as ML experimenters**. An agent plans, writes code, calls tools, and manages files to complete compact ML tasks. Heavily inspired by Stanford's MLAgentBench (https://arxiv.org/pdf/2310.03302) and OpenAI's MLEBench (https://arxiv.org/pdf/2410.07095).

![](figs/architecture.png)

**Task format:**
- **Actions:** edit files, run tests/metrics, parse tracebacks, update prompt/tool/memory.
- **Modalities:** start with **text/tabular**; optional image/vision later.
- **Stop criteria:** produce a **valid structured output** (e.g., metrics JSON) or meet a **task-native target** (e.g., ≥10% over a starter baseline) within token/time/iteration budgets.

## Setup

```bash
python3 script.py --config config.json
```

### Create & activate a virtual environment
Using a venv keeps the Kaggle CLI, Phoenix SDK, and OpenAI client pinned to the same versions across machines. Run the following once per clone:

```bash
python3 -m venv venv
source venv/bin/activate         # macOS/Linux
# .\venv\Scripts\activate        # Windows PowerShell
```

Every time you work in the repo, activate the venv first so the `kaggle` CLI and `pip` resolve to `venv/bin`.

### Environment Variables
- Copy the provided `.env` template (or create one) in the repo root. The code loads it automatically on import, so no manual `export` is needed. Keep this file untracked.

#### 1. Install dependencies
```bash
pip install openai arize-phoenix-otel kaggle pandas scikit-learn
```

#### 2. Correct configuration

`.env` file required keys:
```
OPENAI_API_KEY=<OpenAI key>
PHOENIX_API_KEY=<system key from Phoenix dashboard>
PHOENIX_PROJECT_NAME=toy-llm          # any descriptive label for grouping traces
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/<space-id>/v1/traces
# Optional (legacy/custom collectors):
# PHOENIX_CLIENT_HEADERS=""   # URL-encoded header for pre-Jun-24-2025 spaces
# PHOENIX_GRPC_PORT="4317"                     # override only if your collector runs elsewhere
```
- Go to https://app.phoenix.arize.com/management/spaces
- You'll see spaces. I created a new space called `dsc180a`. Launch that, and then go to settings.
- **Phoenix API key**: Use a **System Key**, not a user key. Retrieve it from *Dashboard of Space → Settings → API Keys* in Phoenix.  
- **Endpoint**: Copy the *Hostname* field and append `/v1/traces` for direct trace export (Cloud ingest is HTTP-only as of May 2025).  
- **Project name**: Any string; we default to `toy-llm` and reference it when filtering traces in Phoenix. This can be changed.

### Observability (Arize Phoenix)
Set `"telemetry": { "phoenix": { "enabled": true } }` in `config.json` (already true in the repo). During a run the system emits both local JSONL traces (`traces/run.jsonl`) and Phoenix spans (LLM API calls, iteration metadata, evaluations).

#### Phoenix setup & verification
1. **Create credentials**  
   - Sign in at [app.phoenix.arize.com](https://app.phoenix.arize.com/login).  
   - Create or select a *Space*, then open **Settings → API Keys**.  
   - Copy the *API Key* and *Hostname* (collector endpoint). If your space was created before **Jun 24, 2025**, also note the legacy API-header requirement described in the [Phoenix docs](https://arize.com/docs/phoenix/integrations/python/beeai/beeai-tracing-python).
2. **Populate `.env`**  
   - Add the following entries:
     ```
     OPENAI_API_KEY=""
     PHOENIX_API_KEY=""
     PHOENIX_PROJECT_NAME="toy-llm"        # or your space/project label
     PHOENIX_COLLECTOR_ENDPOINT="https://app.phoenix.arize.com/s/<space-id>/v1/traces"
     # Optional:
     # PHOENIX_CLIENT_HEADERS=""   # only needed for legacy spaces
     # PHOENIX_GRPC_PORT="4317"                     # change if using a custom collector port
     ```
3. **Enable telemetry**  
   - Set `"telemetry": { "phoenix": { "enabled": true } }` in `config.json` (or rely on auto-enable once the env vars exist).
4. **Run & verify**  
   - Execute `python3 script.py --config config.json`.  
   - On success/failure (of task) you will still get the JSON result, plus Phoenix spans will stream to your space.  
   - In the Phoenix UI, open the space’s *Traces* tab and filter by project name (default `toy-llm`) to confirm you see spans like `toy_llm.run`, `toy_llm.iteration`, and `toy_llm.openai_call`.  
   - Locally, you can also inspect `traces/run.jsonl` for the legacy JSON trace log.
5. **Troubleshooting tips**  
   - No spans appearing? Double-check the API key, endpoint URL, and that `arize-phoenix-otel` is installed in the same virtualenv.  

#### 3. Key issues we hit (and fixes)

| Issue | Symptom | Solution / Notes |
| --- | --- | --- |
| Dashboard hostname used without `/v1/traces` | 405 / 500 from collector | Add `/v1/traces` so the exporter targets the OTLP HTTP ingest endpoint directly |
| Invalid or user-level API key | 401 Unauthorized | Regenerate/copy a **System Key** from *Settings → API Keys* |
| Protocol warning (“defaulting to HTTP”) | Warning during run but spans still show up | Safe to ignore as long as traces arrive in Phoenix (Cloud is HTTP-only as of May 2025) |
| Docs don’t explicitly mention `/v1/traces` | Confusion when endpoints rejected | Empirically confirmed the OTLP subpath is required for Cloud ingest |

#### 4. Verification steps
1. Run `python3 script.py --config config.json`.  
2. In Phoenix, open your space → find the project card (e.g., “Toy LLM System”) → confirm trace counts/latency update.  
3. Drill into a trace to see spans `toy_llm.run`, `toy_llm.iteration`, and `toy_llm.openai_call`.  
4. Ignore protocol warnings if traces arrive correctly in Phoenix UI.
5. Locally inspect `traces/run.jsonl` for a backup log of the run.

---

## Toy Bench (Q1 scaffolding)

The `toy_bench/toy_tabular` package adds a tiny internal benchmark for prototyping context policies and Phoenix tracing before moving to MLAgentBench.

1. **Install extra dependency**
   ```bash
   pip install scikit-learn
   ```
2. **Run the standalone agent loop**
   ```bash
   python -m toy_bench.toy_tabular.toy_agent
   ```
   This performs a baseline run plus three LLM-guided (or heuristic) config proposals, saving artifacts inside `toy_bench/toy_tabular/workspace/`.
3. **Run under Phoenix tracing (opt-in)**
   ```bash
   python run_toy_bench.py --config config.json --num-steps 3
   ```
   The script reuses the existing telemetry settings in `config.json`/`.env`, wraps the run in a `toybench.toy_tabular_run` span, and records step-level spans if tracing is enabled.

All generated data (`data.npy`, `labels.npy`, `results.json`) remains local to the workspace and is ignored by git. Adjust defaults (e.g., number of steps) via the `"toy_bench"` block in `config.json`.

---

## NOMAD Kaggle Benchmark

**What is this dataset?**  
The [Nomad 2018 Predict Transparent Conductors](https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors) challenge asks you to predict the **bandgap energy (eV)** of candidate transparent conductors. Each row (≈2.4k total) represents a simulated crystal with:

- crystal symmetry metadata (`spacegroup`, `number_of_total_atoms`)
- composition ratios for Al/Ga/In
- lattice vectors/angles
- pre-computed formation energy per atom

The bandgap determines whether a material conducts and remains optically transparent, so the benchmark evaluates how well the model captures subtle relationships between composition, lattice geometry, and energy.

**How we use it**  
We keep everything under `benchmarks/nomad/` so it stays isolated from the toy bench. A helper script downloads the raw CSVs into `kaggle-data/nomad/raw/`, another script converts them into normalized NumPy arrays plus summary stats, and an iterative LLM loop tunes a `HistGradientBoostingRegressor` against MAE while logging every step to Phoenix. Each iteration passes the previous configs + metrics and a condensed dataset context blob to the LLM so it has full awareness of what happened earlier.

### Prerequisites
1. **Join the competition** – sign in to Kaggle, open the [Nomad competition page](https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors), click *Late Submission* and accept the terms/rules. Without this, the API (and often the UI) will block downloads with a 403 error.
2. **Kaggle credentials** – place `~/.kaggle/kaggle.json` (or export `KAGGLE_USERNAME`/`KAGGLE_KEY`) so the Kaggle CLI/SDK can authenticate.
3. **Dependencies** – ensure `pip install kaggle pandas scikit-learn` (already listed in the main setup section).

### Workflow

1. **Stage raw CSVs (manual-first)**
   ```bash
   mkdir -p kaggle-data/nomad/raw
   kaggle competitions download -c nomad2018-predict-transparent-conductors -p kaggle-data/nomad/raw
   (cd kaggle-data/nomad/raw && unzip nomad2018-predict-transparent-conductors.zip && unzip train.csv.zip && unzip test.csv.zip && unzip sample_submission.csv.zip)
   ```
   The `kaggle competitions download` command assumes you already joined/accepted rules for this competition. If you cannot gain access, download from the Kaggle UI and drop `train.csv`, `test.csv`, and `sample_submission.csv` into `kaggle-data/nomad/raw`.  
   Optional helpers:
   - Copy from an existing mirror: `python scripts/fetch_nomad.py --local-source /path/to/raw_dir`
   - Re-download with API access: `python scripts/fetch_nomad.py --dataset competitions/nomad2018-predict-transparent-conductors`

2. **Prepare arrays + context metadata**
   ```bash
   python scripts/prepare_nomad.py --float32
   ```
   This validates the presence of `train.csv` and `test.csv`, then emits `features.npy`, `targets.npy`, `dataset_context.json`, etc., into `benchmarks/nomad/workspace/`.

3. **Run the iterative bench with Phoenix tracing**
   ```bash
   python run_nomad_bench.py --config config.json --num-steps 3
   ```
   The LLM (or heuristic fallback) proposes new hyperparameters for a `HistGradientBoostingRegressor` using prior metrics + configs, and each iteration is logged to Phoenix (`nomad.bench.iteration` spans). Adjust defaults via the `"nomad_bench"` block in `config.json`.

Each NOMAD iteration records the previous step’s metric/config along with dataset context so Phoenix traces show exactly what information the model had when proposing changes. The final JSON result contains the full history so you can audit how the agent explored the hyperparameter space.

## Agentic context policies (short vs long) and modes (agentic vs controller)

We now support explicit context policies and reasoning modes so you can compare:
- `policy_type`: `short_context` (tight retrieval, encourages clarifications) vs `long_context` (large retrieval budget, fewer clarifications).
- `reasoning_mode`: `agentic` (ReAct-style loop with tools) vs `controller` (legacy single-shot loop).

Configuration
- See `config.json`:
  - Top-level `policy_type` and `reasoning_mode` defaults.
  - `context_policies` block defines retrieval budgets per policy.
  - `agentic` block defines agent loop parameters (model, temperature, max_steps, max_tokens, log path, result schema).
- You can override via CLI flags on any entry point that supports them.

Run the NOMAD task with different settings
- Agentic + short context:
  ```bash
  python run_nomad_bench.py --config config.json --policy-type short_context --reasoning-mode agentic --num-steps 3
  ```
- Agentic + long context:
  ```bash
  python run_nomad_bench.py --config config.json --policy-type long_context --reasoning-mode agentic --num-steps 3
  ```
- Controller (legacy) + short context (default):
  ```bash
  python run_nomad_bench.py --config config.json --policy-type short_context --reasoning-mode controller --num-steps 3
  ```

What gets logged
- Phoenix spans:
  - `nomad.bench.run`: one span per full NOMAD run.
  - `nomad.bench.iteration`: one per training iteration (after applying a proposal) with metrics/configs.
  - `agent.runner.step`: only in agentic mode; one per LLM/tool step inside an iteration (captures thought/action/usage).
  - All spans include `policy_type` and `reasoning_mode`.
- Local traces (default, no extra config): each NOMAD run is a single JSONL file under `traces/nomad/` named `nomad_<timestamp>.jsonl` with events `run.start`, `op.*`, `step.summary`, and `agent.iteration` (the last embeds the full iteration payload, including clarifying questions/answers and usage). Clarifying questions are answered within the same iteration and carried into subsequent iterations via `clarification_hints`, so the clarifier tool can reuse past answers.

Current NOMAD workflow (high level overview)
1) Baseline
   - Train `HistGradientBoostingRegressor` once; log metrics/config to traces as step 0.
2) Per iteration (agentic mode)
   - Build state with dataset context, recent history (configs/metrics/clarifications), and `clarification_hints` carried from prior Q&A.
   - LLM chooses an action: retrieve, summarize, ask clarifying, or final_answer.
   - Tools run; observations (including clarifier answers) are fed back into the step and promoted into `clarification_hints` so later iterations can reuse them.
   - If a proposal is produced, apply it, retrain, and record results.
3) Logging
   - Phoenix spans for run/iteration.
   - Local JSONL per run in `traces/nomad/nomad_<timestamp>.jsonl` with `agent.iteration` entries that capture prompts, steps, tool outputs, clarifying flags/questions/answers, and usage.
4) Inspecting outputs
   - Final metrics/configs are printed to the console.
   - Detailed per-step traces live in the JSONL file; Phoenix shows spans if telemetry is enabled.

Quick reproduction checklist
1) Install deps and set `.env` (OpenAI + Phoenix keys).  
2) Prepare NOMAD data (`scripts/prepare_nomad.py`).  
3) Choose policy/mode and run:
   ```bash
   python run_nomad_bench.py --config config.json --policy-type short_context --reasoning-mode agentic --num-steps 3
   ```
4) Inspect:
   - Phoenix UI spans (`nomad.bench.run`, `nomad.bench.iteration`, `agent.runner.step` when agentic).
   - `traces/nomad/nomad_<timestamp>.jsonl` for step-by-step details (including clarifiers).
   - CLI JSON output for final config/metrics.

> **Tip:** After installing the Kaggle CLI inside the venv, you can verify it’s wired up by running `kaggle --version`. If the command succeeds only when the venv is active, you’re configured correctly.

## Visualize With Plots
This command generates plots based on the data in `all_results.json` (file that contains all results from each iteration of a single run).
```bash
python toy_bench/toy_tabular/workspace/plot.py
```

Currently, plots plot...

1. **Iteration vs Accuracy**