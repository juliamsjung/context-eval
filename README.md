# DSC180A-Q1Project

## Overview

Evaluating how context policies shape the reliability and efficiency of LLM agents that design, run, and interpret ML experiments.

Inspired by [MLAgentBench](https://arxiv.org/pdf/2310.03302) and [MLEBench](https://arxiv.org/pdf/2410.07095).

![Architecture](figs/new_architecture.png)

## Quick Start

```bash
# Toy benchmark (logistic regression tuning)
python run_toy_bench.py --num-steps 3

# NOMAD benchmark (materials science regression)
python run_nomad_bench.py --num-steps 3 --show-task --show-metric

# With resource_summary visible to agent
python run_nomad_bench.py --num-steps 3 --show-task --show-metric --show-resources

# With custom model and temperature
python run_toy_bench.py --num-steps 3 --model gpt-4o --temperature 0.5
```

---

## Setup

### 1. Clone & Create Environment

```bash
git clone <repo-url>
cd DSC180A-Q1Project
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create `.env` in repo root:

```
OPENAI_API_KEY=<your-openai-key>
```

---

## Kaggle Dataset Setup

We use Kaggle datasets as data sources. Benchmarks run **offline** using prepared artifacts—no Kaggle access during experiments.

### Prerequisites

1. **Create Kaggle API credentials**:
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings) → Scroll to "API" → Create Legacy API Key
   - This downloads `kaggle.json`. It should be saved in your `/Downloads` folder.

2. **Install the credentials**:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Join the competitions** (accept rules on each page):
   - [NOMAD 2018](https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors) - Click "Late Submission"
   - [Leaf Classification](https://www.kaggle.com/c/leaf-classification) - Click "Late Submission"

---

### NOMAD Dataset (Materials Science Regression - Tabular Data)

```bash
# 1. Fetch from Kaggle
python3 scripts/fetch_nomad.py

# 2. Unzip the nested archives
cd kaggle-data/nomad/raw
unzip -o nomad2018-predict-transparent-conductors.zip
unzip -o train.csv.zip
unzip -o test.csv.zip
cd ../../..

# 3. Prepare offline artifacts
python3 scripts/prepare_nomad.py --float32
```

Prepared data saved to `src/benchmarks/nomad/workspace/`.

---

### Leaf Dataset (Species Classification - Image Data (Tabular Data of the Image's Features))

```bash
# 1. Fetch from Kaggle
python3 scripts/fetch_leaf.py

# 2. Unzip the nested archives
cd kaggle-data/leaf/raw
unzip -o leaf-classification.zip
unzip -o train.csv.zip
unzip -o test.csv.zip
cd ../../..

# 3. Prepare offline artifacts
python3 scripts/prepare_leaf.py --float32
```

Prepared data saved to `src/benchmarks/leaf/workspace/`.

---

## CLI Options

| Flag | Description |
|------|-------------|
| `--num-steps` | Number of optimization iterations |
| `--seed` | Random seed |
| `--run-id` | Custom run identifier |
| `--output-dir` | Custom output directory for traces |
| `--show-task` | Include task description in LLM prompt |
| `--show-metric` | Include metric description in LLM prompt |
| `--show-resources` | Include resource_summary (tokens, cost, latency) in LLM prompt |
| `--history-window` | Number of history entries to include (default: 5, 0=none) |
| `--model` | LLM model to use (default: gpt-4o-mini) |
| `--temperature` | LLM temperature setting (default: 0) |

## Running Benchmarks

### Available Benchmarks

| Script | Dataset | Task Type |
|--------|---------|-----------|
| `run_toy_bench.py` | Synthetic | Logistic regression tuning |
| `run_nomad_bench.py` | NOMAD 2018 | Materials science regression |

### Examples

```bash
# Basic run with default settings
python run_toy_bench.py --num-steps 5

# Full context: task description + metric description + history
python run_nomad_bench.py --num-steps 5 \
    --show-task --show-metric --history-window 5

# Minimal context: no task/metric descriptions, no history
python run_nomad_bench.py --num-steps 5 \
    --history-window 0

# Custom run with seed for reproducibility
python run_nomad_bench.py --num-steps 10 \
    --show-task --show-metric --seed 42 --run-id my-experiment
```

---

## Project Structure

```
scripts/                # Data fetching and preparation
├── fetch_nomad.py
├── fetch_leaf.py
├── prepare_nomad.py
└── prepare_leaf.py

src/
├── benchmarks/         # BaseBenchmark and task-specific implementations
│   ├── base.py
│   ├── nomad/          # NOMAD benchmark
│   ├── leaf/           # Leaf benchmark (WIP)
│   └── toy/            # Toy benchmark
├── context/            # Agent-visible context construction and policies
│   ├── ContextBundle
│   ├── ContextAxes
│   └── ContextBuilder
├── trace/              # Observability layer (RunLogger, JSONL trace schema)
│   ├── RunLogger
│   └── TRACE_ONLY_FIELDS
└── utils/              # CLI utilities, configuration helpers

kaggle-data/            # Raw Kaggle downloads (gitignored)
traces/                 # Output JSONL traces
```

