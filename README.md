# context-eval

## Overview

ContextEval is a benchmarking framework that isolates and studies the causal effects of context visibility on LLM agent behavior in iterative machine learning workflows, without modifying model architecture or prompt structure.

## Quick Start

```bash
# Single benchmark runs
python run_toy_bench.py --num-steps 5
python run_nomad_bench.py --num-steps 5 --show-task --show-metric

# Experiment grids (48 runs across all context axis combinations)
./scripts/run_grid.sh toy --dry-run    # preview commands
./scripts/run_grid.sh toy              # run full grid
./scripts/run_grid.sh nomad
```

Results saved to `logs/` (traces in `logs/traces/`, run summaries in `logs/runs/`).

---

## Setup

### 1. Clone & Create Environment

```bash
git clone <repo-url>
cd context-eval
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
   - [Jigsaw Toxic Comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) - Click "Late Submission"

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

### Jigsaw Dataset (Toxic Comment Classification - Text Data)

```bash
# 1. Fetch from Kaggle
python3 scripts/fetch_jigsaw.py

# 2. Unzip the nested archives
cd kaggle-data/jigsaw/raw
unzip -o jigsaw-toxic-comment-classification-challenge.zip
unzip -o train.csv.zip
cd ../../..

# 3. Prepare offline artifacts
python3 scripts/prepare_jigsaw.py
```

Prepared data saved to `src/benchmarks/jigsaw/workspace/`.

---

## CLI Options

### Context Axes
These flags control what information the LLM agent sees:

| Flag | Default | Description |
|------|---------|-------------|
| `--show-task` | off | Include task description in LLM prompt |
| `--show-metric` | off | Include metric description in LLM prompt |
| `--show-bounds` | off | Include parameter bounds (valid ranges) in LLM prompt |
| `--feedback-depth` | 1 | Feedback depth: number of visible outcome signals (1=current only, 5=current+4 history) |

### Experiment Settings

| Flag | Default | Description |
|------|---------|-------------|
| `--num-steps` | 3 | Number of optimization iterations |
| `--seed` | 0 | Random seed for reproducibility |
| `--run-id` | auto | Custom run identifier |
| `--experiment-id` | default | Experiment ID for grouping runs (must be filesystem-safe: `[a-zA-Z0-9_-]+`) |
| `--model` | gpt-4o-mini | LLM model to use |
| `--temperature` | 0 | LLM temperature setting |

### Developer Tools

| Flag | Default | Description |
|------|---------|-------------|
| `--verbose` | off | Enable step-by-step logging |
| `--debug-show-llm` | off | Print full LLM request and response for debugging |
| `--debug-show-diff` | off | Show config changes at each step with score delta |

## Running Benchmarks

### Available Benchmarks

| Script | Dataset | Task Type |
|--------|---------|-----------|
| `run_toy_bench.py` | Synthetic | Logistic regression tuning |
| `run_nomad_bench.py` | NOMAD 2018 | Materials science regression |
| `run_jigsaw_bench.py` | Jigsaw Toxic Comments | Multi-label text classification |

