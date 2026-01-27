# DSC180A-Q1Project

## Overview

Evaluating how context policies shape the reliability and efficiency of LLM agents that design, run, and interpret ML experiments.

Inspired by [MLAgentBench](https://arxiv.org/pdf/2310.03302) and [MLEBench](https://arxiv.org/pdf/2410.07095).

![Architecture](figs/new_architecture.png)

## Quick Start

```bash
# Toy benchmark (logistic regression tuning)
python run_toy_bench.py --config config.json --num-steps 3

# NOMAD benchmark (materials science regression)
python run_nomad_bench.py --config config.json --num-steps 3
```

## Setup

1. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install openai pandas scikit-learn
   ```

3. **Configure API keys** - Create `.env` in repo root:
   ```
   OPENAI_API_KEY=<your-key>
   ```

## Configuration

Edit `config.json` to adjust:
- `model`: LLM model (default: `gpt-4o-mini`)
- `context_policies`: Configure chunk limits and summary sizes for short/long context modes

## CLI Options

| Flag | Description |
|------|-------------|
| `--config` | Path to config file |
| `--num-steps` | Number of optimization iterations |
| `--seed` | Random seed |
| `--run-id` | Custom run identifier |
| `--output-dir` | Custom output directory for traces |

## Project Structure

```
src/
├── benchmarks/     # BaseBenchmark + implementations
│   ├── base.py
│   ├── nomad/      # NOMAD benchmark
│   └── toy/        # Toy benchmark
└── utils/          # Config, logging, CLI utilities

traces/             # Output JSONL traces
```

## NOMAD Data Setup

### 1. Download Data Files

**Option A: Manual Download (Recommended)**
1. Go to [NOMAD Kaggle Competition](https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/data)
2. Accept competition rules and download the data files
3. Place `train.csv` and `test.csv` in `kaggle-data/nomad/raw/`

**Option B: Kaggle CLI**
```bash
# Requires Kaggle CLI + competition acceptance
kaggle competitions download -c nomad2018-predict-transparent-conductors -p kaggle-data/nomad/raw
cd kaggle-data/nomad/raw && unzip *.zip
```

### 2. Create Task Description Files

Manually create these files in `src/benchmarks/nomad/workspace/`:

- **`task_description.txt`** - Copy the competition overview/description from the Kaggle page
- **`metric_description.txt`** - Copy the evaluation metric description from the Kaggle page

### 3. Prepare Dataset

```bash
python scripts/prepare_nomad.py --float32
```

This generates `features.npy`, `targets.npy`, and metadata files in the workspace.
