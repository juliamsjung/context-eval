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
- `policy_type`: `short_context` or `long_context`
- `reasoning_mode`: `agentic` (ReAct loop) or `controller` (single-shot)

## CLI Options

| Flag | Description |
|------|-------------|
| `--config` | Path to config file |
| `--num-steps` | Number of optimization iterations |
| `--policy-type` | `short_context` / `long_context` |
| `--reasoning-mode` | `agentic` / `controller` |
| `--seed` | Random seed |
| `--run-id` | Custom run identifier |
| `--output-dir` | Custom output directory for traces |

## Project Structure

```
src/
├── agent/          # AgentRunner, policies, tools
├── benchmarks/     # BaseBenchmark + implementations
│   ├── base.py
│   ├── nomad/      # NOMAD benchmark
│   └── toy/        # Toy benchmark
└── utils/          # Config, logging, CLI utilities

traces/             # Output JSONL traces
```

## NOMAD Data Setup

```bash
# Requires Kaggle CLI + competition acceptance
kaggle competitions download -c nomad2018-predict-transparent-conductors -p kaggle-data/nomad/raw
cd kaggle-data/nomad/raw && unzip *.zip
python scripts/prepare_nomad.py --float32
```
