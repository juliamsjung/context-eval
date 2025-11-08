# DSC180A-Q1Project

## Overview
Our project aims to explore how to evaluate LLM systems with a focus on the context that shapes their performance. Rather than evaluating the models or agents directly, we are interested in studying the surrounding factors that influence outcomes, including prompts, tools, external data, and memory. Since this is a broad and relatively new area of research, our initial efforts have centered on developing a strong conceptual foundation and identifying feasible directions before beginning implementation.

## What We’ve Done (Weeks 1–5)
During the first two to three weeks, our group focused on reading existing papers and reviewing resources related to LLM evaluation. We studied existing metrics, benchmark designs, and evaluation frameworks to understand how performance is currently measured and what aspects of context are often overlooked. This background research helped us define the gaps that our project could address.

Then, we transitioned to idea generation and refinement. We began with five possible directions and met regularly with our mentor to discuss feedback and feasibility. Through several iterations, we narrowed these ideas to two promising approaches and conducted additional research to understand their potential. Eventually, we decided to combine both directions into a single, cohesive project.

A major reason why this process took additional time is the nature of our core research question: “How could or should we evaluate context?” This question represents an open challenge in the field, since there is currently no definitive answer or established approach. Because the Quarter 1 project is recreating an existing solution, it was difficult to find prior work that directly addresses our specific question. This required us to spend more time exploring the problem space and understanding where our contribution could fit.

## Task: ML Experimentation (Q1 Scope)
We evaluate agents **as ML experimenters**. An agent plans, writes code, calls tools, and manages files to complete compact ML tasks. Heavily inspired by Stanford's MLAgentBench (https://arxiv.org/pdf/2310.03302) and OpenAI's MLEBench (https://arxiv.org/pdf/2410.07095).

![](figs/architecture.png)

**Task format:**
- **Actions:** edit files, run tests/metrics, parse tracebacks, update prompt/tool/memory.
- **Modalities:** start with **text/tabular**; optional image/vision later.
- **Stop criteria:** produce a **valid structured output** (e.g., metrics JSON) or meet a **task-native target** (e.g., ≥10% over a starter baseline) within token/time/iteration budgets.

## Stucture (Placeholder)

- `code.py` contains library code — functions that will power a toy LLM system (soon via the OpenAI SDK with a GPT-5 key). For now, it exposes a minimal `run()` so the repo is executable.
- `config.json` contains parameters for the functions in `code.py` (e.g., model id, simple task input, basic budgets).
- `script.py` imports `code`, loads `config.json`, and calls functions from the `code` module. (This could also be a notebook, `script.ipynb`.)

Post checkpoint steps: Develop a **toy LLM system** using OpenAI SDK + GPT-5 key (as recommended by our mentor), then a small replication of a lightweight agent task to exercise the loop and metrics.

---

## Setup (Placeholder)

```bash
python3 script.py --config config.json