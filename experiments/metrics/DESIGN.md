# Behavioral Stability Metrics — Design Rationale

## Motivation

The `StabilityMetric` class measures **how** an agent searches the hyperparameter space, independent of the final benchmark score it achieves. This enables analysis questions like:

- Does the agent oscillate between configurations or make steady progress?
- Does context visibility (task description, history window, resource usage) influence search patterns?
- Are certain context conditions associated with more erratic exploration?

Unlike task-specific performance metrics (e.g., accuracy, RMSLE), stability metrics are **universal** — they apply to any hyperparameter optimization benchmark without domain-specific knowledge.

---

## Metric 1: Configuration Churn

**What it measures:** The magnitude of change between consecutive hyperparameter configurations, averaged over the full run.

**Why it matters:** High churn indicates large, potentially unfocused jumps in the search space. Low churn suggests incremental refinement. Comparing churn across context conditions can reveal whether agents are more stable when given certain types of information.

### Normalization Scheme

Different hyperparameters have different units and ranges. To produce a meaningful aggregate distance, we normalize each parameter's contribution:

#### Numerical Parameters (`int`, `float`)

```
distance = |value_a - value_b| / (max(|value_a|, |value_b|) + ε)
```

**Rationale:**
- Fractional difference relative to the larger magnitude → invariant to scale.
- Bounded to `[0, 1]` per parameter → prevents large-valued params (e.g., `max_iter=1000`) from dominating small-valued ones (e.g., `lr=0.01`).
- Adding `ε = 1e-9` avoids division by zero when both values are zero.

**Example:**
```python
lr: 0.1 → 0.05  =>  |0.05| / 0.1 = 0.5
max_depth: 10 → 5  =>  |5| / 10 = 0.5
```

#### Categorical Parameters (`str`, `bool`, enums)

```
distance = 0.0  if value_a == value_b  else  1.0
```

**Rationale:**
- Binary penalty — no notion of "partial similarity" between categories (e.g., `"adam"` vs. `"sgd"` is a full change).
- Matches the normalization range `[0, 1]` of numerical params.

**Example:**
```python
optimizer: "adam" → "sgd"  =>  1.0
optimizer: "adam" → "adam" =>  0.0
```

#### Missing Keys

If a key appears in only one config, it counts as a categorical change (`+1.0`).

**Rationale:**
- Represents a structural change (adding/removing a hyperparameter).
- Treats absence as a distinct state from any concrete value.

---

### Aggregation

The pairwise distance is the **sum** of per-parameter distances:

```
distance(config_t, config_{t+1}) = Σ_{key} distance(config_t[key], config_{t+1}[key])
```

Average churn is the mean of all consecutive pairwise distances:

```
average_churn = (1 / N) * Σ_{t=0}^{N-1} distance(config_t, config_{t+1})
```

where `N = total_steps - 1` is the number of transitions.

---

## Metric 2: Instability Score (Oscillation)

**What it measures:** The fraction of steps that revisit a configuration previously seen in the trace.

**Why it matters:** Cycling back to old configurations suggests the agent lacks a coherent search strategy or is failing to retain information about what it has already tried. This can indicate issues with context representation or memory.

### Cycle Detection via Canonical Hashing

To determine if two configurations are identical, we use:

```python
json.dumps(config, sort_keys=True)
```

**Rationale:**
- **Sort keys** → insertion order doesn't matter (`{"lr": 0.1, "depth": 5}` == `{"depth": 5, "lr": 0.1}`).
- **Exact match required** → any change in value (even `0.1 → 0.100001`) counts as a new config. This is intentional: we're detecting true revisits, not approximate returns.

### Counting Repeated Steps

```
instability_score = repeated_steps / total_steps
```

where a **repeated step** is any step whose configuration hash has appeared **at least once before** in the trace.

**Example:**
```
Step 0: {"lr": 0.1}  → first occurrence, not a repeat
Step 1: {"lr": 0.05} → first occurrence, not a repeat
Step 2: {"lr": 0.1}  → matches step 0 → REPEAT
Step 3: {"lr": 0.1}  → matches step 0 and 2 → REPEAT

instability_score = 2 / 4 = 0.5
```

**Interpretation:**
- `0.0` → every step was a novel configuration (no wasted revisits).
- `1.0` → every step was a repeat (theoretically impossible unless the agent never explores).

---

## Design Trade-offs

### Why Not Euclidean Distance?

Euclidean distance requires all parameters to be numeric and assumes a meaningful metric space. Our benchmarks mix numerical and categorical parameters (e.g., `{"lr": 0.1, "optimizer": "adam"}`), so we use component-wise normalization instead.

### Why Sum Instead of Mean for Churn?

Summing distances preserves information about **how many** parameters changed. If two parameters change simultaneously, the churn is `2.0` (not `1.0`). This distinguishes between:
- Incremental single-param tweaks (`churn ≈ 0.5`)
- Wholesale config replacements (`churn ≈ 3.0+`)

### Why Exact Matching for Oscillation?

We considered using a threshold (e.g., "configs within 10% distance are the same"), but:
- Introduces an arbitrary tuning parameter.
- Blurs the distinction between "revisiting" and "refinement."

Exact matching is simpler and more interpretable: a cycle means the agent returned to a **precisely identical** configuration.

---

## Future Extensions

Potential enhancements for future work:

1. **Weighted Churn:** Let domain experts assign importance weights to each hyperparameter (e.g., `lr` might be more critical than `random_seed`).
2. **Approximate Cycle Detection:** Use locality-sensitive hashing or clustering to detect "near-cycles" (e.g., the agent keeps returning to a region of the space).
3. **Directedness Metric:** Measure if churn is correlated with score improvement (high churn + high improvement = exploration; high churn + no improvement = thrashing).
4. **Context-Conditional Analysis:** Stratify churn/instability by context axes (e.g., "instability is 2× higher when `show_task=False`").

---

## References

This metric design was developed to support research questions in the **ContextEval** framework, which isolates the causal effects of context visibility on LLM agent behavior in iterative ML workflows.

For questions or suggestions, see the main [README.md](../../README.md).