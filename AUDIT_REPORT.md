# Hidden / Unused Artifact Audit Report

**Date:** 2025-02-05  
**Scope:** Complete read-only inspection of ContextEval repository  
**Goal:** Identify unused, legacy, misleading, or vestigial code that could confuse experimenters or reviewers

---

## Executive Summary

**Status: WARN** ⚠️

The codebase is **largely clean** with a few minor unused artifacts that should be removed for clarity. No blocking issues found, but several low-to-medium risk items that could confuse reviewers.

---

## 1. Confirmed Clean ✅

### CLI Flags & Arguments
- **All CLI flags are actively used:**
  - `--num-steps` → passed to `run_nomad_bench` / `run_toy_tabular` → `BenchmarkConfig`
  - `--seed` → passed through → `BenchmarkConfig`
  - `--output-dir` → used to set `TRACES_ROOT`
  - `--run-id` → passed through → `benchmark.run()`
  - `--model` → passed through → `BenchmarkConfig`
  - `--temperature` → passed through → `BenchmarkConfig`
  - `--show-task` → passed through → `BenchmarkConfig` → `ContextAxes`
  - `--show-metric` → passed through → `BenchmarkConfig` → `ContextAxes`
  - `--show-resources` → passed through → `BenchmarkConfig` → `ContextAxes`
  - `--history-window` → passed through → `BenchmarkConfig` → `ContextAxes`

### Benchmark Methods
- **All abstract methods are implemented and used:**
  - `get_default_config()` → called in `BaseBenchmark.run()` line 395
  - `run_training()` → called in `BaseBenchmark.run()` lines 413, 455
  - `fallback_config()` → called in `BaseBenchmark.run()` line 438
  - `sanitize_config()` → called in `BaseBenchmark.run()` line 443
  - `_get_primary_score()` → used by `ContextBuilder`
  - `_get_llm_system_prompt()` → used in `_direct_llm_propose()` line 276
  - `_build_llm_user_prompt()` → used in `_direct_llm_propose()` line 277

### Context / Trace Layers
- **All ContextAxes fields are used:** `history_window`, `show_task`, `show_metric`, `show_resources`
- **All ContextBundle fields are used:** `current_config`, `latest_score`, `recent_history`, `task_description`, `metric_description`, `resource_summary`
- **All trace fields are logged and traceable:** No "just in case" logging found

### Tests
- **All tests are valid and test active features:**
  - `test_context_leakage.py` tests the trace/context boundary enforcement
  - Tests cover `ContextBundle`, `ContextAxes`, `ContextBuilder`, `TRACE_ONLY_FIELDS`
  - No tests for removed features found

### Documentation
- **README.md is accurate:** All `--config` references removed, no legacy feature mentions

---

## 2. Unused / Legacy Artifacts Found

### HIGH RISK ⚠️

**None found.**

### MEDIUM RISK ⚠️

#### 1. Unused Constant: `DEFAULT_HISTORY_WINDOW`
- **Location:** `src/benchmarks/nomad/benchmark.py:14`
- **Code:** `DEFAULT_HISTORY_WINDOW = 5`
- **Why unused:** Never referenced anywhere in the codebase. The default is now hardcoded in CLI (`cli.py:44`) and `BenchmarkConfig` (`base.py:115`).
- **Risk:** Medium - Could confuse reviewers who see this constant and wonder if it's used.
- **Recommendation:** **REMOVE** - Dead code, no semantic meaning.

#### 2. Unused Method: `read_context()`
- **Location:** 
  - `src/benchmarks/nomad/env.py:40-44`
  - `src/benchmarks/leaf/env.py:41-45`
  - `src/benchmarks/mercor/env.py:42-46`
- **Code:** `def read_context(self) -> Dict[str, Any]:`
- **Why unused:** Method is defined but never called. `dataset_context.json` files are created by prepare scripts but never read by benchmark code.
- **Risk:** Medium - Suggests context loading capability that doesn't exist, could mislead about architecture.
- **Recommendation:** **REMOVE** - If dataset context is not part of the current design, remove these methods. If it's planned for future use, document it clearly.

#### 3. Unused Parameter: `extra_args` in `parse_benchmark_args()`
- **Location:** `src/utils/cli.py:13, 20, 47-49`
- **Code:** `extra_args: Optional[List[Tuple[str, Dict[str, Any]]]] = None`
- **Why unused:** Parameter exists and is processed, but no caller ever passes `extra_args`. Both `run_nomad_bench.py` and `run_toy_bench.py` call it without this parameter.
- **Risk:** Medium - Suggests extensibility that isn't used, adds complexity.
- **Recommendation:** **REMOVE** - If not needed, remove. If kept for future benchmarks, add a comment explaining when it would be used.

### LOW RISK ℹ️

#### 4. Unused Workspace Files
- **Locations:**
  - `src/benchmarks/nomad/workspace/dataset_context.json` - Created but `read_context()` never called
  - `src/benchmarks/nomad/workspace/prepared_meta.json` - Created but never read
  - `src/benchmarks/toy/workspace/all_results.json` - Used in `train.py` but gitignored, appears to be debug-only
- **Why unused:** Files are created by prepare scripts but not consumed by benchmark execution.
- **Risk:** Low - These are workspace artifacts, not code. However, if they're not used, they add confusion.
- **Recommendation:** **DOCUMENT or REMOVE** - Either document that these are metadata/debug files, or remove them if truly unused.

#### 5. Dead Code Path: `proposal_source = "agent"`
- **Location:** `src/benchmarks/base.py:130` (comment)
- **Code:** `proposal_source: str  # "llm", "agent", "heuristic", "baseline"`
- **Why unused:** Comment mentions "agent" as a possible value, but code only ever sets:
  - `"baseline"` (line 418)
  - `"llm"` (line 282)
  - `"heuristic"` (line 283, 439)
- **Risk:** Low - Comment is misleading but doesn't affect execution.
- **Recommendation:** **UPDATE COMMENT** - Remove "agent" from the comment, or document why it's reserved for future use.

---

## 3. Potentially Confusing Artifacts

### 1. Workspace File Purpose Ambiguity
- **Issue:** Several workspace files exist but their purpose isn't clear:
  - `dataset_context.json` - Created but never read
  - `prepared_meta.json` - Created but never read
  - `all_results.json` - Used in toy benchmark but gitignored
- **Confusion Risk:** Reviewers might think these are part of the active system.
- **Recommendation:** Add comments in prepare scripts explaining these are metadata/debug files, or remove if truly unused.

### 2. `extra_args` Parameter Suggests Extensibility
- **Issue:** `parse_benchmark_args()` has `extra_args` parameter that's never used.
- **Confusion Risk:** Suggests the system supports benchmark-specific CLI args, but none exist.
- **Recommendation:** Remove if not needed, or add example usage in a comment.

---

## 4. Final Verdict

**Status: WARN** ⚠️

The codebase is **research-ready** with minor cleanup recommended. The unused artifacts are low-risk but should be addressed for clarity:

### Blockers: None
- No blocking issues found
- All active code paths are clean
- Architecture is sound

### Recommended Actions (Non-blocking):
1. **Remove** `DEFAULT_HISTORY_WINDOW` constant (1 line)
2. **Remove or document** `read_context()` methods (3 methods)
3. **Remove or document** `extra_args` parameter (1 parameter)
4. **Update comment** for `proposal_source` to remove "agent" reference
5. **Document or remove** unused workspace files (`dataset_context.json`, `prepared_meta.json`)

### Codebase Quality Assessment:
- ✅ **CLI flags:** All used, no dead flags
- ✅ **Benchmark methods:** All implemented and called
- ✅ **Context/Trace layers:** Clean separation, no leakage
- ✅ **Tests:** Valid, test active features
- ✅ **Documentation:** Accurate, no legacy references
- ⚠️ **Constants:** 1 unused constant found
- ⚠️ **Methods:** 3 unused methods found
- ⚠️ **Parameters:** 1 unused parameter found

**Overall:** The codebase is in excellent shape. The identified issues are minor and easily addressed. No architectural concerns.

---

## Appendix: Detailed Findings

### Unused Constant Details

```python
# src/benchmarks/nomad/benchmark.py:14
DEFAULT_HISTORY_WINDOW = 5  # ❌ Never referenced
```

**Usage check:**
```bash
$ grep -r "DEFAULT_HISTORY_WINDOW" src/
src/benchmarks/nomad/benchmark.py:14  # Only definition, no usage
```

### Unused Method Details

```python
# src/benchmarks/nomad/env.py:40-44
def read_context(self) -> Dict[str, Any]:
    """Read the dataset context file."""
    if not self.context_path.exists():
        return {}
    return json.loads(self.context_path.read_text())
```

**Usage check:**
```bash
$ grep -r "\.read_context\|read_context(" src/
# No matches found - method never called
```

### Unused Parameter Details

```python
# src/utils/cli.py:13
def parse_benchmark_args(
    description: str,
    extra_args: Optional[List[Tuple[str, Dict[str, Any]]]] = None,  # ❌ Never used
) -> argparse.Namespace:
```

**Usage check:**
```bash
$ grep -r "parse_benchmark_args" run_*.py
run_nomad_bench.py:12:    args = parse_benchmark_args("Run NOMAD benchmark",)  # No extra_args
run_toy_bench.py:12:    args = parse_benchmark_args("Run Toy tabular benchmark",)  # No extra_args
```

---

**Report Generated:** 2025-02-05  
**Auditor:** Claude Code (CC)  
**Methodology:** Comprehensive grep-based analysis + semantic code search
