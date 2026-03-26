# NetLab

Metrics and autonomous research tools for [NetGraph](https://github.com/networmix/NetGraph) network simulations.

## What It Does

NetLab has two capabilities:

1. **Metrics pipeline** — computes verified reliability metrics (BAC, latency, alpha) from ngraph simulation results. Per-direction, occurrence-count-weighted, hand-verified against 252 assertions.

2. **Autoresearch** — LLM-driven topology exploration. Describe a connectivity idea in natural language, and the system generates a valid ngraph scenario, runs the simulation, computes metrics, and produces a structural interpretation. The LLM also proposes the next experiment, closing the research loop.

## Installation

```bash
pip install netlab
```

From source:

```bash
git clone https://github.com/networmix/NetLab
cd NetLab
make dev
```

## Metrics

### CLI

```bash
# Compute metrics for all scenarios in a directory
netlab metrics path/to/scenarios/

# Summary tables only, no plots
netlab metrics path/to/scenarios/ --no-plots

# Filter specific scenarios
netlab metrics path/to/scenarios/ --only small_clos,small_dragonfly
```

### Python API

```python
from metrics.bac import compute_bac
from metrics.latency import compute_latency_stretch
from metrics.msd import compute_alpha_star

# Load ngraph results
import json
with open("scenario.results.json") as f:
    results = json.load(f)

# Capacity
alpha = compute_alpha_star(results)
print(f"alpha_star: {alpha.alpha_star}")

# Bandwidth availability (aggregate + per direction)
bac = compute_bac(results, step_name="tm_placement")
print(f"BAC AUC: {bac.auc_normalized:.4f}")
for label, pf in bac.per_flow.items():
    print(f"  {label}: AUC={pf.auc_normalized:.4f}")

# Latency stretch
lat = compute_latency_stretch(results)
print(f"baseline p99: {lat.baseline['p99']:.4f}")
print(f"failure p99:  {lat.failures['p99']:.4f}")
```

### Metrics Reference

| Metric | What it measures |
|--------|-----------------|
| **BAC** | Delivered bandwidth distribution across failure iterations. AUC, quantiles, availability at thresholds, BW at probability levels. Per-direction breakdown. |
| **Latency** | Volume-weighted stretch (cost / baseline cost). p50, p95, p99 percentiles, SLO compliance, WES (weighted excess stretch). |
| **Alpha (MSD)** | Maximum demand multiplier the topology supports before saturation. |
| **SPS** | Fraction of source-destination demand satisfied under failures. |
| **CostPower** | CapEx and power normalized by offered demand and reliable bandwidth. |
| **IterOps** | Failure iteration counts, unique pattern counts, timing. |

All metrics correctly handle ngraph's Monte Carlo deduplication (`occurrence_count` expansion).

## Autoresearch

LLM-driven topology research with verified metrics. Three stages:

```
Hypothesis (natural language)
    ↓
[Generation Loop] LLM → ngraph YAML → inspect → validate → iterate
    ↓
[Simulation] ngraph run (expensive, once)
    ↓
[Analysis] Metrics pipeline (verified) → LLM interprets → proposes next hypothesis
```

### Quick Start

```python
from pathlib import Path
from netlab.autoresearch.hypothesis_manager import HypothesisManager
from netlab.autoresearch.backend import ClaudeCLIBackend
import sys

manager = HypothesisManager(
    project_dir=Path("/tmp/my_research"),
    backend=ClaudeCLIBackend(model="sonnet"),
    ngraph_bin=str(Path(sys.executable).parent / "ngraph"),
)

cycle = manager.run_cycle("""
2-site topology, 3 backbone planes, 100 Gbps cross-site per plane.
Internal 500 Gbps. BB nodes with role: bb.
Demands: 100 Gbps each direction, ECMP.
Failure: single random BB node, 20 iterations.
""")

print(cycle.analysis.metrics_report)      # verified numbers
print(cycle.analysis.interpretation)       # LLM explanation
print(cycle.analysis.next_hypothesis)      # what to test next
```

### Multi-Cycle Research

```python
hypothesis = "your initial idea..."
for i in range(5):
    cycle = manager.run_cycle(hypothesis)
    print(f"Cycle {cycle.cycle_id}: {cycle.status}")
    hypothesis = cycle.analysis.next_hypothesis  # LLM proposes next
```

### What Gets Persisted

```
project_dir/
  cycle_log.jsonl           # one-line summary per cycle
  cycles/001/
    hypothesis.yml          # what was tested
    scenario.yml            # generated ngraph YAML
    results/                # ngraph simulation output
    metrics_report.md       # verified numbers (machine-generated)
    interpretation.md       # structural explanation (LLM-generated)
    next_hypothesis.md      # suggested next experiment (LLM-generated)
    status.yml              # analyzed | failed | skipped
```

### Key Design Decision

The LLM never extracts numbers from results. The metrics pipeline (same code that passed 252 hand-calculated assertions) computes all numbers programmatically. The LLM receives verified metrics and provides only interpretation — connecting numbers to topology structure.

### CLI

```bash
# Template-based runner (parameter sweep with LLM feedback)
netlab autoresearch init --base-scenario scenario.yml --output project/
netlab autoresearch run project/ --backend claude-cli --model sonnet

# DC-BB structural analysis and parametric sweep
netlab autoresearch structural-analysis
netlab autoresearch sweep abc1 --output-dir results/
netlab autoresearch cross-sweep --output-dir results/
```

## Repository Structure

```
metrics/                    # Verified metrics pipeline
  common.py                 # Shared: expand_flow_results, canonical_dc
  bac.py                    # Bandwidth availability curve
  latency.py                # Latency stretch analysis
  msd.py                    # Maximum supported demand
  sps.py                    # Structural pair survivability
  iterops.py                # Iteration counts and timing
  aggregate.py              # Cross-seed aggregation
  costpower.py              # Cost and power normalization
  matrixdump.py             # Per-pair placement matrices
netlab/
  cli.py                    # CLI entry point
  metrics_cmd.py            # Metrics command orchestration
  autoresearch/
    generation_loop.py      # Inner Loop 1: idea → validated YAML
    analysis_loop.py        # Inner Loop 2: metrics → interpretation
    metrics_report.py       # Programmatic metrics → markdown
    hypothesis_manager.py   # Outer loop: hypothesis cycles + persistence
    backend.py              # LLM backends (Claude CLI, Codex CLI, OpenAI, mock)
    scenario_generator.py   # DC-BB topology generator
    sweep.py                # Parametric sweep runner
tests/
  data/mini_dcbb.yaml       # 10-node verification scenario
  test_mini_dcbb_verification.py  # 252 hand-calculated assertions
```

## Development

```bash
make dev        # Setup environment
make check      # Pre-commit + tests + lint
make test       # Tests only
make lint       # Linting only
make qt         # Quick tests (skip slow)
```

## Requirements

- Python 3.11+
- [ngraph](https://github.com/networmix/NetGraph) >= 0.21.0
- [netgraph-core](https://github.com/networmix/NetGraph-Core) >= 0.7.0

## License

[MIT License](LICENSE)
