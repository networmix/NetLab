# NetLab

Metrics aggregation and statistical analysis for [NetGraph](https://github.com/networmix/NetGraph) simulation results.

## Overview

NetLab processes NetGraph workflow outputs (JSON artifacts) to compute statistical metrics across random seeds and failure scenarios. Provides CLI and Python API for batch analysis, cross-seed aggregation, and visualization.

## Features

### Metrics

- **BAC (Bandwidth Availability Curve)**: Delivered bandwidth quantiles, availability at thresholds, AUC normalization
- **Latency**: Stretch distributions (p50/p99), tail degradation ratios, SLO compliance, WES (Weighted Excess Stretch)
- **IterationOps**: Per-iteration SPF calls, flows created, reoptimization calls, placement iterations
- **SPS (Structural Pair Survivability)**: Fraction of src-dst pairs meeting demand under failures
- **MSD (Maximum Supported Demand)**: Alpha-star multiplier for traffic matrix scaling capacity
- **CostPower**: CapEx/Power totals, USD/Watt per Gbit (offered and at p99.9 reliability)

### Cross-Seed Analysis

- Positional alignment of time-series data by iteration index
- Median and IQR (interquartile range) computation
- Variable-length series handling with NaN padding

### Visualization

- Cross-seed plots with median curves and IQR bands
- Baseline-normalized delta comparisons
- Statistical significance heatmaps (p-values)

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

## Usage

### CLI

```bash
# Compute metrics for scenarios
netlab metrics tests/data/scenarios/

# With summary tables and plots
netlab metrics tests/data/scenarios/ --summary

# Filter specific scenarios
netlab metrics tests/data/scenarios/ --only small_clos,small_dragonfly

# Skip plot generation
netlab metrics tests/data/scenarios/ --no-plots
```

### Python API

```python
from metrics.bac import compute_bac_metrics
from metrics.aggregate import summarize_across_seeds

# Compute BAC for a workflow step
bac = compute_bac_metrics(iterations, offered_bw, step_name="max_demand")
print(f"p99 availability: {bac['q_pct'][0.99]:.2%}")

# Aggregate across seeds
summary = summarize_across_seeds(series_by_seed, label="latency_p99")
```

## Repository Structure

```
metrics/            # Core metrics modules
  bac.py            # Bandwidth availability
  latency.py        # Latency percentiles
  iterops.py        # Iteration analysis
  aggregate.py      # Cross-seed aggregation
  plot_*.py         # Visualization
netlab/             # CLI
  cli.py            # Command-line interface
  metrics_cmd.py    # Metrics command implementation
tests/              # Test suite
lib/                # Config files
```

## Development

```bash
make dev        # Setup environment
make check      # Tests and linting
make test       # Tests only
make lint       # Linting only
```

## Requirements

- Python 3.11+
- Dependencies: numpy, pandas, matplotlib, seaborn, scipy, rich, ngraph

## License

AGPL-3.0-or-later
