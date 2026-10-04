# Metrics and output files

`netlab metrics` writes per-seed results, scenario summaries, and project tables.
Project metrics are medians across seeds unless stated otherwise. Failure samples
are weighted by each pattern's `occurrence_count`; BAC also includes the baseline
as one sample. Availability describes the configured failure experiment, not
measured network uptime.

MSD demand records use `source`, `target`, and `volume`; placement flow records
use `source`, `destination`, and `demand`. Each failure pattern must have a positive
integer `occurrence_count`.

## Capacity and bandwidth

| Column | Definition |
|--------|------------|
| `scenario` | Scenario directory name. |
| `seeds` | Number of seed summaries included. |
| `node_count`, `link_count` | Median topology counts across seeds. |
| `alpha_star` | Demand multiplier returned by the MSD search, subject to its bounds, resolution, and placement settings. |
| `bac_auc` | Mean of `min(delivered / baseline_delivered, 1)` across samples. |
| `bw_p99`, `bw_p999` | Greatest observed bandwidth met or exceeded by at least 99% or 99.9% of samples, divided by baseline delivery. |

BAC uses baseline delivered bandwidth as its normalization value, called
`offered` in the API. For sorted samples and availability probability `p`,
`BW@p` selects zero-based rank `n - ceil(n*p/100)`. Decimal probabilities use
exact count boundaries: for `[0,...,9]`, `BW@90 = 1`, since nine of ten values
are at least 1. This inverse-survival threshold differs from a descriptive
percentile. `quantiles_abs`, `quantiles_pct`, SPS `tails`, pair matrices and
sweep percentile arrays use the NumPy `lower` estimator, rank `floor((n-1)*q)`.
Raw ratios and curves are uncapped; AUC is capped at 1.

BAC, pair matrices, and SPS sum priority classes with the same exact source and
destination. Fully disconnected iterations retain their zero bandwidth samples.
Survival curves use exact inclusive thresholds, combining ties without linear
interpolation.

Optional structural pair survivability (SPS) uses independently computed MaxFlow
capacities: `sum(min(pair_capacity, pair_demand)) / sum(pair_demand)` per failure
iteration. It does not measure simultaneous traffic placement. Enable it with
`--enable-maxflow` and a `node_to_node_capacity_matrix` step. Its baseline must
cover the exact placement source/destination pairs. For physical pairwise
traffic, use full-name capture groups in MaxFlow selectors, such as `^(A)$`
and `^(B)$`; selectors without captures label groups with the regex itself.
Group or virtual endpoint labels that do not match placement pairs raise an error.
`SPS@p` uses the same inverse-survival threshold as BAC, over failure samples only.

## Latency stretch

Each exact source/destination pair has its own reference: the minimum
cost carrying nonzero volume in its baseline `cost_distribution`. Stretch is
path cost divided by that reference. Percentiles and shares are weighted by
delivered volume. Failure summaries take the median across iterations, then
project tables take the median across seeds.

| Column | Definition |
|--------|------------|
| `lat_base_p50` | Baseline median stretch. |
| `lat_fail_p99` | Median of the failure iterations' volume-weighted p99 stretches. |
| `lat_TD99` | Failure p99 summary divided by baseline p99. |
| `lat_SLO_1_2_drop` | Baseline share with stretch ≤1.2 minus the median failure share. |
| `lat_best_path_drop` | Baseline share at the reference cost minus the median failure share. |
| `lat_WES_delta` | Median failure weighted excess stretch minus baseline weighted excess stretch. |

Weighted excess stretch is the volume-weighted mean of `max(stretch - 1, 0)`.
Latency metrics describe delivered traffic with a positive baseline cost reference;
dropped traffic contributes no path cost. `reference_coverage` in per-seed JSON
records the included fraction of delivered volume. It is zero for missing path
details and undefined when nothing is delivered. Fully disconnected iterations
retain missing entries and are excluded from latency medians. Read latency alongside BAC and reference coverage.
`lat_fail_p99` describes a typical iteration's tail, not the worst failure.

## Iterations, timing, and cost

| Column | Definition |
|--------|------------|
| `iters_fail` | Sum of failure-pattern occurrence counts. |
| `iters_total` | Failure iterations plus one baseline. |
| `unique_patterns` | Number of distinct failure records. |
| `tm_duration_total_sec` | Recorded `tm_placement` duration in seconds. |
| `tm_duration_per_iter_sec` | Total duration divided by total iterations. |
| `capex_total` | Total capital cost reported by the `cost_power` step. |
| `USD_per_Gbit_offered`, `Watt_per_Gbit_offered` | Cost or power divided by `alpha_star * base_total_demand`. |
| `USD_per_Gbit_p999`, `Watt_per_Gbit_p999` | Cost or power divided by absolute bandwidth at 99.9% availability. |

Missing cost data is unavailable. Cost ratios are undefined when
their denominator is zero or unavailable. JSON writes undefined numeric metrics
as `null`, and CSV uses empty numeric fields. Reported zero cost or power stays zero.
Time per iteration is amortized over the requested failure samples plus baseline,
including repeated patterns; it is not the runtime of a single unique solve.
Timing depends on the execution environment; compare runs under similar conditions.

## Project files

| File | Contents |
|------|----------|
| `project.csv` | One row per scenario with the metrics above. |
| `project_per_seed_abs.csv` | Absolute per-seed metrics used by `abs_*.png` plots. |
| `project_baseline_normalized.csv` | Scenario summaries relative to the selected baseline. |
| `project_baseline_normalized_per_seed.csv` | Per-seed ratios (`*_r`, target 1) and deltas (`*_d`, target 0). |
| `normalized_insights.csv` | Means, common-seed counts, and paired-test p-values; Holm correction is applied across scenarios within each metric. |

Normalized comparisons pair matching seeds with the baseline. The text summary
shows absolute metrics, normalized metrics, and comparison statistics.
`netlab test <root> A B` runs paired t-tests between two scenarios on the per-seed
metrics in `<root>_metrics`; `--alpha` sets the significance level. Inference
requires at least three finite matched seeds. Constant observed differences use
point CIs and p=0 (nonzero difference) or p=1 (zero); this computational convention
does not establish certainty about a population. Holm families contain available
scenario comparisons within each metric. Heatmap markers use adjusted p-values.

The baseline is chosen consistently in tables and plots: an explicit argument,
then `NGRAPH_BASELINE_SCENARIO`, then a name containing `baseline`, then the first
alphabetically. An explicitly named missing baseline is an error. Ratios with
zero denominators remain undefined even in the baseline's own row.

Pooling weights every sample equally: seeds with more iterations contribute more
weight. Seed medians and IQR use equal seed weight. Independent failure-pattern
positions are never aligned across seeds.

## Figures

| Figure | Contents |
|--------|----------|
| `BAC.png` | Pooled availability `P(delivered ≥ threshold)` versus percent of baseline delivery. IQR bands across seed curves are shown with at least three seeds. |
| `BAC_delta_vs_baseline.png` | Difference in availability from the selected baseline, by delivered-bandwidth threshold. Positive values mean higher availability. |
| `Latency_p99.png` | Pooled exceedance `P(iteration p99 stretch ≥ threshold)` across failure iterations. Lower values mean fewer iterations exceed the threshold. |
| `IterationOps.png` | Median and IQR across seeds for failure iterations, distinct patterns, and seconds per iteration. |
| `effects_heatmap.png` | Mean normalized effects; markers use Holm-adjusted p-values below 0.05. |
| `abs_*.png`, `norm_*.png` | Per-seed values and scenario medians, in absolute or baseline-normalized units. |

## Scenario and seed files

Each scenario directory contains `alpha_summary.json`, `bac_summary.json`,
`latency_summary.csv`, `iterops_summary.csv`, `costpower_summary.csv`, and
`network_stats_summary.csv`, plus scenario plots when enabled.

Its `seed*/` directories contain metric JSON files, BAC sample CSVs, per-iteration
latency data, and placement matrices. [tests/data/README.md](tests/data/README.md)
describes the checked-in examples and how to regenerate them.

## Reproducibility and interpretation

Each `netlab metrics` invocation replaces the complete generated report tree,
including when `--only` selects a subset. Old seeds, optional metrics, and
plots are removed. A failing run preserves the previous report and its provenance.
Filename and recorded seeds must agree, source hashes must remain unchanged during
analysis, and provenance records numeric-library versions and analysis settings.

Finite sample percentiles cannot establish multi-nine uptime.
MSD is bounded by its search settings and engine precision;
a disconnected network for which no positive alpha is feasible is rejected by
NetGraph. A supplied zero-baseline result can still be analyzed in absolute units,
with normalized metrics undefined.

Reproduce the independent verification with:

```sh
venv/bin/python dev/verify_metrics.py --plots
```

See [dev/METRICS_VERIFICATION.md](dev/METRICS_VERIFICATION.md) for evidence,
defects found, experiment scope, and validation results.
