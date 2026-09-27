# Metrics and output files

`netlab metrics` writes per-seed results, scenario summaries, and project tables.
Project metrics are medians across seeds unless stated otherwise. Failure samples
are weighted by each pattern's `occurrence_count`; BAC also includes the baseline
as one sample. Availability describes the configured failure experiment, not
measured network uptime.

## Capacity and bandwidth

| Column | Definition |
|--------|------------|
| `scenario` | Scenario directory name. |
| `seeds` | Number of seed summaries included. |
| `node_count`, `link_count` | Median topology counts across seeds. |
| `alpha_star` | Demand multiplier returned by the MSD search, subject to its bounds, resolution, and placement settings. |
| `bac_auc` | Mean of `min(delivered / baseline_delivered, 1)` across samples. |
| `bw_p99`, `bw_p999` | Lower-tail delivered-bandwidth quantile at 0.01 or 0.001, divided by baseline delivery. |

BAC uses baseline delivered bandwidth as its normalization value, called
`offered` in the API. Quantiles use the lower observed sample. `bw_p99` and
`bw_p999` are not capped at 1; AUC is. Per-direction BAC combines priority classes
with the same source and destination.

Optional structural pair survivability (SPS) uses independently computed MaxFlow
capacities: `sum(min(pair_capacity, pair_demand)) / sum(pair_demand)` per failure
iteration. It does not measure simultaneous traffic placement. Enable it with
`--enable-maxflow` and a `node_to_node_capacity_matrix` step.

## Latency stretch

For each source/destination pair, the reference cost is the minimum cost carrying
nonzero volume in the baseline flow's `cost_distribution`. Stretch is path cost
divided by that reference. Percentiles and shares are weighted by delivered
volume. Failure summaries are medians of the per-iteration values, followed by a
median across seeds for project tables.

| Column | Definition |
|--------|------------|
| `lat_base_p50` | Baseline median stretch. |
| `lat_fail_p99` | Median of the failure iterations' volume-weighted p99 stretches. |
| `lat_TD99` | Failure p99 summary divided by baseline p99. |
| `lat_SLO_1_2_drop` | Baseline share with stretch ≤1.2 minus the median failure share. |
| `lat_best_path_drop` | Baseline share at the reference cost minus the median failure share. |
| `lat_WES_delta` | Median failure weighted excess stretch minus baseline weighted excess stretch. |

Weighted excess stretch is the volume-weighted mean of `max(stretch - 1, 0)`.
Latency metrics describe delivered traffic; dropped traffic contributes no path
cost. Read them alongside BAC. `lat_fail_p99` is a typical iteration's tail
summary, not the worst failure's p99.

## Iterations, timing, and cost

| Column | Definition |
|--------|------------|
| `iters_fail` | Sum of failure-pattern occurrence counts. |
| `iters_total` | Failure iterations plus one baseline. |
| `unique_patterns` | Number of distinct failure records. |
| `tm_duration_total_sec` | Recorded `tm_placement` duration in seconds. |
| `tm_duration_per_iter_sec` | Total duration divided by total iterations. |
| `capex_total` | Total capital cost reported by network statistics. |
| `USD_per_Gbit_offered`, `Watt_per_Gbit_offered` | Cost or power divided by `alpha_star * base_total_demand`. |
| `USD_per_Gbit_p999`, `Watt_per_Gbit_p999` | Cost or power divided by absolute bandwidth at 99.9% availability. |

Cost ratios are undefined when their denominator is zero or unavailable.
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

## Figures

| Figure | Contents |
|--------|----------|
| `BAC.png` | Pooled availability `P(delivered ≥ threshold)` versus percent of baseline delivery. IQR bands across seed curves are shown with at least three seeds. |
| `BAC_delta_vs_baseline.png` | Difference in availability from the selected baseline, by delivered-bandwidth threshold. Positive values mean higher availability. |
| `Latency_p99.png` | Pooled exceedance `P(iteration p99 stretch > threshold)` across failure iterations. Lower values mean fewer iterations exceed the threshold. |
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
