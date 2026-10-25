# Read NetLab metrics

`netlab metrics` turns NetGraph results into per-seed metric files, scenario
summaries and project tables. A seed is one repeatable set of random choices.
Every numeric project column except `seeds` is the median of that metric's
per-seed values.

A failure pattern can stand for several sampled iterations. Its positive integer
`occurrence_count` is its weight: a pattern with count 10 contributes ten samples.
Bandwidth availability describes these configured samples, not measured uptime.

## Required results

The metrics CLI expects workflow steps named `msd_baseline`
(`MaximumSupportedDemand`), `tm_placement` (`TrafficMatrixPlacement`, with flow
details), and `network_statistics` (`NetworkStats`). Optional pair-capacity
analysis uses `node_to_node_capacity_matrix`; equipment totals use `cost_power`.

`tm_placement` must place the traffic scaled by the MSD result. Set
`alpha_from_step: msd_baseline` and `alpha_from_field: data.alpha_star` on that
step. NetLab checks that the baseline and every failure sample request exactly
`alpha_star` times the total MSD base demand, so all placement metrics describe
the network at its supported load.

MSD demand records use `source`, `target` and `volume`. Placement flow records
use `source`, `destination`, `demand` and `placed`, the delivered part. Keep full
endpoint names: different devices need separate bandwidth and path-cost references.

## Capacity and delivered bandwidth

**Maximum supported demand (MSD)** searches for a multiplier of the configured
traffic. Its result, `alpha_star`, depends on the search bounds, resolution and
placement policy. For example, `alpha_star=2` means the search accepted twice the
base traffic. It is not a claim about other traffic distributions.

**Bandwidth availability (BAC)** compares delivered traffic in each sample with
the amount delivered in the no-failure baseline. The AUC score is the area under
the availability curve, calculated here as the mean capped bandwidth ratio.
BAC includes that baseline as one sample, followed by the occurrence-weighted
failure samples.

| Column | Meaning |
| --- | --- |
| `scenario` | Scenario directory name |
| `seeds` | Number of seed summaries included |
| `node_count`, `link_count` | Median topology counts across seeds |
| `alpha_star` | Demand multiplier returned by MSD |
| `bac_auc` | Mean of `min(delivered / baseline_delivered, 1)` across BAC samples |
| `bw_p90`, `bw_p95`, `bw_p99`, `bw_p999` | Bandwidth met or exceeded by at least 90%, 95%, 99% or 99.9% of BAC samples, divided by baseline delivery |

For example, suppose the baseline delivers 10 Gbps and two failure samples deliver
10 and 5 Gbps. BAC uses ratios `[1, 1, 0.5]`, so `bac_auc = 2.5/3`.
Raw bandwidth ratios and curves can exceed 1; the AUC score caps each ratio at 1.
The API calls the baseline delivered amount `offered`. Project bandwidth columns
are ratios: `0.5` means 50% of baseline delivery. BAC plots label that value as 50%.

### Bandwidth at an availability level

`BW@p` is the greatest observed bandwidth that at least `p` percent of samples
meet or exceed. For sorted samples, its zero-based index is
`n - ceil(n*p/100)`. Decimal probabilities use exact count boundaries.

For `[0,1,2,3,4,5,6,7,8,9]`, `BW@90 = 1`: nine of the ten samples are at least 1.
This is different from an ordinary descriptive percentile. The fields
`quantiles_abs`, `quantiles_pct`, SPS `tails`, pair matrices and sweep percentile
arrays use NumPy's `lower` estimator: index `floor((n-1)*q)` for quantile `q`.

BAC, pair matrices and SPS combine priority classes with the same exact source
and destination. A total outage keeps its zero-bandwidth sample. Availability
curves count values at or above each threshold, including ties, without
interpolating between observations.

### Independent pair capacity

**Structural pair survivability (SPS)** asks how much of each pair's demand its
independently calculated maximum flow could carry. For each failure sample:

`SPS = sum(min(pair_capacity, pair_demand)) / sum(pair_demand)`

For two pairs with demand 10 each and capacities 10 and 5, SPS is `15/20 = 0.75`.
Each maximum flow is solved independently; this does not prove that both pairs
can carry those amounts at the same time.

Enable it with `--enable-maxflow`. The `node_to_node_capacity_matrix` step must
cover every placement endpoint pair under its exact names. To label physical
endpoints with their full names, use capture groups such as `^(A)$` and `^(B)$`
in the MaxFlow selectors. Without captures, NetGraph labels a selected group with the selector
text itself. Labels that do not match placement pairs cause an error.
`SPS@p` uses the same availability-threshold rule as BAC, over failure samples only.

## Latency stretch

Stretch compares path cost with a reference cost; it does not measure elapsed
delay. Each source/destination pair uses the cheapest path that carried positive
volume in its baseline `cost_distribution`. If that cost is 2 and a failure
sends traffic along a path of cost 3, the stretch is `3/2 = 1.5`.

Percentiles and traffic shares are weighted by delivered volume. A failure
summary takes the median across iterations. A project summary then takes the
median across seeds. A volume-weighted p99 is the smallest stretch that at least
99% of the included volume does not exceed.

| Column | Meaning |
| --- | --- |
| `lat_base_p50` | Baseline median stretch |
| `lat_fail_p99` | Median of the failure iterations' volume-weighted p99 stretches |
| `lat_TD99` | Failure p99 summary divided by baseline p99 |
| `lat_SLO_1_2_drop` | Baseline share at stretch ≤1.2 minus the median failure share |
| `lat_best_path_drop` | Baseline share at the reference cost minus the median failure share |
| `lat_WES_delta` | Median failure weighted excess stretch minus baseline weighted excess stretch |

In the column names, SLO means a service-level objective: here, stretch at most
1.2. Weighted excess stretch (WES) is the volume-weighted mean of
`max(stretch - 1, 0)`.
For equal delivered volumes at stretch 1 and 1.5, it is `(0 + 0.5)/2 = 0.25`.

Dropped traffic has no delivered path cost. Latency metrics include only delivered
traffic with a positive baseline reference. The per-seed JSON field
`reference_coverage` says what fraction of delivered volume has that reference.
It is zero when path details are missing and undefined when nothing is delivered.
A total outage keeps a missing latency entry and is excluded from latency medians.

Read latency with BAC and reference coverage. A low stretch after a failure can
coexist with large traffic loss. `lat_fail_p99` describes a typical iteration's
tail, not the worst failure or the p99 of all delivered traffic pooled together.

## Iterations, timing and equipment

| Column | Meaning |
| --- | --- |
| `iters_fail` | Sum of failure-pattern occurrence counts |
| `iters_total` | Failure iterations plus one baseline |
| `unique_patterns` | Number of distinct failure records |
| `tm_duration_total_sec` | Recorded `tm_placement` duration in seconds |
| `tm_duration_per_iter_sec` | That duration divided by total iterations |
| `capex_total` | Equipment purchase cost reported by `cost_power` |
| `USD_per_Gbit_offered`, `Watt_per_Gbit_offered` | Cost or power divided by `alpha_star * base_total_demand` |
| `USD_per_Gbit_p999`, `Watt_per_Gbit_p999` | Cost or power divided by absolute bandwidth at 99.9% availability |

Missing equipment data is unavailable; reported zero cost or power stays zero.
Ratios are undefined if their denominator is zero or unavailable. Undefined
numbers are written as `null` in JSON and empty numeric fields in CSV.

Time per iteration divides the whole recorded duration by the requested samples
plus baseline, including repeated failure patterns. It is an average accounting
value, not the measured runtime of one unique network solve. Compare timings only
under similar execution conditions.

## Compare scenarios and seeds

A project metric usually gives each seed equal weight. Pooled curves give each
sample equal weight, so a seed with more iterations contributes more observations.
For example, pooling a ten-sample seed and a hundred-sample seed gives the second
seed ten times as much weight. A median of their seed summaries treats them equally.
Failure sample positions are not matched across different seeds.

Normalized tables compare matching seeds with a selected baseline scenario.
Ratios (`*_r`) have target 1; differences (`*_d`) have target 0. A zero or missing
denominator stays undefined, even in the baseline's own row.

To choose the baseline scenario on the command line, set
`NGRAPH_BASELINE_SCENARIO=<name>`; Python callers can pass the name directly.
Otherwise NetLab takes the first name, alphabetically, that contains `baseline`
in any letter case, or else the first name alphabetically. A requested baseline
that does not exist is an error.

`netlab test <root> A B` runs a paired t-test on the seeds that A and B share,
for six per-seed metrics: `alpha_star`, BW@99.9% as a fraction of baseline
delivery, `lat_fail_p99`, `USD_per_Gbit_p999`, `Watt_per_Gbit_p999` and
`capex_total`. `--alpha` sets the significance level. A metric is tested only
with at least three seeds where both values are finite, and the test assumes
those paired observations suit a t-test. Confidence intervals describe the
estimated mean difference. Constant observed differences use point intervals and
p=0 for a nonzero difference or p=1 for zero; this calculation does not
establish population-level certainty. `netlab test` compares one pair of
scenarios, so its adjusted p-value equals p.

For each scenario other than the baseline, `normalized_insights.csv` tests
whether its per-seed ratios differ from 1 and its differences from 0. Holm's
multiple-comparison correction is applied across scenarios within each metric.
Heatmap markers use those adjusted p-values.

## Output files

The report is written next to the scenario root: `scenarios/` produces
`scenarios_metrics/`.

| Project file | Contents |
| --- | --- |
| `project.csv` | One summary row per scenario |
| `project_per_seed_abs.csv` | Absolute per-seed metrics |
| `project_baseline_normalized.csv` | Scenario summaries relative to the selected baseline |
| `project_baseline_normalized_per_seed.csv` | Per-seed ratios and differences |
| `normalized_insights.csv` | Means, matched-seed counts and comparison statistics |

| Figure | What to look for |
| --- | --- |
| `BAC.png` | Availability `P(delivered ≥ threshold)` versus percent of baseline delivery; higher is better |
| `BAC_delta_vs_baseline.png` | Availability difference from the selected baseline; positive means higher availability |
| `Latency_p99.png` | Fraction of failure iterations with p99 stretch at or above each threshold; lower is better |
| `IterationOps.png` | Median and interquartile range of iteration counts, distinct patterns and seconds per iteration |
| `effects_heatmap.png` | Mean normalized effects, with markers for adjusted p-values below 0.05 |
| `abs_*.png`, `norm_*.png` | Per-seed values and scenario medians in absolute or normalized units |

The interquartile range is the interval from the 25th to the 75th percentile.
BAC seed bands are shown with at least three seeds.

Each scenario has `alpha_summary.json`, `bac_summary.json`,
`latency_summary.csv`, `iterops_summary.csv`, `costpower_summary.csv` and
`network_stats_summary.csv`. Its `seed*/` directories contain metric JSON,
BAC samples, per-iteration latency data and placement matrices.
See [the saved test examples](tests/data/README.md) for regeneration instructions.

## Save a repeatable report

A metrics calculation replaces the complete report tree, even when `--only`
selects a subset. Removed seeds, optional metrics and plots are removed from the
report too. A failing calculation preserves the existing report and its provenance,
the record of inputs, settings and software versions used to produce it.

The seed in a result filename (`__seed<N>`) must match the `scenario.seed`
recorded inside the file. If a result file changes while the analysis runs, the
analysis fails. The provenance records numeric-library versions and settings.

Finite failure samples cannot establish multi-nine uptime. MSD remains bounded
by its search settings and engine precision. NetGraph rejects an MSD search when
no positive multiplier is feasible. Supplied results with zero baseline delivery
can still be analyzed in absolute bandwidth units; their normalized values are
undefined.

To check the calculations yourself, follow the
[metric verification guide](dev/METRICS_VERIFICATION.md). It uses small networks
with independently calculated expected answers.
