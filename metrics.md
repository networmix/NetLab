# Read NetLab metrics

`netlab metrics` turns NetGraph results into per-seed metric files, scenario
summaries and project tables. A seed is one repeatable set of random choices.
Every numeric project column except `seeds` is the median of that metric's
per-seed values.

## Samples and the baseline reference

A placement step with a failure policy draws `iterations` random failure
patterns. NetGraph groups equal draws into cases, and each case's `occurrences`
counts its draws: a case with 10 occurrences contributes ten samples. A draw that
fails nothing belongs to the `baseline` case and is a sample like any other.
NetGraph also places the traffic once with nothing failed: the baseline
reference, counted once and never sampled. Bandwidth availability and
`iters_total` add the reference to the samples; latency compares the samples
with it. A step without a failure policy has no samples, only the reference.

Bandwidth availability describes these configured samples, not measured uptime.

## Bandwidth units

NetGraph counts bandwidth in whole units of a quantum `q`, which it chooses once
when it prepares a scenario and reports as `units_report.q`. NetLab multiplies
each exported unit count by `q` once and reads nothing else from the units report.
Declared capacities, demand volumes and NetworkStats totals are physical values
and are never scaled. In mini DC-BB, `q = 1/256`: the baseline placement's
153,600 delivered units are 600 of physical bandwidth, and NetworkStats reports
the declared link capacities, 4,300 in total. NetGraph rounds capacities and
demands to whole units of `q`, so a delivery can differ slightly from the flow of
the exact physical network.

## Required results

`netlab metrics` reads `<name>.results.json` files that NetLab wrote with
`netlab run`, `run_simulation` or `SimulationBatch`. Each needs its run record
`<name>.results.run.json` beside it, with status `success` and a measured wall
time for every workflow step; NetLab refuses a result without one. A results
file written by `ngraph run` has no run record.

The workflow needs steps named `msd_baseline` (`MaximumSupportedDemand`),
`tm_placement` (`TrafficMatrixPlacement`) and `network_statistics`
(`NetworkStats`). Latency needs `outputs: {cost_distribution: true}` on
`tm_placement`. Optional pair-capacity analysis uses
`node_to_node_capacity_matrix`; equipment totals use `cost_power`. NetLab finds
each step by its name in the result's `steps` list.

`tm_placement` must place the MSD's demands scaled by the MSD result: set
`alpha_from_step: msd_baseline` on that step, and NetGraph scales its demands by
that step's `alpha_feasible`. NetLab refuses a result whose `tm_placement`
declares a different demand set or different demands than `msd_baseline`, or
whose alpha is not exactly `alpha_feasible`.

Before it computes metrics, NetLab checks the `tm_placement` step's whole-unit
accounting: case occurrences add up to the number of draws the step requested;
in every row the reference request equals delivery plus shortfall; case totals
equal their row sums; a cost report's volume equals its row's delivery; and the
baseline's requests equal the demand the step resolved. A result that fails a
check is refused.

The MSD step lists its demands under `question.demands`, with `source`, `target`
and `physical_volume`. Placement rows name their endpoints under
`endpoints.sources` and `endpoints.destinations` and report
`reference_request_units` and `result.delivered_units`. Keep full endpoint
names: different devices need separate bandwidth and path-cost references.

## Capacity and delivered bandwidth

**Maximum supported demand (MSD)** searches for a multiplier of the configured
traffic. NetLab's `alpha_star` is NetGraph's `alpha_feasible`: the largest
multiplier the search tested and found feasible. It depends on the search
bounds, `alpha_tolerance` and routing. For example, `alpha_star=2` means the
search accepted twice the base traffic. It is not a claim about other traffic
distributions. When no tested multiplier is feasible, NetGraph reports a null
`alpha_feasible` with `search_status: none_feasible`; NetLab then reports the run
as failed with the reason `none_feasible`, never as `alpha_star = 0`.

**Bandwidth availability (BAC)** compares delivered traffic in each observation
with the amount delivered in the baseline reference. The AUC score is the area
under the availability curve, calculated here as the mean capped bandwidth ratio.
BAC's first observation is the baseline reference, followed by every sampled
draw.

| Column | Meaning |
| --- | --- |
| `scenario` | Scenario directory name |
| `seeds` | Number of seed summaries included |
| `node_count`, `link_count` | Median topology counts across seeds |
| `alpha_star` | Demand multiplier returned by MSD |
| `bac_auc` | Mean of `min(delivered / baseline_delivered, 1)` across BAC observations |
| `bw_p90`, `bw_p95`, `bw_p99`, `bw_p999` | Bandwidth met or exceeded by at least 90%, 95%, 99% or 99.9% of BAC observations, divided by baseline delivery |

For example, suppose the baseline delivers 10 Gbps and two sampled draws deliver
10 and 5 Gbps. BAC uses ratios `[1, 1, 0.5]`, so `bac_auc = 2.5/3`.
Raw bandwidth ratios and curves can exceed 1; the AUC score caps each ratio at 1.
The API calls the baseline delivered amount `offered`. Project bandwidth columns
are ratios: `0.5` means 50% of baseline delivery. BAC plots label that value as 50%.

Draws that fail nothing count as samples. Give mini DC-BB
(`tests/data/mini_dcbb.yaml`) a policy that fails each backbone device with
probability 0.1, and draw 100 patterns with seed 42. 71 draws fail no device
and deliver all traffic, 25 deliver half and 4 deliver nothing. BAC then has 101
observations, and its score is
`(1 + 71 + 25 × 0.5) / 101 = 169/202 ≈ 0.837`. Leaving out the 71 empty draws
would give `(1 + 12.5) / 30 = 0.45`.

### Bandwidth at an availability level

`BW@p` is the greatest observed bandwidth that at least `p` percent of
observations meet or exceed. For sorted observations, its zero-based index is
`n - ceil(n*p/100)`. Decimal probabilities use exact count boundaries.

For `[0,1,2,3,4,5,6,7,8,9]`, `BW@90 = 1`: nine of the ten observations are at
least 1.
This is different from an ordinary descriptive percentile. The fields
`quantiles_abs`, `quantiles_pct`, SPS `tails`, pair matrices and sweep percentile
arrays use NumPy's `lower` estimator: index `floor((n-1)*q)` for quantile `q`.

BAC, pair matrices and SPS combine priority classes with the same exact source
and destination. A `combine` demand's row, whose endpoints group several nodes,
stays one pair labelled by the demand's selectors and is never split into node
pairs: `^abc1/rsw2?$` to `^xyz1/rsw$` becomes `abc1/rsw2?→xyz1/rsw`. With
`group_mode: per_group` or `group_pairwise`, each group keeps its own row and
per-pair series, labelled by the selectors followed by the group's node list.
A total outage keeps its zero-bandwidth observation. Availability
curves count values at or above each threshold, including ties, without
interpolating between observations.

### Independent pair capacity

**Structural pair survivability (SPS)** asks how much of each pair's demand its
independently calculated maximum flow could carry. For each sampled draw:

`SPS = sum(min(pair_capacity, pair_demand)) / sum(pair_demand)`

For two pairs with demand 10 each and capacities 10 and 5, SPS is `15/20 = 0.75`.
Each maximum flow is solved independently; this does not prove that both pairs
can carry those amounts at the same time.

Enable it with `--enable-maxflow`. The `node_to_node_capacity_matrix` step must
cover every placement endpoint pair under its exact names. Use capture groups in
the MaxFlow selectors, such as `^(N[1-4])$` with `mode: pairwise`, so that each
row names one source and one destination node. Without captures, a selector
matches one group and the row names all its nodes together. NetLab accepts such
a row only when a `combine` placement demand puts the same nodes in one row
under the same selectors (the default `group_mode: flatten`); otherwise it
refuses it, because a group's capacity cannot be assigned to individual demands.
Missing placement pairs cause an error. Extra MaxFlow pairs are allowed; SPS
uses only the placement pairs. `SPS@p` uses the same availability-threshold rule
as BAC, over the sampled draws only, without the baseline reference.

## Latency stretch

Stretch compares path cost with a reference cost; it does not measure elapsed
delay. Each source/destination pair's reference is the lowest cost that carried
positive volume in its baseline cost report (`outputs.cost_distribution`). If
that cost is 2 and a failure sends traffic along a path of cost 3, the stretch is
`3/2 = 1.5`.

Percentiles and traffic shares are weighted by delivered volume. A failure
summary takes the median across the sampled draws. A project summary then takes
the median across seeds. A volume-weighted p99 is the smallest stretch that at
least 99% of the included volume does not exceed.

| Column | Meaning |
| --- | --- |
| `lat_base_p50` | Baseline median stretch |
| `lat_fail_p99` | Median of the sampled draws' volume-weighted p99 stretches |
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
It is zero when the cost report is missing and undefined when nothing is delivered.
A total outage keeps a missing latency entry and is excluded from latency medians.

Read latency with BAC and reference coverage. A low stretch after a failure can
coexist with large traffic loss. `lat_fail_p99` describes a typical draw's
tail, not the worst failure or the p99 of all delivered traffic pooled together.

### Cost reports of traffic-engineering placements

NetGraph reports a placement row's costs in one of two forms:

- `delivered_by_path_cost`, for shortest-path routing such as
  `SHORTEST_PATHS_ECMP`: how much volume took paths of each cost.
- `marginal_curve`, for traffic-engineering presets such as `TE_MAXFLOW_UNLIM`:
  the extra cost of each further unit of a minimum-cost flow.

NetLab reads both forms the same way for the percentiles, SLO shares and
best-path share: each cost, divided by the reference, weighted by its volume. For
a marginal curve these are percentiles of marginal cost, not of the costs of the
paths the traffic takes. The two are equal when no later unit of the flow
reroutes units placed before it. WES of a `marginal_curve` row uses the row's
mean path cost instead, `total_cost` divided by delivered volume, which does not
depend on how the flow is split into paths.

The case where they differ: S reaches T over S-A-T and S-B-T, each of cost 4,
and over S-A-B-T, of cost 3; every link has capacity 1. In the baseline another
demand occupies A-B, so two units from S to T take the two paths of cost 4, and
the reference is 4. A failure removes that demand. The minimum-cost flow now
finds one unit at cost 3 (S-A-B-T) and one at cost 5 (S-B, back along A-B to A,
then A-T), which together again use the two paths of cost 4. The curve reports
costs 3 and 5, so the failure p50 stretch is 0.75 and p99 is 1.25. The mean path
cost is `8/2 = 4`, so WES is 0.

On the saved small scenarios in `tests/data`, which use `TE_MAXFLOW_UNLIM`, the
per-seed latency summaries read from the curve equal those computed from path
costs, although some individual draws differ. This is not a bound for other
networks.

## Iterations, timing and equipment

| Column | Meaning |
| --- | --- |
| `iters_fail` | Sampled draws, including draws that fail nothing: the sum of case occurrences |
| `iters_total` | Sampled draws plus the one baseline reference |
| `unique_patterns` | Number of distinct sampled cases, the no-failure case included when drawn |
| `tm_duration_total_sec` | Wall time of the `tm_placement` step in seconds, from the run record |
| `tm_duration_per_iter_sec` | That time divided by `iters_total` |
| `capex_total` | Equipment purchase cost reported by `cost_power` |
| `USD_per_Gbit_offered`, `Watt_per_Gbit_offered` | Cost or power divided by `alpha_star * base_total_demand` |
| `USD_per_Gbit_p999`, `Watt_per_Gbit_p999` | Cost or power divided by absolute bandwidth at 99.9% availability |

Missing equipment data is unavailable; reported zero cost or power stays zero.
Ratios are undefined if their denominator is zero or unavailable. Undefined
numbers are written as `null` in JSON and empty numeric fields in CSV.

NetLab measures the wall time of each workflow step itself, through NetGraph's
`step_hook`, and saves the times in the run record beside the results. Scenario
preparation is not part of any step's time. A cached result keeps the times of
the run that produced it. Time per iteration divides the step's time by the
sampled draws plus the baseline reference, repeated draws included. It is an
average accounting value, not the measured runtime of one unique network solve.
Compare timings only under similar execution conditions.

In the mini DC-BB example above, `iters_fail` is 100, `iters_total` is 101 and
`unique_patterns` is 9.

## Compare scenarios and seeds

A project metric usually gives each seed equal weight. Pooled BAC curves give each
bandwidth observation equal weight, including each seed's baseline reference. A
seed with ten draws contributes eleven observations; one with a hundred draws
contributes 101. Pooled latency curves include only draws with finite latency
values. A median of seed summaries treats the seeds equally. Draw positions are
not matched across different seeds.

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
| `Latency_p99.png` | Fraction of sampled draws with p99 stretch at or above each threshold; lower is better |
| `IterationOps.png` | Median and interquartile range of iteration counts, distinct patterns and seconds per iteration |
| `effects_heatmap.png` | Mean normalized effects, with markers for adjusted p-values below 0.05 |
| `abs_*.png`, `norm_*.png` | Per-seed values and scenario medians in absolute or normalized units |

The interquartile range is the interval from the 25th to the 75th percentile.
BAC seed bands are shown with at least three seeds.

Each scenario has `alpha_summary.json`, `bac_summary.json`,
`latency_summary.csv`, `iterops_summary.csv`, `costpower_summary.csv` and
`network_stats_summary.csv`. Its `seed*/` directories contain metric JSON,
BAC observations, per-draw latency data and placement matrices.
See [the saved test examples](tests/data/README.md) for regeneration instructions.

## Save a repeatable report

A metrics calculation replaces the complete report tree, even when `--only`
selects a subset. Removed seeds, optional metrics and plots are removed from the
report too. A failing calculation preserves the existing report and its provenance,
the record of inputs, settings and software versions used to produce it.

The seed in a result filename (`__seed<N>`) must match the `metadata.seed`
recorded inside the file. If a result file or its run record changes while the
analysis runs, the analysis fails. The provenance records numeric-library
versions and settings, and NetLab's version with `commit` and `dirty`:

- a clean Git checkout records its commit and `dirty: false`;
- a checkout with uncommitted changes records a null `commit` and `dirty: true`;
- a GitHub install records the installed commit and `dirty: false`;
- anything else, including a Git error, records both as null.

A finite number of sampled draws cannot establish multi-nine uptime. MSD remains
bounded by its search settings and the bandwidth quantum `q`. Supplied results
with zero baseline delivery can still be analyzed in absolute bandwidth units;
their normalized values are undefined.

To check the calculations yourself, follow the
[metric verification guide](dev/METRICS_VERIFICATION.md). It uses small networks
with independently calculated expected answers.
