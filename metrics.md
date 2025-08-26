# Topline Project Metrics and Plots

This document describes the project-level metrics and produced figures. All results for each multi-scenario run are stored in a set of nested folders inside a per-run top folder.

## Metrics (project-level)

- **scenario**: Scenario name (directory name under the per-run top-level folder)
  - What it is: Human-readable identifier for the network topology/config being analyzed. Taken directly from the scenario directory name under the per-run top-level folder. In the overall pipeline, scenario directory names come from the names of the TopoGen config files used for the run.

- **seeds**: Number of seeds included for the scenario
  - What it is: Count of independent runs of each of the scenarios contributing to the metrics.
  - How it's calculated: From discovered per-seed summaries.

- **node_count**: Median node count across seeds for the scenario
  - What it is: Number of nodes in the topology, per seed; summarized by median.
  - How it's calculated: Read from `network_stats_summary.csv` per scenario and take the median.

- **link_count**: Median link count across seeds for the scenario
  - What it is: Number of links in the topology, per seed; summarized by median.
  - How it's calculated: Read from `network_stats_summary.csv` per scenario and take the median.

- **alpha_star**: Maximum supported demand multiplier found by the MSD step (per seed), then median across seeds.
  - What it is: The largest traffic scaling factor (relative to the baseline traffic matrix) the network can place without violating placement constraints.
  - How it's calculated: From the `msd_baseline` step per seed; we take the median across seeds.
  - How to interpret: Higher is better; measures capacity headroom given topology and constraints. By design, TopoGen tries to preserve capacity invariant in the model to make it easier to compare across scenarios, hence, this should be the same or very similar for all scenarios in a run. If it's not, it's a sign of a problem with the generated model.
  - Pseudocode:
    - per seed: `alpha_star_seed = results.steps.msd_baseline.data.alpha_star`
    - scenario: `alpha_star = median(alpha_star_seed over seeds)`

- **bw_p99**: Bandwidth at 99% availability, normalized by baseline-offered (per seed), median across seeds.
  - What it is: Delivered bandwidth level met or exceeded in 99% of iterations, divided by baseline delivered (offered); capped at 1.0.
  - How it's calculated: Lower-tail quantile of total delivered series at q = 1 − 0.99; divide by total offered; cap at 1.0; then median across seeds.
  - How to interpret: Higher is better; tail reliability of delivered capacity.

- **bw_p999**: Bandwidth at 99.9% availability, normalized by baseline-offered (per seed), median across seeds.
  - What it is: Same as `bw_p99` with 99.9% reliability threshold.
  - How it's calculated: q = 1 − 0.999; divide by offered; cap at 1.0; then median across seeds.
  - How to interpret: Higher is better; stringent tail SLO.

- **bac_auc**: BAC AUC (area under curve) — mean of delivered/offered clipped to 1.0 across iterations (per seed), median across seeds.
  - What it is: Average normalized delivered capacity across iterations; equivalent to area under the availability curve (with uniform iteration weight).
  - How it's calculated: For each iteration, compute delivered/offered, cap at 1.0; average across iterations; median across seeds.
  - How to interpret: Higher is better; summarizes resilience beyond single tail points.

- **lat_base_p50**: Baseline latency stretch p50 (volume-weighted) per seed; median across seeds.
  - What it is: Typical latency stretch in the baseline (no failure) case.
  - How it's calculated: Compute per-flow stretch relative to baseline min-cost path, volume-weighted; take weighted p50; median across seeds.
  - How to interpret: Near 1.0 is ideal; if more than half the volume is above min-cost, the p50 would exceed 1.0.

- **lat_fail_p99**: Median across failure iterations (within a seed) of latency stretch p99; then median across seeds.
  - What it is: Tail latency stretch under failures.
  - How it's calculated: For each failure iteration, compute weighted p99 stretch; median across iterations; median across seeds.
  - How to interpret: Lower is better. Shows the close-to-worst-case latency stretch under failures.

- **lat_TD99**: Tail degradation at 99% = (failure p99) / (baseline p99), per seed; median across seeds.
  - What it is: Ratio of failure to baseline tail latency.
  - How it's calculated: `TD99_seed = fail_p99_seed / base_p99_seed`; then median across seeds.
  - How to interpret: 1.0 = parity; >1.0 worse tails under failure.

- **lat_SLO_1_2_drop**: Drop in share meeting SLO=1.2 stretch: baseline_share(≤1.2) − failures_share(≤1.2) per seed; median across seeds.
  - What it is: Lost fraction of traffic volume that meets 1.2× stretch SLO moving from baseline to failures.
  - How it's calculated: Compute volume share with stretch ≤1.2 in baseline and in failures (per-iteration median), then baseline − failures; median across seeds.
  - How to interpret: Closer to 0.0 is better.

- **lat_best_path_drop**: Drop in best-path share: baseline_best_share − failures_best_share per seed; median across seeds.
  - What it is: Reduction in volume fraction still on the baseline minimum-cost path under failures.
  - How it's calculated: Share equal to baseline min-cost in baseline and failures; baseline − failures; median across seeds.
  - How to interpret: Closer to 0.0 is better.

- **lat_WES_delta**: Weighted Excess Stretch delta: failures_WES − baseline_WES per seed; median across seeds.
  - What it is: Change in average excess stretch beyond 1.0, volume-weighted.
  - How it's calculated: `WES = E[(stretch−1)+]` with volume weights; failures median − baseline; median across seeds.
  - How to interpret: It shows average “extra stretch” beyond ideal (1.0) that traffic experiences, volume-weighted, under failures vs baseline. Closer to 0.0 is better; positive means worse stretch. It combines both how much and how often paths are longer than ideal. Small inflations on lots of traffic and rare big inflations both move it upward in proportion to severity and volume.

- **spf_calls_per_iter**, **flows_created_per_iter**, **reopt_calls_per_iter**: Median across seeds of per-iteration SPF calls, flows created, reoptimization calls.
  - What they are: Control-plane and placement effort proxies per iteration.
  - How they're calculated: From `iterops_summary.csv` per scenario; median across seeds.
  - How to interpret: Lower SPF/reopt generally better; flows created reflects demand fragmentation and might be a proxy for required number of LSPs to avoid stranded traffic or capacity.

- **tm_duration_total_sec**, **tm_duration_per_iter_sec**: tm_placement timing (seconds), total and per-iteration medians across seeds.
  - What they are: Runtime diagnostics for placement, dominated by SPF calls.
  - How they're calculated: From per-seed timing summaries; median across seeds.
  - How to interpret: Lower is better. In combination with the number of SPF calls, it shows the relative cost of the placement step.

- **USD_per_Gbit_offered**, **Watt_per_Gbit_offered**: Cost/Power divided by offered at alpha*; median across seeds.
  - What they are: Efficiency per offered Gbps at `alpha*`.
  - How they're calculated: `capex_total / (alpha_star * base_total_demand)`; similarly for watts.
  - How to interpret: Lower is better.

- **USD_per_Gbit_p999**, **Watt_per_Gbit_p999**: Cost/Power divided by reliable bandwidth at p99.9; median across seeds.
  - What they are: Dollars or watts per Gbps available with 99.9% probability.
  - How they're calculated: `capex_total / bw_at_probability_abs[99.9]`; similar for watts.
  - How to interpret: Lower is better.

## Project-level summaries

It is a text report generated from the project CSVs and includes:

- A table titled “Consolidated project metrics” with columns:
  - `seeds`, `node_count`, `link_count`, `alpha_star`, `bw_p99`, `bw_p999`, `bac_auc`, `lat_base_p50`, `lat_fail_p99`, `lat_TD99`, `lat_SLO_1_2_drop`, `lat_best_path_drop`, `lat_WES_delta`, `spf_calls_per_iter`, `flows_created_per_iter`, `reopt_calls_per_iter`, `tm_duration_total_sec`, `tm_duration_per_iter_sec`, `USD_per_Gbit_offered`, `Watt_per_Gbit_offered`, `USD_per_Gbit_p999`, `Watt_per_Gbit_p999`, `capex_total`.
- A table titled “Baseline-normalized metrics (scenario / baseline)” with normalized ratios/deltas per scenario:
  - Ratios (`*_r`) vs 1.0: `bw_p99_pct_r`, `bw_p999_pct_r`, `auc_norm_r`, `USD_per_Gbit_offered_r`, `Watt_per_Gbit_offered_r`, `USD_per_Gbit_p999_r`, `Watt_per_Gbit_p999_r`, plus node/link ratios.
  - Deltas (`*_d`) vs 0.0: `lat_SLO_1_2_drop_d`, `lat_best_path_drop_d`, `lat_WES_delta_d`.
- A section “All baseline-normalized comparisons vs target (mean, n, adj_p)” summarizing paired t-tests vs target (1.0 for ratios, 0.0 for deltas). Columns show mean value, sample size (common seeds), and Holm-adjusted p-values per scenario.

## Project-level figures

- **BAC.png**
  - What it shows: Cross-seed pooled BAC availability curves per scenario: A(x) = P(delivered ≥ x), x in % of offered. Includes IQR bands across seeds when ≥3 seeds.
  - How it’s computed: Pool normalized delivered samples across seeds from `seed*/bac.json`, form empirical availability (1 − CDF), and plot stepwise curves. See `metrics/plot_cross_seed_bac.py`.
  - How to read: Curves further upper-right are better. Narrow IQR band indicates stable behavior. Arguably, this is the most important figure for scenario comparison.

- **BAC_delta_vs_baseline.png**
  - What it shows: Availability improvement vs baseline (default 80–100% band). Positive = better than baseline at that delivered threshold.
  - How it’s computed: Pooled availability for each scenario and baseline are interpolated on a common grid; delta = scenario − baseline availability. See `metrics/plot_bac_delta_vs_baseline.py`.
  - How to read: Lines above zero indicate higher availability vs baseline; guides at 80/90/95% aid comparison.

- **Latency_p99.png**
  - What it shows: Cross-seed pooled latency exceedance curves for p99 stretch (baseline-referenced). Y = P(stretch > x) = 1 − CDF(stretch).
  - How it’s computed: From `seed*/latency.json` per-iteration p99 over failures; pool across seeds, compute exceedance (1 − CDF). IQR bands from per-seed curves on a common grid. See `metrics/plot_cross_seed_latency.py`.
  - How to read: For a fixed stretch x (e.g., 5), lower curve is better (smaller fraction of iterations exceed x). For a fixed tail probability y (e.g., 0.01), lower x is better (smaller p99-stretch at that exceedance).

- **effects_heatmap.png**
  - What it shows: Baseline-normalized effect sizes per metric per scenario, with significance markers for p < 0.05.
  - How it’s computed: From `normalized_insights.csv`, effect = mean − 1.0 for ratio metrics and mean − 0.0 for delta metrics; Holm-adjusted p-values used for annotation. See `metrics/plot_significance_heatmap.py`.
  - How to read: Red/blue indicates direction and magnitude vs baseline; dots mark statistically significant differences at p < 0.05.

## Other project artifacts

- Absolute distributions with per-seed jitter and scenario medians:
  - `abs_AUC.png` (BAC AUC), `abs_BW_p99.png`, `abs_Latency_fail_p99.png`, `abs_USD_per_Gbit_offered.png`, `abs_USD_per_Gbit_p999.png`, `abs_nodes.png`, `abs_links.png`, `abs_CapEx.png`.
- Baseline-normalized distributions with per-seed jitter and scenario medians:
  - `norm_AUC.png`, `norm_BW_p99.png`, `norm_Latency_fail_p99.png`, `norm_USD_per_Gbit_offered.png`, `norm_USD_per_Gbit_p999.png`, `norm_nodes.png`, `norm_links.png`.
- Cross-seed iteration ops summary:
  - `IterationOps.png` (SPF/iter, flows created/iter, reopt/iter) across scenarios.

## CSV artifacts

- `project.csv`: Consolidated project metrics table (one row per scenario). Columns include those described in Metrics.
- `project_baseline_normalized.csv`: Per-scenario metrics normalized to baseline (ratios and deltas). Used for normalized plots.
- `normalized_insights.csv`: Wide-format table of baseline-normalized comparisons vs target with columns per metric: `<metric>__mean`, `<metric>__n`, `<metric>__p`, plus `<metric>__p_adj` (Holm-adjusted within metric).
- `project_per_seed_abs.csv`: Per-seed absolute metrics across scenarios (e.g., `auc_norm`, `bw_p99_pct`, `lat_fail_p99`, `USD_per_Gbit_offered`, `USD_per_Gbit_p999`, `capex_total`, `node_count`, `link_count`). Used for “abs_*.png” figures.
- `project_baseline_normalized_per_seed.csv`: Per-seed baseline-normalized metrics (ratios `_r` vs 1.0, deltas `_d` vs 0.0) for non-baseline scenarios. Used for “norm_*.png” figures.

## Scenario and seed-level outputs

Within each scenario directory (e.g., `Clos_L16_S4/`):

- Scenario-level summaries: `alpha_summary.json`, `bac_summary.json`, `latency_summary.csv`, `iterops_summary.csv`, `costpower_summary.csv`, `network_stats_summary.csv`.
- Scenario-level figures: `BAC.png`, `Latency_p99.png`, `IterationOps.png` (single-scenario views).

All per-seed artifacts are stored in the per-seed folder within the scenario directory.
