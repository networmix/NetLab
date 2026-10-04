# Metrics verification

The numerical audit covers the current NetLab metric pipeline and its research
consumers. Evidence and rendered figures live in `build/metrics-verification/`.
The runnable experiment is `dev/verify_metrics.py`; regression tests are in
`tests/metrics/test_numerical_verification.py` and `test_native_verification.py`.

## Reproduce

```sh
venv/bin/python dev/verify_metrics.py --plots
venv/bin/python -m pytest tests/metrics tests/test_current_ngraph_integration.py tests/test_mini_dcbb_verification.py --no-cov
make check-ci
```

The experiment performs 38 configured native runs: 16 exhaustive failure masks,
six successful model variants, one expected disconnected-MSD rejection, and
15 scenario/seed runs. It also constructs a 136-sample weighted population from
the native outcomes. Those 136 observations are reweighted native results, not
136 additional engine executions.

## Checks

| Area | Evidence | Result |
| --- | --- | --- |
| BW@p / SPS@p | Rational counting oracle at 13 sample lengths from 1 to 10,001; exact 90/95/99/99.9/99.99% boundaries and replication | Passed after rank correction |
| Empirical survival | 100 random tied integer samples, direct threshold counts; cross-seed plots and saved curves | Passed |
| Latency | 100 independently volume-expanded distributions; distinct devices, priorities, zero delivery, cost-unit rescaling, missing/partial references | Passed after endpoint and tolerance fixes |
| MSD | Native diamond capacity, pairwise volume splitting, combine, bidirectional traffic, bounded search | Passed; no feasible positive alpha produces an engine error |
| Pair matrices / SPS | All 16 edge-failure subsets vs NetworkX maximum flow and two-path arithmetic | Passed; mismatched group labels rejected |
| Cost / power | Four 100-dollar/10-watt routers and eight 2-dollar/1-watt optics; global and per-metro totals 416 dollars/48 watts; independent divisions | Passed; missing is distinct from zero |
| Iterations | Occurrence expansion, 136 weighted failures, metadata reconciliation, nonnegative time | Passed; timing is amortized, not a throughput benchmark |
| Seed aggregation | Three capacity configurations × five real seeds, 9/19/29/39/49 failures per seed; manual medians, ratios, deltas, pooled counts and IQR | Passed across all non-timing project metrics |
| Inference | SciPy paired t and CIs over scales 1e-150..1e150; invariance at 1e-200 and 1e200; constant decimals; missing pairs; hand-calculated Holm | Passed after stable scaling and validation |
| Input contract | Mutation of native failure/baseline demand, placed, dropped, costs, summaries, alpha, occurrence counts, duration and endpoints | Passed; invalid numbers and missing required fields rejected |
| Artifacts | Reduced-input rerun, MaxFlow removal, late-seed failure, source mutation during analysis, filename seed mismatch, strict JSON | Passed; complete-tree publication and retained prior report on failure |
| Plots | Inclusive pre-steps, exact observation thresholds, common baseline, seed IQR, adjusted significance, visual inspection | Passed; single/pooled BAC, delta, latency and heatmap renders inspected |
| Research consumers | Shared BAC computation, explicit availability field, consistent lower percentiles, unrounded stored values | Updated and covered by the full research suite |

## Defects found and fixed

The first new numerical suite had 18 failures before correction. Evidence is in
`before.json` and `before-tests.log`; later regression logs are saved separately.

| Finding | Reproducer | Correct behavior |
| --- | --- | --- |
| Incorrect availability rank | `[0,...,9]` returned BW@90=0 | BW@90=1: exactly 9/10 samples meet it |
| Endpoint aliasing | Unchanged device paths of cost 1 and 10 produced p99 stretch 10 | Each device has its own reference; p99=1 |
| Unmatched MaxFlow selectors | Native regex group labels produced SPS=0 on a fully connected network | Exact endpoint coverage is required; incompatible groups raise a diagnostic |
| Missing cost represented as free | No CostPower step produced zero unit cost | Unavailable values serialize as null |
| Independent sample positions aligned | Permuting one seed changed positional median/IQR | Positional reduction removed; compare scalar seed summaries and survival curves |
| Stale/mixed reports | Removed seed and MaxFlow files survived; a failing later seed partially rewrote output | Entire successful report replaces prior output; failure preserves prior bytes |
| Undefined baseline normalized to one | Missing/zero denominators in the baseline row became 1 | They remain unavailable |
| Extreme-scale inference | Nonconstant differences at 1e-200 became deterministic; at 1e200 variance overflowed | Scale before variance; preserve p/t and rescale CI |
| Best-path tolerance depended on units | Doubling path cost at 1e-12 scale still counted as best path | Relative tolerance, no arbitrary absolute floor |
| Plot/report inconsistency | Single-seed curve used post-steps; baseline choice and clipping differed | Inclusive pre-steps, common selector, uncapped delivery curves |
| Partial latency coverage hidden | Delivered traffic without a baseline reference vanished from the denominator | `reference_coverage` explicitly reports the included delivered fraction |

## Interpretation

- BAC deliberately samples one baseline plus occurrence-weighted failures.
  SPS, pair matrices, and failure latency use failures only. These summaries
  describe the configured experiment; they do not measure uptime.
- BW@p uses sorted zero-based rank `n - ceil(n*p/100)`, with exact decimal count
  boundaries. Descriptive high percentiles retain NumPy's `lower` estimator;
  they are not availability guarantees and need not be replication invariant.
- A seed median uses equal seed weight; pooling uses equal observation weight.
  Seeds with more observations therefore contribute more to pooled results.
- MaxFlow capacities are independent per pair; their sum does not establish
  simultaneous multicommodity feasibility. Group-to-device reassignment cannot
  be inferred safely from labels alone and is explicitly unsupported.
- Latency is conditional on delivered volume with a positive baseline reference.
  Outages retain missing entries; reference coverage distinguishes absent path
  details and partially referenceable delivery. Inspect BAC alongside latency.
- Zero baseline delivery has defined absolute bandwidth and undefined ratios.
  NetGraph's MSD workflow rejects a network with no positive feasible multiplier.
- Pairwise demand expansion divides volume across matched pairs/groups, as
  confirmed by source inspection and the native experiment.
- Statistical tests require three finite matched seeds and the usual paired-test
  assumptions. Constant-sample p=0/1 is a computational convention, not certainty
  about the population. Finite samples cannot establish multi-nine uptime.
- The audit covers arithmetic and integration for the listed models. Recorded
  durations depend on the environment and are not benchmark results.

Sources for definitions: [NumPy quantile methods](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html),
[SciPy paired t-test and confidence interval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html).

## Validation results

Results from the metric verification run.

- Independent native experiment: passed, including 435 failure trials across the
  15 multi-seed placement studies and an independently calculated reference table.
- Numerical/native targeted checks: passed; complete metrics/current-engine suite
  passed 138 tests after restoring strict demand selectors.
- `make check-ci`: **516 passed in 153.43 s**, overall coverage 80%; Ruff passes;
  Pyright has zero errors and three missing-stub warnings (SciPy/CairoSVG).
- Wheel and sdist build; `twine check` and `pip check` pass.
- The rebuilt wheel was force-installed into a separate environment with its own
  resolved dependencies. The import path was asserted to be its `site-packages`.
  A real mini-DCBB run reproduced AUC=6/11; the full 15-study report matched the
  source-tree report across every numeric project column at rtol=1e-12.
- The eight saved geographic fixture analyses were regenerated. Released
  changelog history is unchanged byte-for-byte. No commit or push was performed.

Key evidence files: `experiments.log`, `native-summary.json`,
`multiseed-expected.json`, `multiseed_metrics/project.csv`, `check-ci.log`,
`build.log`, `twine.log`, `pip-check.log`, and `wheel-smoke.log`.
