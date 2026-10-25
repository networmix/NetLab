# Check metric calculations

`dev/verify_metrics.py` creates small networks, runs NetGraph and compares NetLab's
metrics with answers calculated separately. The networks are small enough to
check by hand or with a separate maximum-flow calculation.

This page explains how to run those checks and interpret them. The metric
reference is [metrics.md](../metrics.md).

## Run the checks

From the repository root, install `.[dev]` in the project's virtual environment.
Then run:

```bash
venv/bin/python dev/verify_metrics.py --plots
venv/bin/python -m pytest tests/metrics tests/test_current_ngraph_integration.py tests/test_mini_dcbb_verification.py --no-cov
```

The first command writes to `build/metrics-verification/`. Use `--output PATH`
to choose another directory. `--plots` also renders report figures. The output
contains the NetGraph results and the independently calculated answers; the input
networks are defined in `dev/verify_metrics.py`. A failed comparison raises an
error and makes the command fail. The second
command runs the metric regression and integration tests. For formatting, lint,
type checks and the complete test suite, run `make check-ci`.

## A network you can check by hand

The verification network has two routes between the same endpoints:

| Route | Capacity | Total cost |
| --- | ---: | ---: |
| Through node `u` | 5 | 2 |
| Through node `v` | 3 | 4 |

With both routes available, it carries 8 units. If a link on the first route
fails, only 3 units can be delivered. The experiment includes one baseline and
one failure sample, so its bandwidth ratios are `1` and `3/8`. Their mean,
the capped BAC score, is `(1 + 3/8) / 2 = 0.6875`.

The failure's delivered traffic uses the route of cost 4. The cheapest baseline
route costs 2, so its latency stretch is `4/2 = 2`. This ratio measures path cost,
not elapsed network delay.

There are four routers at 100 dollars and 10 watts each, plus eight optics at
2 dollars and 1 watt each. The totals are `4*100 + 8*2 = 416` dollars and
`4*10 + 8*1 = 48` watts. MSD accepts the 8 units unchanged (`alpha_star = 1`),
so `USD_per_Gbit_offered` is `416 / (1 * 8) = 52`.

The script repeats the experiment for all 16 subsets of its four physical links.
It checks capacities against both NetworkX maximum flow and the two-route sum.
It also checks weighted samples, placement variants and several random seeds.

## What the checks cover

| Area | Independent check |
| --- | --- |
| Bandwidth at a stated availability | Count samples that meet each threshold, including exact decimal boundaries and repeated samples |
| Availability curves | Count observations at or above each distinct delivered-bandwidth value |
| Latency stretch | Expand delivered-volume distributions and use a separate baseline cost for each endpoint pair |
| Pair capacity and SPS | Compare each failed graph with maximum flow and the two-route arithmetic |
| Cost and power | Add device totals and divide by independently calculated bandwidth |
| Seed summaries | Calculate medians, ratios, differences and pooled counts separately |
| Statistical comparisons | Compare paired tests with SciPy, including very small/large scales and constant differences |
| Invalid inputs | Change demand, delivery, costs, occurrence counts and endpoint fields; require a diagnostic |
| Report replacement | Reduce inputs or force a late failure; check that output is complete and failure keeps the saved report |

Regression cases are in `tests/metrics/test_numerical_verification.py`,
`tests/metrics/test_native_verification.py` and the other files under `tests/metrics/`.
The CLI tests also check tables and report replacement.

## Inspect the output

| File or directory | Contents |
| --- | --- |
| `diamond-mask-*.json` | NetGraph results for each failed-link subset |
| `native-summary.json` | Expected and measured values for the small network |
| `variant-*.json` | Results for the additional model cases |
| `disconnected-msd.json` | The expected rejection when no positive demand multiplier is feasible |
| `multiseed/` | NetGraph results for three capacity variants, each with five seeds |
| `multiseed-expected.json` | Independently calculated summaries across seeds |
| `multiseed_metrics/` | Generated tables and optional figures for those results |

Read bandwidth and latency together. A total outage has zero delivered bandwidth
and no meaningful delivered-path latency. Pairwise maximum flow treats each pair
independently; its sum does not prove that all pairs can carry traffic at once.
These experiments check arithmetic and integration, not real-world uptime or
execution speed.

For the statistical methods, see
[NumPy quantiles](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)
and [SciPy's paired t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html).
