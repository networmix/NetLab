# Compare data-center-to-backbone connections

These experiments compare ways to connect data-center (DC) devices to backbone
(BB) devices at two sites. DC rows represent spine rows. BB rows represent
backbone planes: groups of devices that can share a failure. Matching planes
connect the sites; there are no direct links between different planes at one site.

Each topology directory contains a `scenario.yml` with its devices, connections
and link capacities. Directory names list the grids in this order: DC-A, BB-A,
BB-B, DC-B. `dc4x9` means 4 DC rows of 9 devices, and `bb4x4` means 4 backbone
planes of 4 devices. For example, `dc4x9_bb4x4_bb4x4_dc4x9_one_to_one` has 36 DC
and 16 BB devices at each site. The suffix describes how DC rows connect to
backbone planes.

## Run an included topology

From the repository root, create the development environment with `make dev`.
Then run:

```bash
source venv/bin/activate
cd experiments/dc-bb-interconnect
python run.py --list
python run.py dc4x9_bb4x4_bb4x4_dc4x9_one_to_one --seeds 42 43 44
```

The runner adds shared inputs to the topology's `scenario.yml`: hardware
components from `../_shared/components.yml`, failure policies from `policies/`,
demands from `demands/` and the workflow that the topology names from
`workflows/`. Each topology defines its own blueprints. The seed becomes the
scenario seed, which selects the sampled failures. A range such as `--seeds 42:50`
includes 42 through 49.

The `failure_analysis` workflow first runs `msd_baseline`, a
`MaximumSupportedDemand` search for the largest multiplier of the inter-site
traffic the network can carry. Six placement steps then place that scaled
traffic under 1,000 sampled failures each. Every sample fails one randomly chosen
element of the step's type: a DC-to-BB link (`tm_dc_bb_link`), an inter-site
BB link (`tm_bb_bb_link`), a DC device (`tm_dc_node`), a BB device
(`tm_bb_node`), all inter-site links of one plane (`tm_plane`) or all devices of
one DC row at one site (`tm_dc_row`).

To write the merged scenario without running the analysis:

```bash
python run.py dc4x9_bb4x4_bb4x4_dc4x9_one_to_one --dry-run
```

To analyze saved results, compare topologies or draw the network:

```bash
python run.py --metrics
python run.py --compare
python run.py dc4x9_bb4x4_bb4x4_dc4x9_one_to_one --visualize --split
```

`--visualize` draws a saved result as `results/<topology>/topology.svg`.
`--split` also writes `topology_split.svg`, with disconnected components stacked
so they can be inspected separately. `--force` reruns cached simulations.

## Find the results

Results are saved under `results/<topology>/` relative to this experiment
directory. Each `<topology>__seed<N>/` subdirectory holds the merged scenario and
its NetGraph results. `summary.json` combines the seeds:

- `alpha_star_by_seed` lists each seed's MSD multiplier, NetGraph's
  `alpha_feasible`, or `null` when a result has no `msd_baseline` step;
  `alpha_star` is the median of the available values, zero included. A seed
  whose search finds no feasible multiplier has no saved result: its run fails
  with the reason `none_feasible`.
- `failure_analysis` gives, for each placement step, the minimum, mean and
  maximum fraction of the requested traffic that was delivered, over the sampled
  draws of all seeds.
- `worst_failures` names the steps whose minimum fraction is lowest.

`python run.py --compare` prints these summaries side by side and saves
`results/comparison.json`. `Alpha* %Max` is each topology's `alpha_star` as a
percentage of the largest one in the comparison.

Read the fractions together with `alpha_star`. Each topology is loaded to its
own MSD multiplier, so a topology with a lower `alpha_star` can deliver a larger
fraction of a smaller load and still carry less traffic. See
[the metric definitions](../../metrics.md).

## Connection patterns

| Pattern | Connections |
| --- | --- |
| `one_to_one` | Each DC row attaches to the corresponding BB plane |
| `full_mesh` | Each DC row attaches to every BB plane |
| `two_to_one` | Two DC rows share one BB plane |
| `balanced_sparse`, `balanced_dense` and named variants | Selected connections listed in that topology's scenario |

Most small examples use 4×9 DC grids and four backbone planes of four devices at
each site; `dc4x9_bb2x4_bb2x4_dc4x9_two_to_one` has two planes. The larger
`one_to_one` example uses a 16×36 DC grid and 16 planes of four devices at each
site. Use `--list` and inspect `scenario.yml` to select the configuration you want
to compare.
