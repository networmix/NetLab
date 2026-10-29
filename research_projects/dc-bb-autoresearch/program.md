# Data-center-to-backbone interconnect design

Choose how data-center (DC) devices connect to backbone (BB) devices at two
sites, called ABC1 and XYZ1. Each proposal picks a group count and a layout for
each site. NetLab builds that network, runs it in NetGraph and reports the result.
Compare traffic capacity, delivery under failures and the number of ports needed.
A design is not better just because it has fewer links or a higher normalized score.

## The network being modeled

| Part | Grid and devices | Backbone ports per DC device |
| --- | --- | --- |
| ABC1 | 16 rows × 36 devices = 576 DC devices, named FADU in the model | At most 16 |
| XYZ1 | 64 device positions × 24 planes = 1,536 DC devices, named XSW in the model | At most 4 |
| Backbone at each site | 64 planes × 4 devices = 256 devices | |

A plane is a set of backbone devices that can fail together. DC-to-backbone links
carry 400 Gbps. The two backbones are joined plane by plane: every backbone
device in a plane at ABC1 connects to every device in the same plane at XYZ1, once
on each of two long-haul paths. That gives `64 × 4 × 4 × 2 = 2,048` long-haul
links of 800 Gbps.

The traffic is 100,000 Gbps from the ABC1 rack switches to XYZ1 and 100,000 Gbps
back. `alpha_star` multiplies both directions: `alpha_star = 2` means 200,000 Gbps
each way. The traffic also crosses each site's internal switches, which have their
own capacities. These values come from `DcBbScenarioConfig` in
`netlab/autoresearch/dcbb_config.py` and from
`netlab/autoresearch/scenario_generator.py`.

## Divide devices into connection groups

Each side divides both its DC grid and its BB grid into `G` rectangular groups.
Every device belongs to one group. Within a group, every DC device connects to
every BB device. A DC device therefore has `256 / G` backbone links, and a side
has `DC devices × 256 / G` DC-to-BB links.

For example, ABC1 with `G=16` has `576/16 = 36` DC devices and `256/16 = 16` BB
devices per group. Each DC device needs 16 BB-facing ports, for
`576 × 16 = 9,216` DC-to-BB links on that side. With `G=32`, each DC device needs
8 such ports. The port limits are why ABC1 needs `G` of at least 16 and XYZ1 at
least 64.

A layout string has the form
`DC_row_groups x DC_column_groups _ BB_row_groups x BB_column_groups`.
The two products must both equal `G`, and each count must divide its grid
dimension. These are partition counts, not block sizes. The BB grid has 64 rows,
one per plane, and 4 columns, one per device position.

For ABC1, `4x4_16x1` at `G=16` divides the DC grid into 4×4 groups of 4×9
devices. It divides the BB grid into 16×1 groups of 4×4 devices: 4 planes and all
4 device positions. Matching groups are fully connected. For XYZ1, `16x4_16x4` at
`G=64` gives DC blocks of 4×6 and BB blocks of 4×1.

## Check failure resilience

The structural check calculates connection loss from the layout without running
traffic. It looks at three failures: one plane at one site, one device position
across a plane group of four consecutive planes, and one BB device. A layout
passes when, under each of them, both of these hold:

1. Every DC device keeps at least one BB connection.
2. Every DC device keeps at least 75% of its BB connections.

For ABC1 `4x4_16x1`, a DC device has 16 links to a block of 4 planes × 4 devices.
Losing one plane removes 4 of them (25%). Losing one device position across the
plane group also removes 4, and losing one BB device removes 1. The device keeps
75%, so the layout passes. For ABC1 `8x4_16x2` at `G=32`, the block is 4 planes ×
2 devices; losing one device position removes 4 of 8 links, so it fails.

Under these rules, every ABC1 layout with `G=16` passes, and no ABC1 layout with
`G=32` or `G=64` does. No XYZ1 layout passes. Keeping 75% of at most 4 links
allows each failure to remove only one link of four, so only `G=64` could pass. A
4×1 block loses one link to a plane failure but all four to a device-position
failure; 2×2 and 1×4 blocks lose two or four links to one plane. With `G=128` or
`G=256`, one BB device already removes half or all of an XSW's links. Compare
XYZ1 layouts by their simulated delivery.

Prefer the largest `G` that passes, because a larger `G` uses fewer ports. The
structural rules leave out failures of a whole plane group and of a long-haul
path. Simulate those. A structural pass does not prove that traffic is delivered
under every failure mode.

## How a proposal is evaluated

NetLab generates the scenario for your parameters, checks it and runs NetGraph.
The workflow first runs `msd_baseline`, a `MaximumSupportedDemand` search for
`alpha_star`. It then places the traffic at that multiplier under sampled
failure draws: one placement step for each failure mode defined in
`netlab/autoresearch/dcbb_failures.py`, plus `tm_combined`, whose samples pick one
of those modes with equal probability. The modes cover long-haul paths, planes,
plane groups, device positions, one to eight BB devices and random outages of BB
devices and links. Every step derives its random seed from the run's seed
(`netlab autoresearch run --seed`), so all proposals in a run use the same seed.

The project's [objective.yml](objective.yml) maximizes `alpha_star`, read from
`result.alpha_feasible` of the `msd_baseline` step, and has no constraints. Your
score is `alpha_star` alone: the connection-retention rules and failure delivery
are not part of it, so weigh them yourself. Results that fail to generate or
validate are reported with their error. A proposal whose search finds no feasible
multiplier fails with the reason `none_feasible`; it never scores zero.

`alpha_star` is the largest demand multiplier the bounded MSD search tested and
found feasible. BAC AUC is the mean delivered/baseline ratio, capped at 1, over
the baseline reference and the sampled draws. Also compare absolute delivered
bandwidth: a smaller network can have a higher normalized BAC while carrying less
traffic. See [the metric definitions](../../metrics.md).

## Run comparisons from the command line

From the NetLab repository root with the development environment active:

```bash
netlab autoresearch structural-analysis --output layouts.json
netlab autoresearch sweep abc1 --output-dir results/abc1
netlab autoresearch cross-sweep --output-dir results/cross
```

`structural-analysis` prints every layout's connection loss for each failure type
and saves it to `layouts.json`. `sweep abc1` simulates one layout for each distinct
combination of `G` and BB block shape on ABC1, with XYZ1 held at `G=64` and
`16x4_16x4`. `sweep xyz1` does the same for XYZ1, with ABC1 held at `G=64` and
`16x4_16x4`. `cross-sweep` simulates every combination of those ABC1 and XYZ1
choices. Each entry in `results.jsonl` records `alpha_star`, the BAC AUC of
`tm_combined` and the BAC of every failure mode. The output directory's
`sweep.json` records the seed, iteration count and package versions.

## Questions to answer

- Which layouts satisfy the connection-retention rules, and how many ports do they need?
- Do those layouts still deliver the required traffic under plane-group and long-haul failures?
- Does increasing `G` save ports at an acceptable cost in absolute capacity and delivered bandwidth?
- How does the block shape change capacity and delivery for the same `G`?
- How would different failure-mode weights change the ranking?

## Propose parameters

Choose values from the parameter space listed with each request; it comes from
[hypothesis_template.yml](hypothesis_template.yml). Group counts are integers:
16, 32 or 64 for ABC1 and 64, 128 or 256 for XYZ1. A layout must match its chosen
group count. Return the choice as YAML, with the group counts written as plain
numbers, not quoted strings:

```yaml
params:
  g_abc1: 16
  g_xyz1: 64
  layout_abc1: "4x4_16x1"
  layout_xyz1: "16x4_16x4"
```
