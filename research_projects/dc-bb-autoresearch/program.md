# DC-BB Interconnect Optimization

## Topology

Two data centers (ABC1 and XYZ1) connected through a shared 64-plane backbone.

**ABC1 (DCType1):** 576 FADU devices in a 16×36 grid (16 HGRIDs × 36 FADU/HGRID). Up to 16 BB-facing ports per FADU.

**XYZ1 (DCTypeF):** 1,536 XSW devices in a 64×24 grid (64 devices/plane × 24 planes). Up to 4 BB-facing ports per XSW.

**Backbone:** 256 BB devices per site (64 planes × 4 devices/plane). Dual long-haul paths (Path_A, Path_B) cross-connect at 800 Gbps per link. Total cross-site: 2,048 links.

## Mesh Group Interconnect

G groups partition both DC and BB grids into equal rectangular blocks. Full mesh within each group. Notation `ArxBc <> CrxDc`: DC block (A rows × B cols) paired with BB block (C rows × D cols).

BB block shape controls how connections overlap failure domains: planes, plane groups, and device indices. DC block shape controls which devices share a group.

## Design Rules

For single plane-site, device-index, and BB-device failures:

1. Every DC device retains at least one BB connection.
2. Each device retains at least 75% of its BB connections.
3. Maximize G subject to these rules to reduce port count.

The structural feasibility check excludes plane-group and long-haul failures.
Evaluate those separately in simulations; passing the structural check does not
establish resilience to every failure mode.

## Reference results

These tables are study inputs without saved run provenance. Reproduce the selected
configurations before using the values to rank designs. The current generator
runs each configured failure mode separately and combines them with equal weight.
Record the NetGraph version, seed, iteration count, and policies for new results.

Metrics:
- **alpha_star**: Maximum demand multiplier the topology supports (higher = more capacity)
- **BAC AUC**: Bandwidth Availability Curve area under curve (higher = more resilient under failures)

### ABC1 Results (other side fixed at G_xyz1=64)

| G | k_dc | BB Block | alpha* | BAC AUC | Feasible |
|---|------|----------|--------|---------|----------|
| 16 | 16 | 16rx1c | 9.21 | 1.0000 | ✓ |
| 16 | 16 | 4rx4c | 9.21 | 1.0000 | ✓ |
| 16 | 16 | 8rx2c | 9.21 | 0.9994 | ✓ |
| 32 | 8 | 8rx1c | 9.21 | 1.0000 | ✗ |
| 32 | 8 | 4rx2c | 9.21 | 1.0000 | ✗ |
| 32 | 8 | 2rx4c | 9.21 | 1.0000 | ✗ |
| 64 | 4 | 4rx1c | 9.21 | 0.8900 | ✗ |
| 64 | 4 | 2rx2c | 9.21 | 0.8885 | ✗ |
| 64 | 4 | 1rx4c | 9.21 | 0.8891 | ✗ |

The listed ABC1 cases have alpha 9.21. Their BAC differs by group count: about
1.0 for G=16/32 and 0.89 for G=64. Check the simulated link utilization before
attributing the shared alpha to a bottleneck.

### XYZ1 Results (other side fixed at G_abc1=64)

| G | k_dc | BB Block | alpha* | BAC AUC | Feasible |
|---|------|----------|--------|---------|----------|
| 64 | 4 | 4rx1c | 9.21 | 0.8900 | ✗ |
| 64 | 4 | 2rx2c | 9.21 | 0.8851 | ✗ |
| 64 | 4 | 1rx4c | 9.21 | 0.8943 | ✗ |
| 128 | 2 | 2rx1c | 9.21 | 0.8697 | ✗ |
| 128 | 2 | 1rx2c | 9.21 | 0.8697 | ✗ |
| 256 | 1 | 1rx1c | 6.14 | 0.9405 | ✗ |

In these inputs, XYZ1 G=256 has lower alpha and higher normalized BAC than
G=64/128. Its DC-BB capacity is 1,536 × 400 Gbps = 614.4 Tbps. Compare absolute
delivered bandwidth as well as normalized BAC. None of the listed XYZ1 layouts
passes the connection-retention rules.

### Cross-Side Results (54 combinations, top by BAC)

| G_abc1 | BB_abc1 | G_xyz1 | BB_xyz1 | alpha* | BAC AUC |
|--------|---------|--------|---------|--------|---------|
| 16/32 | any | 64 | any | 9.21 | 1.0000 |
| 16/32 | any | 128 | any | 9.21 | 0.9651-0.9660 |
| 64 | 1rx4c | 256 | 1rx1c | 6.14 | 0.9511 |
| 16 | 16rx1c | 256 | 1rx1c | 6.14 | 0.9488 |
| 64 | any | 64 | any | 9.21 | 0.8849-0.8946 |
| 64 | any | 128 | any | 9.21 | 0.8694-0.8700 |

## Research Questions

Use fresh simulation results to answer:

1. **Deployment recommendation:** Which ABC1 × XYZ1 combination warrants further testing? Consider the tradeoff between port cost (higher G = fewer ports), capacity (alpha), and resilience (BAC).

2. **G=32 on ABC1:** It achieves BAC = 1.0 like G=16 but uses half the BB ports (8 vs 16 per FADU). The structural analysis flags it as infeasible under the connection-retention rules. Is the simulation BAC of 1.0 trustworthy, or is 200 iterations insufficient to reveal the vulnerability?

3. **XYZ1 G=256 vs G=64:** Is the 33% capacity reduction acceptable for the BAC improvement? Under what traffic load assumptions?

4. **DC-side block choice:** For the recommended G values, does the DC-side factorization matter for operational reasons (cabling, traffic locality, failure blast radius)?

5. **Sensitivity to failure weights:** How does the ranking change when the combined policy weights change? Report the weights used and compare per-mode results.

## Parameters

- `g_abc1`: 16, 32, 64
- `g_xyz1`: 64, 128, 256
- `layout_abc1`: Grid partition counts; both products must equal `g_abc1`.
- `layout_xyz1`: Grid partition counts; both products must equal `g_xyz1`.

A layout is `DC_row_groups x DC_column_groups _ BB_row_groups x BB_column_groups`.
Block dimensions are grid dimensions divided by these group counts.

Return parameter choices as YAML:
```yaml
params:
  g_abc1: "16"
  g_xyz1: "64"
  layout_abc1: "4x4_16x1"
  layout_xyz1: "16x4_16x4"
```
