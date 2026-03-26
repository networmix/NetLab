# DC-BB Interconnect Optimization

## Topology

Two data centers (ABC1 and XYZ1) connected through a shared 64-plane backbone.

**ABC1 (DCType1):** 576 FADU devices in a 16×36 grid (16 HGRIDs × 36 FADU/HGRID). Up to 16 BB-facing ports per FADU.

**XYZ1 (DCTypeF):** 1,536 XSW devices in a 64×24 grid (64 devices/plane × 24 planes). Up to 4 BB-facing ports per XSW.

**Backbone:** 256 BB devices per site (64 planes × 4 devices/plane). Dual long-haul paths (Path_A, Path_B) cross-connect at 800 Gbps per link. Total cross-site: 2,048 links.

## Mesh Group Interconnect

G groups partition both DC and BB grids into equal rectangular blocks. Full mesh within each group. Notation `ArxBc <> CrxDc`: DC block (A rows × B cols) paired with BB block (C rows × D cols).

The **BB block shape** determines failure resilience — how a group's BB connections align with failure domains (planes, plane groups, device indices). The **DC block shape** determines which physical devices share a mesh group — affecting cabling and traffic locality.

## Design Rules

1. **No hanging:** Every DC device retains ≥1 BB connection after any single non-LH failure.
2. **≥75% retention:** Each device keeps ≥75% of BB connections under any single non-LH failure (excluding plane_group which affects only ~1.6% of devices per event).
3. **Maximize G** subject to Rules 1-2 (larger G = fewer ports = lower cost).

## Simulation Results

All configs were simulated with ngraph (200 failure iterations, 7 weighted failure modes). Metrics:
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

**Key finding:** Alpha is identical (9.21) across all ABC1 configs — ABC1 DC-BB (921.6 Gbps aggregate) is always the capacity bottleneck regardless of G or layout. BAC differentiates: G=16/32 achieve BAC ≈ 1.0, G=64 drops to 0.89.

### XYZ1 Results (other side fixed at G_abc1=64)

| G | k_dc | BB Block | alpha* | BAC AUC | Feasible |
|---|------|----------|--------|---------|----------|
| 64 | 4 | 4rx1c | 9.21 | 0.8900 | ✗ |
| 64 | 4 | 2rx2c | 9.21 | 0.8851 | ✗ |
| 64 | 4 | 1rx4c | 9.21 | 0.8943 | ✗ |
| 128 | 2 | 2rx1c | 9.21 | 0.8697 | ✗ |
| 128 | 2 | 1rx2c | 9.21 | 0.8697 | ✗ |
| 256 | 1 | 1rx1c | 6.14 | 0.9405 | ✗ |

**Key finding:** G=64 and G=128 achieve full alpha (9.21, bottlenecked at ABC1). G=256 has lower alpha (6.14, XYZ1 becomes bottleneck at 614.4 Gbps) but best BAC (0.94). No XYZ1 config passes structural feasibility — 4 BB ports is too few for the 75% rule.

### Cross-Side Results (54 combinations, top by BAC)

| G_abc1 | BB_abc1 | G_xyz1 | BB_xyz1 | alpha* | BAC AUC |
|--------|---------|--------|---------|--------|---------|
| 16/32 | any | 64 | any | 9.21 | 1.0000 |
| 16/32 | any | 128 | any | 9.21 | 0.9651-0.9660 |
| 64 | 1rx4c | 256 | 1rx1c | 6.14 | 0.9511 |
| 16 | 16rx1c | 256 | 1rx1c | 6.14 | 0.9488 |
| 64 | any | 64 | any | 9.21 | 0.8849-0.8946 |
| 64 | any | 128 | any | 9.21 | 0.8694-0.8700 |

**Key findings:**
1. **Best resilience + full capacity:** ABC1 G=16 or G=32 with XYZ1 G=64. BAC = 1.0 with alpha = 9.21.
2. **ABC1 G matters for BAC:** G=16/32 gives BAC ≈ 1.0; G=64 drops to 0.89. The BB block within G doesn't significantly affect cross-side BAC.
3. **XYZ1 G=256 tradeoff:** Better BAC (0.94) but 33% less capacity (alpha 6.14 vs 9.21).
4. **Cross-side interaction:** ABC1 BB block affects BAC slightly when XYZ1 is at G=256 (range 0.93-0.95), but not when XYZ1 is at G=64 (all ≈ 1.0).

## Research Questions

Given the simulation data above:

1. **Deployment recommendation:** Which ABC1 × XYZ1 combination should be deployed? Consider the tradeoff between port cost (higher G = fewer ports), capacity (alpha), and resilience (BAC).

2. **G=32 on ABC1:** It achieves BAC = 1.0 like G=16 but uses half the BB ports (8 vs 16 per FADU). The structural analysis flags it as infeasible (device-index causes 50% loss). Is the simulation BAC of 1.0 trustworthy, or is 200 iterations insufficient to reveal the vulnerability?

3. **XYZ1 G=256 vs G=64:** Is the 33% capacity reduction acceptable for the BAC improvement? Under what traffic load assumptions?

4. **DC-side block choice:** For the recommended G values, does the DC-side factorization matter for operational reasons (cabling, traffic locality, failure blast radius)?

5. **Sensitivity to failure weights:** The Monte Carlo uses fixed weights (LH path 10%, plane group 15%, plane-site 15%, device-index 10%, single BB 15%, dual BB 10%, random link 25%). How sensitive is the ranking to these weights?

## Parameters

- `g_abc1`: 16, 32, 64
- `g_xyz1`: 64, 128, 256
- `layout_abc1`: Grid factorization (MUST match g_abc1 — see template comments)
- `layout_xyz1`: Grid factorization (MUST match g_xyz1)

Return your parameter choices as YAML:
```yaml
params:
  g_abc1: "16"
  g_xyz1: "64"
  layout_abc1: "4x4_16x1"
  layout_xyz1: "16x4_16x4"
```
