# DC-BB Interconnect Optimization

## Topology

Two data centers (ABC1 and XYZ1) connected through a shared 64-plane backbone.

**ABC1 (DCType1):** 576 FADU devices arranged in a 16×36 grid (16 HGRIDs × 36 FADU/HGRID). Each FADU has up to 16 BB-facing ports.

**XYZ1 (DCTypeF):** 1,536 XSW devices arranged in a 64×24 grid (64 devices/plane × 24 planes). Each XSW has up to 4 BB-facing ports.

**Backbone:** 256 BB devices per site, arranged in a 64×4 grid (64 planes × 4 devices/plane). Dual long-haul paths (Path_A, Path_B) cross-connect the two sites.

## Mesh Group Interconnect

DC devices connect to BB devices through **mesh groups**. G groups partition both the DC grid and the BB grid into G equal-sized rectangular blocks. All DC devices in group i connect to all BB devices in group i (full mesh within group).

### Notation: `ArxBc <> CrxDc`

This describes the **block shape** of ONE mesh group:
- Left: **A rows × B columns** of DC devices
- Right: **C rows × D columns** of BB devices
- A×B = DC_total/G devices per group on DC side
- C×D = BB_total/G devices per group on BB side
- Each group has A×B×C×D links (full mesh)

The DC and BB factorizations are independent — they must produce the same G but can partition their grids differently.

**Physical meaning of rows and columns:**
- FADU grid: rows = HGRIDs (failure-independent), columns = FADU index within HGRID
- XSW grid: rows = device index within plane, columns = plane index
- BB grid: rows = planes, columns = device index within plane

**Why the BB block shape matters:** BB rows are planes. The failure model includes plane-site failures (all devices in one plane at one site), plane group failures (4 consecutive planes), and device-index-across-plane-group failures (same device index across 4 planes). The BB block shape determines how many of a group's BB devices fall within a single failure domain.

### Example: G=64 on ABC1 (k_dc=4 BB per FADU)

| Layout | BB block | Plane-site failure | Dev-idx-across-PG |
|---|---|---|---|
| `1r×9c <> 1r×4c` | 1 plane, all 4 devs | **100% loss** (all BB in same plane) | 25% loss |
| `1r×9c <> 2r×2c` | 2 planes, 2 devs each | **50% loss** (ECMP stranding) | 50% loss |
| `1r×9c <> 4r×1c` | 4 planes, 1 dev each | 25% loss | **100% loss** (block aligns with PG) |

Same G, same link count, same total capacity — but completely different failure behavior.

## Design Rules

### Rule 1: No hanging devices
After any single non-LH-path failure, every DC device must retain ≥1 BB connection. A hanging device forces traffic reconvergence through the DC Clos — adding latency and creating congestion on alternative paths.

**Formally:** For every DC device d and every single failure f (excluding LH path):
`surviving_BB_connections(d, f) ≥ 1`

### Rule 2: ≥75% capacity retention under single failure
Excluding LH path failures (which inherently lose 50% cross-site capacity), any single failure should leave each device with ≥75% of its BB connections. ECMP distributes traffic equally across connections — losing >25% means significant traffic disruption.

**Formally:** For every DC device d and every single failure f (excluding LH path):
`surviving_BB_connections(d, f) / total_BB_connections(d) ≥ 0.75`

This requires `total_BB_connections(d) ≥ 4` AND the layout must spread those connections across at least 4 independent failure domains.

### Rule 3: Largest viable G
Larger G = fewer ports per device = lower cost. Prefer the largest G that satisfies Rules 1 and 2.

**Formally:** Maximize G subject to Rules 1 and 2 being satisfied for all devices and all non-LH failure types.

## Structural Analysis Results

### ABC1 side (FADU, 16 BB-facing ports)

| G | k_dc | Best BB block | Plane-site | Dev-idx-PG | Plane group | Rules |
|---|---|---|---|---|---|---|
| **16** | 16 | **16r×1c** | 6% loss | **25% loss** | 25% loss | ✓ pass |
| 32 | 8 | 8r×1c (best) | 12% loss | **50% loss** | 50% loss | ✗ fail |
| 64 | 4 | 4r×1c (best) | 25% loss | **100% loss** | 100% loss | ✗ fail |

**Answer for ABC1: G=16 with BB block `16r×1c`** (spread each group's 16 BB devices across 16 different planes, 1 device per plane). The DC-side block shape doesn't affect failure properties — all 3 options (`4r×9c`, `2r×18c`, `1r×36c`) are equivalent for resilience.

### XYZ1 side (XSW, 4 BB-facing ports)

| G | k_dc | Best BB block | Plane-site | Dev-idx-PG | Rules |
|---|---|---|---|---|---|
| 64 | 4 | 4r×1c | 25% loss | **100% loss** | ✗ fail |
| 128 | 2 | 2r×1c | 50% loss | **100% loss** | ✗ fail |
| 256 | 1 | — | 100% loss | 100% loss | ✗ fail |

**No G value satisfies both rules on XYZ1.** The XSW port budget (4 ports) is the binding constraint. With only 4 BB connections, any correlated failure removing ≥2 connections exceeds 25% loss.

**Best available for XYZ1: G=64, BB block `4r×1c`** — minimizes plane-site loss (25%) and avoids hanging under that failure type. But device-index-across-plane-group remains catastrophic.

## What to Explore

Given the structural analysis, the research questions are:

1. **Verify the structural predictions with simulation.** Does the `16r×1c` BB block on ABC1 actually achieve the predicted resilience? Does `4r×1c` on XYZ1?

2. **Cross-side interaction.** ABC1 and XYZ1 share the backbone. Does the ABC1 layout choice affect XYZ1 performance (or vice versa)?

3. **DC-side block sensitivity.** On ABC1 with G=16, the 3 DC-side blocks (`4r×9c`, `2r×18c`, `1r×36c`) should be equivalent for BB-failure resilience. But do they differ for DC-internal failure scenarios?

4. **XYZ1 tradeoff.** Given that XYZ1 can't satisfy the 75% rule, which failure types matter most? Is the device-index-across-plane-group failure realistic enough to optimize against, or should we accept it and optimize plane-site resilience instead?

## Parameters

- `g_abc1`: 16, 32, 64 (mesh group count for ABC1-BB)
- `g_xyz1`: 64, 128, 256 (mesh group count for XYZ1-BB)
- `layout_abc1`: Block shape for ABC1 mesh groups (valid options depend on G)
- `layout_xyz1`: Block shape for XYZ1 mesh groups (valid options depend on G)

**Constraint:** Layout must be valid for the chosen G. If you pick an invalid combination, the system will report a generation error.

## Output

Return your parameter choices as YAML:
```yaml
params:
  g_abc1: "16"
  g_xyz1: "64"
  layout_abc1: "4x4_16x1"
  layout_xyz1: "8x8_64x1"
```
