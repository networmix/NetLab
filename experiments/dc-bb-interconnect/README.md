# DC-BB Interconnect Study

Assessment of Data Center (DC) to Backbone (BB) interconnect topologies across two sites.

## Network Model

```
Site A                              Site B
┌─────────────────┐                ┌─────────────────┐
│  Data Center    │                │  Data Center    │
│  (rows x cols)  │                │  (rows x cols)  │
└────────┬────────┘                └────────┬────────┘
         │                                  │
         ▼                                  ▼
┌─────────────────┐                ┌─────────────────┐
│    Backbone     │◄──────────────►│    Backbone     │
│ (planes x cols) │  Inter-site    │ (planes x cols) │
└─────────────────┘                └─────────────────┘
```

## Assumptions

### Data Center (DC) Layer

- Modeled as a matrix with rows and columns
- Rows represent independent rows in the DC spine layer
- Columns represent devices per row
- Traffic originates/terminates at or behind DC layer nodes
- Supported shapes: 16x36, 24x64

### Backbone (BB) Layer

- Modeled as a matrix with planes and columns (nodes per plane)
- Planes are independent within a site (no cross-plane connectivity intra-site)
- BB attaches to DC spine layer
- Supported shapes: 64x4, 32x4, 16x4, 8x4, 16x1, 16x2

### Inter-site Connectivity

- Only BB layers are interconnected between sites
- Connectivity is plane-to-plane (plane N at Site A connects to plane N at Site B)
- Planes within a site are NOT connected to each other

### Asymmetry Support

- DC shapes can differ between sites
- BB shapes can differ between sites (both plane count and nodes-per-plane)
- When BB plane counts differ, only common planes (min of both) have inter-site links

## Topology Naming

Format: `dc{R}x{C}_bb{P}x{N}_bb{P}x{N}_dc{R}x{C}_{pattern}`

- DC-A: Site A Data Center (rows x cols)
- BB-A: Site A Backbone (planes x nodes)
- BB-B: Site B Backbone (planes x nodes)
- DC-B: Site B Data Center (rows x cols)
- pattern: Interconnect pattern (one_to_one, full_mesh, etc.)

Examples:

- `dc16x36_bb32x4_bb32x4_dc16x36_one_to_one` - symmetric 32-plane BB
- `dc16x36_bb16x4_bb16x4_dc16x36_full_mesh` - symmetric 16-plane BB
- `dc16x36_bb32x4_bb16x4_dc24x64_one_to_one` - asymmetric configuration

## Usage

```bash
# List available topologies
python3 run.py --list

# Run specific topology with seeds
python3 run.py dc16x36_bb32x4_bb32x4_dc16x36_one_to_one --seeds 42 43 44

# Run with seed range
python3 run.py dc16x36_bb32x4_bb32x4_dc16x36_one_to_one --seeds 42:50

# Force re-run (ignore cache)
python3 run.py dc16x36_bb32x4_bb32x4_dc16x36_one_to_one --force

# Compute metrics only (from existing results)
python3 run.py --metrics

# Dry run (generate merged scenario without running)
python3 run.py dc16x36_bb32x4_bb32x4_dc16x36_one_to_one --dry-run
```

## Output

Results saved to `results/{topology}/`:

- `summary.json` - aggregated failure analysis metrics
- `{topology}__seed{N}/` - per-seed raw ngraph results

## Interconnect Patterns

- **one_to_one**: Each DC row connects to a fixed subset of BB planes
- **full_mesh**: Every DC row connects to all BB planes
- **balanced_sparse**: DC rows connect to subset of planes with rotation
- **balanced_dense**: Higher connectivity with skip pattern
