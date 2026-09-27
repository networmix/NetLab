# DC-BB interconnect experiments

Compare connections between data-center (DC) and backbone (BB) devices at two
sites. DC rows represent spine rows; BB rows represent planes. Matching backbone
planes connect the sites. Backbone planes have no direct links to one another
within a site.

Topology directories are named
`dc{rows}x{devices}_bb{planes}x{devices}_bb{planes}x{devices}_dc{rows}x{devices}_{pattern}`,
ordered DC-A, BB-A, BB-B, DC-B. Each directory's `scenario.yml` defines the actual
connections and capacities. Use `--list` to see the available configurations.

## Run

From the repository root, with the development environment installed:

```bash
source venv/bin/activate
cd experiments/dc-bb-interconnect
python run.py --list

# Run an included topology with three seeds
python run.py dc4x9_bb4x4_bb4x4_dc4x9_one_to_one --seeds 42 43 44

# Seed range: 42 through 49
python run.py dc4x9_bb4x4_bb4x4_dc4x9_one_to_one --seeds 42:50

# Write the merged scenario without simulating
python run.py dc4x9_bb4x4_bb4x4_dc4x9_one_to_one --dry-run

# Analyze existing results and compare topologies
python run.py --metrics
python run.py --compare
```

Add `--force` to rerun cached simulations. Results are written to
`results/{topology}/`, with raw NetGraph outputs in `{topology}__seed{N}/` and
aggregated failure statistics in `summary.json`.

## Included patterns

- `one_to_one`: each DC row attaches to its corresponding BB plane.
- `full_mesh`: each DC row attaches to every BB plane.
- `balanced_sparse`, `balanced_dense`, and other named variants: partial
  connections defined in their scenario files.

The small configurations use 4×9 DC grids and four BB planes per site, except
`two_to_one`, which has two planes. The larger `one_to_one` case uses a 16×36 DC
grid and 16×4 backbone per site.
