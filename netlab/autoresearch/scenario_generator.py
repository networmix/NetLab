"""DC-BB scenario generation: mesh group algorithm, node builder, and configuration.

This module implements the GCD-based mesh group assignment algorithm
for partitioning DC and BB devices into groups that form full-mesh
interconnects. It also provides the DcBbScenarioConfig dataclass
shared across all scenario generation steps, and the _build_nodes
function that creates all topology nodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DcBbScenarioConfig:
    """Configuration for DC-BB scenario generation."""

    # ABC1 (DCType1) parameters
    abc1_hgrids: int = 16
    abc1_fadu_per_hgrid: int = 36
    abc1_planes: int = 8
    abc1_ssw_per_plane: int = 36
    abc1_pods_per_building: int = 96
    abc1_buildings: int = 5
    abc1_rsw_per_pod: int = 48

    # XYZ1 (DCTypeF) parameters
    xyz1_xsw_planes: int = 24
    xyz1_xsw_per_plane: int = 64
    xyz1_ssw_per_megapod: int = 24
    xyz1_fsw_per_megapod: int = 32
    xyz1_megapods: int = 72

    # Backbone parameters
    bb_planes: int = 64
    bb_devices_per_plane: int = 4

    # Link capacities
    dc_bb_link_capacity: float = 400.0
    bb_bb_link_capacity: float = 800.0

    # DC-BB interconnect parameters (the things autoresearch varies)
    g_abc1: int = 64
    g_xyz1: int = 64
    layout_abc1: tuple = (16, 4, 16, 4)
    layout_xyz1: tuple = (16, 4, 16, 4)

    # Workflow parameters
    seed: int = 42
    msd_resolution: float = 0.01
    failure_iterations: int = 200


def _compute_mesh_groups(
    dc_rows: int,
    dc_cols: int,
    bb_rows: int,
    bb_cols: int,
    g: int,
    layout: tuple[int, int, int, int],
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    """Compute mesh group assignments for DC and BB device grids.

    Partitions a dc_rows x dc_cols grid of DC devices and a
    bb_rows x bb_cols grid of BB devices into G groups. Each group
    is a contiguous rectangular block in both the DC and BB grids.

    Args:
        dc_rows: Number of DC device rows.
        dc_cols: Number of DC device columns.
        bb_rows: Number of BB device rows.
        bb_cols: Number of BB device columns.
        g: Number of mesh groups.
        layout: (gr_dc, gc_dc, gr_bb, gc_bb) — how G factorizes
                into row/column splits for each grid.

    Returns:
        List of G tuples (dc_devices, bb_devices), where each element
        is a list of (row, col) index tuples.

    Raises:
        ValueError: If the layout is incompatible with the grid
                    dimensions or G.
    """
    gr_dc, gc_dc, gr_bb, gc_bb = layout

    if gr_dc * gc_dc != g:
        raise ValueError(
            f"DC layout {gr_dc}x{gc_dc}={gr_dc * gc_dc} does not match G={g}"
        )
    if gr_bb * gc_bb != g:
        raise ValueError(
            f"BB layout {gr_bb}x{gc_bb}={gr_bb * gc_bb} does not match G={g}"
        )
    if dc_rows % gr_dc != 0:
        raise ValueError(f"dc_rows={dc_rows} not divisible by gr_dc={gr_dc}")
    if dc_cols % gc_dc != 0:
        raise ValueError(f"dc_cols={dc_cols} not divisible by gc_dc={gc_dc}")
    if bb_rows % gr_bb != 0:
        raise ValueError(f"bb_rows={bb_rows} not divisible by gr_bb={gr_bb}")
    if bb_cols % gc_bb != 0:
        raise ValueError(f"bb_cols={bb_cols} not divisible by gc_bb={gc_bb}")

    dc_block_rows = dc_rows // gr_dc
    dc_block_cols = dc_cols // gc_dc
    bb_block_rows = bb_rows // gr_bb
    bb_block_cols = bb_cols // gc_bb

    groups: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    for group_id in range(g):
        # DC block position
        gi = group_id // gc_dc
        gj = group_id % gc_dc

        dc_devs: list[tuple[int, int]] = []
        for r in range(gi * dc_block_rows, (gi + 1) * dc_block_rows):
            for c in range(gj * dc_block_cols, (gj + 1) * dc_block_cols):
                dc_devs.append((r, c))

        # BB block position (independent factorization)
        bi = group_id // gc_bb
        bj = group_id % gc_bb

        bb_devs: list[tuple[int, int]] = []
        for r in range(bi * bb_block_rows, (bi + 1) * bb_block_rows):
            for c in range(bj * bb_block_cols, (bj + 1) * bb_block_cols):
                bb_devs.append((r, c))

        groups.append((dc_devs, bb_devs))

    return groups


def get_viable_g_values(
    dc_total: int, bb_total: int, dc_ports: int, bb_ports: int
) -> list[int]:
    """Return sorted list of viable G values given device counts and port limits.

    G is viable when:
    1. G evenly divides both dc_total and bb_total.
    2. Per-device degree in each group respects port limits:
       k_dc = bb_total / G <= dc_ports
       k_bb = dc_total / G <= bb_ports

    Args:
        dc_total: Total number of DC devices.
        bb_total: Total number of BB devices.
        dc_ports: Maximum ports per DC device (limits connections to BB).
        bb_ports: Maximum ports per BB device (limits connections to DC).

    Returns:
        Sorted list of viable G values.
    """
    # G must divide both totals. The set of common divisors of dc_total
    # and bb_total is the set of divisors of gcd(dc_total, bb_total).
    g_common = math.gcd(dc_total, bb_total)

    viable: list[int] = []
    for candidate in _divisors(g_common):
        # But G must actually divide each total individually (which is
        # guaranteed since candidate | gcd), and we also need dc_total/G
        # and bb_total/G to be integers (also guaranteed). Check ports:
        k_dc = bb_total // candidate
        k_bb = dc_total // candidate
        if k_dc <= dc_ports and k_bb <= bb_ports:
            viable.append(candidate)

    return sorted(viable)


def _divisors(n: int) -> list[int]:
    """Return all positive divisors of n in ascending order."""
    if n <= 0:
        return []
    divs: list[int] = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def get_valid_layouts(
    g: int,
    dc_rows: int,
    dc_cols: int,
    bb_rows: int,
    bb_cols: int,
) -> list[tuple[int, int, int, int]]:
    """Return all valid (gr_dc, gc_dc, gr_bb, gc_bb) factorizations for G.

    Valid means:
    - gr_dc * gc_dc == g
    - gr_bb * gc_bb == g
    - dc_rows divisible by gr_dc
    - dc_cols divisible by gc_dc
    - bb_rows divisible by gr_bb
    - bb_cols divisible by gc_bb

    Args:
        g: Number of mesh groups.
        dc_rows: DC grid row count.
        dc_cols: DC grid column count.
        bb_rows: BB grid row count.
        bb_cols: BB grid column count.

    Returns:
        List of valid (gr_dc, gc_dc, gr_bb, gc_bb) tuples, sorted.
    """
    dc_facts = _factorizations(g, dc_rows, dc_cols)
    bb_facts = _factorizations(g, bb_rows, bb_cols)

    layouts: list[tuple[int, int, int, int]] = []
    for gr_dc, gc_dc in dc_facts:
        for gr_bb, gc_bb in bb_facts:
            layouts.append((gr_dc, gc_dc, gr_bb, gc_bb))

    return sorted(layouts)


def _factorizations(g: int, rows: int, cols: int) -> list[tuple[int, int]]:
    """Return all (gr, gc) where gr*gc == g, rows%gr == 0, cols%gc == 0."""
    results: list[tuple[int, int]] = []
    for gr in _divisors(g):
        gc = g // gr
        if rows % gr == 0 and cols % gc == 0:
            results.append((gr, gc))
    return results


def validate_layout(
    g: int,
    layout: tuple[int, int, int, int],
    dc_rows: int,
    dc_cols: int,
    bb_rows: int,
    bb_cols: int,
) -> bool:
    """Check if a layout is valid for the given dimensions.

    Args:
        g: Number of mesh groups.
        layout: (gr_dc, gc_dc, gr_bb, gc_bb).
        dc_rows: DC grid row count.
        dc_cols: DC grid column count.
        bb_rows: BB grid row count.
        bb_cols: BB grid column count.

    Returns:
        True if the layout is valid, False otherwise.
    """
    gr_dc, gc_dc, gr_bb, gc_bb = layout
    return (
        gr_dc * gc_dc == g
        and gr_bb * gc_bb == g
        and dc_rows % gr_dc == 0
        and dc_cols % gc_dc == 0
        and bb_rows % gr_bb == 0
        and bb_cols % gc_bb == 0
    )


# ---------------------------------------------------------------------------
# Node builder
# ---------------------------------------------------------------------------

# DCTypeF structural constant: FSW are arranged in 4 rows within each MegaPod.
# This is fixed by the MegaPod architecture and not parameterized in the config.
_XYZ1_FSW_ROWS = 4


def _build_nodes(config: DcBbScenarioConfig) -> dict:
    """Build all node definitions for the DC-BB topology.

    Creates nodes for:
    - ABC1 (DCType1): RSW, FSW, SSW, FADU
    - Backbone: BB devices on both abc1 and xyz1 sides
    - XYZ1 (DCTypeF): RSW, FSW, SSW, XSW

    Args:
        config: Scenario configuration with device counts per layer.

    Returns:
        Dict mapping node_name -> {"attrs": {"role": ..., "site": ..., ...}}.
    """
    nodes: dict[str, dict] = {}

    # --- ABC1 (DCType1) ---

    # Collapsed RSW: 1 per pod
    for p in range(1, config.abc1_pods_per_building + 1):
        nodes[f"abc1/pod{p}/rsw"] = {
            "attrs": {"role": "rsw", "site": "abc1"},
        }

    # FSW: 1 per pod per plane
    for p in range(1, config.abc1_pods_per_building + 1):
        for pl in range(1, config.abc1_planes + 1):
            nodes[f"abc1/pod{p}/fsw/plane{pl}"] = {
                "attrs": {"role": "fsw", "site": "abc1", "plane": pl, "pod": p},
            }

    # SSW: per plane per index
    for pl in range(1, config.abc1_planes + 1):
        for i in range(1, config.abc1_ssw_per_plane + 1):
            nodes[f"abc1/ssw/plane{pl}/idx{i}"] = {
                "attrs": {"role": "ssw", "site": "abc1", "plane": pl, "index": i},
            }

    # FADU: per hgrid per index
    for h in range(1, config.abc1_hgrids + 1):
        for i in range(1, config.abc1_fadu_per_hgrid + 1):
            nodes[f"abc1/fadu/hgrid{h}/idx{i}"] = {
                "attrs": {"role": "fadu", "site": "abc1", "hgrid": h, "index": i},
            }

    # --- Backbone ---

    # BB devices on both sites
    for site in ["abc1", "xyz1"]:
        for pl in range(1, config.bb_planes + 1):
            for d in range(1, config.bb_devices_per_plane + 1):
                nodes[f"bb/{site}/plane{pl}/dev{d}"] = {
                    "attrs": {
                        "role": "bb",
                        "site": site,
                        "plane": pl,
                        "device": d,
                    },
                }

    # --- XYZ1 (DCTypeF) ---

    # Collapsed RSW: 1 per MegaPod (we model 1 MegaPod)
    nodes["xyz1/mp1/rsw"] = {
        "attrs": {"role": "rsw", "site": "xyz1"},
    }

    # FSW: arranged in rows × devices-per-row within the MegaPod
    fsw_devs_per_row = config.xyz1_fsw_per_megapod // _XYZ1_FSW_ROWS
    for r in range(1, _XYZ1_FSW_ROWS + 1):
        for d in range(1, fsw_devs_per_row + 1):
            nodes[f"xyz1/mp1/fsw/row{r}/dev{d}"] = {
                "attrs": {"role": "fsw", "site": "xyz1", "row": r, "device": d},
            }

    # SSW: 1 per XSW plane within the MegaPod
    for pl in range(1, config.xyz1_ssw_per_megapod + 1):
        nodes[f"xyz1/mp1/ssw/plane{pl}"] = {
            "attrs": {"role": "ssw", "site": "xyz1", "plane": pl},
        }

    # XSW: per plane per device (shared across all MegaPods)
    for pl in range(1, config.xyz1_xsw_planes + 1):
        for d in range(1, config.xyz1_xsw_per_plane + 1):
            nodes[f"xyz1/xsw/plane{pl}/dev{d}"] = {
                "attrs": {
                    "role": "xsw",
                    "site": "xyz1",
                    "plane": pl,
                    "device": d,
                },
            }

    return nodes
