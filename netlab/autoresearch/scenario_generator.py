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


# ---------------------------------------------------------------------------
# Internal DC link builder
# ---------------------------------------------------------------------------


def _build_internal_dc_links(config: DcBbScenarioConfig) -> list[dict]:
    """Build all internal DC links within ABC1 and XYZ1.

    Creates intra-DC Clos links (RSW-FSW, FSW-SSW, SSW-FADU/XSW) for both
    data centers. Does NOT create DC-BB or BB cross-site links.

    Capacities are scaled by the building/megapod multiplier:
    - ABC1: ×config.abc1_buildings (default 5)
    - XYZ1: ×config.xyz1_megapods (default 72)

    Args:
        config: Scenario configuration with device counts and scaling factors.

    Returns:
        List of link dicts, each with source, target, capacity, cost, attrs.
    """
    links: list[dict] = []

    # --- ABC1 internal links ---

    abc1_scale = config.abc1_buildings

    # RSW ↔ FSW: each RSW connects to all FSW in its pod (one per plane)
    # Capacity = rsw_per_pod × 200G × buildings
    rsw_fsw_cap = config.abc1_rsw_per_pod * 200.0 * abc1_scale
    for p in range(1, config.abc1_pods_per_building + 1):
        rsw = f"abc1/pod{p}/rsw"
        for pl in range(1, config.abc1_planes + 1):
            links.append(
                {
                    "source": rsw,
                    "target": f"abc1/pod{p}/fsw/plane{pl}",
                    "capacity": rsw_fsw_cap,
                    "cost": 1.0,
                    "attrs": {"link_type": "rsw_fsw", "site": "abc1"},
                }
            )

    # FSW ↔ SSW: each FSW connects to all SSW in its plane
    # Capacity = 200G × buildings
    fsw_ssw_cap = 200.0 * abc1_scale
    for p in range(1, config.abc1_pods_per_building + 1):
        for pl in range(1, config.abc1_planes + 1):
            fsw = f"abc1/pod{p}/fsw/plane{pl}"
            for i in range(1, config.abc1_ssw_per_plane + 1):
                links.append(
                    {
                        "source": fsw,
                        "target": f"abc1/ssw/plane{pl}/idx{i}",
                        "capacity": fsw_ssw_cap,
                        "cost": 1.0,
                        "attrs": {"link_type": "fsw_ssw", "site": "abc1"},
                    }
                )

    # SSW ↔ FADU: SSW(plane_pl, idx_i) connects to FADU(hgrid_h, idx_i)
    # for all h ∈ [1, hgrids]. Capacity = 2 × 200G × buildings.
    ssw_fadu_cap = 2 * 200.0 * abc1_scale
    for pl in range(1, config.abc1_planes + 1):
        for i in range(1, config.abc1_ssw_per_plane + 1):
            ssw = f"abc1/ssw/plane{pl}/idx{i}"
            for h in range(1, config.abc1_hgrids + 1):
                links.append(
                    {
                        "source": ssw,
                        "target": f"abc1/fadu/hgrid{h}/idx{i}",
                        "capacity": ssw_fadu_cap,
                        "cost": 1.0,
                        "attrs": {"link_type": "ssw_fadu", "site": "abc1"},
                    }
                )

    # --- XYZ1 internal links ---

    xyz1_scale = config.xyz1_megapods
    xyz1_link_rate = 400.0  # base link rate for XYZ1 (DCTypeF uses 400G)

    # RSW ↔ FSW: single RSW connects to all FSW
    # Capacity = 400G × megapods
    rsw_fsw_cap_xyz = xyz1_link_rate * xyz1_scale
    rsw_xyz = "xyz1/mp1/rsw"
    fsw_devs_per_row = config.xyz1_fsw_per_megapod // _XYZ1_FSW_ROWS
    for r in range(1, _XYZ1_FSW_ROWS + 1):
        for d in range(1, fsw_devs_per_row + 1):
            links.append(
                {
                    "source": rsw_xyz,
                    "target": f"xyz1/mp1/fsw/row{r}/dev{d}",
                    "capacity": rsw_fsw_cap_xyz,
                    "cost": 1.0,
                    "attrs": {"link_type": "rsw_fsw", "site": "xyz1"},
                }
            )

    # FSW ↔ SSW: each FSW connects to all SSW
    # Capacity = 400G × megapods
    fsw_ssw_cap_xyz = xyz1_link_rate * xyz1_scale
    for r in range(1, _XYZ1_FSW_ROWS + 1):
        for d in range(1, fsw_devs_per_row + 1):
            fsw = f"xyz1/mp1/fsw/row{r}/dev{d}"
            for pl in range(1, config.xyz1_ssw_per_megapod + 1):
                links.append(
                    {
                        "source": fsw,
                        "target": f"xyz1/mp1/ssw/plane{pl}",
                        "capacity": fsw_ssw_cap_xyz,
                        "cost": 1.0,
                        "attrs": {"link_type": "fsw_ssw", "site": "xyz1"},
                    }
                )

    # SSW ↔ XSW: SSW(plane_pl) connects to all XSW in plane_pl
    # Capacity = 400G × megapods
    ssw_xsw_cap = xyz1_link_rate * xyz1_scale
    for pl in range(1, config.xyz1_ssw_per_megapod + 1):
        ssw = f"xyz1/mp1/ssw/plane{pl}"
        for d_x in range(1, config.xyz1_xsw_per_plane + 1):
            links.append(
                {
                    "source": ssw,
                    "target": f"xyz1/xsw/plane{pl}/dev{d_x}",
                    "capacity": ssw_xsw_cap,
                    "cost": 1.0,
                    "attrs": {"link_type": "ssw_xsw", "site": "xyz1"},
                }
            )

    return links


# ---------------------------------------------------------------------------
# BB cross-site link builder
# ---------------------------------------------------------------------------


def _build_bb_cross_site_links(config: DcBbScenarioConfig) -> list[dict]:
    """Build cross-site links between ABC1-side and XYZ1-side BB devices.

    Creates a 4x4 full mesh per plane, replicated on Path_A and Path_B.
    Total: 64 planes × 16 pairs × 2 paths = 2,048 links.
    """
    links: list[dict] = []
    for pl in range(1, config.bb_planes + 1):
        pg = (pl - 1) // 4 + 1
        for a in range(1, config.bb_devices_per_plane + 1):
            for x in range(1, config.bb_devices_per_plane + 1):
                for path in ["path_a", "path_b"]:
                    links.append(
                        {
                            "source": f"bb/abc1/plane{pl}/dev{a}",
                            "target": f"bb/xyz1/plane{pl}/dev{x}",
                            "capacity": config.bb_bb_link_capacity,
                            "cost": 10,
                            "risk_groups": [
                                path,
                                f"plane_group_{pg}",
                                f"plane_{pl}_site_abc1",
                                f"plane_{pl}_site_xyz1",
                                f"pg_{pg}_idx_{a}_abc1",
                                f"pg_{pg}_idx_{x}_xyz1",
                            ],
                            "attrs": {
                                "link_type": "bb_cross_site",
                                "plane": pl,
                                "path": path,
                            },
                        }
                    )
    return links


# ---------------------------------------------------------------------------
# DC-BB interconnect link builder
# ---------------------------------------------------------------------------


def _build_dc_bb_links(config: DcBbScenarioConfig) -> list[dict]:
    """Build mesh-group-based links between DC devices and BB devices.

    Uses the mesh group algorithm to partition FADU<->BB(abc1-side) and
    XSW<->BB(xyz1-side) into groups, then creates full-mesh links within
    each group.

    Raises:
        AssertionError: If per-device degree exceeds port constraints
            (k_fadu <= 16, k_xsw <= 4).
    """
    bb_total = config.bb_planes * config.bb_devices_per_plane

    # --- Port constraint checks ---
    k_fadu = bb_total // config.g_abc1
    k_xsw = bb_total // config.g_xyz1
    assert k_fadu <= 16, (
        f"k_fadu={k_fadu} exceeds 16-port limit (G_abc1={config.g_abc1})"
    )
    assert k_xsw <= 4, f"k_xsw={k_xsw} exceeds 4-port limit (G_xyz1={config.g_xyz1})"

    links: list[dict] = []

    # --- ABC1: FADU <-> BB(abc1) ---
    abc1_groups = _compute_mesh_groups(
        dc_rows=config.abc1_hgrids,
        dc_cols=config.abc1_fadu_per_hgrid,
        bb_rows=config.bb_planes,
        bb_cols=config.bb_devices_per_plane,
        g=config.g_abc1,
        layout=config.layout_abc1,
    )
    for dc_devs, bb_devs in abc1_groups:
        for dr, dc in dc_devs:
            fadu_name = f"abc1/fadu/hgrid{dr + 1}/idx{dc + 1}"
            for br, bc in bb_devs:
                bb_plane = br + 1
                bb_dev = bc + 1
                bb_name = f"bb/abc1/plane{bb_plane}/dev{bb_dev}"
                pg = (bb_plane - 1) // 4 + 1
                links.append(
                    {
                        "source": fadu_name,
                        "target": bb_name,
                        "capacity": config.dc_bb_link_capacity,
                        "cost": 5,
                        "risk_groups": [
                            f"plane_{bb_plane}_site_abc1",
                            f"plane_group_{pg}",
                            f"pg_{pg}_idx_{bb_dev}_abc1",
                        ],
                        "attrs": {"link_type": "dc_bb", "side": "abc1"},
                    }
                )

    # --- XYZ1: XSW <-> BB(xyz1) ---
    # The XSW grid uses 64x24 orientation (rows=device, cols=plane)
    # to match GCD factorization requirements.
    xyz1_groups = _compute_mesh_groups(
        dc_rows=config.xyz1_xsw_per_plane,
        dc_cols=config.xyz1_xsw_planes,
        bb_rows=config.bb_planes,
        bb_cols=config.bb_devices_per_plane,
        g=config.g_xyz1,
        layout=config.layout_xyz1,
    )
    for dc_devs, bb_devs in xyz1_groups:
        for dr, dc in dc_devs:
            # dr = device index within plane, dc = plane index
            xsw_name = f"xyz1/xsw/plane{dc + 1}/dev{dr + 1}"
            for br, bc in bb_devs:
                bb_plane = br + 1
                bb_dev = bc + 1
                bb_name = f"bb/xyz1/plane{bb_plane}/dev{bb_dev}"
                pg = (bb_plane - 1) // 4 + 1
                links.append(
                    {
                        "source": xsw_name,
                        "target": bb_name,
                        "capacity": config.dc_bb_link_capacity,
                        "cost": 5,
                        "risk_groups": [
                            f"plane_{bb_plane}_site_xyz1",
                            f"plane_group_{pg}",
                            f"pg_{pg}_idx_{bb_dev}_xyz1",
                        ],
                        "attrs": {"link_type": "dc_bb", "side": "xyz1"},
                    }
                )

    return links


# ---------------------------------------------------------------------------
# Risk group builder
# ---------------------------------------------------------------------------


def _build_risk_groups(config: DcBbScenarioConfig) -> list[dict]:
    """Build all risk group definitions for the DC-BB topology.

    Creates four categories of risk groups:
    - 2 path groups (path_a, path_b) for long-haul path failures
    - bb_planes//4 plane groups (groups of 4 consecutive planes)
    - bb_planes * 2 plane-site groups (per plane, per site)
    - (bb_planes//4) * bb_devices_per_plane * 2 device-index-across-plane-group groups

    Default config: 2 + 16 + 128 + 128 = 274 groups.

    Args:
        config: Scenario configuration with BB plane/device counts.

    Returns:
        List of risk group dicts, each with "name" and "attrs" keys.
    """
    groups: list[dict] = []

    # Path risk groups
    groups.append({"name": "path_a", "attrs": {"type": "long_haul_path"}})
    groups.append({"name": "path_b", "attrs": {"type": "long_haul_path"}})

    # Plane groups (4 consecutive planes per group)
    for g in range(1, config.bb_planes // 4 + 1):
        groups.append(
            {
                "name": f"plane_group_{g}",
                "attrs": {
                    "type": "plane_group",
                    "planes": list(range((g - 1) * 4 + 1, g * 4 + 1)),
                },
            }
        )

    # Per-plane-site groups
    for pl in range(1, config.bb_planes + 1):
        for site in ["abc1", "xyz1"]:
            groups.append(
                {
                    "name": f"plane_{pl}_site_{site}",
                    "attrs": {"type": "plane_site", "plane": pl, "site": site},
                }
            )

    # Device-index-across-plane-group
    for g in range(1, config.bb_planes // 4 + 1):
        for d in range(1, config.bb_devices_per_plane + 1):
            for site in ["abc1", "xyz1"]:
                groups.append(
                    {
                        "name": f"pg_{g}_idx_{d}_{site}",
                        "attrs": {"type": "device_index_across_planes"},
                    }
                )

    return groups


# ---------------------------------------------------------------------------
# Demand builder
# ---------------------------------------------------------------------------


def _build_demands(config: DcBbScenarioConfig) -> dict:
    """Build demand definitions for the DC-BB topology.

    Creates a bidirectional aggregate demand between ABC1 and XYZ1 using
    ``combine`` mode. Each direction carries 100 Tbps (100,000 Gbps) as a
    reference volume; MSD scales this by alpha to find the maximum
    supportable demand.

    Args:
        config: Scenario configuration (currently unused but accepted
            for interface consistency with other builders).

    Returns:
        Dict with a single key ``baseline_traffic_matrix`` mapping to a
        list of two demand entries (one per direction).
    """
    return {
        "baseline_traffic_matrix": [
            {
                "source": "^abc1/pod.*/rsw$",
                "target": "^xyz1/mp1/rsw$",
                "volume": 100000.0,
                "mode": "combine",
                "flow_policy": "SHORTEST_PATHS_ECMP",
            },
            {
                "source": "^xyz1/mp1/rsw$",
                "target": "^abc1/pod.*/rsw$",
                "volume": 100000.0,
                "mode": "combine",
                "flow_policy": "SHORTEST_PATHS_ECMP",
            },
        ]
    }


# ---------------------------------------------------------------------------
# Failure policy builder
# ---------------------------------------------------------------------------


def _build_failure_policy(config: DcBbScenarioConfig) -> dict:
    """Build the weighted Monte Carlo failure policy for the DC-BB topology.

    Combines seven failure modes into a single policy named
    ``dc_bb_failures``:

    1. Long-haul path failure (path_a or path_b) -- weight 0.10
    2. Plane group failure (4 consecutive planes) -- weight 0.15
    3. Per-plane-site failure (all devices in one plane at one site) -- 0.15
    4. Device-index-across-plane-group failure -- weight 0.10
    5. Single random BB device failure -- weight 0.15
    6. Two random BB device failures -- weight 0.10
    7. Random BB-BB link failures (availability 0.99) -- weight 0.25

    Args:
        config: Scenario configuration (currently unused but accepted
            for interface consistency with other builders).

    Returns:
        Dict with key ``dc_bb_failures`` mapping to the policy definition.
    """
    return {
        "dc_bb_failures": {
            "attrs": {
                "description": (
                    "Monte Carlo: BB path, plane group, device, and link failures"
                ),
            },
            "modes": [
                # Mode 1: Long-haul path failure (correlated across all 64 planes)
                {
                    "weight": 0.10,
                    "rules": [
                        {
                            "scope": "risk_group",
                            "mode": "choice",
                            "count": 1,
                            "conditions": [
                                {
                                    "attr": "type",
                                    "op": "==",
                                    "value": "long_haul_path",
                                },
                            ],
                        },
                    ],
                },
                # Mode 2: Plane group failure (4 consecutive planes)
                {
                    "weight": 0.15,
                    "rules": [
                        {
                            "scope": "risk_group",
                            "mode": "choice",
                            "count": 1,
                            "conditions": [
                                {
                                    "attr": "type",
                                    "op": "==",
                                    "value": "plane_group",
                                },
                            ],
                        },
                    ],
                },
                # Mode 3: All devices of one plane at one site (4 devices)
                {
                    "weight": 0.15,
                    "rules": [
                        {
                            "scope": "risk_group",
                            "mode": "choice",
                            "count": 1,
                            "conditions": [
                                {
                                    "attr": "type",
                                    "op": "==",
                                    "value": "plane_site",
                                },
                            ],
                        },
                    ],
                },
                # Mode 4: Device index across plane group at one site
                {
                    "weight": 0.10,
                    "rules": [
                        {
                            "scope": "risk_group",
                            "mode": "choice",
                            "count": 1,
                            "conditions": [
                                {
                                    "attr": "type",
                                    "op": "==",
                                    "value": "device_index_across_planes",
                                },
                            ],
                        },
                    ],
                },
                # Mode 5: Single random BB device failure
                {
                    "weight": 0.15,
                    "rules": [
                        {
                            "scope": "node",
                            "mode": "choice",
                            "count": 1,
                            "conditions": [
                                {
                                    "attr": "role",
                                    "op": "==",
                                    "value": "bb",
                                },
                            ],
                        },
                    ],
                },
                # Mode 6: Two random BB device failures
                {
                    "weight": 0.10,
                    "rules": [
                        {
                            "scope": "node",
                            "mode": "choice",
                            "count": 2,
                            "conditions": [
                                {
                                    "attr": "role",
                                    "op": "==",
                                    "value": "bb",
                                },
                            ],
                        },
                    ],
                },
                # Mode 7: Random BB-BB link failures (availability 0.99)
                {
                    "weight": 0.25,
                    "rules": [
                        {
                            "scope": "link",
                            "mode": "random",
                            "probability": 0.01,
                            "conditions": [
                                {
                                    "attr": "link_type",
                                    "op": "==",
                                    "value": "bb_cross_site",
                                },
                            ],
                        },
                    ],
                },
            ],
        }
    }


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------


def _build_workflow(config: DcBbScenarioConfig) -> list[dict]:
    """Build the workflow steps for the DC-BB scenario.

    Creates two steps:

    1. **MaximumSupportedDemand** (``msd_baseline``): finds the maximum
       alpha multiplier the topology supports for the baseline traffic
       matrix under no-failure conditions.
    2. **TrafficMatrixPlacement** (``tm_placement``): places the baseline
       traffic matrix under the ``dc_bb_failures`` failure policy to
       compute the Bandwidth Availability Curve (BAC).

    Args:
        config: Scenario configuration with seed, resolution, and
            iteration count.

    Returns:
        List of two workflow step dicts.
    """
    return [
        {
            "type": "MaximumSupportedDemand",
            "name": "msd_baseline",
            "demands": "baseline_traffic_matrix",
            "flow_policy": "SHORTEST_PATHS_ECMP",
            "seed": config.seed,
            "resolution": config.msd_resolution,
        },
        {
            "type": "TrafficMatrixPlacement",
            "name": "tm_placement",
            "demands": "baseline_traffic_matrix",
            "flow_policy": "SHORTEST_PATHS_ECMP",
            "failure_policy": "dc_bb_failures",
            "iterations": config.failure_iterations,
            "parallelism": 8,
            "seed": config.seed,
            "metadata": {"baseline": True},
        },
    ]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def validate_config(config: DcBbScenarioConfig) -> list[str]:
    """Validate a DcBbScenarioConfig for consistency and feasibility.

    Checks:
    - g_abc1 is a viable G value for the ABC1 DC-BB interconnect.
    - g_xyz1 is a viable G value for the XYZ1 DC-BB interconnect.
    - layout_abc1 is valid for the ABC1 grid dimensions.
    - layout_xyz1 is valid for the XYZ1 grid dimensions.
    - Port constraints are satisfied (k_fadu <= 16, k_xsw <= 4).

    Args:
        config: Scenario configuration to validate.

    Returns:
        List of error strings. Empty list means the config is valid.
    """
    errors: list[str] = []

    bb_total = config.bb_planes * config.bb_devices_per_plane

    # ABC1 side: FADU grid is hgrids x fadu_per_hgrid
    abc1_dc_total = config.abc1_hgrids * config.abc1_fadu_per_hgrid
    abc1_dc_ports = 16  # FADU has 16 BB-facing ports
    abc1_bb_ports = config.abc1_fadu_per_hgrid  # BB(abc1) connects to FADUs

    viable_g_abc1 = get_viable_g_values(
        dc_total=abc1_dc_total,
        bb_total=bb_total,
        dc_ports=abc1_dc_ports,
        bb_ports=abc1_bb_ports,
    )
    if config.g_abc1 not in viable_g_abc1:
        errors.append(
            f"g_abc1={config.g_abc1} is not viable; viable values: {viable_g_abc1}"
        )

    # XYZ1 side: XSW grid is xsw_per_plane x xsw_planes
    xyz1_dc_total = config.xyz1_xsw_per_plane * config.xyz1_xsw_planes
    xyz1_dc_ports = 4  # XSW has 4 BB-facing ports
    xyz1_bb_ports = config.xyz1_xsw_planes  # BB(xyz1) connects to XSWs

    viable_g_xyz1 = get_viable_g_values(
        dc_total=xyz1_dc_total,
        bb_total=bb_total,
        dc_ports=xyz1_dc_ports,
        bb_ports=xyz1_bb_ports,
    )
    if config.g_xyz1 not in viable_g_xyz1:
        errors.append(
            f"g_xyz1={config.g_xyz1} is not viable; viable values: {viable_g_xyz1}"
        )

    # Layout validation for ABC1
    if not validate_layout(
        g=config.g_abc1,
        layout=config.layout_abc1,
        dc_rows=config.abc1_hgrids,
        dc_cols=config.abc1_fadu_per_hgrid,
        bb_rows=config.bb_planes,
        bb_cols=config.bb_devices_per_plane,
    ):
        errors.append(
            f"layout_abc1={config.layout_abc1} is not valid for "
            f"g_abc1={config.g_abc1}, "
            f"dc={config.abc1_hgrids}x{config.abc1_fadu_per_hgrid}, "
            f"bb={config.bb_planes}x{config.bb_devices_per_plane}"
        )

    # Layout validation for XYZ1
    if not validate_layout(
        g=config.g_xyz1,
        layout=config.layout_xyz1,
        dc_rows=config.xyz1_xsw_per_plane,
        dc_cols=config.xyz1_xsw_planes,
        bb_rows=config.bb_planes,
        bb_cols=config.bb_devices_per_plane,
    ):
        errors.append(
            f"layout_xyz1={config.layout_xyz1} is not valid for "
            f"g_xyz1={config.g_xyz1}, "
            f"dc={config.xyz1_xsw_per_plane}x{config.xyz1_xsw_planes}, "
            f"bb={config.bb_planes}x{config.bb_devices_per_plane}"
        )

    # Port constraint checks
    k_fadu = bb_total // config.g_abc1 if config.g_abc1 > 0 else bb_total
    if k_fadu > 16:
        errors.append(
            f"Port constraint violated: k_fadu={k_fadu} > 16 (G_abc1={config.g_abc1})"
        )

    k_xsw = bb_total // config.g_xyz1 if config.g_xyz1 > 0 else bb_total
    if k_xsw > 4:
        errors.append(
            f"Port constraint violated: k_xsw={k_xsw} > 4 (G_xyz1={config.g_xyz1})"
        )

    return errors


# ---------------------------------------------------------------------------
# End-to-end scenario generation
# ---------------------------------------------------------------------------


def _fix_failure_rules(failure_policy: dict) -> dict:
    """Post-process failure policy to nest conditions inside match blocks.

    The _build_failure_policy builder puts ``conditions`` directly on the
    rule dict. The ngraph parser expects them inside a ``match`` wrapper.
    This function moves ``conditions`` into ``match.conditions`` for every
    rule in every mode.

    Args:
        failure_policy: Dict from _build_failure_policy (policy_name -> def).

    Returns:
        Fixed failure policy dict (modified in place and returned).
    """
    for _policy_name, policy_def in failure_policy.items():
        for mode in policy_def.get("modes", []):
            for rule in mode.get("rules", []):
                if "conditions" in rule and "match" not in rule:
                    rule["match"] = {"conditions": rule.pop("conditions")}
    return failure_policy


def _fix_workflow_steps(workflow: list[dict]) -> list[dict]:
    """Post-process workflow steps for compatibility with ngraph step constructors.

    Fixes applied:
    - Renames ``demands`` to ``demand_set`` (builder uses wrong key name).
    - Removes ``flow_policy`` from MaximumSupportedDemand steps (not a valid
      parameter for the MSD step constructor).
    - Removes ``metadata`` from TrafficMatrixPlacement steps (not a valid
      parameter for the TMP step constructor).

    Args:
        workflow: List of step dicts from _build_workflow.

    Returns:
        Fixed workflow list (modified in place and returned).
    """
    # Keys that are not accepted by the respective step constructors
    _MSD_INVALID_KEYS = {"flow_policy", "metadata"}
    _TMP_INVALID_KEYS = {"metadata", "flow_policy"}

    for step in workflow:
        if "demands" in step and "demand_set" not in step:
            step["demand_set"] = step.pop("demands")

        step_type = step.get("type", "")
        if step_type == "MaximumSupportedDemand":
            for key in _MSD_INVALID_KEYS:
                step.pop(key, None)
        elif step_type == "TrafficMatrixPlacement":
            for key in _TMP_INVALID_KEYS:
                step.pop(key, None)

    return workflow


def generate_scenario(config: DcBbScenarioConfig) -> dict:
    """Generate a complete NetGraph scenario dict from a DcBbScenarioConfig.

    Assembles all builder outputs into a single dict that can be serialized
    to YAML and consumed by ``ngraph inspect`` or ``ngraph run``.

    Args:
        config: Validated scenario configuration.

    Returns:
        Complete scenario dict with keys: seed, network, risk_groups,
        demands, failures, workflow.

    Raises:
        ValueError: If the config fails validation.
    """
    errors = validate_config(config)
    if errors:
        raise ValueError("Invalid config: " + "; ".join(errors))

    nodes = _build_nodes(config)
    links = (
        _build_internal_dc_links(config)
        + _build_bb_cross_site_links(config)
        + _build_dc_bb_links(config)
    )
    risk_groups = _build_risk_groups(config)
    demands = _build_demands(config)
    failure_policy = _fix_failure_rules(_build_failure_policy(config))
    workflow = _fix_workflow_steps(_build_workflow(config))

    return {
        "seed": config.seed,
        "network": {
            "nodes": nodes,
            "links": links,
        },
        "risk_groups": risk_groups,
        "demands": demands,
        "failures": failure_policy,
        "workflow": workflow,
    }
