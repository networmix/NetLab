"""DC-BB dimensions, mesh layouts and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


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

    # DC-BB interconnect parameters (what autoresearch varies)
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
        gi = group_id // gc_dc
        gj = group_id % gc_dc
        dc_devs: list[tuple[int, int]] = []
        for r in range(gi * dc_block_rows, (gi + 1) * dc_block_rows):
            for c in range(gj * dc_block_cols, (gj + 1) * dc_block_cols):
                dc_devs.append((r, c))

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
    """Return sorted list of viable G values given device counts and port limits."""
    g_common = math.gcd(dc_total, bb_total)
    viable: list[int] = []
    for candidate in _divisors(g_common):
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
    """Return all valid (gr_dc, gc_dc, gr_bb, gc_bb) factorizations for G."""
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
    """Check if a layout is valid for the given dimensions."""
    if len(layout) != 4 or any(type(v) is not int or v < 1 for v in layout):
        return False
    gr_dc, gc_dc, gr_bb, gc_bb = layout
    return (
        gr_dc * gc_dc == g
        and gr_bb * gc_bb == g
        and dc_rows % gr_dc == 0
        and dc_cols % gc_dc == 0
        and bb_rows % gr_bb == 0
        and bb_cols % gc_bb == 0
    )


def validate_config(config: DcBbScenarioConfig) -> list[str]:
    """Validate a DcBbScenarioConfig for consistency and feasibility."""
    errors: list[str] = []
    for field in fields(config):
        value = getattr(config, field.name)
        if (
            field.type == "int"
            and field.name not in {"seed", "failure_iterations"}
            and (type(value) is not int or value < 1)
        ):
            errors.append(f"{field.name} must be a positive integer")
        if field.type == "float" and (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            errors.append(f"{field.name} must be positive and finite")
    if type(config.seed) is not int:
        errors.append("seed must be an integer")
    if type(config.failure_iterations) is not int or config.failure_iterations < 0:
        errors.append("failure_iterations must be a nonnegative integer")
    if errors:
        return errors
    if config.xyz1_ssw_per_megapod != config.xyz1_xsw_planes:
        errors.append("xyz1_ssw_per_megapod must equal xyz1_xsw_planes")
    if config.xyz1_fsw_per_megapod % 4:
        errors.append("xyz1_fsw_per_megapod must be divisible by four rows")
    if config.bb_planes % 4:
        errors.append("bb_planes must be divisible by four planes per failure group")
    bb_total = config.bb_planes * config.bb_devices_per_plane

    abc1_dc_total = config.abc1_hgrids * config.abc1_fadu_per_hgrid
    viable_g_abc1 = get_viable_g_values(
        abc1_dc_total, bb_total, 16, config.abc1_fadu_per_hgrid
    )
    if config.g_abc1 not in viable_g_abc1:
        errors.append(
            f"g_abc1={config.g_abc1} is not viable; viable values: {viable_g_abc1}"
        )

    xyz1_dc_total = config.xyz1_xsw_per_plane * config.xyz1_xsw_planes
    viable_g_xyz1 = get_viable_g_values(
        xyz1_dc_total, bb_total, 4, config.xyz1_xsw_planes
    )
    if config.g_xyz1 not in viable_g_xyz1:
        errors.append(
            f"g_xyz1={config.g_xyz1} is not viable; viable values: {viable_g_xyz1}"
        )

    if not validate_layout(
        config.g_abc1,
        config.layout_abc1,
        config.abc1_hgrids,
        config.abc1_fadu_per_hgrid,
        config.bb_planes,
        config.bb_devices_per_plane,
    ):
        errors.append(
            f"layout_abc1={config.layout_abc1} is not valid for g_abc1={config.g_abc1}, "
            f"dc={config.abc1_hgrids}x{config.abc1_fadu_per_hgrid}, "
            f"bb={config.bb_planes}x{config.bb_devices_per_plane}"
        )

    if not validate_layout(
        config.g_xyz1,
        config.layout_xyz1,
        config.xyz1_xsw_per_plane,
        config.xyz1_xsw_planes,
        config.bb_planes,
        config.bb_devices_per_plane,
    ):
        errors.append(
            f"layout_xyz1={config.layout_xyz1} is not valid for g_xyz1={config.g_xyz1}, "
            f"dc={config.xyz1_xsw_per_plane}x{config.xyz1_xsw_planes}, "
            f"bb={config.bb_planes}x{config.bb_devices_per_plane}"
        )

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
