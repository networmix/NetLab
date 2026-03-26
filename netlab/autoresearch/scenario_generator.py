"""DC-BB scenario generation using ngraph DSL patterns.

Generates DSL-idiomatic YAML that ngraph expands correctly:
- Internal Clos links via expand + mesh patterns (~6 definitions for 35K links)
- DC-BB mesh group links via expand + mesh per group
- Risk groups assigned via link_rules post-creation
- Post-expansion validation against expected counts

Preserves from original: mesh group algorithm, config/validation,
risk group definitions, failure policy, demands, workflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from netlab.autoresearch.scenario_validation import (
    ExpectedCounts,
    compute_expected_counts,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Mesh group algorithm (preserved, proven by 27 tests)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# G-value and layout utilities (preserved)
# ---------------------------------------------------------------------------


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
# Config validation (preserved)
# ---------------------------------------------------------------------------


def validate_config(config: DcBbScenarioConfig) -> list[str]:
    """Validate a DcBbScenarioConfig for consistency and feasibility."""
    errors: list[str] = []
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


# ---------------------------------------------------------------------------
# Node builder (DSL-idiomatic: brackets for Clos, explicit for mesh groups)
# ---------------------------------------------------------------------------


def _build_nodes(config: DcBbScenarioConfig) -> dict[str, dict]:
    """Build network.nodes dict.

    Uses '/' separators as word boundaries (pl1/ cannot match pl10/).
    Internal Clos nodes use bracket expansion. Mesh-group-dependent
    nodes (FADU, XSW, BB) created explicitly with mg{G} in path.
    """
    nodes: dict[str, dict] = {}

    pods = config.abc1_pods_per_building
    planes = config.abc1_planes
    ssw_pp = config.abc1_ssw_per_plane
    hgrids = config.abc1_hgrids
    xsw_pl = config.xyz1_xsw_planes
    xsw_pp = config.xyz1_xsw_per_plane
    fsw_rows = 4
    fsw_devs = config.xyz1_fsw_per_megapod // fsw_rows

    # ABC1 Internal Clos
    nodes[f"abc1/pod[1-{pods}]/rsw"] = {"attrs": {"role": "rsw", "site": "abc1"}}
    nodes[f"abc1/pod[1-{pods}]/fsw/pl[1-{planes}]"] = {
        "attrs": {"role": "fsw", "site": "abc1"}
    }
    nodes[f"abc1/ssw/pl[1-{planes}]/ix[1-{ssw_pp}]"] = {
        "attrs": {"role": "ssw", "site": "abc1"}
    }

    # ABC1 FADU + BB (explicit, mesh group in path)
    abc1_groups = _compute_mesh_groups(
        hgrids,
        config.abc1_fadu_per_hgrid,
        config.bb_planes,
        config.bb_devices_per_plane,
        config.g_abc1,
        config.layout_abc1,
    )
    for gid, (dc_devs, bb_devs) in enumerate(abc1_groups):
        for r, c in dc_devs:
            nodes[f"abc1/fadu/mg{gid}/hg{r + 1}/ix{c + 1}"] = {
                "attrs": {
                    "role": "fadu",
                    "site": "abc1",
                    "hgrid": r + 1,
                    "index": c + 1,
                }
            }
        for r, c in bb_devs:
            name = f"bb/abc1/mg{gid}/pl{r + 1}/dv{c + 1}"
            if name not in nodes:
                nodes[name] = {
                    "attrs": {
                        "role": "bb",
                        "site": "abc1",
                        "plane": r + 1,
                        "device": c + 1,
                    }
                }

    # XYZ1 Internal Clos
    nodes["xyz1/mp1/rsw"] = {"attrs": {"role": "rsw", "site": "xyz1"}}
    nodes[f"xyz1/mp1/fsw/rw[1-{fsw_rows}]/dv[1-{fsw_devs}]"] = {
        "attrs": {"role": "fsw", "site": "xyz1"}
    }
    nodes[f"xyz1/mp1/ssw/pl[1-{xsw_pl}]"] = {"attrs": {"role": "ssw", "site": "xyz1"}}

    # XYZ1 XSW + BB (explicit, mesh group in path)
    xyz1_groups = _compute_mesh_groups(
        xsw_pp,
        xsw_pl,
        config.bb_planes,
        config.bb_devices_per_plane,
        config.g_xyz1,
        config.layout_xyz1,
    )
    for gid, (dc_devs, bb_devs) in enumerate(xyz1_groups):
        for r, c in dc_devs:
            nodes[f"xyz1/xsw/mg{gid}/pl{c + 1}/dv{r + 1}"] = {
                "attrs": {
                    "role": "xsw",
                    "site": "xyz1",
                    "plane": c + 1,
                    "device": r + 1,
                }
            }
        for r, c in bb_devs:
            name = f"bb/xyz1/mg{gid}/pl{r + 1}/dv{c + 1}"
            if name not in nodes:
                nodes[name] = {
                    "attrs": {
                        "role": "bb",
                        "site": "xyz1",
                        "plane": r + 1,
                        "device": c + 1,
                    }
                }

    return nodes


# ---------------------------------------------------------------------------
# Internal Clos links (DSL expand + mesh patterns)
# ---------------------------------------------------------------------------


def _build_internal_links(config: DcBbScenarioConfig) -> list[dict]:
    """Build internal Clos links using DSL expand + mesh.

    6 link definitions expand to 35,360 links.
    """
    links: list[dict] = []
    abc1_scale = config.abc1_buildings
    xyz1_scale = config.xyz1_megapods

    pod_list = list(range(1, config.abc1_pods_per_building + 1))
    plane_list = list(range(1, config.abc1_planes + 1))
    ix_list = list(range(1, config.abc1_ssw_per_plane + 1))
    xpl_list = list(range(1, config.xyz1_xsw_planes + 1))

    # ABC1: RSW→FSW
    links.append(
        {
            "source": "abc1/pod${p}/rsw$",
            "target": "abc1/pod${p}/fsw/",
            "expand": {"vars": {"p": pod_list}, "mode": "cartesian"},
            "pattern": "mesh",
            "capacity": config.abc1_rsw_per_pod * 200.0 * abc1_scale,
            "cost": 1.0,
            "attrs": {"link_type": "rsw_fsw", "site": "abc1"},
        }
    )

    # ABC1: FSW→SSW
    links.append(
        {
            "source": "abc1/pod${p}/fsw/pl${q}$",
            "target": "abc1/ssw/pl${q}/",
            "expand": {"vars": {"p": pod_list, "q": plane_list}, "mode": "cartesian"},
            "pattern": "mesh",
            "capacity": 200.0 * abc1_scale,
            "cost": 1.0,
            "attrs": {"link_type": "fsw_ssw", "site": "abc1"},
        }
    )

    # ABC1: SSW→FADU (index-matched: /ix{I}$ ensures exact match)
    links.append(
        {
            "source": "abc1/ssw/pl${q}/ix${i}$",
            "target": "abc1/fadu/.*/ix${i}$",
            "expand": {"vars": {"q": plane_list, "i": ix_list}, "mode": "cartesian"},
            "pattern": "mesh",
            "capacity": 2 * 200.0 * abc1_scale,
            "cost": 1.0,
            "attrs": {"link_type": "ssw_fadu", "site": "abc1"},
        }
    )

    # XYZ1: FSW→RSW
    links.append(
        {
            "source": "xyz1/mp1/fsw/",
            "target": "xyz1/mp1/rsw$",
            "pattern": "mesh",
            "capacity": 400.0 * xyz1_scale,
            "cost": 1.0,
            "attrs": {"link_type": "rsw_fsw", "site": "xyz1"},
        }
    )

    # XYZ1: SSW→FSW
    links.append(
        {
            "source": "xyz1/mp1/ssw/",
            "target": "xyz1/mp1/fsw/",
            "pattern": "mesh",
            "capacity": 400.0 * xyz1_scale,
            "cost": 1.0,
            "attrs": {"link_type": "fsw_ssw", "site": "xyz1"},
        }
    )

    # XYZ1: XSW→SSW (plane-matched: /pl{Q}/ word boundary)
    links.append(
        {
            "source": "xyz1/xsw/.*/pl${q}/",
            "target": "xyz1/mp1/ssw/pl${q}$",
            "expand": {"vars": {"q": xpl_list}, "mode": "cartesian"},
            "pattern": "mesh",
            "capacity": 400.0 * xyz1_scale,
            "cost": 1.0,
            "attrs": {"link_type": "ssw_xsw", "site": "xyz1"},
        }
    )

    return links


# ---------------------------------------------------------------------------
# DC-BB mesh group links (DSL expand + mesh per group)
# ---------------------------------------------------------------------------


def _build_dc_bb_links(config: DcBbScenarioConfig) -> list[dict]:
    """Build DC-BB links: one expand+mesh definition per mesh group."""
    links: list[dict] = []

    # ABC1
    links.append(
        {
            "source": "abc1/fadu/mg${g}/",
            "target": "bb/abc1/mg${g}/",
            "expand": {"vars": {"g": list(range(config.g_abc1))}, "mode": "cartesian"},
            "pattern": "mesh",
            "capacity": config.dc_bb_link_capacity,
            "cost": 5,
            "attrs": {"link_type": "dc_bb", "side": "abc1"},
        }
    )

    # XYZ1
    links.append(
        {
            "source": "xyz1/xsw/mg${g}/",
            "target": "bb/xyz1/mg${g}/",
            "expand": {"vars": {"g": list(range(config.g_xyz1))}, "mode": "cartesian"},
            "pattern": "mesh",
            "capacity": config.dc_bb_link_capacity,
            "cost": 5,
            "attrs": {"link_type": "dc_bb", "side": "xyz1"},
        }
    )

    return links


# ---------------------------------------------------------------------------
# BB cross-site links (dual paths, per-plane mesh)
# ---------------------------------------------------------------------------


def _build_bb_cross_site_links(config: DcBbScenarioConfig) -> list[dict]:
    """Build BB cross-site links: per-plane mesh, dual paths via attrs tag."""
    links: list[dict] = []
    pl_list = list(range(1, config.bb_planes + 1))

    for path_label in ["a", "b"]:
        links.append(
            {
                "source": "bb/abc1/.*/pl${p}/",
                "target": "bb/xyz1/.*/pl${p}/",
                "expand": {"vars": {"p": pl_list}, "mode": "cartesian"},
                "pattern": "mesh",
                "capacity": config.bb_bb_link_capacity,
                "cost": 10,
                "attrs": {"link_type": "bb_cross_site", "path": path_label},
            }
        )

    return links


# ---------------------------------------------------------------------------
# Link rules for risk group assignment
# ---------------------------------------------------------------------------


def _build_link_rules(config: DcBbScenarioConfig) -> list[dict]:
    """Build link_rules for risk group assignment on DC-BB and cross-site links."""
    rules: list[dict] = []
    ppg = 4

    # DC-BB risk groups
    for side in ["abc1", "xyz1"]:
        dc_prefix = "abc1/fadu/" if side == "abc1" else "xyz1/xsw/"
        for pl in range(1, config.bb_planes + 1):
            pg = (pl - 1) // ppg + 1
            for dv in range(1, config.bb_devices_per_plane + 1):
                rules.append(
                    {
                        "source": dc_prefix,
                        "target": f"bb/{side}/.*/pl{pl}/dv{dv}$",
                        "risk_groups": [
                            f"plane_{pl}_site_{side}",
                            f"plane_group_{pg}",
                            f"pg_{pg}_idx_{dv}_{side}",
                        ],
                    }
                )

    # BB cross-site risk groups
    for pl in range(1, config.bb_planes + 1):
        pg = (pl - 1) // ppg + 1
        for da in range(1, config.bb_devices_per_plane + 1):
            for dx in range(1, config.bb_devices_per_plane + 1):
                for path_label in ["a", "b"]:
                    rules.append(
                        {
                            "source": f"bb/abc1/.*/pl{pl}/dv{da}$",
                            "target": f"bb/xyz1/.*/pl{pl}/dv{dx}$",
                            "link_match": {
                                "conditions": [
                                    {"attr": "path", "op": "==", "value": path_label}
                                ],
                            },
                            "risk_groups": [
                                f"path_{path_label}",
                                f"plane_{pl}_site_abc1",
                                f"plane_{pl}_site_xyz1",
                                f"plane_group_{pg}",
                                f"pg_{pg}_idx_{da}_abc1",
                                f"pg_{pg}_idx_{dx}_xyz1",
                            ],
                        }
                    )

    return rules


# ---------------------------------------------------------------------------
# Risk groups (preserved)
# ---------------------------------------------------------------------------


def _build_risk_groups(config: DcBbScenarioConfig) -> list[dict]:
    """Build all risk group definitions (274 total)."""
    groups: list[dict] = []

    groups.append({"name": "path_a", "attrs": {"type": "long_haul_path"}})
    groups.append({"name": "path_b", "attrs": {"type": "long_haul_path"}})

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

    for pl in range(1, config.bb_planes + 1):
        for site in ["abc1", "xyz1"]:
            groups.append(
                {
                    "name": f"plane_{pl}_site_{site}",
                    "attrs": {"type": "plane_site", "plane": pl, "site": site},
                }
            )

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
# Failure policy (preserved)
# ---------------------------------------------------------------------------


def _fix_failure_rules(failure_policy: dict) -> dict:
    """Nest conditions inside match blocks for ngraph compatibility."""
    for _policy_name, policy_def in failure_policy.items():
        for mode in policy_def.get("modes", []):
            for rule in mode.get("rules", []):
                if "conditions" in rule and "match" not in rule:
                    rule["match"] = {"conditions": rule.pop("conditions")}
    return failure_policy


# Condition shorthands
def _RG(typ: str) -> list[dict[str, str]]:
    return [{"attr": "type", "op": "==", "value": typ}]


_BB = [{"attr": "role", "op": "==", "value": "bb"}]
_DCBB = [{"attr": "link_type", "op": "==", "value": "dc_bb"}]
_XSITE = [{"attr": "link_type", "op": "==", "value": "bb_cross_site"}]

# Failure modes: (name, rule_dict)
# Three categories:
#   1. Correlated (risk-group-based) — shared infrastructure events
#   2. Deterministic fixed-count — exact N devices/groups, stress tests
#   3. Availability-based — independent per-entity probability
_FAILURE_MODES = [
    # --- Correlated single failures ---
    (
        "lh_path",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 1,
            "conditions": _RG("long_haul_path"),
        },
    ),
    (
        "plane_group",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 1,
            "conditions": _RG("plane_group"),
        },
    ),
    (
        "plane_site",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 1,
            "conditions": _RG("plane_site"),
        },
    ),
    (
        "dev_index",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 1,
            "conditions": _RG("device_index_across_planes"),
        },
    ),
    # --- Correlated scaled failures (stress test) ---
    (
        "2x_plane_site",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 2,
            "conditions": _RG("plane_site"),
        },
    ),
    (
        "4x_plane_site",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 4,
            "conditions": _RG("plane_site"),
        },
    ),
    (
        "2x_plane_group",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 2,
            "conditions": _RG("plane_group"),
        },
    ),
    (
        "2x_dev_index",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 2,
            "conditions": _RG("device_index_across_planes"),
        },
    ),
    # --- Deterministic fixed-count BB device failures ---
    ("1x_bb", {"scope": "node", "mode": "choice", "count": 1, "conditions": _BB}),
    ("2x_bb", {"scope": "node", "mode": "choice", "count": 2, "conditions": _BB}),
    ("4x_bb", {"scope": "node", "mode": "choice", "count": 4, "conditions": _BB}),
    ("8x_bb", {"scope": "node", "mode": "choice", "count": 8, "conditions": _BB}),
    # --- Availability-based (independent per-entity) ---
    (
        "bb_avail_2pct",
        {"scope": "node", "mode": "random", "probability": 0.02, "conditions": _BB},
    ),
    (
        "bb_avail_5pct",
        {"scope": "node", "mode": "random", "probability": 0.05, "conditions": _BB},
    ),
    (
        "bb_avail_10pct",
        {"scope": "node", "mode": "random", "probability": 0.10, "conditions": _BB},
    ),
    (
        "dcbb_avail",
        {"scope": "link", "mode": "random", "probability": 0.01, "conditions": _DCBB},
    ),
    (
        "xsite_avail",
        {"scope": "link", "mode": "random", "probability": 0.01, "conditions": _XSITE},
    ),
]

FAILURE_MODE_NAMES = [name for name, _ in _FAILURE_MODES]


def _build_failure_policy(config: DcBbScenarioConfig) -> dict:
    """Build failure policies: one per mode + one combined (equal weight).

    Each single-mode policy runs that failure type exclusively.
    The combined policy samples all modes with equal probability.
    """
    policies: dict = {}

    for name, rule in _FAILURE_MODES:
        policies[f"fm_{name}"] = {
            "modes": [{"weight": 1.0, "rules": [dict(rule)]}],
        }

    w = 1.0 / len(_FAILURE_MODES)
    policies["fm_combined"] = {
        "attrs": {
            "description": f"All {len(_FAILURE_MODES)} failure modes, equal weight"
        },
        "modes": [{"weight": w, "rules": [dict(rule)]} for _, rule in _FAILURE_MODES],
    }

    return policies


# ---------------------------------------------------------------------------
# Demands (with $ anchors)
# ---------------------------------------------------------------------------


def _build_demands(config: DcBbScenarioConfig) -> dict:
    """Build bidirectional combine-mode demands between ABC1 and XYZ1 RSWs."""
    volume = 100_000.0
    return {
        "baseline_traffic_matrix": [
            {
                "source": "^abc1/pod.*/rsw$",
                "target": "^xyz1/mp1/rsw$",
                "volume": volume,
                "mode": "combine",
                "flow_policy": "SHORTEST_PATHS_ECMP",
            },
            {
                "source": "^xyz1/mp1/rsw$",
                "target": "^abc1/pod.*/rsw$",
                "volume": volume,
                "mode": "combine",
                "flow_policy": "SHORTEST_PATHS_ECMP",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Workflow (preserved)
# ---------------------------------------------------------------------------


def _build_workflow(config: DcBbScenarioConfig) -> list[dict]:
    """Build MSD + per-mode TMP steps + combined TMP.

    Creates 9 workflow steps:
      1. msd_baseline — find alpha_star
      2-8. tm_{mode} — BAC under each failure mode independently
      9. tm_combined — BAC under all modes with equal weight
    """
    steps: list[dict] = [
        {
            "type": "MaximumSupportedDemand",
            "name": "msd_baseline",
            "demand_set": "baseline_traffic_matrix",
            "seed": config.seed,
            "resolution": config.msd_resolution,
        },
    ]

    # One TMP per single-mode policy
    for name in FAILURE_MODE_NAMES:
        steps.append(
            {
                "type": "TrafficMatrixPlacement",
                "name": f"tm_{name}",
                "demand_set": "baseline_traffic_matrix",
                "failure_policy": f"fm_{name}",
                "iterations": config.failure_iterations,
                "parallelism": 8,
                "seed": config.seed,
                "alpha_from_step": "msd_baseline",
                "alpha_from_field": "data.alpha_star",
            }
        )

    # Combined TMP
    steps.append(
        {
            "type": "TrafficMatrixPlacement",
            "name": "tm_combined",
            "demand_set": "baseline_traffic_matrix",
            "failure_policy": "fm_combined",
            "iterations": config.failure_iterations,
            "parallelism": 8,
            "seed": config.seed,
            "alpha_from_step": "msd_baseline",
            "alpha_from_field": "data.alpha_star",
        }
    )

    return steps


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_scenario(config: DcBbScenarioConfig) -> dict:
    """Generate a complete ngraph scenario dict.

    Uses DSL-idiomatic patterns: expand + mesh for links,
    link_rules for risk group assignment.

    Args:
        config: Validated scenario configuration.

    Returns:
        Complete scenario dict ready for yaml.dump + ngraph run.

    Raises:
        ValueError: If config fails validation.
    """
    errors = validate_config(config)
    if errors:
        raise ValueError("Invalid config: " + "; ".join(errors))

    return {
        "seed": config.seed,
        "network": {
            "nodes": _build_nodes(config),
            "links": (
                _build_internal_links(config)
                + _build_dc_bb_links(config)
                + _build_bb_cross_site_links(config)
            ),
            "link_rules": _build_link_rules(config),
        },
        "risk_groups": _build_risk_groups(config),
        "demands": _build_demands(config),
        "failures": _fix_failure_rules(_build_failure_policy(config)),
        "workflow": _build_workflow(config),
    }


def generate_scenario_with_validation(
    config: DcBbScenarioConfig,
) -> tuple[dict, ExpectedCounts]:
    """Generate scenario and return expected counts for validation.

    Use this when you need to validate the expanded graph.
    """
    scenario = generate_scenario(config)
    expected = compute_expected_counts(
        abc1_pods=config.abc1_pods_per_building,
        abc1_planes=config.abc1_planes,
        abc1_ssw_per_plane=config.abc1_ssw_per_plane,
        abc1_hgrids=config.abc1_hgrids,
        abc1_fadu_per_hgrid=config.abc1_fadu_per_hgrid,
        xyz1_xsw_per_plane=config.xyz1_xsw_per_plane,
        xyz1_xsw_planes=config.xyz1_xsw_planes,
        xyz1_ssw_per_megapod=config.xyz1_ssw_per_megapod,
        xyz1_fsw_per_megapod=config.xyz1_fsw_per_megapod,
        bb_planes=config.bb_planes,
        bb_devices_per_plane=config.bb_devices_per_plane,
        g_abc1=config.g_abc1,
        g_xyz1=config.g_xyz1,
    )
    return scenario, expected
