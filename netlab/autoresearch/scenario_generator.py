"""Build DC-BB scenario dictionaries from grid, mesh, and failure settings."""

from __future__ import annotations

from netlab.autoresearch.scenario_validation import (
    ExpectedCounts,
    compute_expected_counts,
)

from .dcbb_config import DcBbScenarioConfig, _compute_mesh_groups, validate_config
from .dcbb_failures import (
    FAILURE_MODE_NAMES,
    _build_failure_policy,
    _build_link_rules,
    _build_risk_groups,
)


def _build_nodes(config: DcBbScenarioConfig) -> dict[str, dict]:
    """Build node definitions with mesh groups encoded in device paths.

    Slashes prevent plane-prefix collisions (``pl1/`` cannot match ``pl10/``).
    Clos nodes use bracket expansion; FADU, XSW, and BB nodes are explicit.
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


def _build_internal_links(config: DcBbScenarioConfig) -> list[dict]:
    """Build internal Clos links with expansion and mesh patterns."""
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


def _build_workflow(config: DcBbScenarioConfig) -> list[dict]:
    """Build MSD, one placement step per failure mode, and a combined placement step."""
    steps: list[dict] = [
        {
            "type": "MaximumSupportedDemand",
            "name": "msd_baseline",
            "demand_set": "baseline_traffic_matrix",
            "seed": config.seed,
            "resolution": config.msd_resolution,
        },
    ]

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


def generate_scenario(config: DcBbScenarioConfig) -> dict:
    """Validate the configuration and generate a NetGraph scenario dictionary.

    Raises ValueError if the configuration is inconsistent.
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
        "failures": _build_failure_policy(config),
        "workflow": _build_workflow(config),
    }


def generate_scenario_with_validation(
    config: DcBbScenarioConfig,
) -> tuple[dict, ExpectedCounts]:
    """Return a scenario and the expected counts for its expanded network."""
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


def generate_from_parameters(**params) -> dict:
    """DC-BB research adapter: decode the documented grid-factorization strings."""
    for name in ("layout_abc1", "layout_xyz1"):
        if name in params and isinstance(params[name], str):
            params[name] = tuple(
                int(part) for part in params[name].replace("x", "_").split("_")
            )
    return generate_scenario(DcBbScenarioConfig(**params))
