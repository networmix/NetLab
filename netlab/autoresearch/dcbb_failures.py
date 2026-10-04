"""DC-BB failure domains, membership and workflow policies."""

from __future__ import annotations

from .dcbb_config import DcBbScenarioConfig


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


def _build_risk_groups(config: DcBbScenarioConfig) -> list[dict]:
    """Define long-haul path, plane, plane-group, and device-index risk groups."""
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


# Condition shorthands
def _RG(typ: str) -> list[dict[str, str]]:
    return [{"attr": "type", "op": "==", "value": typ}]


_BB = [{"attr": "role", "op": "==", "value": "bb"}]
_DCBB = [{"attr": "link_type", "op": "==", "value": "dc_bb"}]
_XSITE = [{"attr": "link_type", "op": "==", "value": "bb_cross_site"}]

# Failure modes: (name, rule_dict)
# Three categories:
#   1. Correlated (risk-group-based) — shared infrastructure events
#   2. Fixed-count — choose N devices/groups per iteration
#   3. Availability-based — independent per-entity probability
_FAILURE_MODES = [
    (
        "lh_path",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 1,
            "match": {"conditions": _RG("long_haul_path")},
        },
    ),
    (
        "plane_group",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 1,
            "match": {"conditions": _RG("plane_group")},
        },
    ),
    (
        "plane_site",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 1,
            "match": {"conditions": _RG("plane_site")},
        },
    ),
    (
        "dev_index",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 1,
            "match": {"conditions": _RG("device_index_across_planes")},
        },
    ),
    (
        "2x_plane_site",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 2,
            "match": {"conditions": _RG("plane_site")},
        },
    ),
    (
        "4x_plane_site",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 4,
            "match": {"conditions": _RG("plane_site")},
        },
    ),
    (
        "2x_plane_group",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 2,
            "match": {"conditions": _RG("plane_group")},
        },
    ),
    (
        "2x_dev_index",
        {
            "scope": "risk_group",
            "mode": "choice",
            "count": 2,
            "match": {"conditions": _RG("device_index_across_planes")},
        },
    ),
    (
        "1x_bb",
        {"scope": "node", "mode": "choice", "count": 1, "match": {"conditions": _BB}},
    ),
    (
        "2x_bb",
        {"scope": "node", "mode": "choice", "count": 2, "match": {"conditions": _BB}},
    ),
    (
        "4x_bb",
        {"scope": "node", "mode": "choice", "count": 4, "match": {"conditions": _BB}},
    ),
    (
        "8x_bb",
        {"scope": "node", "mode": "choice", "count": 8, "match": {"conditions": _BB}},
    ),
    (
        "bb_avail_2pct",
        {
            "scope": "node",
            "mode": "random",
            "probability": 0.02,
            "match": {"conditions": _BB},
        },
    ),
    (
        "bb_avail_5pct",
        {
            "scope": "node",
            "mode": "random",
            "probability": 0.05,
            "match": {"conditions": _BB},
        },
    ),
    (
        "bb_avail_10pct",
        {
            "scope": "node",
            "mode": "random",
            "probability": 0.10,
            "match": {"conditions": _BB},
        },
    ),
    (
        "dcbb_avail",
        {
            "scope": "link",
            "mode": "random",
            "probability": 0.01,
            "match": {"conditions": _DCBB},
        },
    ),
    (
        "xsite_avail",
        {
            "scope": "link",
            "mode": "random",
            "probability": 0.01,
            "match": {"conditions": _XSITE},
        },
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
