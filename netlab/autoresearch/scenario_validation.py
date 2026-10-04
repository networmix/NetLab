"""Check expanded DC-BB node counts, link counts, and layer membership."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ngraph.model.network import Network
from ngraph.scenario import Scenario


@dataclass
class ExpectedCounts:
    """Expected node and link counts after DSL expansion."""

    nodes: int = 0
    links: int = 0

    # Per-layer link breakdown
    links_rsw_fsw_abc1: int = 0
    links_fsw_ssw_abc1: int = 0
    links_ssw_fadu: int = 0
    links_dc_bb_abc1: int = 0
    links_bb_cross_site: int = 0
    links_dc_bb_xyz1: int = 0
    links_ssw_xsw: int = 0
    links_fsw_ssw_xyz1: int = 0
    links_rsw_fsw_xyz1: int = 0

    @property
    def total_links(self) -> int:
        return (
            self.links_rsw_fsw_abc1
            + self.links_fsw_ssw_abc1
            + self.links_ssw_fadu
            + self.links_dc_bb_abc1
            + self.links_bb_cross_site
            + self.links_dc_bb_xyz1
            + self.links_ssw_xsw
            + self.links_fsw_ssw_xyz1
            + self.links_rsw_fsw_xyz1
        )


def compute_expected_counts(
    *,
    abc1_pods: int = 96,
    abc1_planes: int = 8,
    abc1_ssw_per_plane: int = 36,
    abc1_hgrids: int = 16,
    abc1_fadu_per_hgrid: int = 36,
    xyz1_xsw_per_plane: int = 64,
    xyz1_xsw_planes: int = 24,
    xyz1_ssw_per_megapod: int = 24,
    xyz1_fsw_per_megapod: int = 32,
    bb_planes: int = 64,
    bb_devices_per_plane: int = 4,
    g_abc1: int = 64,
    g_xyz1: int = 64,
) -> ExpectedCounts:
    """Calculate expected node and link counts from the grid dimensions."""
    # Nodes
    abc1_rsw = abc1_pods
    abc1_fsw = abc1_pods * abc1_planes
    abc1_ssw = abc1_planes * abc1_ssw_per_plane
    abc1_fadu = abc1_hgrids * abc1_fadu_per_hgrid
    bb_per_side = bb_planes * bb_devices_per_plane
    xyz1_rsw = 1
    xyz1_fsw = xyz1_fsw_per_megapod
    xyz1_ssw = xyz1_ssw_per_megapod
    xyz1_xsw = xyz1_xsw_per_plane * xyz1_xsw_planes

    nodes = (
        abc1_rsw
        + abc1_fsw
        + abc1_ssw
        + abc1_fadu
        + bb_per_side * 2  # abc1 side + xyz1 side
        + xyz1_rsw
        + xyz1_fsw
        + xyz1_ssw
        + xyz1_xsw
    )

    # Links
    bb_total = bb_planes * bb_devices_per_plane

    rsw_fsw_abc1 = abc1_pods * abc1_planes  # One FSW per plane for each RSW.
    fsw_ssw_abc1 = (
        abc1_pods * abc1_planes * abc1_ssw_per_plane
    )  # Each FSW connects to every SSW in its plane.
    ssw_fadu = (
        abc1_planes * abc1_ssw_per_plane * abc1_hgrids
    )  # One FADU per horizontal grid for each SSW.
    dc_bb_abc1 = abc1_fadu * (bb_total // g_abc1)  # each FADU → k_dc BB devices
    bb_cross_site = (
        bb_planes * bb_devices_per_plane**2 * 2
    )  # Two full meshes between sites per plane.
    dc_bb_xyz1 = xyz1_xsw * (bb_total // g_xyz1)  # each XSW → k_dc BB devices
    ssw_xsw = (
        xyz1_ssw_per_megapod * xyz1_xsw_per_plane
    )  # Each SSW connects to every XSW in its plane.
    fsw_ssw_xyz1 = (
        xyz1_fsw_per_megapod * xyz1_ssw_per_megapod
    )  # Full mesh between FSW and SSW.
    rsw_fsw_xyz1 = xyz1_fsw_per_megapod  # One RSW connects to every FSW.

    links = (
        rsw_fsw_abc1
        + fsw_ssw_abc1
        + ssw_fadu
        + dc_bb_abc1
        + bb_cross_site
        + dc_bb_xyz1
        + ssw_xsw
        + fsw_ssw_xyz1
        + rsw_fsw_xyz1
    )

    return ExpectedCounts(
        nodes=nodes,
        links=links,
        links_rsw_fsw_abc1=rsw_fsw_abc1,
        links_fsw_ssw_abc1=fsw_ssw_abc1,
        links_ssw_fadu=ssw_fadu,
        links_dc_bb_abc1=dc_bb_abc1,
        links_bb_cross_site=bb_cross_site,
        links_dc_bb_xyz1=dc_bb_xyz1,
        links_ssw_xsw=ssw_xsw,
        links_fsw_ssw_xyz1=fsw_ssw_xyz1,
        links_rsw_fsw_xyz1=rsw_fsw_xyz1,
    )


def validate_scenario_file(scenario_path: Path, expected: ExpectedCounts) -> list[str]:
    """Load via NetGraph's API and validate counts and layer membership."""
    network = Scenario.from_yaml(scenario_path.read_text(encoding="utf-8")).network
    return validate_expanded_network(network, expected)


def validate_expanded_network(
    network: Network,
    expected: ExpectedCounts,
) -> list[str]:
    """Validate an expanded Network object against expected counts.

    Check total node/link counts and links per layer. Load the network with
    Scenario.from_yaml() before calling this function.
    """
    errors = []

    if len(network.nodes) != expected.nodes:
        errors.append(
            f"Node count: expected {expected.nodes}, got {len(network.nodes)}"
        )

    if len(network.links) != expected.links:
        errors.append(
            f"Link count: expected {expected.links}, got {len(network.links)}"
        )

    layer_counts: Counter = Counter()
    for link in network.links.values():
        src, tgt = link.source, link.target
        if "rsw" in src and "fsw" in tgt or "fsw" in src and "rsw" in tgt:
            if "abc1" in src:
                layer_counts["rsw_fsw_abc1"] += 1
            else:
                layer_counts["rsw_fsw_xyz1"] += 1
        elif "fsw" in src and "ssw" in tgt or "ssw" in src and "fsw" in tgt:
            if "abc1" in src:
                layer_counts["fsw_ssw_abc1"] += 1
            else:
                layer_counts["fsw_ssw_xyz1"] += 1
        elif "ssw" in src and "fadu" in tgt or "fadu" in src and "ssw" in tgt:
            layer_counts["ssw_fadu"] += 1
        elif "ssw" in src and "xsw" in tgt or "xsw" in src and "ssw" in tgt:
            layer_counts["ssw_xsw"] += 1
        elif ("fadu" in src and "bb/" in tgt) or ("bb/" in src and "fadu" in tgt):
            layer_counts["dc_bb_abc1"] += 1
        elif ("xsw" in src and "bb/" in tgt) or ("bb/" in src and "xsw" in tgt):
            layer_counts["dc_bb_xyz1"] += 1
        elif (
            "bb/abc1" in src
            and "bb/xyz1" in tgt
            or "bb/xyz1" in src
            and "bb/abc1" in tgt
        ):
            layer_counts["bb_cross_site"] += 1
        else:
            layer_counts[f"unknown_{src[:20]}_{tgt[:20]}"] += 1

    expected_layers = {
        "rsw_fsw_abc1": expected.links_rsw_fsw_abc1,
        "fsw_ssw_abc1": expected.links_fsw_ssw_abc1,
        "ssw_fadu": expected.links_ssw_fadu,
        "dc_bb_abc1": expected.links_dc_bb_abc1,
        "bb_cross_site": expected.links_bb_cross_site,
        "dc_bb_xyz1": expected.links_dc_bb_xyz1,
        "ssw_xsw": expected.links_ssw_xsw,
        "fsw_ssw_xyz1": expected.links_fsw_ssw_xyz1,
        "rsw_fsw_xyz1": expected.links_rsw_fsw_xyz1,
    }

    for layer_name, exp_count in expected_layers.items():
        actual = layer_counts.get(layer_name, 0)
        if actual != exp_count:
            errors.append(f"Layer {layer_name}: expected {exp_count}, got {actual}")

    for key, count in layer_counts.items():
        if key.startswith("unknown_"):
            errors.append(f"Unexpected link type: {key} ({count} links)")

    return errors


def validate_no_cross_group_links(
    network: Network,
) -> list[str]:
    """Check that each DC-BB link joins nodes in the same mesh group."""
    errors = []
    for _link_id, link in network.links.items():
        if link.attrs.get("link_type") != "dc_bb":
            continue
        src_mg = _extract_mg(link.source)
        tgt_mg = _extract_mg(link.target)
        if src_mg is None or tgt_mg is None:
            errors.append(
                f"Cannot extract mesh group from DC-BB link: {link.source} -> {link.target}"
            )
        elif src_mg != tgt_mg:
            errors.append(
                f"Cross-group DC-BB link: {link.source} (mg{src_mg}) -> {link.target} (mg{tgt_mg})"
            )
    return errors


def _extract_mg(path: str) -> Optional[str]:
    """Extract mesh group from a node path like 'abc1/fadu/mg03/...'."""
    m = re.search(r"/mg(\d+)/", path)
    return m.group(1) if m else None
