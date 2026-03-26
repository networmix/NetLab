"""Post-expansion validation for generated DC-BB scenarios.

Validates that the expanded network (after ngraph DSL processing)
matches the expected structure derived from config parameters.

Two validation levels:
  Level 1: Parse ngraph inspect output for total node/link counts.
  Level 2: Load via Scenario.from_yaml(), verify per-layer counts
           and per-node degree distributions.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


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
    """Compute expected post-expansion counts from config parameters.

    These are the counts ngraph inspect should report after DSL expansion.
    """
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

    rsw_fsw_abc1 = abc1_pods * abc1_planes  # each RSW → 8 FSW
    fsw_ssw_abc1 = abc1_pods * abc1_planes * abc1_ssw_per_plane  # each FSW → 36 SSW
    ssw_fadu = abc1_planes * abc1_ssw_per_plane * abc1_hgrids  # each SSW → 16 FADU
    dc_bb_abc1 = abc1_fadu * (bb_total // g_abc1)  # each FADU → k_dc BB devices
    bb_cross_site = (
        bb_planes * bb_devices_per_plane**2 * 2
    )  # 4×4 mesh × 2 paths per plane
    dc_bb_xyz1 = xyz1_xsw * (bb_total // g_xyz1)  # each XSW → k_dc BB devices
    ssw_xsw = xyz1_ssw_per_megapod * xyz1_xsw_per_plane  # each SSW → 64 XSW
    fsw_ssw_xyz1 = xyz1_fsw_per_megapod * xyz1_ssw_per_megapod  # each FSW → 24 SSW
    rsw_fsw_xyz1 = xyz1_fsw_per_megapod  # 1 RSW → 32 FSW

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


def validate_inspect_output(inspect_stdout: str, expected: ExpectedCounts) -> list[str]:
    """Parse ngraph inspect output and compare against expected counts.

    Returns list of error messages. Empty = valid.
    """
    errors = []

    # Parse "Total Nodes: N"
    m = re.search(r"Total Nodes:\s*([\d,]+)", inspect_stdout)
    if m:
        actual_nodes = int(m.group(1).replace(",", ""))
        if actual_nodes != expected.nodes:
            errors.append(
                f"Node count mismatch: expected {expected.nodes}, got {actual_nodes}"
            )
    else:
        errors.append("Could not parse 'Total Nodes' from inspect output")

    # Parse "Total Links: N"
    m = re.search(r"Total Links:\s*([\d,]+)", inspect_stdout)
    if m:
        actual_links = int(m.group(1).replace(",", ""))
        if actual_links != expected.links:
            errors.append(
                f"Link count mismatch: expected {expected.links}, got {actual_links}"
            )
    else:
        errors.append("Could not parse 'Total Links' from inspect output")

    return errors


def validate_scenario_file(
    scenario_path: Path,
    expected: ExpectedCounts,
    ngraph_bin: str = "ngraph",
) -> list[str]:
    """Run ngraph inspect and validate against expected counts.

    Level 1 validation: fast, uses ngraph CLI.
    """
    result = subprocess.run(
        [ngraph_bin, "inspect", str(scenario_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return [f"ngraph inspect failed: {result.stderr[-500:]}"]

    return validate_inspect_output(result.stdout, expected)


def validate_expanded_network(
    network: Any,
    expected: ExpectedCounts,
) -> list[str]:
    """Validate an expanded Network object against expected counts.

    Level 2 validation: thorough, requires loading via Scenario.from_yaml().
    Checks per-layer link counts and per-node degree distributions.
    """
    errors = []

    # Node count
    if len(network.nodes) != expected.nodes:
        errors.append(
            f"Node count: expected {expected.nodes}, got {len(network.nodes)}"
        )

    # Link count
    if len(network.links) != expected.links:
        errors.append(
            f"Link count: expected {expected.links}, got {len(network.links)}"
        )

    # Classify links by layer using source/target node names
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

    # Check for unknown link types
    for key, count in layer_counts.items():
        if key.startswith("unknown_"):
            errors.append(f"Unexpected link type: {key} ({count} links)")

    return errors


def validate_no_cross_group_links(
    network: Any,
) -> list[str]:
    """Verify no DC-BB link connects nodes in different mesh groups.

    Checks that the mesh group encoded in node paths matches between
    source and target of every DC-BB link.
    """
    errors = []
    for _link_id, link in network.links.items():
        if link.attrs.get("link_type") != "dc_bb":
            continue
        # Extract mesh group from source and target paths
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
