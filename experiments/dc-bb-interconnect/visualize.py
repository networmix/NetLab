#!/usr/bin/env python3
"""
DC-BB Topology Visualizer

Generates organized SVG diagrams from ngraph results with BuildGraph export.
Layout: Semantic attribute-based positioning - DC rows horizontal, BB planes in middle.
Site A DCs at top, BB planes in middle, Site B DCs at bottom.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import networkx as nx  # noqa: E402

from netlab.visualize import GraphVisualizer, StyleConfig, layout_row  # noqa: E402


class DcBbVisualizer(GraphVisualizer):
    """
    DC-BB specific 3-tier layout visualizer.

    Layout structure:
    - Site A DC rows: horizontal band at top
    - BB planes: vertical boxes in middle (Site A nodes top, Site B nodes bottom)
    - Site B DC rows: horizontal band at bottom
    """

    # DC-BB specific layout constants (not in StyleConfig)
    DC_ROW_GAP = 24
    BB_PLANE_GAP = 30
    BB_NODE_SPACING = 20

    def __init__(self, graph: nx.Graph, style: Optional[StyleConfig] = None):
        """Initialize DC-BB visualizer with appropriate style."""
        if style is None:
            style = StyleConfig(
                node_radius=8,
                node_spacing=20,
                group_padding=16,
                section_gap=180,
                canvas_padding=40,
                node_colors={"A": "#4CAF50", "B": "#2196F3"},
                edge_colors={"bb_intersite": "#FF9800"},
                box_colors={
                    "dc_A": ("#E8F5E9", "#4CAF50"),
                    "dc_B": ("#E3F2FD", "#2196F3"),
                },
                highlight_edge_types={"bb_intersite"},
            )
        super().__init__(graph, style)
        self._groups: Optional[dict] = None

    def layout(self) -> Dict[str, Tuple[float, float]]:
        """Position nodes by semantic attributes in 3-tier layout."""
        groups = self._group_nodes()
        self._groups = groups
        positions = {}

        # Discover all DC rows and BB planes
        dc_ids = sorted(set(groups["dc"]["A"].keys()) | set(groups["dc"]["B"].keys()))
        bb_plane_ids = sorted(
            set(groups["bb"]["A"].keys()) | set(groups["bb"]["B"].keys())
        )

        # Find max nodes per DC row (for consistent spacing)
        max_dc_nodes = 0
        for site in ["A", "B"]:
            for dc_id in dc_ids:
                max_dc_nodes = max(max_dc_nodes, len(groups["dc"][site].get(dc_id, [])))

        # Calculate DC row box width
        dc_row_box_width = (
            max_dc_nodes * self.style.node_spacing + self.style.group_padding * 2
        )

        # Find max BB nodes per site per plane
        max_bb_nodes_per_plane = 0
        for site in ["A", "B"]:
            for plane_id in bb_plane_ids:
                max_bb_nodes_per_plane = max(
                    max_bb_nodes_per_plane, len(groups["bb"][site].get(plane_id, []))
                )

        # Calculate BB plane box width
        bb_plane_width = max_bb_nodes_per_plane * self.BB_NODE_SPACING + 32
        self._bb_plane_width = bb_plane_width

        # Calculate total widths for centering
        num_dc_rows = len(dc_ids)
        num_planes = len(bb_plane_ids)

        dc_band_width = (
            num_dc_rows * dc_row_box_width + (num_dc_rows - 1) * self.DC_ROW_GAP
        )
        bb_band_width = (
            num_planes * bb_plane_width + (num_planes - 1) * self.BB_PLANE_GAP
        )

        # Canvas width is the maximum of the two bands plus padding
        content_width = max(dc_band_width, bb_band_width)
        self.canvas_width = content_width + self.style.canvas_padding * 2

        # Y positions for each section
        y_dc_a = (
            self.style.canvas_padding
            + self.style.node_radius
            + self.style.group_padding
        )
        y_bb_start = (
            y_dc_a
            + self.style.node_radius
            + self.style.group_padding
            + self.style.section_gap
        )

        # Calculate BB section height
        bb_section_height = (
            self.style.node_radius * 2 + 40 + self.style.node_radius * 2 + 60
        )

        y_dc_b = y_bb_start + bb_section_height + self.style.section_gap

        self.canvas_height = (
            y_dc_b
            + self.style.node_radius
            + self.style.group_padding
            + self.style.canvas_padding
        )

        # --- Position DC nodes - Site A (top) ---
        dc_start_x = self.style.canvas_padding + (content_width - dc_band_width) / 2
        for i, dc_id in enumerate(dc_ids):
            nodes = groups["dc"]["A"].get(dc_id, [])
            row_x = (
                dc_start_x
                + i * (dc_row_box_width + self.DC_ROW_GAP)
                + self.style.group_padding
                + self.style.node_radius
            )
            row_positions = layout_row(
                nodes,
                row_x,
                y_dc_a,
                self.style.node_spacing,
                sort_key=self._get_node_index,
            )
            positions.update(row_positions)

        # --- Position DC nodes - Site B (bottom) ---
        for i, dc_id in enumerate(dc_ids):
            nodes = groups["dc"]["B"].get(dc_id, [])
            row_x = (
                dc_start_x
                + i * (dc_row_box_width + self.DC_ROW_GAP)
                + self.style.group_padding
                + self.style.node_radius
            )
            row_positions = layout_row(
                nodes,
                row_x,
                y_dc_b,
                self.style.node_spacing,
                sort_key=self._get_node_index,
            )
            positions.update(row_positions)

        # --- Position BB nodes (middle section) ---
        bb_start_x = self.style.canvas_padding + (content_width - bb_band_width) / 2
        y_bb_a = y_bb_start + self.style.node_radius + 20
        y_bb_b = y_bb_start + bb_section_height - self.style.node_radius - 20

        for i, plane_id in enumerate(bb_plane_ids):
            plane_left_x = bb_start_x + i * (bb_plane_width + self.BB_PLANE_GAP)

            # Site A BB nodes (top row within plane)
            nodes_a = groups["bb"]["A"].get(plane_id, [])
            start_x = plane_left_x + 16 + self.style.node_radius
            positions.update(
                layout_row(
                    nodes_a,
                    start_x,
                    y_bb_a,
                    self.BB_NODE_SPACING,
                    sort_key=self._get_node_index,
                )
            )

            # Site B BB nodes (bottom row within plane)
            nodes_b = groups["bb"]["B"].get(plane_id, [])
            positions.update(
                layout_row(
                    nodes_b,
                    start_x,
                    y_bb_b,
                    self.BB_NODE_SPACING,
                    sort_key=self._get_node_index,
                )
            )

        self.positions = positions
        return positions

    def _parse_node_id(self, node_id: str) -> dict:
        """Parse node ID to extract DC row or BB plane."""
        parts = node_id.split("/")
        result = {}

        if len(parts) >= 2:
            if parts[0].startswith("Site"):
                result["site"] = parts[0][-1]

            layer_part = parts[1]
            if layer_part.startswith("DC-"):
                result["dc_id"] = int(layer_part.split("-")[1])
            elif layer_part.startswith("BB-"):
                result["bb_plane_id"] = int(layer_part.split("-")[1])

        return result

    def _group_nodes(self) -> dict:
        """Group nodes by role, site, and ID."""
        groups = {
            "dc": {"A": defaultdict(list), "B": defaultdict(list)},
            "bb": {"A": defaultdict(list), "B": defaultdict(list)},
        }

        for node, attrs in self.graph.nodes(data=True):
            role = attrs.get("role")
            site = attrs.get("site")

            if not role or not site:
                continue

            parsed = self._parse_node_id(node)

            if role == "dc":
                dc_id = parsed.get("dc_id", attrs.get("dc_row_id"))
                if dc_id is not None:
                    groups["dc"][site][dc_id].append(node)
            elif role == "bb":
                plane_id = parsed.get("bb_plane_id", attrs.get("bb_plane_id"))
                if plane_id is not None:
                    groups["bb"][site][plane_id].append(node)

        return groups

    def _get_node_index(self, node: str) -> int:
        """Extract numeric index from node ID."""
        parts = node.split("/")
        if parts:
            try:
                return int(parts[-1])
            except ValueError:
                pass
        return 0

    def get_groups(self) -> Dict[str, List[str]]:
        """Return DC rows and BB planes as groups for box drawing."""
        if self._groups is None:
            self._groups = self._group_nodes()

        result = {}

        # DC row groups
        for site in ["A", "B"]:
            for dc_id, nodes in self._groups["dc"][site].items():
                result[f"dc_{site}_{dc_id}"] = nodes

        # BB plane groups
        bb_plane_ids = sorted(
            set(self._groups["bb"]["A"].keys()) | set(self._groups["bb"]["B"].keys())
        )
        for plane_id in bb_plane_ids:
            nodes_a = self._groups["bb"]["A"].get(plane_id, [])
            nodes_b = self._groups["bb"]["B"].get(plane_id, [])
            result[f"bb_plane_{plane_id}"] = nodes_a + nodes_b

        return result

    def get_group_style(self, group_name: str) -> Tuple[str, str]:
        """Get style for a group based on its type."""
        if group_name.startswith("dc_A"):
            return self.style.box_colors.get("dc_A", ("#E8F5E9", "#4CAF50"))
        elif group_name.startswith("dc_B"):
            return self.style.box_colors.get("dc_B", ("#E3F2FD", "#2196F3"))
        elif group_name.startswith("bb_plane"):
            return (self.style.default_box_fill, self.style.default_box_stroke)
        return super().get_group_style(group_name)

    def get_group_label(self, group_name: str) -> Optional[str]:
        """Get label for a group."""
        if group_name.startswith("dc_"):
            # Extract DC-N from dc_A_N or dc_B_N
            parts = group_name.split("_")
            if len(parts) >= 3:
                return f"DC-{parts[2]}"
        elif group_name.startswith("bb_plane_"):
            plane_id = group_name.replace("bb_plane_", "")
            return f"Plane {plane_id}"
        return None


def visualize_topology(
    results_path: Path,
    output_path: Optional[Path] = None,
    split: bool = False,
) -> Path:
    """Visualize a topology from results JSON.

    Args:
        results_path: Path to .results.json file
        output_path: Optional output path for SVG (default: same dir as results)
        split: If True, also generate a split view with disconnected components stacked

    Returns:
        Path to generated SVG
    """
    if output_path is None:
        output_path = results_path.parent / "topology.svg"

    viz = DcBbVisualizer.from_results(results_path)
    viz.layout()
    viz.render_svg(output_path)

    if split:
        # Create a fresh visualizer for split view (positions get modified)
        viz_split = DcBbVisualizer.from_results(results_path)
        viz_split.layout()
        split_path = output_path.with_stem(output_path.stem + "_split")
        result = viz_split.render_svg_split(split_path)
        if result:
            print(f"Generated split view: {split_path}")

    return output_path


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SVG visualization from ngraph results"
    )
    parser.add_argument(
        "results",
        type=Path,
        help="Path to .results.json file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output SVG path (default: topology.svg in results dir)",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Also generate split view with disconnected components stacked",
    )

    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: {args.results} not found")
        return 1

    visualize_topology(args.results, args.output, split=args.split)
    return 0


if __name__ == "__main__":
    exit(main())
