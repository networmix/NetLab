"""Render network graphs as SVG or PNG with configurable layouts and styles."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx

from netlab.artifacts import atomic_path


@dataclass
class EdgeStyle:
    """Style configuration for an edge."""

    color: str = "#BDBDBD"
    width: float = 1.0
    opacity: float = 0.6


@dataclass
class StyleConfig:
    """Node, edge, group-box, and spacing settings. Lengths use SVG coordinate units."""

    # Sizing
    node_radius: float = 8
    node_spacing: float = 20
    group_padding: float = 16
    section_gap: float = 100
    canvas_padding: float = 40

    # Color scheme: maps attribute values to colors
    node_colors: Dict[str, str] = field(default_factory=dict)
    edge_colors: Dict[str, str] = field(default_factory=dict)
    box_colors: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    # Which attributes drive coloring
    node_color_attr: str = "site"
    edge_color_attr: str = "link_type"

    # Defaults
    default_node_color: str = "#9E9E9E"
    default_edge_color: str = "#BDBDBD"
    default_box_fill: str = "#F5F5F5"
    default_box_stroke: str = "#BDBDBD"

    # Edge styling
    edge_width: float = 1.0
    edge_opacity: float = 0.6
    highlighted_edge_width: float = 2.0
    highlighted_edge_opacity: float = 0.9
    highlight_edge_types: Set[str] = field(default_factory=set)

    # Unconnected node styling
    unconnected_node_stroke: str = "#F44336"  # Red
    unconnected_node_stroke_width: float = 2.0


class GraphVisualizer(ABC):
    """Render a NetworkX graph with subclass-defined positions.

    Implement ``layout``; override color, edge-style, and group hooks as needed.
    Use ``render_svg`` for SVG and optional PNG output.
    """

    def __init__(self, graph: nx.Graph, style: Optional[StyleConfig] = None):
        """Use the supplied graph and style, or the default StyleConfig."""
        self.graph = graph
        self.style = style or StyleConfig()
        self.positions: Dict[str, Tuple[float, float]] = {}
        self.canvas_width: float = 0
        self.canvas_height: float = 0

    @classmethod
    def from_results(
        cls,
        results_path: Path,
        style: Optional[StyleConfig] = None,
        build_graph_step: str = "build_graph",
    ) -> "GraphVisualizer":
        """Load the graph exported by ``build_graph_step`` in a NetGraph result file."""
        with open(results_path) as f:
            results = json.load(f)

        steps = results.get("steps", {})
        if build_graph_step not in steps:
            raise ValueError(
                f"No {build_graph_step} step in results. Add BuildGraph to workflow."
            )

        graph_data = steps[build_graph_step]["data"]["graph"]
        graph = nx.node_link_graph(graph_data, edges="edges")

        return cls(graph, style)

    @abstractmethod
    def layout(self) -> Dict[str, Tuple[float, float]]:
        """Set node positions and canvas dimensions; return the position mapping."""
        pass

    def get_node_color(self, node: str, attrs: dict) -> str:
        """Choose a node color from its configured attribute; override for custom rules."""
        value = attrs.get(self.style.node_color_attr)
        if value is None:
            return self.style.default_node_color
        return self.style.node_colors.get(str(value), self.style.default_node_color)

    def get_edge_style(self, u: str, v: str, data: dict) -> EdgeStyle:
        """Choose edge color, width, and opacity from its attributes."""
        edge_type = data.get(self.style.edge_color_attr)
        is_highlighted = edge_type in self.style.highlight_edge_types

        if edge_type is None:
            color = self.style.default_edge_color
        else:
            color = self.style.edge_colors.get(
                str(edge_type), self.style.default_edge_color
            )
        width = (
            self.style.highlighted_edge_width
            if is_highlighted
            else self.style.edge_width
        )
        opacity = (
            self.style.highlighted_edge_opacity
            if is_highlighted
            else self.style.edge_opacity
        )

        return EdgeStyle(color=color, width=width, opacity=opacity)

    def get_groups(self) -> Dict[str, List[str]]:
        """Return group names and their node IDs for boxes; empty by default."""
        return {}

    def get_group_style(self, group_name: str) -> Tuple[str, str]:
        """Return a group box's (fill, stroke) colors."""
        return self.style.box_colors.get(
            group_name, (self.style.default_box_fill, self.style.default_box_stroke)
        )

    def get_group_label(self, group_name: str) -> Optional[str]:
        """Return the group label, or None to omit it."""
        return group_name

    def render_svg(
        self,
        output_path: Path,
        png: bool = True,
        png_scale: int = 2,
    ) -> None:
        """Write SVG and optionally a PNG at ``png_scale`` times the canvas size."""
        if not self.positions:
            self.layout()

        svg = self._create_svg_root()
        self._draw_background(svg)

        boxes_group = ET.SubElement(svg, "g", id="boxes")
        links_group = ET.SubElement(svg, "g", id="links")
        nodes_group = ET.SubElement(svg, "g", id="nodes")
        labels_group = ET.SubElement(svg, "g", id="labels")

        # Draw in order (boxes underneath, then edges, then nodes)
        self._draw_all_groups(boxes_group, labels_group)
        self._draw_all_edges(links_group)
        self._draw_all_nodes(nodes_group)

        self._write_svg(svg, output_path)
        print(f"Generated: {output_path}")

        if png:
            self._write_png(output_path, scale=png_scale)

    def render_svg_split(
        self,
        output_path: Path,
        gap: float = 500,
        max_components: int = 64,
        min_component_size: int = 2,
        png: bool = True,
        png_scale: int = 2,
    ) -> Optional[Path]:
        """Stack disconnected components and render them to SVG, with optional PNG.

        Ignore components smaller than ``min_component_size``. Return None when
        fewer than two or more than ``max_components`` eligible components remain.
        """
        if not self.positions:
            self.layout()

        num_components = self.split_disconnected_components(
            gap=gap,
            max_components=max_components,
            min_component_size=min_component_size,
        )

        if num_components <= 1:
            return None

        self.render_svg(output_path, png=png, png_scale=png_scale)
        return output_path

    def _create_svg_root(self) -> ET.Element:
        """Create the SVG root element."""
        return ET.Element(
            "svg",
            xmlns="http://www.w3.org/2000/svg",
            width=str(int(self.canvas_width)),
            height=str(int(self.canvas_height)),
            viewBox=f"0 0 {int(self.canvas_width)} {int(self.canvas_height)}",
        )

    def _draw_background(self, svg: ET.Element) -> None:
        """Add white background."""
        ET.SubElement(
            svg,
            "rect",
            width="100%",
            height="100%",
            fill="white",
        )

    def _draw_all_groups(
        self, boxes_group: ET.Element, labels_group: ET.Element
    ) -> None:
        """Draw all group boxes."""
        groups = self.get_groups()
        for group_name, nodes in groups.items():
            positioned = [n for n in nodes if n in self.positions]
            if not positioned:
                continue

            bounds = self.compute_bounds(positioned)
            fill, stroke = self.get_group_style(group_name)
            label = self.get_group_label(group_name)

            self.draw_box(
                boxes_group,
                bounds[0],
                bounds[1],
                bounds[2] - bounds[0],
                bounds[3] - bounds[1],
                fill,
                stroke,
            )

            if label:
                label_x = (bounds[0] + bounds[2]) / 2
                label_y = bounds[1] - 6
                self.draw_label(labels_group, label_x, label_y, label, stroke)

    def _draw_all_edges(self, parent: ET.Element) -> None:
        """Draw all edges, with non-highlighted first."""
        highlighted = []
        normal = []

        for u, v, data in self.graph.edges(data=True):
            if u not in self.positions or v not in self.positions:
                continue
            edge_type = data.get(self.style.edge_color_attr)
            if edge_type in self.style.highlight_edge_types:
                highlighted.append((u, v, data))
            else:
                normal.append((u, v, data))

        for u, v, data in normal + highlighted:
            style = self.get_edge_style(u, v, data)
            x1, y1 = self.positions[u]
            x2, y2 = self.positions[v]
            self.draw_edge(parent, x1, y1, x2, y2, style)

    def _draw_all_nodes(self, parent: ET.Element) -> None:
        """Draw all nodes."""
        for node, (x, y) in self.positions.items():
            attrs = self.graph.nodes[node]
            color = self.get_node_color(node, attrs)

            is_connected = self.graph.degree[node] > 0
            if is_connected:
                stroke = "white"
                stroke_width = 1.0
            else:
                stroke = self.style.unconnected_node_stroke
                stroke_width = self.style.unconnected_node_stroke_width

            self.draw_node(
                parent, x, y, color, stroke=stroke, stroke_width=stroke_width
            )

    def _write_svg(self, svg: ET.Element, output_path: Path) -> None:
        """Write SVG to file."""
        tree = ET.ElementTree(svg)
        ET.indent(tree, space="  ")
        with atomic_path(output_path) as temporary:
            tree.write(temporary, encoding="unicode", xml_declaration=True)

    def _write_png(self, svg_path: Path, scale: int = 2) -> None:
        """Generate PNG from SVG."""
        try:
            import cairosvg
        except ImportError as e:
            raise ImportError(
                "cairosvg is required for PNG generation. "
                "Install with: pip install cairosvg\n"
                "On macOS, you may also need: brew install cairo"
            ) from e
        png_path = svg_path.with_suffix(".png")
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=scale)
        print(f"Generated: {png_path}")

    def draw_node(
        self,
        parent: ET.Element,
        x: float,
        y: float,
        color: str,
        radius: Optional[float] = None,
        stroke: str = "white",
        stroke_width: float = 1.0,
    ) -> ET.Element:
        """Append and return a circle; default its radius to style.node_radius."""
        if radius is None:
            radius = self.style.node_radius

        attrib = {
            "cx": str(x),
            "cy": str(y),
            "r": str(radius),
            "fill": color,
            "stroke": stroke,
            "stroke-width": str(stroke_width),
        }
        return ET.SubElement(parent, "circle", attrib)

    def draw_edge(
        self,
        parent: ET.Element,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        style: EdgeStyle,
    ) -> ET.Element:
        """Append and return an SVG line with the supplied edge style."""
        attrib = {
            "x1": str(x1),
            "y1": str(y1),
            "x2": str(x2),
            "y2": str(y2),
            "stroke": style.color,
            "stroke-width": str(style.width),
            "stroke-opacity": str(style.opacity),
        }
        return ET.SubElement(parent, "line", attrib)

    def draw_box(
        self,
        parent: ET.Element,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: str,
        stroke: str,
        rx: float = 6,
        stroke_width: float = 1,
    ) -> ET.Element:
        """Append and return an SVG rectangle with corner radius ``rx``."""
        attrib = {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "rx": str(rx),
            "fill": fill,
            "stroke": stroke,
            "stroke-width": str(stroke_width),
        }
        return ET.SubElement(parent, "rect", attrib)

    def draw_label(
        self,
        parent: ET.Element,
        x: float,
        y: float,
        text: str,
        color: str = "#616161",
        font_size: int = 11,
        font_weight: str = "500",
    ) -> ET.Element:
        """Append and return a text label centered horizontally at (x, y)."""
        attrib = {
            "x": str(x),
            "y": str(y),
            "text-anchor": "middle",
            "font-size": str(font_size),
            "font-family": "sans-serif",
            "font-weight": font_weight,
            "fill": color,
        }
        elem = ET.SubElement(parent, "text", attrib)
        elem.text = text
        return elem

    def compute_bounds(
        self,
        nodes: List[str],
        padding: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) for positioned nodes.

        Default padding is node_radius + group_padding / 2.
        """
        if padding is None:
            padding = self.style.node_radius + self.style.group_padding / 2

        xs = [self.positions[n][0] for n in nodes if n in self.positions]
        ys = [self.positions[n][1] for n in nodes if n in self.positions]

        if not xs or not ys:
            return (0, 0, 0, 0)

        return (
            min(xs) - padding,
            min(ys) - padding,
            max(xs) + padding,
            max(ys) + padding,
        )

    def split_disconnected_components(
        self,
        gap: float = 500,
        max_components: int = 64,
        min_component_size: int = 2,
    ) -> int:
        """Stack eligible components vertically, updating positions and canvas height.

        Ignore components smaller than ``min_component_size``. Return the component
        count (0 or 1 means no split); return 0 when the limit is exceeded.
        """
        if not self.positions:
            return 0

        # Find connected components (use undirected view for visual connectivity)
        if self.graph.is_directed():
            undirected = self.graph.to_undirected()
        else:
            undirected = self.graph

        all_components = list(nx.connected_components(undirected))
        components = [
            comp for comp in all_components if len(comp) >= min_component_size
        ]

        if len(components) <= 1:
            return len(components)
        if len(components) > max_components:
            return 0

        positioned_components = []
        for comp in components:
            positioned_nodes = [n for n in comp if n in self.positions]
            if positioned_nodes:
                positioned_components.append(positioned_nodes)

        if len(positioned_components) <= 1:
            return len(positioned_components)

        component_bounds = []
        for nodes in positioned_components:
            ys = [self.positions[n][1] for n in nodes]
            xs = [self.positions[n][0] for n in nodes]
            min_y, max_y = min(ys), max(ys)
            min_x, max_x = min(xs), max(xs)
            component_bounds.append(
                {
                    "nodes": nodes,
                    "min_y": min_y,
                    "max_y": max_y,
                    "min_x": min_x,
                    "max_x": max_x,
                    "height": max_y - min_y,
                }
            )

        component_bounds.sort(key=lambda c: c["min_y"])

        current_y = self.style.canvas_padding
        for comp_info in component_bounds:
            y_offset = current_y - comp_info["min_y"] + self.style.node_radius

            for node in comp_info["nodes"]:
                x, y = self.positions[node]
                self.positions[node] = (x, y + y_offset)

            current_y += comp_info["height"] + gap + self.style.node_radius * 2

        self.canvas_height = (
            max(
                current_y,
                max(y for _, y in self.positions.values()) + self.style.node_radius,
            )
            + self.style.canvas_padding
        )

        return len(positioned_components)


def layout_row(
    nodes: List[str],
    start_x: float,
    y: float,
    spacing: float,
    sort_key: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Return node positions in a horizontal row, optionally ordered by ``sort_key``."""
    if sort_key:
        nodes = sorted(nodes, key=sort_key)
    return {node: (start_x + i * spacing, y) for i, node in enumerate(nodes)}


def layout_column(
    nodes: List[str],
    x: float,
    start_y: float,
    spacing: float,
    sort_key: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Return node positions in a vertical column, optionally ordered by ``sort_key``."""
    if sort_key:
        nodes = sorted(nodes, key=sort_key)
    return {node: (x, start_y + i * spacing) for i, node in enumerate(nodes)}


def layout_grid(
    nodes: List[str],
    start_x: float,
    start_y: float,
    cols: int,
    spacing_x: float,
    spacing_y: float,
    sort_key: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Return node positions in a grid with ``cols`` columns and the supplied spacing."""
    if cols < 1:
        raise ValueError("cols must be positive")
    if sort_key:
        nodes = sorted(nodes, key=sort_key)
    positions = {}
    for i, node in enumerate(nodes):
        row, col = divmod(i, cols)
        positions[node] = (start_x + col * spacing_x, start_y + row * spacing_y)
    return positions


def merge_layouts(
    *layouts: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    """Merge position mappings; later mappings override duplicate node IDs."""
    result = {}
    for layout in layouts:
        result.update(layout)
    return result
