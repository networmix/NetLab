"""
Graph Visualizer Module

Provides a base class for graph visualization with reusable rendering primitives
and configurable styling.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx


@dataclass
class EdgeStyle:
    """Style configuration for an edge."""

    color: str = "#BDBDBD"
    width: float = 1.0
    opacity: float = 0.6


@dataclass
class StyleConfig:
    """
    Visual styling configuration for graph visualization.

    Attributes:
        node_radius: Radius of node circles
        node_spacing: Default spacing between nodes
        group_padding: Padding around group boxes
        section_gap: Gap between layout sections
        canvas_padding: Padding around the canvas edge

        node_colors: Maps attribute values to node colors
        edge_colors: Maps attribute values to edge colors
        box_colors: Maps group names to (fill, stroke) tuples

        node_color_attr: Node attribute used for coloring
        edge_color_attr: Edge attribute used for coloring

        default_node_color: Default node color
        default_edge_color: Default edge color
        default_box_fill: Default box fill color
        default_box_stroke: Default box stroke color

        edge_width: Default edge width
        edge_opacity: Default edge opacity
        highlighted_edge_width: Width for highlighted edges
        highlighted_edge_opacity: Opacity for highlighted edges
        highlight_edge_types: Set of edge types to highlight

        unconnected_node_stroke: Stroke color for unconnected nodes
        unconnected_node_stroke_width: Stroke width for unconnected nodes
    """

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
    """
    Base class for graph visualization.

    Provides:
    - Reusable rendering primitives (draw_node, draw_edge, draw_box)
    - SVG/PNG generation
    - Configurable styling via StyleConfig

    Subclasses must implement:
    - layout(): Compute node positions

    Subclasses may override:
    - get_node_color(): Custom node coloring logic
    - get_edge_style(): Custom edge styling logic
    - get_groups(): Define node groups for box drawing
    - get_group_style(): Custom group styling logic
    """

    def __init__(self, graph: nx.Graph, style: Optional[StyleConfig] = None):
        """
        Initialize the visualizer.

        Args:
            graph: NetworkX graph to visualize
            style: Optional style configuration
        """
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
        """
        Create visualizer from ngraph results JSON.

        Args:
            results_path: Path to .results.json file
            style: Optional style configuration
            build_graph_step: Name of BuildGraph step in results

        Returns:
            GraphVisualizer instance
        """
        with open(results_path) as f:
            results = json.load(f)

        steps = results.get("steps", {})
        if build_graph_step not in steps:
            raise ValueError(
                f"No {build_graph_step} step in results. Add BuildGraph to workflow."
            )

        graph_data = steps[build_graph_step]["data"]["graph"]
        # Handle both 'edges' and 'links' keys for compatibility
        edges_key = "edges" if "edges" in graph_data else "links"
        graph = nx.node_link_graph(graph_data, edges=edges_key)

        return cls(graph, style)

    # --- Abstract method: subclass MUST implement ---

    @abstractmethod
    def layout(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute node positions.

        Must:
        - Populate self.positions with {node_id: (x, y)}
        - Set self.canvas_width and self.canvas_height
        - Return self.positions
        """
        pass

    # --- Hooks: subclass MAY override ---

    def get_node_color(self, node: str, attrs: dict) -> str:
        """
        Get color for a node based on style config.

        Override for custom coloring logic.

        Args:
            node: Node ID
            attrs: Node attributes

        Returns:
            Color string (hex or name)
        """
        value = attrs.get(self.style.node_color_attr)
        if value is None:
            return self.style.default_node_color
        return self.style.node_colors.get(str(value), self.style.default_node_color)

    def get_edge_style(self, u: str, v: str, data: dict) -> EdgeStyle:
        """
        Get style for an edge based on style config.

        Override for custom edge styling logic.

        Args:
            u: Source node ID
            v: Target node ID
            data: Edge attributes

        Returns:
            EdgeStyle instance
        """
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
        """
        Return node groups for box drawing.

        Override to enable group box drawing.

        Returns:
            Dict mapping group name to list of node IDs
        """
        return {}

    def get_group_style(self, group_name: str) -> Tuple[str, str]:
        """
        Get (fill, stroke) colors for a group box.

        Override for custom group styling logic.

        Args:
            group_name: Name of the group

        Returns:
            Tuple of (fill_color, stroke_color)
        """
        return self.style.box_colors.get(
            group_name, (self.style.default_box_fill, self.style.default_box_stroke)
        )

    def get_group_label(self, group_name: str) -> Optional[str]:
        """
        Get label text for a group box.

        Override to provide custom labels.

        Args:
            group_name: Name of the group

        Returns:
            Label string, or None for no label
        """
        return group_name

    # --- Rendering ---

    def render_svg(
        self,
        output_path: Path,
        png: bool = True,
        png_scale: int = 2,
    ) -> None:
        """
        Render graph to SVG (and optionally PNG).

        Args:
            output_path: Output SVG path
            png: Whether to also generate PNG
            png_scale: Scale factor for PNG (default: 2x)
        """
        if not self.positions:
            self.layout()

        svg = self._create_svg_root()
        self._draw_background(svg)

        # Create groups for layering
        boxes_group = ET.SubElement(svg, "g", id="boxes")
        links_group = ET.SubElement(svg, "g", id="links")
        nodes_group = ET.SubElement(svg, "g", id="nodes")
        labels_group = ET.SubElement(svg, "g", id="labels")

        # Draw in order (boxes underneath, then edges, then nodes)
        self._draw_all_groups(boxes_group, labels_group)
        self._draw_all_edges(links_group)
        self._draw_all_nodes(nodes_group)

        # Write SVG
        self._write_svg(svg, output_path)
        print(f"Generated: {output_path}")

        # Generate PNG if requested
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
        """
        Render split view if multiple disconnected components exist.

        Applies split_disconnected_components() before rendering.
        Returns None if split wasn't applied (1 component or too many).

        Args:
            output_path: Output SVG path
            gap: Vertical gap between stacked components
            max_components: Skip splitting if more than this many components
            min_component_size: Ignore components smaller than this
            png: Whether to also generate PNG
            png_scale: Scale factor for PNG

        Returns:
            Path to generated SVG, or None if split wasn't applied
        """
        if not self.positions:
            self.layout()

        # Apply split
        num_components = self.split_disconnected_components(
            gap=gap,
            max_components=max_components,
            min_component_size=min_component_size,
        )

        # If no split applied (1 component or too many), return None
        if num_components <= 1:
            return None

        # Render the split view
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
                # Add label above the box
                label_x = (bounds[0] + bounds[2]) / 2
                label_y = bounds[1] - 6
                self.draw_label(labels_group, label_x, label_y, label, stroke)

    def _draw_all_edges(self, parent: ET.Element) -> None:
        """Draw all edges, with non-highlighted first."""
        # Separate highlighted and non-highlighted edges
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

        # Draw normal edges first
        for u, v, data in normal:
            style = self.get_edge_style(u, v, data)
            x1, y1 = self.positions[u]
            x2, y2 = self.positions[v]
            self.draw_edge(parent, x1, y1, x2, y2, style)

        # Draw highlighted edges on top
        for u, v, data in highlighted:
            style = self.get_edge_style(u, v, data)
            x1, y1 = self.positions[u]
            x2, y2 = self.positions[v]
            self.draw_edge(parent, x1, y1, x2, y2, style)

    def _draw_all_nodes(self, parent: ET.Element) -> None:
        """Draw all nodes."""
        for node, (x, y) in self.positions.items():
            attrs = self.graph.nodes[node]
            color = self.get_node_color(node, attrs)

            # Check if node is connected (has any edges)
            is_connected = len(list(self.graph.edges(node))) > 0
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
        tree.write(output_path, encoding="unicode", xml_declaration=True)

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

    # --- Drawing primitives ---

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
        """
        Draw a node circle.

        Args:
            parent: Parent SVG element
            x: X coordinate
            y: Y coordinate
            color: Fill color
            radius: Circle radius (default: style.node_radius)
            stroke: Stroke color
            stroke_width: Stroke width

        Returns:
            The created circle element
        """
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
        """
        Draw an edge line.

        Args:
            parent: Parent SVG element
            x1: Start X coordinate
            y1: Start Y coordinate
            x2: End X coordinate
            y2: End Y coordinate
            style: Edge style configuration

        Returns:
            The created line element
        """
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
        """
        Draw a rounded rectangle box.

        Args:
            parent: Parent SVG element
            x: X coordinate
            y: Y coordinate
            width: Box width
            height: Box height
            fill: Fill color
            stroke: Stroke color
            rx: Corner radius
            stroke_width: Stroke width

        Returns:
            The created rect element
        """
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
        """
        Draw a text label.

        Args:
            parent: Parent SVG element
            x: X coordinate (center)
            y: Y coordinate
            text: Label text
            color: Text color
            font_size: Font size
            font_weight: Font weight

        Returns:
            The created text element
        """
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
        """
        Compute bounding box for positioned nodes.

        Args:
            nodes: List of node IDs
            padding: Padding around nodes (default: node_radius + group_padding/2)

        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
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
        """
        Vertically stack disconnected components after semantic layout.

        Modifies self.positions in place and updates self.canvas_height.

        Args:
            gap: Vertical gap between stacked components
            max_components: Skip splitting if more than this many components
            min_component_size: Ignore components smaller than this

        Returns:
            Number of components found (0 if no split was applied)
        """
        if not self.positions:
            return 0

        # Find connected components (use undirected view for visual connectivity)
        if self.graph.is_directed():
            undirected = self.graph.to_undirected()
        else:
            undirected = self.graph

        # Get components and filter by size
        all_components = list(nx.connected_components(undirected))
        components = [
            comp for comp in all_components if len(comp) >= min_component_size
        ]

        # Skip if only 1 component or too many
        if len(components) <= 1:
            return len(components)
        if len(components) > max_components:
            return 0  # Signal no split applied due to too many

        # Filter to only positioned nodes
        positioned_components = []
        for comp in components:
            positioned_nodes = [n for n in comp if n in self.positions]
            if positioned_nodes:
                positioned_components.append(positioned_nodes)

        if len(positioned_components) <= 1:
            return len(positioned_components)

        # Compute bounding box for each component and sort by min-y
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

        # Sort by min-y (top-most component first)
        component_bounds.sort(key=lambda c: c["min_y"])

        # Stack components vertically
        current_y = self.style.canvas_padding
        for comp_info in component_bounds:
            # Calculate offset to move this component
            y_offset = current_y - comp_info["min_y"] + self.style.node_radius

            # Apply offset to all nodes in this component
            for node in comp_info["nodes"]:
                x, y = self.positions[node]
                self.positions[node] = (x, y + y_offset)

            # Move current_y to below this component
            current_y += comp_info["height"] + gap + self.style.node_radius * 2

        # Update canvas height
        self.canvas_height = current_y + self.style.canvas_padding

        return len(positioned_components)


# --- Layout helper functions ---


def layout_row(
    nodes: List[str],
    start_x: float,
    y: float,
    spacing: float,
    sort_key: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Position nodes in a horizontal row.

    Args:
        nodes: List of node IDs
        start_x: Starting X coordinate
        y: Y coordinate for all nodes
        spacing: Spacing between nodes
        sort_key: Optional function to sort nodes

    Returns:
        Dict mapping node ID to (x, y) position
    """
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
    """
    Position nodes in a vertical column.

    Args:
        nodes: List of node IDs
        x: X coordinate for all nodes
        start_y: Starting Y coordinate
        spacing: Spacing between nodes
        sort_key: Optional function to sort nodes

    Returns:
        Dict mapping node ID to (x, y) position
    """
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
    """
    Position nodes in a grid.

    Args:
        nodes: List of node IDs
        start_x: Starting X coordinate
        start_y: Starting Y coordinate
        cols: Number of columns
        spacing_x: Horizontal spacing
        spacing_y: Vertical spacing
        sort_key: Optional function to sort nodes

    Returns:
        Dict mapping node ID to (x, y) position
    """
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
    """
    Merge multiple position dictionaries.

    Args:
        layouts: Position dictionaries to merge

    Returns:
        Combined position dictionary
    """
    result = {}
    for layout in layouts:
        result.update(layout)
    return result
