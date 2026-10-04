"""SVG graph rendering keeps every positioned component inside the canvas."""

import xml.etree.ElementTree as ET

import networkx as nx

from netlab.visualize import GraphVisualizer


class TinyVisualizer(GraphVisualizer):
    def layout(self):
        self.positions = {
            "A": (20, 20),
            "B": (40, 20),
            "C": (20, 50),
            "D": (40, 50),
            "isolated": (900, 800),
        }
        self.canvas_width, self.canvas_height = 1000, 900
        return self.positions


def test_split_svg_keeps_isolated_nodes_and_draws_each_edge_once(tmp_path):
    graph = nx.Graph([("A", "B"), ("C", "D")])
    graph.add_node("isolated")
    visualizer = TinyVisualizer(graph)
    output = tmp_path / "nested/graph.svg"
    assert visualizer.render_svg_split(output, gap=20, png=False) == output
    root = ET.parse(output).getroot()
    namespace = {"s": "http://www.w3.org/2000/svg"}
    links = root.find("s:g[@id='links']", namespace)
    assert links is not None and len(links) == 2
    assert float(root.attrib["width"]) > 900
    assert float(root.attrib["height"]) > 800
    nodes = root.find("s:g[@id='nodes']", namespace)
    assert nodes is not None and len(nodes) == 5
