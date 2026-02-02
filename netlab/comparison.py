"""
Comparison Table Module

Provides utilities for building and printing comparison tables across
multiple scenarios/topologies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class TableRow:
    """Definition of a row in the comparison table."""

    label: str
    extractor: Callable[[dict], Any]
    formatter: Optional[Callable[[Any], str]] = None
    default: str = "?"


@dataclass
class MetricRowGroup:
    """Definition of a group of metric rows."""

    metrics_key: str  # Key in summary containing metrics dict
    metric_field: str  # Field to extract from each metric (e.g., "min_ratio")
    label_map: Dict[str, str] = field(default_factory=dict)
    formatter: Optional[Callable[[Any], str]] = None
    default: str = "N/A"


class ComparisonTableBuilder:
    """
    Builder for comparison tables across scenarios.

    Example usage:
        builder = ComparisonTableBuilder(summaries)
        builder.add_row("Network", lambda s: f"{s['network']['node_count']}N")
        builder.add_row("Alpha*", lambda s: s.get("alpha_star"), format_float)
        builder.add_metric_rows("failure_analysis", "min_ratio",
                               label_map={"tm_dc_node": "DC Node"})
        builder.print_table(title="TOPOLOGY COMPARISON")
    """

    def __init__(self, summaries: Dict[str, dict]):
        """
        Initialize with scenario summaries.

        Args:
            summaries: Dict mapping scenario/topology name to summary dict
        """
        self.summaries = summaries
        self.scenarios = list(summaries.keys())
        self.rows: List[Union[TableRow, MetricRowGroup]] = []

    def add_row(
        self,
        label: str,
        extractor: Callable[[dict], Any],
        formatter: Optional[Callable[[Any], str]] = None,
        default: str = "?",
    ) -> "ComparisonTableBuilder":
        """
        Add a custom row to the table.

        Args:
            label: Row label (first column)
            extractor: Function to extract value from summary dict
            formatter: Optional function to format the value
            default: Default string if value is None

        Returns:
            self for chaining
        """
        self.rows.append(TableRow(label, extractor, formatter, default))
        return self

    def add_metric_rows(
        self,
        metrics_key: str,
        metric_field: str,
        label_map: Optional[Dict[str, str]] = None,
        formatter: Optional[Callable[[Any], str]] = None,
        default: str = "N/A",
    ) -> "ComparisonTableBuilder":
        """
        Add rows for all metrics in a key.

        Args:
            metrics_key: Key in summary containing metrics dict (e.g., "failure_analysis")
            metric_field: Field to extract from each metric (e.g., "min_ratio")
            label_map: Optional mapping from metric names to display labels
            formatter: Optional function to format values
            default: Default string if value is None

        Returns:
            self for chaining
        """
        self.rows.append(
            MetricRowGroup(
                metrics_key=metrics_key,
                metric_field=metric_field,
                label_map=label_map or {},
                formatter=formatter,
                default=default,
            )
        )
        return self

    def _get_all_metric_keys(self, metrics_key: str) -> List[str]:
        """Get all metric keys across all scenarios."""
        all_keys = set()
        for summary in self.summaries.values():
            metrics = summary.get(metrics_key, {})
            all_keys.update(metrics.keys())
        return sorted(all_keys)

    def _format_value(
        self,
        value: Any,
        formatter: Optional[Callable[[Any], str]],
        default: str,
    ) -> str:
        """Format a value for display."""
        if value is None:
            return default
        if formatter:
            return formatter(value)
        return str(value)

    def print_table(
        self,
        title: str = "COMPARISON",
        min_col_width: int = 12,
        label_width: int = 15,
        transposed: bool = False,
    ) -> None:
        """
        Print the comparison table.

        Args:
            title: Table title
            min_col_width: Minimum column width
            label_width: Width for the label column
            transposed: If True, scenarios are rows and metrics are columns
        """
        if not self.scenarios:
            print("No scenarios to compare.")
            return

        if transposed:
            self._print_table_transposed(title, min_col_width)
            return

        # Calculate column width
        col_width = max(min_col_width, max(len(s) for s in self.scenarios))
        total_width = label_width + 3 + len(self.scenarios) * (col_width + 3)

        # Print header
        print("\n" + "=" * total_width)
        print(title)
        print("=" * total_width)

        # Print scenario names header
        print(f"\n{'':>{label_width}}", end="")
        for scenario in self.scenarios:
            print(f"  {scenario:>{col_width}}", end="")
        print()
        print("-" * total_width)

        # Print rows
        for row in self.rows:
            if isinstance(row, TableRow):
                self._print_row(row, col_width, label_width)
            elif isinstance(row, MetricRowGroup):
                self._print_metric_rows(row, col_width, label_width)

    def _print_table_transposed(
        self,
        title: str,
        col_width: int = 10,
    ) -> None:
        """Print table with scenarios as rows, metrics as columns."""
        # Build column headers and data
        columns: List[tuple[str, Callable[[dict], str]]] = []

        for row in self.rows:
            if isinstance(row, TableRow):
                columns.append(
                    (
                        row.label,
                        lambda s, r=row: self._format_value(
                            r.extractor(s) if s else None, r.formatter, r.default
                        ),
                    )
                )
            elif isinstance(row, MetricRowGroup):
                metric_keys = self._get_all_metric_keys(row.metrics_key)
                for mk in metric_keys:
                    label = row.label_map.get(mk, mk)
                    columns.append(
                        (
                            label,
                            lambda s, g=row, k=mk: self._format_value(
                                s.get(g.metrics_key, {}).get(k, {}).get(g.metric_field)
                                if s
                                else None,
                                g.formatter,
                                g.default,
                            ),
                        )
                    )

        # Calculate widths
        scenario_width = max(len(s) for s in self.scenarios) + 2
        col_width = max(col_width, max(len(c[0]) for c in columns) + 2)
        total_width = scenario_width + len(columns) * col_width

        # Print header
        print(f"\n┌─ {title} ─" + "─" * (total_width - len(title) - 4))

        # Print column headers
        print(f"│ {'Topology':<{scenario_width - 2}}", end="")
        for col_name, _ in columns:
            print(f" {col_name:>{col_width - 1}}", end="")
        print()
        print("├" + "─" * (total_width + 1))

        # Print data rows
        for scenario in self.scenarios:
            summary = self.summaries[scenario]
            # Shorten scenario name if needed
            short_name = scenario
            if len(short_name) > scenario_width - 2:
                short_name = "…" + short_name[-(scenario_width - 3) :]
            print(f"│ {short_name:<{scenario_width - 2}}", end="")
            for _, extractor in columns:
                try:
                    value = extractor(summary)
                except (KeyError, TypeError):
                    value = "?"
                print(f" {value:>{col_width - 1}}", end="")
            print()

        print("└" + "─" * (total_width + 1))

    def _print_row(
        self,
        row: TableRow,
        col_width: int,
        label_width: int,
    ) -> None:
        """Print a single row."""
        print(f"{row.label:<{label_width}}", end="")
        for scenario in self.scenarios:
            summary = self.summaries[scenario]
            try:
                value = row.extractor(summary)
            except (KeyError, TypeError):
                value = None
            formatted = self._format_value(value, row.formatter, row.default)
            print(f"  {formatted:>{col_width}}", end="")
        print()

    def _print_metric_rows(
        self,
        group: MetricRowGroup,
        col_width: int,
        label_width: int,
    ) -> None:
        """Print rows for a metric group."""
        metric_keys = self._get_all_metric_keys(group.metrics_key)

        for metric_key in metric_keys:
            label = group.label_map.get(metric_key, metric_key)
            print(f"{label:<{label_width}}", end="")

            for scenario in self.scenarios:
                summary = self.summaries[scenario]
                metrics = summary.get(group.metrics_key, {})
                metric = metrics.get(metric_key, {})
                value = (
                    metric.get(group.metric_field) if isinstance(metric, dict) else None
                )
                formatted = self._format_value(value, group.formatter, group.default)
                print(f"  {formatted:>{col_width}}", end="")
            print()

    def to_dict(self) -> dict:
        """
        Export comparison data as a dictionary.

        Returns:
            Dict with scenarios, data per scenario, and metric types
        """
        # Collect all failure types
        all_failure_types = set()
        for summary in self.summaries.values():
            all_failure_types.update(summary.get("failure_analysis", {}).keys())

        comparison = {
            "scenarios": self.scenarios,
            "failure_types": sorted(all_failure_types),
            "data": {},
        }

        for scenario, summary in self.summaries.items():
            comparison["data"][scenario] = {
                "network": summary.get("network", {}),
                "alpha_star": summary.get("alpha_star"),
                "failures": {},
                "worst_failures": summary.get("worst_failures", {}),
            }
            for ft in comparison["failure_types"]:
                fa = summary.get("failure_analysis", {}).get(ft, {})
                comparison["data"][scenario]["failures"][ft] = {
                    "min": fa.get("min_ratio"),
                    "avg": fa.get("avg_ratio"),
                    "max": fa.get("max_ratio"),
                }

        return comparison

    def to_json(self, path: Optional[Path] = None) -> str:
        """
        Export comparison data as JSON.

        Args:
            path: Optional path to write JSON file

        Returns:
            JSON string
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(json_str)
        return json_str


# Common formatters
def format_percent(value: float) -> str:
    """Format a value as percentage."""
    return f"{value:.2%}"


def format_network_stats(summary: dict) -> str:
    """Format network stats as "NN/LL" string."""
    net = summary.get("network", {})
    nodes = net.get("node_count", "?")
    links = net.get("link_count", "?")
    return f"{nodes}N/{links}L"
