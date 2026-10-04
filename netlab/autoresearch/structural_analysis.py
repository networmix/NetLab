"""Evaluate DC-BB mesh layouts against connection-retention rules.

Enumerate valid group counts and layouts, then compute worst-case connection
loss from their geometry. This analysis does not run simulations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from netlab.artifacts import write_json_atomic
from netlab.autoresearch.dcbb_config import (
    DcBbScenarioConfig,
    get_valid_layouts,
    get_viable_g_values,
)

# Failure types that structural analysis can evaluate.
# Long-haul path failures are excluded — they always remove 50% cross-site
# capacity regardless of layout.
FAILURE_TYPES = (
    "plane_site",
    "plane_group",
    "device_index_across_pg",
    "single_bb_device",
)


@dataclass
class FailureFingerprint:
    """Worst-case per-device capacity loss fraction for each failure type."""

    plane_site: float = 0.0
    plane_group: float = 0.0
    device_index_across_pg: float = 0.0
    single_bb_device: float = 0.0

    @property
    def worst_feasibility(self) -> float:
        """Maximum loss across plane-site, device-index, and single-device failures.

        Plane-group failures are excluded from the feasibility rule.
        """
        return max(
            self.plane_site,
            self.device_index_across_pg,
            self.single_bb_device,
        )

    @property
    def best_retention(self) -> float:
        """Minimum surviving fraction for feasibility-relevant failure types."""
        return 1.0 - self.worst_feasibility


@dataclass
class ConfigResult:
    """Structural analysis result for one (side, G, layout) configuration."""

    side: str
    g: int
    layout: tuple[int, int, int, int]
    bb_block_rows: int
    bb_block_cols: int
    k_dc: int  # BB connections per DC device
    fingerprint: FailureFingerprint
    feasible: bool
    rule1_pass: bool  # Every device keeps a BB connection.
    rule2_pass: bool  # Meets the configured retention threshold.
    notation: str  # ArxBc <> CrxDc block shape string

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layout"] = list(d["layout"])
        return d


@dataclass
class StructuralAnalysisResult:
    """Structural analysis results for one DC side."""

    side: str
    dc_rows: int
    dc_cols: int
    bb_rows: int
    bb_cols: int
    dc_ports: int
    configs: list[ConfigResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "dc_rows": self.dc_rows,
            "dc_cols": self.dc_cols,
            "bb_rows": self.bb_rows,
            "bb_cols": self.bb_cols,
            "dc_ports": self.dc_ports,
            "configs": [c.to_dict() for c in self.configs],
        }


def _compute_failure_fingerprint(
    bb_block_rows: int,
    bb_block_cols: int,
    bb_planes: int,
    bb_devices_per_plane: int,
    planes_per_group: int = 4,
) -> FailureFingerprint:
    """Compute worst-case capacity loss fractions from BB block geometry.

    BB grid: bb_planes rows x bb_devices_per_plane cols.
    BB block: bb_block_rows consecutive planes x bb_block_cols consecutive devices.
    k = bb_block_rows * bb_block_cols  (BB connections per DC device per group).

    Failure types and what they kill within a group's BB block:
    - plane_site: all devices in one plane at one site → bb_block_cols devices
    - plane_group: all devices in planes_per_group consecutive planes →
      min(bb_block_rows, planes_per_group) * bb_block_cols devices
    - device_index_across_pg: one device index across planes_per_group planes →
      min(bb_block_rows, planes_per_group) devices (one per plane, same col)
    - single_bb_device: exactly 1 device
    """
    k = bb_block_rows * bb_block_cols
    if k == 0:
        return FailureFingerprint()

    plane_site_loss = bb_block_cols / k

    pg_affected_planes = min(bb_block_rows, planes_per_group)
    plane_group_loss = (pg_affected_planes * bb_block_cols) / k

    dev_idx_affected = min(bb_block_rows, planes_per_group)
    device_index_loss = dev_idx_affected / k

    single_device_loss = 1 / k

    return FailureFingerprint(
        plane_site=plane_site_loss,
        plane_group=plane_group_loss,
        device_index_across_pg=device_index_loss,
        single_bb_device=single_device_loss,
    )


def _block_notation(
    dc_block_rows: int,
    dc_block_cols: int,
    bb_block_rows: int,
    bb_block_cols: int,
) -> str:
    """Format block shape as ArxBc <> CrxDc notation."""
    return f"{dc_block_rows}rx{dc_block_cols}c <> {bb_block_rows}rx{bb_block_cols}c"


def analyze_side(
    side: str,
    dc_rows: int,
    dc_cols: int,
    bb_rows: int,
    bb_cols: int,
    dc_ports: int,
    bb_ports: int,
    planes_per_group: int = 4,
    retention_threshold: float = 0.75,
) -> StructuralAnalysisResult:
    """Enumerate layouts and check per-device connection retention after failures.

    Rows and columns describe the DC and BB grids; port limits constrain mesh
    group sizes. ``retention_threshold`` is the minimum surviving connection
    fraction for plane-site, device-index, and single-device failures.
    """
    dc_total = dc_rows * dc_cols
    result = StructuralAnalysisResult(
        side=side,
        dc_rows=dc_rows,
        dc_cols=dc_cols,
        bb_rows=bb_rows,
        bb_cols=bb_cols,
        dc_ports=dc_ports,
    )

    viable_g = get_viable_g_values(dc_total, bb_rows * bb_cols, dc_ports, bb_ports)

    for g in sorted(viable_g, reverse=True):  # largest G first
        layouts = get_valid_layouts(g, dc_rows, dc_cols, bb_rows, bb_cols)
        k_dc = (bb_rows * bb_cols) // g

        for layout in layouts:
            gr_dc, gc_dc, gr_bb, gc_bb = layout
            dc_block_rows = dc_rows // gr_dc
            dc_block_cols = dc_cols // gc_dc
            bb_block_rows = bb_rows // gr_bb
            bb_block_cols = bb_cols // gc_bb

            fp = _compute_failure_fingerprint(
                bb_block_rows=bb_block_rows,
                bb_block_cols=bb_block_cols,
                bb_planes=bb_rows,
                bb_devices_per_plane=bb_cols,
                planes_per_group=planes_per_group,
            )

            rule1 = (
                fp.worst_feasibility < 1.0
            )  # Every device keeps a BB connection. under feasibility-relevant failures
            rule2 = fp.best_retention >= retention_threshold

            notation = _block_notation(
                dc_block_rows, dc_block_cols, bb_block_rows, bb_block_cols
            )

            result.configs.append(
                ConfigResult(
                    side=side,
                    g=g,
                    layout=layout,
                    bb_block_rows=bb_block_rows,
                    bb_block_cols=bb_block_cols,
                    k_dc=k_dc,
                    fingerprint=fp,
                    feasible=rule1 and rule2,
                    rule1_pass=rule1,
                    rule2_pass=rule2,
                    notation=notation,
                )
            )

    return result


def run_structural_analysis(
    config: DcBbScenarioConfig | None = None,
) -> dict[str, StructuralAnalysisResult]:
    """Analyze both DC sides using the supplied configuration or its defaults."""
    if config is None:
        config = DcBbScenarioConfig()

    abc1 = analyze_side(
        side="abc1",
        dc_rows=config.abc1_hgrids,
        dc_cols=config.abc1_fadu_per_hgrid,
        bb_rows=config.bb_planes,
        bb_cols=config.bb_devices_per_plane,
        dc_ports=16,  # FADU has 16 BB-facing ports
        bb_ports=config.abc1_fadu_per_hgrid,  # no practical limit on BB side
    )

    xyz1 = analyze_side(
        side="xyz1",
        dc_rows=config.xyz1_xsw_per_plane,
        dc_cols=config.xyz1_xsw_planes,
        bb_rows=config.bb_planes,
        bb_cols=config.bb_devices_per_plane,
        dc_ports=4,  # XSW has 4 BB-facing ports
        bb_ports=config.xyz1_xsw_per_plane,  # no practical limit on BB side
    )

    return {"abc1": abc1, "xyz1": xyz1}


def save_results(results: dict[str, StructuralAnalysisResult], path: Path) -> None:
    """Save structural analysis results to JSON."""
    data = {side: r.to_dict() for side, r in results.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, data)


def print_summary(results: dict[str, StructuralAnalysisResult]) -> None:
    """Print a human-readable summary table."""
    for side, analysis in results.items():
        feasible = [c for c in analysis.configs if c.feasible]
        total = len(analysis.configs)
        print(f"\n{'=' * 70}")
        print(
            f"  {side.upper()}  ({analysis.dc_rows}x{analysis.dc_cols} DC, "
            f"{analysis.bb_rows}x{analysis.bb_cols} BB)"
        )
        print(f"  {total} configs, {len(feasible)} feasible")
        print(f"{'=' * 70}")
        print(
            f"  {'G':>4}  {'k_dc':>4}  {'Notation':<25}  "
            f"{'PS':>5}  {'PG':>5}  {'DI':>5}  {'1D':>5}  {'Pass':>4}"
        )
        print(
            f"  {'-' * 4}  {'-' * 4}  {'-' * 25}  "
            f"{'-' * 5}  {'-' * 5}  {'-' * 5}  {'-' * 5}  {'-' * 4}"
        )
        for c in analysis.configs:
            fp = c.fingerprint
            mark = " OK " if c.feasible else "FAIL"
            print(
                f"  {c.g:>4}  {c.k_dc:>4}  {c.notation:<25}  "
                f"{fp.plane_site:>5.0%}  {fp.plane_group:>5.0%}  "
                f"{fp.device_index_across_pg:>5.0%}  "
                f"{fp.single_bb_device:>5.0%}  {mark}"
            )
