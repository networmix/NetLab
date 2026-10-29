"""Small DC-BB inputs for real research execution tests."""

from __future__ import annotations

from dataclasses import dataclass

from netlab.autoresearch.dcbb_config import DcBbScenarioConfig


@dataclass
class SmallDcBbScenarioConfig(DcBbScenarioConfig):
    """Minimum geometry, with two draws to exercise sampled outcomes."""

    abc1_hgrids: int = 1
    abc1_fadu_per_hgrid: int = 1
    abc1_planes: int = 1
    abc1_ssw_per_plane: int = 1
    abc1_pods_per_building: int = 1
    abc1_buildings: int = 1
    abc1_rsw_per_pod: int = 1
    xyz1_xsw_planes: int = 1
    xyz1_xsw_per_plane: int = 1
    xyz1_ssw_per_megapod: int = 1
    xyz1_fsw_per_megapod: int = 4
    xyz1_megapods: int = 1
    bb_planes: int = 4
    bb_devices_per_plane: int = 1
    g_abc1: int = 1
    g_xyz1: int = 1
    layout_abc1: tuple = (1, 1, 1, 1)
    layout_xyz1: tuple = (1, 1, 1, 1)
    failure_iterations: int = 2
    msd_resolution: float = 0.0001
