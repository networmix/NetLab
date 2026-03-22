"""Tests for _build_dc_bb_links in scenario_generator.py."""

from __future__ import annotations

import pytest

from netlab.autoresearch.scenario_generator import (
    DcBbScenarioConfig,
    _build_dc_bb_links,
    _build_nodes,
)


@pytest.fixture
def default_config():
    return DcBbScenarioConfig()


@pytest.fixture
def default_links(default_config):
    return _build_dc_bb_links(default_config)


# ---------------------------------------------------------------------------
# Link counts
# ---------------------------------------------------------------------------


class TestLinkCounts:
    """Verify total and per-side link counts with default config (G=64)."""

    def test_abc1_link_count(self, default_links):
        abc1 = [lk for lk in default_links if lk["attrs"]["side"] == "abc1"]
        assert len(abc1) == 2304

    def test_xyz1_link_count(self, default_links):
        xyz1 = [lk for lk in default_links if lk["attrs"]["side"] == "xyz1"]
        assert len(xyz1) == 6144

    def test_total_link_count(self, default_links):
        assert len(default_links) == 8448


# ---------------------------------------------------------------------------
# Per-device degree
# ---------------------------------------------------------------------------


class TestPerDeviceDegree:
    """Each FADU connects to exactly k_fadu BB devices; each XSW to k_xsw."""

    def test_fadu_degree(self, default_links):
        abc1 = [lk for lk in default_links if lk["attrs"]["side"] == "abc1"]
        # Count how many BB devices each FADU connects to
        fadu_targets: dict[str, set[str]] = {}
        for lk in abc1:
            fadu_targets.setdefault(lk["source"], set()).add(lk["target"])
        # 576 FADU, each with degree 4
        assert len(fadu_targets) == 576
        for fadu, targets in fadu_targets.items():
            assert len(targets) == 4, f"{fadu} has degree {len(targets)}, expected 4"

    def test_xsw_degree(self, default_links):
        xyz1 = [lk for lk in default_links if lk["attrs"]["side"] == "xyz1"]
        xsw_targets: dict[str, set[str]] = {}
        for lk in xyz1:
            xsw_targets.setdefault(lk["source"], set()).add(lk["target"])
        # 1536 XSW, each with degree 4
        assert len(xsw_targets) == 1536
        for xsw, targets in xsw_targets.items():
            assert len(targets) == 4, f"{xsw} has degree {len(targets)}, expected 4"


# ---------------------------------------------------------------------------
# Link attributes
# ---------------------------------------------------------------------------


class TestLinkAttributes:
    """Verify capacity, cost, and attrs on every link."""

    def test_capacity(self, default_links):
        for lk in default_links:
            assert lk["capacity"] == 400.0

    def test_cost(self, default_links):
        for lk in default_links:
            assert lk["cost"] == 5

    def test_link_type(self, default_links):
        for lk in default_links:
            assert lk["attrs"]["link_type"] == "dc_bb"

    def test_side_values(self, default_links):
        sides = {lk["attrs"]["side"] for lk in default_links}
        assert sides == {"abc1", "xyz1"}

    def test_all_links_have_risk_groups(self, default_links):
        for lk in default_links:
            assert "risk_groups" in lk, (
                f"Link {lk['source']}->{lk['target']} missing risk_groups"
            )
            assert len(lk["risk_groups"]) == 3


# ---------------------------------------------------------------------------
# Risk group assignments on DC-BB links
# ---------------------------------------------------------------------------


class TestDcBbRiskGroups:
    """Verify risk group assignments derived from BB endpoint."""

    def test_abc1_link_has_plane_site_group(self, default_links):
        """ABC1 DC-BB links include plane_P_site_abc1 from BB endpoint."""
        abc1 = [lk for lk in default_links if lk["attrs"]["side"] == "abc1"]
        for lk in abc1:
            # Extract plane from target name bb/abc1/plane{P}/dev{D}
            parts = lk["target"].split("/")
            plane = int(parts[2].replace("plane", ""))
            expected_rg = f"plane_{plane}_site_abc1"
            assert expected_rg in lk["risk_groups"], (
                f"{lk['target']}: expected {expected_rg} in {lk['risk_groups']}"
            )

    def test_xyz1_link_has_plane_site_group(self, default_links):
        """XYZ1 DC-BB links include plane_P_site_xyz1 from BB endpoint."""
        xyz1 = [lk for lk in default_links if lk["attrs"]["side"] == "xyz1"]
        for lk in xyz1:
            parts = lk["target"].split("/")
            plane = int(parts[2].replace("plane", ""))
            expected_rg = f"plane_{plane}_site_xyz1"
            assert expected_rg in lk["risk_groups"], (
                f"{lk['target']}: expected {expected_rg} in {lk['risk_groups']}"
            )

    def test_abc1_link_has_plane_group(self, default_links):
        """ABC1 DC-BB links include plane_group_G from BB endpoint's plane."""
        abc1 = [lk for lk in default_links if lk["attrs"]["side"] == "abc1"]
        for lk in abc1:
            parts = lk["target"].split("/")
            plane = int(parts[2].replace("plane", ""))
            pg = (plane - 1) // 4 + 1
            expected_rg = f"plane_group_{pg}"
            assert expected_rg in lk["risk_groups"]

    def test_abc1_link_has_device_index_group(self, default_links):
        """ABC1 DC-BB links include pg_G_idx_D_abc1 from BB endpoint."""
        abc1 = [lk for lk in default_links if lk["attrs"]["side"] == "abc1"]
        for lk in abc1:
            parts = lk["target"].split("/")
            plane = int(parts[2].replace("plane", ""))
            dev = int(parts[3].replace("dev", ""))
            pg = (plane - 1) // 4 + 1
            expected_rg = f"pg_{pg}_idx_{dev}_abc1"
            assert expected_rg in lk["risk_groups"]

    def test_xyz1_link_has_device_index_group(self, default_links):
        """XYZ1 DC-BB links include pg_G_idx_D_xyz1 from BB endpoint."""
        xyz1 = [lk for lk in default_links if lk["attrs"]["side"] == "xyz1"]
        for lk in xyz1:
            parts = lk["target"].split("/")
            plane = int(parts[2].replace("plane", ""))
            dev = int(parts[3].replace("dev", ""))
            pg = (plane - 1) // 4 + 1
            expected_rg = f"pg_{pg}_idx_{dev}_xyz1"
            assert expected_rg in lk["risk_groups"]

    def test_spot_check_specific_abc1_link(self, default_links):
        """Spot-check: FADU->bb/abc1/plane7/dev2 should have specific risk groups."""
        abc1 = [lk for lk in default_links if lk["attrs"]["side"] == "abc1"]
        target_links = [lk for lk in abc1 if lk["target"] == "bb/abc1/plane7/dev2"]
        assert len(target_links) > 0, "No links to bb/abc1/plane7/dev2"
        for lk in target_links:
            assert lk["risk_groups"] == [
                "plane_7_site_abc1",
                "plane_group_2",
                "pg_2_idx_2_abc1",
            ]

    def test_spot_check_specific_xyz1_link(self, default_links):
        """Spot-check: XSW->bb/xyz1/plane7/dev3 should have specific risk groups."""
        xyz1 = [lk for lk in default_links if lk["attrs"]["side"] == "xyz1"]
        target_links = [lk for lk in xyz1 if lk["target"] == "bb/xyz1/plane7/dev3"]
        assert len(target_links) > 0, "No links to bb/xyz1/plane7/dev3"
        for lk in target_links:
            assert lk["risk_groups"] == [
                "plane_7_site_xyz1",
                "plane_group_2",
                "pg_2_idx_3_xyz1",
            ]


# ---------------------------------------------------------------------------
# Node name validity
# ---------------------------------------------------------------------------


class TestNodeNameValidity:
    """All source/target names match nodes produced by _build_nodes."""

    def test_all_sources_are_valid_nodes(self, default_config, default_links):
        nodes = _build_nodes(default_config)
        for lk in default_links:
            assert lk["source"] in nodes, f"source {lk['source']} not in nodes"

    def test_all_targets_are_valid_nodes(self, default_config, default_links):
        nodes = _build_nodes(default_config)
        for lk in default_links:
            assert lk["target"] in nodes, f"target {lk['target']} not in nodes"

    def test_abc1_source_names_are_fadu(self, default_links):
        abc1 = [lk for lk in default_links if lk["attrs"]["side"] == "abc1"]
        for lk in abc1:
            assert lk["source"].startswith("abc1/fadu/")

    def test_abc1_target_names_are_bb(self, default_links):
        abc1 = [lk for lk in default_links if lk["attrs"]["side"] == "abc1"]
        for lk in abc1:
            assert lk["target"].startswith("bb/abc1/")

    def test_xyz1_source_names_are_xsw(self, default_links):
        xyz1 = [lk for lk in default_links if lk["attrs"]["side"] == "xyz1"]
        for lk in xyz1:
            assert lk["source"].startswith("xyz1/xsw/")

    def test_xyz1_target_names_are_bb(self, default_links):
        xyz1 = [lk for lk in default_links if lk["attrs"]["side"] == "xyz1"]
        for lk in xyz1:
            assert lk["target"].startswith("bb/xyz1/")


# ---------------------------------------------------------------------------
# Different G values
# ---------------------------------------------------------------------------


class TestDifferentGValues:
    """Different G values produce different link counts."""

    def test_g_abc1_16_produces_9216_links(self):
        cfg = DcBbScenarioConfig(g_abc1=16, layout_abc1=(4, 4, 16, 1))
        links = _build_dc_bb_links(cfg)
        abc1 = [lk for lk in links if lk["attrs"]["side"] == "abc1"]
        assert len(abc1) == 9216

    def test_g_abc1_32_produces_4608_links(self):
        cfg = DcBbScenarioConfig(g_abc1=32, layout_abc1=(8, 4, 8, 4))
        links = _build_dc_bb_links(cfg)
        abc1 = [lk for lk in links if lk["attrs"]["side"] == "abc1"]
        assert len(abc1) == 4608

    def test_g_abc1_16_degree_is_16(self):
        cfg = DcBbScenarioConfig(g_abc1=16, layout_abc1=(4, 4, 16, 1))
        links = _build_dc_bb_links(cfg)
        abc1 = [lk for lk in links if lk["attrs"]["side"] == "abc1"]
        fadu_targets: dict[str, set[str]] = {}
        for lk in abc1:
            fadu_targets.setdefault(lk["source"], set()).add(lk["target"])
        for fadu, targets in fadu_targets.items():
            assert len(targets) == 16, f"{fadu} has degree {len(targets)}"

    def test_g_xyz1_128_produces_3072_links(self):
        # XSW grid is 64x24 for mesh groups; layout must divide those dims.
        cfg = DcBbScenarioConfig(g_xyz1=128, layout_xyz1=(16, 8, 32, 4))
        links = _build_dc_bb_links(cfg)
        xyz1 = [lk for lk in links if lk["attrs"]["side"] == "xyz1"]
        # 1536 XSW * (256/128) = 1536 * 2 = 3072
        assert len(xyz1) == 3072

    def test_g_xyz1_256_produces_1536_links(self):
        # XSW grid is 64x24 for mesh groups; layout must divide those dims.
        cfg = DcBbScenarioConfig(g_xyz1=256, layout_xyz1=(64, 4, 64, 4))
        links = _build_dc_bb_links(cfg)
        xyz1 = [lk for lk in links if lk["attrs"]["side"] == "xyz1"]
        # 1536 XSW * (256/256) = 1536 * 1 = 1536
        assert len(xyz1) == 1536


# ---------------------------------------------------------------------------
# Port constraint assertions
# ---------------------------------------------------------------------------


class TestPortConstraints:
    """Port constraint assertions fire on invalid G values."""

    def test_fadu_port_violation(self):
        # G_abc1=8 would give k_fadu = 256/8 = 32 > 16
        with pytest.raises(AssertionError, match="k_fadu=32 exceeds 16-port limit"):
            _build_dc_bb_links(DcBbScenarioConfig(g_abc1=8, layout_abc1=(2, 4, 8, 1)))

    def test_xsw_port_violation(self):
        # G_xyz1=32 would give k_xsw = 256/32 = 8 > 4
        with pytest.raises(AssertionError, match="k_xsw=8 exceeds 4-port limit"):
            _build_dc_bb_links(DcBbScenarioConfig(g_xyz1=32, layout_xyz1=(8, 4, 8, 4)))


# ---------------------------------------------------------------------------
# Custom capacity
# ---------------------------------------------------------------------------


class TestCustomCapacity:
    """Link capacity follows config.dc_bb_link_capacity."""

    def test_custom_capacity(self):
        cfg = DcBbScenarioConfig(dc_bb_link_capacity=200.0)
        links = _build_dc_bb_links(cfg)
        for lk in links:
            assert lk["capacity"] == 200.0


# ---------------------------------------------------------------------------
# No duplicate links
# ---------------------------------------------------------------------------


class TestNoDuplicates:
    """No duplicate (source, target) pairs within each side."""

    def test_no_duplicate_abc1_links(self, default_links):
        abc1 = [lk for lk in default_links if lk["attrs"]["side"] == "abc1"]
        pairs = [(lk["source"], lk["target"]) for lk in abc1]
        assert len(pairs) == len(set(pairs)), "Duplicate ABC1 links found"

    def test_no_duplicate_xyz1_links(self, default_links):
        xyz1 = [lk for lk in default_links if lk["attrs"]["side"] == "xyz1"]
        pairs = [(lk["source"], lk["target"]) for lk in xyz1]
        assert len(pairs) == len(set(pairs)), "Duplicate XYZ1 links found"
