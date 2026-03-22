"""Tests for _build_risk_groups in scenario_generator.py."""

from collections import Counter

import pytest

from netlab.autoresearch.scenario_generator import (
    DcBbScenarioConfig,
    _build_bb_cross_site_links,
    _build_dc_bb_links,
    _build_risk_groups,
)


@pytest.fixture
def default_config():
    return DcBbScenarioConfig()


@pytest.fixture
def default_groups(default_config):
    return _build_risk_groups(default_config)


# ---------------------------------------------------------------------------
# Total and category counts
# ---------------------------------------------------------------------------


class TestCounts:
    """Verify total and per-category risk group counts."""

    def test_total_count(self, default_groups):
        assert len(default_groups) == 274

    def test_path_group_count(self, default_groups):
        path_groups = [
            g for g in default_groups if g["attrs"]["type"] == "long_haul_path"
        ]
        assert len(path_groups) == 2

    def test_plane_group_count(self, default_groups):
        plane_groups = [
            g for g in default_groups if g["attrs"]["type"] == "plane_group"
        ]
        assert len(plane_groups) == 16

    def test_plane_site_count(self, default_groups):
        plane_site_groups = [
            g for g in default_groups if g["attrs"]["type"] == "plane_site"
        ]
        assert len(plane_site_groups) == 128

    def test_device_index_count(self, default_groups):
        dev_idx_groups = [
            g
            for g in default_groups
            if g["attrs"]["type"] == "device_index_across_planes"
        ]
        assert len(dev_idx_groups) == 128


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------


class TestUniqueness:
    """Verify all risk group names are unique."""

    def test_all_names_unique(self, default_groups):
        names = [g["name"] for g in default_groups]
        assert len(names) == len(set(names)), (
            f"Duplicate names: {[n for n, c in Counter(names).items() if c > 1]}"
        )


# ---------------------------------------------------------------------------
# Name format and type attributes
# ---------------------------------------------------------------------------


class TestPathGroups:
    """Verify path risk groups."""

    def test_path_a_exists(self, default_groups):
        path_a = [g for g in default_groups if g["name"] == "path_a"]
        assert len(path_a) == 1
        assert path_a[0]["attrs"]["type"] == "long_haul_path"

    def test_path_b_exists(self, default_groups):
        path_b = [g for g in default_groups if g["name"] == "path_b"]
        assert len(path_b) == 1
        assert path_b[0]["attrs"]["type"] == "long_haul_path"


class TestPlaneGroups:
    """Verify plane group risk groups."""

    def test_plane_group_names(self, default_groups):
        pg_names = {
            g["name"] for g in default_groups if g["attrs"]["type"] == "plane_group"
        }
        expected = {f"plane_group_{i}" for i in range(1, 17)}
        assert pg_names == expected

    def test_plane_group_1_planes(self, default_groups):
        pg1 = [g for g in default_groups if g["name"] == "plane_group_1"][0]
        assert pg1["attrs"]["planes"] == [1, 2, 3, 4]

    def test_plane_group_16_planes(self, default_groups):
        pg16 = [g for g in default_groups if g["name"] == "plane_group_16"][0]
        assert pg16["attrs"]["planes"] == [61, 62, 63, 64]

    def test_plane_group_boundary(self, default_groups):
        """Plane group 2 should cover planes 5-8."""
        pg2 = [g for g in default_groups if g["name"] == "plane_group_2"][0]
        assert pg2["attrs"]["planes"] == [5, 6, 7, 8]


class TestPlaneSiteGroups:
    """Verify plane-site risk groups."""

    def test_plane_site_names(self, default_groups):
        ps_names = {
            g["name"] for g in default_groups if g["attrs"]["type"] == "plane_site"
        }
        expected = set()
        for pl in range(1, 65):
            for site in ["abc1", "xyz1"]:
                expected.add(f"plane_{pl}_site_{site}")
        assert ps_names == expected

    def test_plane_site_attrs(self, default_groups):
        """Spot-check plane_1_site_abc1 attributes."""
        pg = [g for g in default_groups if g["name"] == "plane_1_site_abc1"][0]
        assert pg["attrs"]["type"] == "plane_site"
        assert pg["attrs"]["plane"] == 1
        assert pg["attrs"]["site"] == "abc1"

    def test_plane_site_64_xyz1(self, default_groups):
        pg = [g for g in default_groups if g["name"] == "plane_64_site_xyz1"][0]
        assert pg["attrs"]["plane"] == 64
        assert pg["attrs"]["site"] == "xyz1"


class TestDeviceIndexGroups:
    """Verify device-index-across-plane-group risk groups."""

    def test_device_index_names(self, default_groups):
        di_names = {
            g["name"]
            for g in default_groups
            if g["attrs"]["type"] == "device_index_across_planes"
        }
        expected = set()
        for g_idx in range(1, 17):
            for d in range(1, 5):
                for site in ["abc1", "xyz1"]:
                    expected.add(f"pg_{g_idx}_idx_{d}_{site}")
        assert di_names == expected

    def test_device_index_type(self, default_groups):
        di_groups = [
            g
            for g in default_groups
            if g["attrs"]["type"] == "device_index_across_planes"
        ]
        for grp in di_groups:
            assert grp["attrs"]["type"] == "device_index_across_planes"


# ---------------------------------------------------------------------------
# Consistency with _build_bb_cross_site_links
# ---------------------------------------------------------------------------


class TestConsistencyWithLinks:
    """Risk group names must match those referenced by _build_bb_cross_site_links."""

    def test_all_link_risk_groups_defined(self, default_config):
        """Every risk group referenced by BB links exists in the risk group list."""
        groups = _build_risk_groups(default_config)
        group_names = {g["name"] for g in groups}

        links = _build_bb_cross_site_links(default_config)
        referenced = set()
        for link in links:
            referenced.update(link["risk_groups"])

        missing = referenced - group_names
        assert not missing, (
            f"Risk groups referenced by links but not defined: {missing}"
        )

    def test_path_groups_in_links(self, default_config):
        """Both path_a and path_b are used by links."""
        links = _build_bb_cross_site_links(default_config)
        paths = {
            rg for link in links for rg in link["risk_groups"] if rg.startswith("path_")
        }
        assert "path_a" in paths
        assert "path_b" in paths

    def test_plane_group_names_in_links(self, default_config):
        """All 16 plane groups are referenced by links."""
        links = _build_bb_cross_site_links(default_config)
        pg_names = {
            rg
            for link in links
            for rg in link["risk_groups"]
            if rg.startswith("plane_group_")
        }
        groups = _build_risk_groups(default_config)
        defined_pg = {g["name"] for g in groups if g["attrs"]["type"] == "plane_group"}
        assert pg_names == defined_pg

    def test_plane_site_names_in_links(self, default_config):
        """All plane_site groups referenced by links are defined."""
        links = _build_bb_cross_site_links(default_config)
        ps_names = {
            rg
            for link in links
            for rg in link["risk_groups"]
            if rg.startswith("plane_") and "_site_" in rg
        }
        groups = _build_risk_groups(default_config)
        defined_ps = {g["name"] for g in groups if g["attrs"]["type"] == "plane_site"}
        assert ps_names.issubset(defined_ps)

    def test_device_index_groups_referenced_by_bb_links(self, default_config):
        """All 128 device_index_across_planes groups are referenced by BB cross-site links."""
        groups = _build_risk_groups(default_config)
        defined_di = {
            g["name"]
            for g in groups
            if g["attrs"]["type"] == "device_index_across_planes"
        }
        links = _build_bb_cross_site_links(default_config)
        referenced_di = {
            rg for link in links for rg in link["risk_groups"] if rg.startswith("pg_")
        }
        assert defined_di == referenced_di

    def test_device_index_groups_referenced_by_dc_bb_links(self, default_config):
        """DC-BB links reference device_index_across_planes groups."""
        groups = _build_risk_groups(default_config)
        defined_di = {
            g["name"]
            for g in groups
            if g["attrs"]["type"] == "device_index_across_planes"
        }
        links = _build_dc_bb_links(default_config)
        referenced_di = {
            rg for link in links for rg in link["risk_groups"] if rg.startswith("pg_")
        }
        # DC-BB links reference a subset (only one site per link)
        assert referenced_di.issubset(defined_di)
        assert len(referenced_di) > 0

    def test_all_dc_bb_link_risk_groups_defined(self, default_config):
        """Every risk group referenced by DC-BB links exists in the risk group list."""
        groups = _build_risk_groups(default_config)
        group_names = {g["name"] for g in groups}
        links = _build_dc_bb_links(default_config)
        referenced = set()
        for link in links:
            referenced.update(link.get("risk_groups", []))
        missing = referenced - group_names
        assert not missing, (
            f"Risk groups referenced by DC-BB links but not defined: {missing}"
        )


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    """Verify function respects non-default config values."""

    def test_fewer_planes(self):
        cfg = DcBbScenarioConfig(bb_planes=8, bb_devices_per_plane=2)
        groups = _build_risk_groups(cfg)
        # 2 path + 2 plane_group + 16 plane_site + 8 device_idx = 28
        path = [g for g in groups if g["attrs"]["type"] == "long_haul_path"]
        pg = [g for g in groups if g["attrs"]["type"] == "plane_group"]
        ps = [g for g in groups if g["attrs"]["type"] == "plane_site"]
        di = [g for g in groups if g["attrs"]["type"] == "device_index_across_planes"]
        assert len(path) == 2
        assert len(pg) == 2  # 8 // 4 = 2
        assert len(ps) == 16  # 8 * 2
        assert len(di) == 8  # 2 groups * 2 devices * 2 sites
        assert len(groups) == 28

    def test_single_plane_group(self):
        cfg = DcBbScenarioConfig(bb_planes=4, bb_devices_per_plane=1)
        groups = _build_risk_groups(cfg)
        # 2 + 1 + 8 + 2 = 13
        assert len(groups) == 13
        pg = [g for g in groups if g["attrs"]["type"] == "plane_group"]
        assert len(pg) == 1
        assert pg[0]["attrs"]["planes"] == [1, 2, 3, 4]
