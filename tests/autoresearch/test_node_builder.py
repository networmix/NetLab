"""Tests for _build_nodes in scenario_generator.py."""

import pytest

from netlab.autoresearch.scenario_generator import (
    DcBbScenarioConfig,
    _build_nodes,
)


@pytest.fixture
def default_nodes():
    """Nodes built from default config."""
    return _build_nodes(DcBbScenarioConfig())


# ---------------------------------------------------------------------------
# Layer counts
# ---------------------------------------------------------------------------


class TestNodeCounts:
    """Verify node counts per layer and total."""

    def test_total_count(self, default_nodes):
        assert len(default_nodes) == 3833

    def test_abc1_rsw_count(self, default_nodes):
        rsw = [
            n for n in default_nodes if n.startswith("abc1/pod") and n.endswith("/rsw")
        ]
        assert len(rsw) == 96

    def test_abc1_fsw_count(self, default_nodes):
        fsw = [n for n in default_nodes if n.startswith("abc1/pod") and "/fsw/" in n]
        assert len(fsw) == 768

    def test_abc1_ssw_count(self, default_nodes):
        ssw = [n for n in default_nodes if n.startswith("abc1/ssw/")]
        assert len(ssw) == 288

    def test_abc1_fadu_count(self, default_nodes):
        fadu = [n for n in default_nodes if n.startswith("abc1/fadu/")]
        assert len(fadu) == 576

    def test_bb_abc1_count(self, default_nodes):
        bb_abc1 = [n for n in default_nodes if n.startswith("bb/abc1/")]
        assert len(bb_abc1) == 256

    def test_bb_xyz1_count(self, default_nodes):
        bb_xyz1 = [n for n in default_nodes if n.startswith("bb/xyz1/")]
        assert len(bb_xyz1) == 256

    def test_bb_total_count(self, default_nodes):
        bb = [n for n in default_nodes if n.startswith("bb/")]
        assert len(bb) == 512

    def test_xyz1_rsw_count(self, default_nodes):
        rsw = [n for n in default_nodes if n.startswith("xyz1/mp1/rsw")]
        assert len(rsw) == 1

    def test_xyz1_fsw_count(self, default_nodes):
        fsw = [n for n in default_nodes if n.startswith("xyz1/mp1/fsw/")]
        assert len(fsw) == 32

    def test_xyz1_ssw_count(self, default_nodes):
        ssw = [n for n in default_nodes if n.startswith("xyz1/mp1/ssw/")]
        assert len(ssw) == 24

    def test_xsw_count(self, default_nodes):
        xsw = [n for n in default_nodes if n.startswith("xyz1/xsw/")]
        assert len(xsw) == 1536


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


class TestNodeAttributes:
    """Verify every node has required attributes and specific nodes are correct."""

    def test_all_nodes_have_role_and_site(self, default_nodes):
        for name, data in default_nodes.items():
            attrs = data["attrs"]
            assert "role" in attrs, f"{name} missing 'role'"
            assert "site" in attrs, f"{name} missing 'site'"

    def test_abc1_rsw_attrs(self, default_nodes):
        node = default_nodes["abc1/pod1/rsw"]
        assert node["attrs"]["role"] == "rsw"
        assert node["attrs"]["site"] == "abc1"

    def test_abc1_fsw_attrs(self, default_nodes):
        node = default_nodes["abc1/pod5/fsw/plane3"]
        attrs = node["attrs"]
        assert attrs["role"] == "fsw"
        assert attrs["site"] == "abc1"
        assert attrs["plane"] == 3
        assert attrs["pod"] == 5

    def test_abc1_ssw_attrs(self, default_nodes):
        node = default_nodes["abc1/ssw/plane8/idx36"]
        attrs = node["attrs"]
        assert attrs["role"] == "ssw"
        assert attrs["site"] == "abc1"
        assert attrs["plane"] == 8
        assert attrs["index"] == 36

    def test_abc1_fadu_attrs(self, default_nodes):
        node = default_nodes["abc1/fadu/hgrid16/idx1"]
        attrs = node["attrs"]
        assert attrs["role"] == "fadu"
        assert attrs["site"] == "abc1"
        assert attrs["hgrid"] == 16
        assert attrs["index"] == 1

    def test_bb_abc1_attrs(self, default_nodes):
        node = default_nodes["bb/abc1/plane64/dev4"]
        attrs = node["attrs"]
        assert attrs["role"] == "bb"
        assert attrs["site"] == "abc1"
        assert attrs["plane"] == 64
        assert attrs["device"] == 4

    def test_bb_xyz1_attrs(self, default_nodes):
        node = default_nodes["bb/xyz1/plane1/dev1"]
        attrs = node["attrs"]
        assert attrs["role"] == "bb"
        assert attrs["site"] == "xyz1"
        assert attrs["plane"] == 1
        assert attrs["device"] == 1

    def test_xyz1_rsw_attrs(self, default_nodes):
        node = default_nodes["xyz1/mp1/rsw"]
        assert node["attrs"]["role"] == "rsw"
        assert node["attrs"]["site"] == "xyz1"

    def test_xyz1_fsw_attrs(self, default_nodes):
        node = default_nodes["xyz1/mp1/fsw/row4/dev8"]
        attrs = node["attrs"]
        assert attrs["role"] == "fsw"
        assert attrs["site"] == "xyz1"
        assert attrs["row"] == 4
        assert attrs["device"] == 8

    def test_xyz1_ssw_attrs(self, default_nodes):
        node = default_nodes["xyz1/mp1/ssw/plane24"]
        attrs = node["attrs"]
        assert attrs["role"] == "ssw"
        assert attrs["site"] == "xyz1"
        assert attrs["plane"] == 24

    def test_xsw_attrs(self, default_nodes):
        node = default_nodes["xyz1/xsw/plane24/dev64"]
        attrs = node["attrs"]
        assert attrs["role"] == "xsw"
        assert attrs["site"] == "xyz1"
        assert attrs["plane"] == 24
        assert attrs["device"] == 64


# ---------------------------------------------------------------------------
# Naming patterns
# ---------------------------------------------------------------------------


class TestNamingPatterns:
    """Spot-check that specific expected node names exist."""

    def test_abc1_rsw_first_and_last(self, default_nodes):
        assert "abc1/pod1/rsw" in default_nodes
        assert "abc1/pod96/rsw" in default_nodes

    def test_abc1_fsw_boundary(self, default_nodes):
        assert "abc1/pod1/fsw/plane1" in default_nodes
        assert "abc1/pod96/fsw/plane8" in default_nodes

    def test_abc1_ssw_boundary(self, default_nodes):
        assert "abc1/ssw/plane1/idx1" in default_nodes
        assert "abc1/ssw/plane8/idx36" in default_nodes

    def test_abc1_fadu_boundary(self, default_nodes):
        assert "abc1/fadu/hgrid1/idx1" in default_nodes
        assert "abc1/fadu/hgrid16/idx36" in default_nodes

    def test_bb_boundary(self, default_nodes):
        assert "bb/abc1/plane1/dev1" in default_nodes
        assert "bb/abc1/plane64/dev4" in default_nodes
        assert "bb/xyz1/plane1/dev1" in default_nodes
        assert "bb/xyz1/plane64/dev4" in default_nodes

    def test_xyz1_fsw_boundary(self, default_nodes):
        assert "xyz1/mp1/fsw/row1/dev1" in default_nodes
        assert "xyz1/mp1/fsw/row4/dev8" in default_nodes

    def test_xyz1_ssw_boundary(self, default_nodes):
        assert "xyz1/mp1/ssw/plane1" in default_nodes
        assert "xyz1/mp1/ssw/plane24" in default_nodes

    def test_xsw_boundary(self, default_nodes):
        assert "xyz1/xsw/plane1/dev1" in default_nodes
        assert "xyz1/xsw/plane24/dev64" in default_nodes

    def test_no_zero_indices(self, default_nodes):
        """All indices start at 1, never 0."""
        for name in default_nodes:
            # Check that no path segment ends with "0" preceded by a keyword
            for segment in name.split("/"):
                for prefix in ("pod", "plane", "idx", "hgrid", "dev", "row"):
                    if segment.startswith(prefix):
                        idx_str = segment[len(prefix) :]
                        if idx_str.isdigit():
                            assert int(idx_str) >= 1, f"Zero-based index in {name}"


# ---------------------------------------------------------------------------
# Roles distribution
# ---------------------------------------------------------------------------


class TestRolesDistribution:
    """Verify role attribute values match expected sets."""

    def test_all_roles(self, default_nodes):
        roles = {data["attrs"]["role"] for data in default_nodes.values()}
        assert roles == {"rsw", "fsw", "ssw", "fadu", "bb", "xsw"}

    def test_abc1_roles(self, default_nodes):
        abc1_roles = {
            data["attrs"]["role"]
            for name, data in default_nodes.items()
            if data["attrs"]["site"] == "abc1" and not name.startswith("bb/")
        }
        assert abc1_roles == {"rsw", "fsw", "ssw", "fadu"}

    def test_xyz1_roles(self, default_nodes):
        xyz1_roles = {
            data["attrs"]["role"]
            for name, data in default_nodes.items()
            if data["attrs"]["site"] == "xyz1" and not name.startswith("bb/")
        }
        assert xyz1_roles == {"rsw", "fsw", "ssw", "xsw"}


# ---------------------------------------------------------------------------
# Parameterized config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    """Verify _build_nodes respects non-default config values."""

    def test_fewer_pods(self):
        cfg = DcBbScenarioConfig(abc1_pods_per_building=10)
        nodes = _build_nodes(cfg)
        rsw = [n for n in nodes if n.startswith("abc1/pod") and n.endswith("/rsw")]
        assert len(rsw) == 10
        fsw = [n for n in nodes if n.startswith("abc1/pod") and "/fsw/" in n]
        assert len(fsw) == 10 * 8  # 10 pods * 8 planes

    def test_fewer_bb_planes(self):
        cfg = DcBbScenarioConfig(bb_planes=8)
        nodes = _build_nodes(cfg)
        bb = [n for n in nodes if n.startswith("bb/")]
        assert len(bb) == 8 * 4 * 2  # 8 planes * 4 devs * 2 sites

    def test_fewer_xsw_planes(self):
        cfg = DcBbScenarioConfig(xyz1_xsw_planes=4, xyz1_ssw_per_megapod=4)
        nodes = _build_nodes(cfg)
        xsw = [n for n in nodes if n.startswith("xyz1/xsw/")]
        assert len(xsw) == 4 * 64  # 4 planes * 64 devs
        ssw = [n for n in nodes if n.startswith("xyz1/mp1/ssw/")]
        assert len(ssw) == 4

    def test_unique_node_names(self, default_nodes):
        """Node names are unique (dict keys guarantee this, but verify count)."""
        # Build again as a list to check for duplicates in generation logic
        cfg = DcBbScenarioConfig()
        nodes = _build_nodes(cfg)
        assert len(nodes) == 3833
