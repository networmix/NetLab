"""Tests for _build_internal_dc_links in scenario_generator.py."""

import pytest

from netlab.autoresearch.scenario_generator import (
    DcBbScenarioConfig,
    _build_internal_dc_links,
    _build_nodes,
)


@pytest.fixture
def default_links():
    """Links built from default config."""
    return _build_internal_dc_links(DcBbScenarioConfig())


@pytest.fixture
def default_nodes():
    """Nodes built from default config (for cross-referencing)."""
    return _build_nodes(DcBbScenarioConfig())


def _links_by_type(links, link_type):
    return [lk for lk in links if lk["attrs"]["link_type"] == link_type]


def _links_by_site(links, site):
    return [lk for lk in links if lk["attrs"]["site"] == site]


# ---------------------------------------------------------------------------
# Link counts
# ---------------------------------------------------------------------------


class TestLinkCounts:
    """Verify link counts match acceptance criteria."""

    def test_abc1_rsw_fsw_count(self, default_links):
        abc1_rsw_fsw = [
            lk
            for lk in default_links
            if lk["attrs"]["link_type"] == "rsw_fsw" and lk["attrs"]["site"] == "abc1"
        ]
        assert len(abc1_rsw_fsw) == 768

    def test_abc1_fsw_ssw_count(self, default_links):
        abc1_fsw_ssw = [
            lk
            for lk in default_links
            if lk["attrs"]["link_type"] == "fsw_ssw" and lk["attrs"]["site"] == "abc1"
        ]
        assert len(abc1_fsw_ssw) == 27_648

    def test_abc1_ssw_fadu_count(self, default_links):
        abc1_ssw_fadu = [
            lk
            for lk in default_links
            if lk["attrs"]["link_type"] == "ssw_fadu" and lk["attrs"]["site"] == "abc1"
        ]
        assert len(abc1_ssw_fadu) == 4_608

    def test_abc1_total(self, default_links):
        abc1 = _links_by_site(default_links, "abc1")
        assert len(abc1) == 33_024

    def test_xyz1_rsw_fsw_count(self, default_links):
        xyz1_rsw_fsw = [
            lk
            for lk in default_links
            if lk["attrs"]["link_type"] == "rsw_fsw" and lk["attrs"]["site"] == "xyz1"
        ]
        assert len(xyz1_rsw_fsw) == 32

    def test_xyz1_fsw_ssw_count(self, default_links):
        xyz1_fsw_ssw = [
            lk
            for lk in default_links
            if lk["attrs"]["link_type"] == "fsw_ssw" and lk["attrs"]["site"] == "xyz1"
        ]
        assert len(xyz1_fsw_ssw) == 768

    def test_xyz1_ssw_xsw_count(self, default_links):
        xyz1_ssw_xsw = [
            lk
            for lk in default_links
            if lk["attrs"]["link_type"] == "ssw_xsw" and lk["attrs"]["site"] == "xyz1"
        ]
        assert len(xyz1_ssw_xsw) == 1_536

    def test_xyz1_total(self, default_links):
        xyz1 = _links_by_site(default_links, "xyz1")
        assert len(xyz1) == 2_336

    def test_grand_total(self, default_links):
        assert len(default_links) == 35_360


# ---------------------------------------------------------------------------
# Capacities
# ---------------------------------------------------------------------------


class TestCapacities:
    """Verify capacities include scaling factors from config."""

    def test_abc1_rsw_fsw_capacity(self, default_links):
        # 48 RSW × 200G × 5 buildings = 48000
        link = next(
            lk
            for lk in default_links
            if lk["attrs"]["link_type"] == "rsw_fsw" and lk["attrs"]["site"] == "abc1"
        )
        assert link["capacity"] == 48_000.0

    def test_abc1_fsw_ssw_capacity(self, default_links):
        # 200G × 5 buildings = 1000
        link = next(
            lk
            for lk in default_links
            if lk["attrs"]["link_type"] == "fsw_ssw" and lk["attrs"]["site"] == "abc1"
        )
        assert link["capacity"] == 1_000.0

    def test_abc1_ssw_fadu_capacity(self, default_links):
        # 2 × 200G × 5 buildings = 2000
        link = next(
            lk
            for lk in default_links
            if lk["attrs"]["link_type"] == "ssw_fadu" and lk["attrs"]["site"] == "abc1"
        )
        assert link["capacity"] == 2_000.0

    def test_xyz1_rsw_fsw_capacity(self, default_links):
        # 400G × 72 megapods = 28800
        link = next(
            lk
            for lk in default_links
            if lk["attrs"]["link_type"] == "rsw_fsw" and lk["attrs"]["site"] == "xyz1"
        )
        assert link["capacity"] == 28_800.0

    def test_xyz1_fsw_ssw_capacity(self, default_links):
        # 400G × 72 megapods = 28800
        link = next(
            lk
            for lk in default_links
            if lk["attrs"]["link_type"] == "fsw_ssw" and lk["attrs"]["site"] == "xyz1"
        )
        assert link["capacity"] == 28_800.0

    def test_xyz1_ssw_xsw_capacity(self, default_links):
        # 400G × 72 megapods = 28800
        link = next(
            lk
            for lk in default_links
            if lk["attrs"]["link_type"] == "ssw_xsw" and lk["attrs"]["site"] == "xyz1"
        )
        assert link["capacity"] == 28_800.0

    def test_all_costs_are_one(self, default_links):
        for lk in default_links:
            assert lk["cost"] == 1.0, f"Link {lk['source']}->{lk['target']} cost != 1.0"

    def test_custom_abc1_buildings(self):
        """Capacity scales with abc1_buildings."""
        cfg = DcBbScenarioConfig(abc1_buildings=3)
        links = _build_internal_dc_links(cfg)
        rsw_fsw = next(
            lk
            for lk in links
            if lk["attrs"]["link_type"] == "rsw_fsw" and lk["attrs"]["site"] == "abc1"
        )
        assert rsw_fsw["capacity"] == 48 * 200.0 * 3

    def test_custom_xyz1_megapods(self):
        """Capacity scales with xyz1_megapods."""
        cfg = DcBbScenarioConfig(xyz1_megapods=10)
        links = _build_internal_dc_links(cfg)
        fsw_ssw = next(
            lk
            for lk in links
            if lk["attrs"]["link_type"] == "fsw_ssw" and lk["attrs"]["site"] == "xyz1"
        )
        assert fsw_ssw["capacity"] == 400.0 * 10


# ---------------------------------------------------------------------------
# Connectivity correctness
# ---------------------------------------------------------------------------


class TestConnectivity:
    """Verify that the right nodes are connected."""

    def test_rsw_connects_to_own_pod_fsw(self, default_links):
        """abc1/pod1/rsw should connect to abc1/pod1/fsw/plane{1..8}."""
        rsw1_links = [lk for lk in default_links if lk["source"] == "abc1/pod1/rsw"]
        assert len(rsw1_links) == 8
        targets = {lk["target"] for lk in rsw1_links}
        expected = {f"abc1/pod1/fsw/plane{pl}" for pl in range(1, 9)}
        assert targets == expected

    def test_rsw_does_not_cross_pods(self, default_links):
        """abc1/pod1/rsw should NOT connect to abc1/pod2/fsw/*."""
        rsw1_targets = {
            lk["target"] for lk in default_links if lk["source"] == "abc1/pod1/rsw"
        }
        cross_pod = {t for t in rsw1_targets if "pod2" in t}
        assert len(cross_pod) == 0

    def test_fsw_connects_to_all_ssw_in_plane(self, default_links):
        """abc1/pod1/fsw/plane1 should connect to abc1/ssw/plane1/idx{1..36}."""
        fsw_links = [
            lk for lk in default_links if lk["source"] == "abc1/pod1/fsw/plane1"
        ]
        assert len(fsw_links) == 36
        targets = {lk["target"] for lk in fsw_links}
        expected = {f"abc1/ssw/plane1/idx{i}" for i in range(1, 37)}
        assert targets == expected

    def test_fsw_stays_in_plane(self, default_links):
        """abc1/pod1/fsw/plane1 should NOT connect to SSW in plane 2."""
        fsw_targets = {
            lk["target"]
            for lk in default_links
            if lk["source"] == "abc1/pod1/fsw/plane1"
        }
        wrong_plane = {t for t in fsw_targets if "plane2" in t}
        assert len(wrong_plane) == 0

    def test_ssw_connects_to_fadu_all_hgrids(self, default_links):
        """abc1/ssw/plane1/idx1 connects to abc1/fadu/hgrid{1..16}/idx1."""
        ssw_links = [
            lk for lk in default_links if lk["source"] == "abc1/ssw/plane1/idx1"
        ]
        assert len(ssw_links) == 16
        targets = {lk["target"] for lk in ssw_links}
        expected = {f"abc1/fadu/hgrid{h}/idx1" for h in range(1, 17)}
        assert targets == expected

    def test_ssw_fadu_idx_match(self, default_links):
        """SSW idx_i should connect to FADU idx_i (same index), not other indices."""
        ssw_links = [
            lk for lk in default_links if lk["source"] == "abc1/ssw/plane3/idx5"
        ]
        for lk in ssw_links:
            assert lk["target"].endswith("/idx5"), (
                f"SSW idx5 connected to wrong FADU: {lk['target']}"
            )

    def test_xyz1_rsw_connects_to_all_fsw(self, default_links):
        """xyz1/mp1/rsw connects to all 32 FSW."""
        rsw_links = [lk for lk in default_links if lk["source"] == "xyz1/mp1/rsw"]
        assert len(rsw_links) == 32
        targets = {lk["target"] for lk in rsw_links}
        expected = {
            f"xyz1/mp1/fsw/row{r}/dev{d}" for r in range(1, 5) for d in range(1, 9)
        }
        assert targets == expected

    def test_xyz1_fsw_connects_to_all_ssw(self, default_links):
        """Each XYZ1 FSW connects to all 24 SSW."""
        fsw_links = [
            lk for lk in default_links if lk["source"] == "xyz1/mp1/fsw/row1/dev1"
        ]
        assert len(fsw_links) == 24
        targets = {lk["target"] for lk in fsw_links}
        expected = {f"xyz1/mp1/ssw/plane{pl}" for pl in range(1, 25)}
        assert targets == expected

    def test_xyz1_ssw_connects_to_all_xsw_in_plane(self, default_links):
        """xyz1/mp1/ssw/plane1 connects to all 64 XSW in plane 1."""
        ssw_links = [
            lk for lk in default_links if lk["source"] == "xyz1/mp1/ssw/plane1"
        ]
        assert len(ssw_links) == 64
        targets = {lk["target"] for lk in ssw_links}
        expected = {f"xyz1/xsw/plane1/dev{d}" for d in range(1, 65)}
        assert targets == expected

    def test_xyz1_ssw_stays_in_plane(self, default_links):
        """xyz1/mp1/ssw/plane1 should NOT connect to XSW in plane 2."""
        ssw_targets = {
            lk["target"]
            for lk in default_links
            if lk["source"] == "xyz1/mp1/ssw/plane1"
        }
        wrong_plane = {t for t in ssw_targets if "plane2" in t}
        assert len(wrong_plane) == 0


# ---------------------------------------------------------------------------
# Link structure
# ---------------------------------------------------------------------------


class TestLinkStructure:
    """Verify each link dict has the required keys and format."""

    def test_required_keys(self, default_links):
        required = {"source", "target", "capacity", "cost", "attrs"}
        for lk in default_links:
            assert set(lk.keys()) == required, f"Bad keys in link: {lk}"

    def test_attrs_have_link_type_and_site(self, default_links):
        for lk in default_links:
            assert "link_type" in lk["attrs"]
            assert "site" in lk["attrs"]

    def test_link_types(self, default_links):
        types = {lk["attrs"]["link_type"] for lk in default_links}
        assert types == {"rsw_fsw", "fsw_ssw", "ssw_fadu", "ssw_xsw"}

    def test_no_duplicate_links(self, default_links):
        """Every (source, target) pair should be unique."""
        pairs = [(lk["source"], lk["target"]) for lk in default_links]
        assert len(pairs) == len(set(pairs))


# ---------------------------------------------------------------------------
# Node reference validity
# ---------------------------------------------------------------------------


class TestNodeReferences:
    """Verify all link endpoints exist in the node set."""

    def test_all_sources_are_valid_nodes(self, default_links, default_nodes):
        for lk in default_links:
            assert lk["source"] in default_nodes, f"Source not in nodes: {lk['source']}"

    def test_all_targets_are_valid_nodes(self, default_links, default_nodes):
        for lk in default_links:
            assert lk["target"] in default_nodes, f"Target not in nodes: {lk['target']}"


# ---------------------------------------------------------------------------
# Parameterized config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    """Verify link builder respects non-default config values."""

    def test_fewer_pods(self):
        cfg = DcBbScenarioConfig(abc1_pods_per_building=10)
        links = _build_internal_dc_links(cfg)
        rsw_fsw = [
            lk
            for lk in links
            if lk["attrs"]["link_type"] == "rsw_fsw" and lk["attrs"]["site"] == "abc1"
        ]
        # 10 pods × 8 planes = 80
        assert len(rsw_fsw) == 80
        fsw_ssw = [
            lk
            for lk in links
            if lk["attrs"]["link_type"] == "fsw_ssw" and lk["attrs"]["site"] == "abc1"
        ]
        # 10 pods × 8 planes × 36 SSW per plane = 2880
        assert len(fsw_ssw) == 2_880

    def test_fewer_planes(self):
        cfg = DcBbScenarioConfig(abc1_planes=2, abc1_ssw_per_plane=4)
        links = _build_internal_dc_links(cfg)
        rsw_fsw = [
            lk
            for lk in links
            if lk["attrs"]["link_type"] == "rsw_fsw" and lk["attrs"]["site"] == "abc1"
        ]
        # 96 pods × 2 planes = 192
        assert len(rsw_fsw) == 192
        fsw_ssw = [
            lk
            for lk in links
            if lk["attrs"]["link_type"] == "fsw_ssw" and lk["attrs"]["site"] == "abc1"
        ]
        # 96 pods × 2 planes × 4 SSW = 768
        assert len(fsw_ssw) == 768

    def test_fewer_xsw_planes(self):
        cfg = DcBbScenarioConfig(
            xyz1_xsw_planes=4,
            xyz1_ssw_per_megapod=4,
        )
        links = _build_internal_dc_links(cfg)
        ssw_xsw = [lk for lk in links if lk["attrs"]["link_type"] == "ssw_xsw"]
        # 4 SSW × 64 XSW per plane = 256
        assert len(ssw_xsw) == 256

    def test_fewer_fsw(self):
        # xyz1_fsw_per_megapod must be divisible by _XYZ1_FSW_ROWS=4
        cfg = DcBbScenarioConfig(xyz1_fsw_per_megapod=16)
        links = _build_internal_dc_links(cfg)
        rsw_fsw = [
            lk
            for lk in links
            if lk["attrs"]["link_type"] == "rsw_fsw" and lk["attrs"]["site"] == "xyz1"
        ]
        assert len(rsw_fsw) == 16
        fsw_ssw = [
            lk
            for lk in links
            if lk["attrs"]["link_type"] == "fsw_ssw" and lk["attrs"]["site"] == "xyz1"
        ]
        # 16 FSW × 24 SSW = 384
        assert len(fsw_ssw) == 384
