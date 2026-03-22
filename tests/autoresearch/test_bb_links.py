"""Tests for _build_bb_cross_site_links in scenario_generator.py."""

from collections import Counter

import pytest

from netlab.autoresearch.scenario_generator import (
    DcBbScenarioConfig,
    _build_bb_cross_site_links,
)


@pytest.fixture
def default_links():
    """Links built from default config."""
    return _build_bb_cross_site_links(DcBbScenarioConfig())


# ---------------------------------------------------------------------------
# Link count
# ---------------------------------------------------------------------------


class TestLinkCount:
    """Verify total and per-plane link counts."""

    def test_total_count(self, default_links):
        # 64 planes * 4*4 full mesh * 2 paths = 2048
        assert len(default_links) == 2048

    def test_links_per_plane(self, default_links):
        """Each plane should have exactly 32 links (16 pairs x 2 paths)."""
        plane_counts = Counter(link["attrs"]["plane"] for link in default_links)
        assert len(plane_counts) == 64
        for pl, count in plane_counts.items():
            assert count == 32, f"plane {pl} has {count} links, expected 32"


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


class TestCapacity:
    """Verify link capacities."""

    def test_all_links_default_capacity(self, default_links):
        for link in default_links:
            assert link["capacity"] == 800.0

    def test_custom_capacity(self):
        cfg = DcBbScenarioConfig(bb_bb_link_capacity=1600.0)
        links = _build_bb_cross_site_links(cfg)
        for link in links:
            assert link["capacity"] == 1600.0


# ---------------------------------------------------------------------------
# Path distribution
# ---------------------------------------------------------------------------


class TestPathDistribution:
    """Verify path_a / path_b split."""

    def test_half_path_a_half_path_b(self, default_links):
        path_counts = Counter(link["attrs"]["path"] for link in default_links)
        assert path_counts["path_a"] == 1024
        assert path_counts["path_b"] == 1024

    def test_each_plane_has_both_paths(self, default_links):
        for pl in range(1, 65):
            plane_links = [lk for lk in default_links if lk["attrs"]["plane"] == pl]
            paths = {lk["attrs"]["path"] for lk in plane_links}
            assert paths == {"path_a", "path_b"}, f"plane {pl} missing a path"


# ---------------------------------------------------------------------------
# Risk groups
# ---------------------------------------------------------------------------


class TestRiskGroups:
    """Verify risk group assignments."""

    def test_risk_group_structure(self, default_links):
        """Every link has exactly 4 risk groups."""
        for link in default_links:
            assert len(link["risk_groups"]) == 4

    def test_risk_group_contents(self, default_links):
        """Spot-check a specific link's risk groups."""
        # Find a plane 1, path_a link from dev1->dev1
        target = None
        for link in default_links:
            if (
                link["source"] == "bb/abc1/plane1/dev1"
                and link["target"] == "bb/xyz1/plane1/dev1"
                and link["attrs"]["path"] == "path_a"
            ):
                target = link
                break
        assert target is not None
        assert target["risk_groups"] == [
            "path_a",
            "plane_group_1",
            "plane_1_site_abc1",
            "plane_1_site_xyz1",
        ]

    def test_plane_group_mapping(self, default_links):
        """Verify plane_group assignment for boundary planes."""
        expected = {
            1: "plane_group_1",
            4: "plane_group_1",
            5: "plane_group_2",
            8: "plane_group_2",
            9: "plane_group_3",
            61: "plane_group_16",
            64: "plane_group_16",
        }
        for pl, expected_group in expected.items():
            plane_links = [lk for lk in default_links if lk["attrs"]["plane"] == pl]
            assert len(plane_links) > 0, f"no links for plane {pl}"
            for link in plane_links:
                assert expected_group in link["risk_groups"], (
                    f"plane {pl}: expected {expected_group} in {link['risk_groups']}"
                )

    def test_plane_site_risk_groups(self, default_links):
        """Each plane's links include plane_N_site_abc1 and plane_N_site_xyz1."""
        for pl in range(1, 65):
            plane_links = [lk for lk in default_links if lk["attrs"]["plane"] == pl]
            for link in plane_links:
                assert f"plane_{pl}_site_abc1" in link["risk_groups"]
                assert f"plane_{pl}_site_xyz1" in link["risk_groups"]

    def test_16_plane_groups_total(self, default_links):
        """There should be exactly 16 distinct plane groups across all links."""
        groups = set()
        for link in default_links:
            for rg in link["risk_groups"]:
                if rg.startswith("plane_group_"):
                    groups.add(rg)
        assert len(groups) == 16
        assert groups == {f"plane_group_{g}" for g in range(1, 17)}


# ---------------------------------------------------------------------------
# Link attributes
# ---------------------------------------------------------------------------


class TestLinkAttributes:
    """Verify link-level attributes."""

    def test_all_links_have_cost_10(self, default_links):
        for link in default_links:
            assert link["cost"] == 10

    def test_link_type_attr(self, default_links):
        for link in default_links:
            assert link["attrs"]["link_type"] == "bb_cross_site"

    def test_source_target_naming(self, default_links):
        """All sources are abc1-side BB, all targets are xyz1-side BB."""
        for link in default_links:
            assert link["source"].startswith("bb/abc1/plane")
            assert link["target"].startswith("bb/xyz1/plane")

    def test_source_target_same_plane(self, default_links):
        """Source and target are always in the same plane."""
        for link in default_links:
            src_plane = link["source"].split("/")[2]  # "planeN"
            tgt_plane = link["target"].split("/")[2]
            assert src_plane == tgt_plane


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    """Verify function respects non-default config."""

    def test_fewer_planes(self):
        cfg = DcBbScenarioConfig(bb_planes=4)
        links = _build_bb_cross_site_links(cfg)
        # 4 planes * 16 pairs * 2 paths = 128
        assert len(links) == 128

    def test_fewer_devices_per_plane(self):
        cfg = DcBbScenarioConfig(bb_devices_per_plane=2)
        links = _build_bb_cross_site_links(cfg)
        # 64 planes * 2*2 mesh * 2 paths = 512
        assert len(links) == 512

    def test_single_plane_single_device(self):
        cfg = DcBbScenarioConfig(bb_planes=1, bb_devices_per_plane=1)
        links = _build_bb_cross_site_links(cfg)
        # 1 plane * 1 pair * 2 paths = 2
        assert len(links) == 2
        assert links[0]["source"] == "bb/abc1/plane1/dev1"
        assert links[0]["target"] == "bb/xyz1/plane1/dev1"
