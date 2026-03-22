"""Tests for the mesh group algorithm in scenario_generator.py."""

import pytest

from netlab.autoresearch.scenario_generator import (
    DcBbScenarioConfig,
    _compute_mesh_groups,
    get_valid_layouts,
    get_viable_g_values,
    validate_layout,
)

# ---------------------------------------------------------------------------
# Viable G values
# ---------------------------------------------------------------------------


class TestGetViableGValues:
    """Acceptance criteria: exact viable G sets for known configurations."""

    def test_dc16x36_bb64x4(self):
        """dc16x36 + bb64x4 => viable G is exactly {16, 32, 64}.

        dc_total=576, bb_total=256, dc_ports=16, bb_ports=36.
        G=16: k_dc=256/16=16<=16, k_bb=576/16=36<=36
        G=32: k_dc=256/32=8<=16, k_bb=576/32=18<=36
        G=64: k_dc=256/64=4<=16, k_bb=576/64=9<=36
        """
        result = get_viable_g_values(
            dc_total=576, bb_total=256, dc_ports=16, bb_ports=36
        )
        assert set(result) == {16, 32, 64}

    def test_dc64x24_bb64x4(self):
        """dc64x24 + bb64x4 => viable G is exactly {64, 128, 256}.

        dc_total=1536, bb_total=256, dc_ports=4, bb_ports=24.
        G=64: k_dc=256/64=4<=4, k_bb=1536/64=24<=24
        G=128: k_dc=256/128=2<=4, k_bb=1536/128=12<=24
        G=256: k_dc=256/256=1<=4, k_bb=1536/256=6<=24
        """
        result = get_viable_g_values(
            dc_total=1536, bb_total=256, dc_ports=4, bb_ports=24
        )
        assert set(result) == {64, 128, 256}

    def test_returns_sorted(self):
        result = get_viable_g_values(
            dc_total=576, bb_total=256, dc_ports=16, bb_ports=36
        )
        assert result == sorted(result)

    def test_no_viable_g_when_ports_too_small(self):
        # dc_ports=1, bb_ports=1 means only G where both bb_total/G<=1
        # and dc_total/G<=1, so G>=max(dc_total, bb_total).
        # But G must divide both, so effectively no viable G if
        # dc_total and bb_total are large and different.
        result = get_viable_g_values(dc_total=576, bb_total=256, dc_ports=1, bb_ports=1)
        assert result == []

    def test_trivial_g_equals_1(self):
        # G=1 means every device sees every other device.
        # k_dc = bb_total, k_bb = dc_total.
        result = get_viable_g_values(dc_total=4, bb_total=4, dc_ports=4, bb_ports=4)
        assert 1 in result


# ---------------------------------------------------------------------------
# Mesh group computation
# ---------------------------------------------------------------------------


class TestComputeMeshGroups:
    """Test _compute_mesh_groups partitioning and group sizes."""

    def test_every_dc_device_in_exactly_one_group(self):
        """Every DC device (row, col) appears in exactly one group."""
        dc_rows, dc_cols = 16, 36
        bb_rows, bb_cols = 64, 4
        g = 16
        layout = (4, 4, 16, 1)

        groups = _compute_mesh_groups(dc_rows, dc_cols, bb_rows, bb_cols, g, layout)

        all_dc = set()
        for dc_devs, _ in groups:
            dc_set = set(dc_devs)
            # No overlap with previously seen devices
            assert all_dc.isdisjoint(dc_set), "DC device in multiple groups"
            all_dc.update(dc_set)

        expected = {(r, c) for r in range(dc_rows) for c in range(dc_cols)}
        assert all_dc == expected

    def test_every_bb_device_in_exactly_one_group(self):
        """Every BB device (row, col) appears in exactly one group."""
        dc_rows, dc_cols = 16, 36
        bb_rows, bb_cols = 64, 4
        g = 16
        layout = (4, 4, 16, 1)

        groups = _compute_mesh_groups(dc_rows, dc_cols, bb_rows, bb_cols, g, layout)

        all_bb = set()
        for _, bb_devs in groups:
            bb_set = set(bb_devs)
            assert all_bb.isdisjoint(bb_set), "BB device in multiple groups"
            all_bb.update(bb_set)

        expected = {(r, c) for r in range(bb_rows) for c in range(bb_cols)}
        assert all_bb == expected

    def test_group_count_equals_g(self):
        g = 32
        layout = (8, 4, 8, 4)
        groups = _compute_mesh_groups(16, 36, 64, 4, g, layout)
        assert len(groups) == g

    def test_full_mesh_link_count(self):
        """Full mesh within each group: total links = dc_total * bb_total / G."""
        dc_rows, dc_cols = 16, 36
        bb_rows, bb_cols = 64, 4
        g = 64
        layout = (16, 4, 16, 4)

        groups = _compute_mesh_groups(dc_rows, dc_cols, bb_rows, bb_cols, g, layout)

        total_links = sum(len(dc) * len(bb) for dc, bb in groups)
        expected = (dc_rows * dc_cols) * (bb_rows * bb_cols) // g
        assert total_links == expected

    def test_uniform_group_sizes(self):
        """Each group has dc_total/G DC devices and bb_total/G BB devices."""
        dc_rows, dc_cols = 16, 36
        bb_rows, bb_cols = 64, 4
        g = 64
        layout = (16, 4, 16, 4)

        groups = _compute_mesh_groups(dc_rows, dc_cols, bb_rows, bb_cols, g, layout)

        dc_total = dc_rows * dc_cols
        bb_total = bb_rows * bb_cols
        for dc_devs, bb_devs in groups:
            assert len(dc_devs) == dc_total // g
            assert len(bb_devs) == bb_total // g

    def test_port_constraints_respected(self):
        """k_dc <= max_ports, k_bb <= max_ports within each group."""
        dc_rows, dc_cols = 16, 36
        bb_rows, bb_cols = 64, 4
        g = 16
        layout = (4, 4, 16, 1)
        dc_ports = 16
        bb_ports = 36

        groups = _compute_mesh_groups(dc_rows, dc_cols, bb_rows, bb_cols, g, layout)

        for dc_devs, bb_devs in groups:
            # Each DC device connects to all BB devices in the group
            k_dc = len(bb_devs)
            # Each BB device connects to all DC devices in the group
            k_bb = len(dc_devs)
            assert k_dc <= dc_ports, f"k_dc={k_dc} > dc_ports={dc_ports}"
            assert k_bb <= bb_ports, f"k_bb={k_bb} > bb_ports={bb_ports}"

    def test_invalid_layout_dc_product_mismatch(self):
        with pytest.raises(ValueError, match="DC layout"):
            _compute_mesh_groups(16, 36, 64, 4, g=16, layout=(2, 2, 4, 4))

    def test_invalid_layout_bb_product_mismatch(self):
        with pytest.raises(ValueError, match="BB layout"):
            _compute_mesh_groups(16, 36, 64, 4, g=16, layout=(4, 4, 2, 2))

    def test_invalid_layout_dc_rows_not_divisible(self):
        with pytest.raises(ValueError, match="dc_rows"):
            _compute_mesh_groups(16, 36, 64, 4, g=6, layout=(6, 1, 6, 1))

    def test_small_example(self):
        """Verify a small, hand-traceable example."""
        # 4x4 DC grid, 2x2 BB grid, G=4, layout=(2,2,2,2)
        groups = _compute_mesh_groups(4, 4, 2, 2, g=4, layout=(2, 2, 2, 2))
        assert len(groups) == 4

        # Each group: 4 DC devices, 1 BB device
        for dc_devs, bb_devs in groups:
            assert len(dc_devs) == 4
            assert len(bb_devs) == 1

        # Group 0 (gi=0, gj=0): DC rows [0,1], cols [0,1]
        dc0, bb0 = groups[0]
        assert set(dc0) == {(0, 0), (0, 1), (1, 0), (1, 1)}
        # BB group 0 (bi=0, bj=0): BB rows [0], cols [0]
        assert set(bb0) == {(0, 0)}

    def test_large_config_dc64x24_bb64x4(self):
        """Larger config: dc64x24, bb64x4, G=64."""
        dc_rows, dc_cols = 64, 24
        bb_rows, bb_cols = 64, 4
        g = 64
        layout = (16, 4, 16, 4)

        groups = _compute_mesh_groups(dc_rows, dc_cols, bb_rows, bb_cols, g, layout)

        assert len(groups) == 64

        # Verify partition
        all_dc = set()
        all_bb = set()
        for dc_devs, bb_devs in groups:
            assert len(dc_devs) == (64 * 24) // 64  # 24
            assert len(bb_devs) == (64 * 4) // 64  # 4
            dc_set = set(dc_devs)
            bb_set = set(bb_devs)
            assert all_dc.isdisjoint(dc_set)
            assert all_bb.isdisjoint(bb_set)
            all_dc.update(dc_set)
            all_bb.update(bb_set)

        assert len(all_dc) == 64 * 24
        assert len(all_bb) == 64 * 4


# ---------------------------------------------------------------------------
# Layout validation and enumeration
# ---------------------------------------------------------------------------


class TestValidateLayout:
    def test_valid_layout(self):
        assert validate_layout(
            g=64,
            layout=(16, 4, 16, 4),
            dc_rows=16,
            dc_cols=36,
            bb_rows=64,
            bb_cols=4,
        )

    def test_invalid_g_product(self):
        assert not validate_layout(
            g=64,
            layout=(8, 4, 16, 4),  # 8*4=32 != 64
            dc_rows=16,
            dc_cols=36,
            bb_rows=64,
            bb_cols=4,
        )

    def test_invalid_divisibility(self):
        assert not validate_layout(
            g=64,
            layout=(16, 4, 16, 4),
            dc_rows=15,  # 15 % 16 != 0
            dc_cols=36,
            bb_rows=64,
            bb_cols=4,
        )


class TestGetValidLayouts:
    def test_dc16x36_bb64x4_g64(self):
        layouts = get_valid_layouts(g=64, dc_rows=16, dc_cols=36, bb_rows=64, bb_cols=4)
        # (16, 4) is the only DC factorization: 16*4=64, 16%16=0, 36%4=0
        # BB: (16, 4) works: 64%16=0, 4%4=0. Also (64,1): 64%64=0, 4%1=0
        assert len(layouts) > 0
        assert (16, 4, 16, 4) in layouts
        # All must be valid
        for layout in layouts:
            assert validate_layout(64, layout, 16, 36, 64, 4)

    def test_dc16x36_bb64x4_g16(self):
        layouts = get_valid_layouts(g=16, dc_rows=16, dc_cols=36, bb_rows=64, bb_cols=4)
        assert len(layouts) > 0
        # (4, 4) DC factorization: 4*4=16, 16%4=0, 36%4=0
        # (16, 1) BB factorization: 16*1=16, 64%16=0, 4%1=0
        assert (4, 4, 16, 1) in layouts

    def test_empty_when_no_valid_factorization(self):
        # G=7, dc_rows=16, dc_cols=36 — 7 is prime and doesn't divide 16 or 36 cleanly
        layouts = get_valid_layouts(g=7, dc_rows=16, dc_cols=36, bb_rows=64, bb_cols=4)
        assert layouts == []

    def test_all_returned_layouts_are_valid(self):
        for g in [16, 32, 64]:
            layouts = get_valid_layouts(
                g=g, dc_rows=16, dc_cols=36, bb_rows=64, bb_cols=4
            )
            for layout in layouts:
                assert validate_layout(g, layout, 16, 36, 64, 4)


# ---------------------------------------------------------------------------
# DcBbScenarioConfig dataclass
# ---------------------------------------------------------------------------


class TestDcBbScenarioConfig:
    def test_default_values(self):
        cfg = DcBbScenarioConfig()
        assert cfg.abc1_hgrids == 16
        assert cfg.abc1_fadu_per_hgrid == 36
        assert cfg.abc1_planes == 8
        assert cfg.abc1_ssw_per_plane == 36
        assert cfg.abc1_pods_per_building == 96
        assert cfg.abc1_buildings == 5
        assert cfg.abc1_rsw_per_pod == 48

        assert cfg.xyz1_xsw_planes == 24
        assert cfg.xyz1_xsw_per_plane == 64
        assert cfg.xyz1_ssw_per_megapod == 24
        assert cfg.xyz1_fsw_per_megapod == 32
        assert cfg.xyz1_megapods == 72

        assert cfg.bb_planes == 64
        assert cfg.bb_devices_per_plane == 4

        assert cfg.dc_bb_link_capacity == 400.0
        assert cfg.bb_bb_link_capacity == 800.0

        assert cfg.g_abc1 == 64
        assert cfg.g_xyz1 == 64
        assert cfg.layout_abc1 == (16, 4, 16, 4)
        assert cfg.layout_xyz1 == (16, 4, 16, 4)

        assert cfg.seed == 42
        assert cfg.msd_resolution == 0.01
        assert cfg.failure_iterations == 200

    def test_custom_values(self):
        cfg = DcBbScenarioConfig(g_abc1=32, seed=123)
        assert cfg.g_abc1 == 32
        assert cfg.seed == 123
        # Other defaults unchanged
        assert cfg.g_xyz1 == 64


# ---------------------------------------------------------------------------
# Integration: viable G + valid layouts + mesh groups together
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end: viable G -> valid layout -> compute groups -> verify."""

    @pytest.mark.parametrize(
        "dc_rows,dc_cols,bb_rows,bb_cols,dc_ports,bb_ports,expected_g_set",
        [
            (16, 36, 64, 4, 16, 36, {16, 32, 64}),
            (64, 24, 64, 4, 4, 24, {64, 128, 256}),
        ],
    )
    def test_viable_g_to_groups(
        self,
        dc_rows,
        dc_cols,
        bb_rows,
        bb_cols,
        dc_ports,
        bb_ports,
        expected_g_set,
    ):
        dc_total = dc_rows * dc_cols
        bb_total = bb_rows * bb_cols

        viable = get_viable_g_values(dc_total, bb_total, dc_ports, bb_ports)
        assert set(viable) == expected_g_set

        for g in viable:
            layouts = get_valid_layouts(g, dc_rows, dc_cols, bb_rows, bb_cols)
            assert len(layouts) > 0, f"No valid layout for G={g}"

            layout = layouts[0]
            groups = _compute_mesh_groups(dc_rows, dc_cols, bb_rows, bb_cols, g, layout)
            assert len(groups) == g

            # Verify partition
            all_dc = set()
            all_bb = set()
            for dc_devs, bb_devs in groups:
                dc_set = set(dc_devs)
                bb_set = set(bb_devs)
                assert all_dc.isdisjoint(dc_set)
                assert all_bb.isdisjoint(bb_set)
                all_dc.update(dc_set)
                all_bb.update(bb_set)

            assert len(all_dc) == dc_total
            assert len(all_bb) == bb_total

            # Full mesh link count
            total_links = sum(len(dc) * len(bb) for dc, bb in groups)
            assert total_links == dc_total * bb_total // g

            # Port constraints
            for dc_devs, bb_devs in groups:
                assert len(bb_devs) <= dc_ports
                assert len(dc_devs) <= bb_ports
