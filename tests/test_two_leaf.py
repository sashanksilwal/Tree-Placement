# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for two-big-leaf canopy energy balance."""

import pytest
import torch
import utherm.energy_balance.canopy as canopy_module

from utherm.energy_balance.canopy import solve_canopy_eb_two_leaf, solve_canopy_eb
from utherm.energy_balance.config import CanopyProperties

ROWS, COLS = 4, 4


@pytest.fixture
def two_leaf_inputs(device):
    """Standard inputs for two-big-leaf tests."""
    return {
        "kdown_beam": torch.full((ROWS, COLS), 400.0, device=device),
        "kdown_diffuse": torch.full((ROWS, COLS), 100.0, device=device),
        "ldown": torch.full((ROWS, COLS), 350.0, device=device),
        "ta_c": 25.0,
        "ws": 2.0,
        "ea_hpa": 15.0,
        "pressure_kpa": 101.325,
        "solar_elev_deg": 45.0,
        "canopy_mask": torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
        "lai_grid": torch.full((ROWS, COLS), 3.0, device=device),
        "device": device,
    }


class TestTwoLeafPartition:

    def test_longwave_interception_is_partitioned_once(self, device, two_leaf_inputs, monkeypatch):
        """Sunlit and shaded classes must sum to whole-canopy LW interception."""
        captured = []

        def fake_solver(kabs, *args, longwave_absorb_fraction=None, **kwargs):
            captured.append(longwave_absorb_fraction.clone())
            zeros = torch.zeros_like(kabs)
            active = kwargs["valid_mask"]
            return canopy_module.LeafSolveResult(
                temperature=zeros,
                sensible_heat=zeros,
                latent_heat=zeros,
                net_radiation=zeros,
                residual=zeros,
                converged=active,
                railed=torch.zeros_like(active),
                nonfinite=torch.zeros_like(active),
                iterations=torch.zeros_like(kabs, dtype=torch.int32),
            )

        monkeypatch.setattr(canopy_module, "_solve_leaf_temperature_batched", fake_solver)
        props = CanopyProperties.deciduous()
        solve_canopy_eb_two_leaf(props=props, **two_leaf_inputs)
        expected = 1.0 - canopy_module.canopy_hemispherical_transmittance(
            two_leaf_inputs["lai_grid"], props.clumping_factor,
        )
        assert len(captured) == 2
        assert torch.allclose(captured[0] + captured[1], expected, atol=1.0e-6)

    def test_two_leaf_returns_valid(self, device, two_leaf_inputs):
        """Two-leaf model should return valid leaf temperatures."""
        props = CanopyProperties.deciduous()
        result = solve_canopy_eb_two_leaf(props=props, **two_leaf_inputs)
        valid = ~torch.isnan(result['t_leaf'])
        assert valid.any()
        assert result['t_leaf'][valid].mean().item() > 273.15
        assert result['canopy_solver_converged'][valid].all()

    def test_two_leaf_energy_closure(self, device, two_leaf_inputs):
        """Rnet - QH - QE should be near zero."""
        props = CanopyProperties.deciduous()
        result = solve_canopy_eb_two_leaf(props=props, **two_leaf_inputs)
        valid = ~torch.isnan(result['canopy_rnet'])
        residual = result['canopy_rnet'] - result['canopy_qh'] - result['canopy_qe']
        assert residual[valid].abs().mean().item() < 10.0

    def test_sunlit_warmer_than_shaded(self, device):
        """At high solar elevation with beam, sunlit leaves should be warmer overall."""
        # This is an indirect test: two-leaf with beam should produce
        # higher leaf temp than big-leaf with only diffuse
        props = CanopyProperties.deciduous()

        result_beam = solve_canopy_eb_two_leaf(
            kdown_beam=torch.full((ROWS, COLS), 600.0, device=device),
            kdown_diffuse=torch.full((ROWS, COLS), 100.0, device=device),
            ldown=torch.full((ROWS, COLS), 350.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            solar_elev_deg=60.0,
            canopy_mask=torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
            lai_grid=torch.full((ROWS, COLS), 3.0, device=device),
            props=props, device=device,
        )
        result_diffuse = solve_canopy_eb_two_leaf(
            kdown_beam=torch.full((ROWS, COLS), 0.0, device=device),
            kdown_diffuse=torch.full((ROWS, COLS), 100.0, device=device),
            ldown=torch.full((ROWS, COLS), 350.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            solar_elev_deg=60.0,
            canopy_mask=torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
            lai_grid=torch.full((ROWS, COLS), 3.0, device=device),
            props=props, device=device,
        )

        v1 = ~torch.isnan(result_beam['t_leaf'])
        v2 = ~torch.isnan(result_diffuse['t_leaf'])
        assert result_beam['t_leaf'][v1].mean().item() > result_diffuse['t_leaf'][v2].mean().item()


class TestTwoLeafFallback:

    def test_night_fallback_to_big_leaf(self, device):
        """At night (solar_elev <= 0), two-leaf should fall back to big-leaf."""
        props = CanopyProperties.deciduous()
        result = solve_canopy_eb_two_leaf(
            kdown_beam=torch.zeros(ROWS, COLS, device=device),
            kdown_diffuse=torch.zeros(ROWS, COLS, device=device),
            ldown=torch.full((ROWS, COLS), 300.0, device=device),
            ta_c=15.0, ws=1.0, ea_hpa=10.0, pressure_kpa=101.325,
            solar_elev_deg=-5.0,
            canopy_mask=torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
            lai_grid=torch.full((ROWS, COLS), 3.0, device=device),
            props=props, device=device,
        )
        valid = ~torch.isnan(result['t_leaf'])
        assert valid.any()

    def test_low_beam_fallback(self, device):
        """Very low beam (< 0.1 W/m2) should trigger big-leaf fallback."""
        props = CanopyProperties.deciduous()
        result = solve_canopy_eb_two_leaf(
            kdown_beam=torch.full((ROWS, COLS), 0.05, device=device),
            kdown_diffuse=torch.full((ROWS, COLS), 100.0, device=device),
            ldown=torch.full((ROWS, COLS), 350.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            solar_elev_deg=30.0,
            canopy_mask=torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
            lai_grid=torch.full((ROWS, COLS), 3.0, device=device),
            props=props, device=device,
        )
        valid = ~torch.isnan(result['t_leaf'])
        assert valid.any()

    def test_fallback_matches_big_leaf(self, device):
        """Fallback output should match big-leaf for identical inputs."""
        props = CanopyProperties.deciduous()
        kdown = torch.full((ROWS, COLS), 100.0, device=device)

        two_leaf = solve_canopy_eb_two_leaf(
            kdown_beam=torch.zeros(ROWS, COLS, device=device),
            kdown_diffuse=kdown,
            ldown=torch.full((ROWS, COLS), 350.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            solar_elev_deg=-1.0,
            canopy_mask=torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
            lai_grid=torch.full((ROWS, COLS), 3.0, device=device),
            props=props, device=device,
        )
        big_leaf = solve_canopy_eb(
            kdown_above=kdown,
            ldown=torch.full((ROWS, COLS), 350.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            canopy_mask=torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
            lai_grid=torch.full((ROWS, COLS), 3.0, device=device),
            props=props, device=device,
        )

        v = ~torch.isnan(two_leaf['t_leaf'])
        diff = (two_leaf['t_leaf'][v] - big_leaf['t_leaf'][v]).abs()
        assert diff.max().item() < 0.1


class TestTwoLeafEdgeCases:

    def test_very_low_lai(self, device, two_leaf_inputs):
        """LAI near zero should produce minimal canopy effect."""
        props = CanopyProperties.deciduous()
        two_leaf_inputs["lai_grid"] = torch.full((ROWS, COLS), 0.1, device=device)
        result = solve_canopy_eb_two_leaf(props=props, **two_leaf_inputs)
        valid = ~torch.isnan(result['t_leaf'])
        if valid.any():
            t_air_k = 25.0 + 273.15
            # With very low LAI, leaf temp should be close to air temp
            assert (result['t_leaf'][valid] - t_air_k).abs().mean().item() < 10.0

    def test_high_solar_elevation(self, device, two_leaf_inputs):
        """High solar elevation (90 deg) should still work."""
        props = CanopyProperties.deciduous()
        two_leaf_inputs["solar_elev_deg"] = 89.0
        result = solve_canopy_eb_two_leaf(props=props, **two_leaf_inputs)
        valid = ~torch.isnan(result['t_leaf'])
        assert valid.any()

    def test_low_solar_elevation(self, device, two_leaf_inputs):
        """Low positive solar elevation should increase beam extinction."""
        props = CanopyProperties.deciduous()
        two_leaf_inputs["solar_elev_deg"] = 5.0
        result = solve_canopy_eb_two_leaf(props=props, **two_leaf_inputs)
        valid = ~torch.isnan(result['t_leaf'])
        assert valid.any()

    def test_no_mask_gives_nan(self, device, two_leaf_inputs):
        """Empty canopy mask should give all NaN."""
        props = CanopyProperties.deciduous()
        two_leaf_inputs["canopy_mask"] = torch.zeros(ROWS, COLS, dtype=torch.bool, device=device)
        result = solve_canopy_eb_two_leaf(props=props, **two_leaf_inputs)
        assert torch.isnan(result['t_leaf']).all()

    def test_evergreen_properties(self, device, two_leaf_inputs):
        """Evergreen canopy should also produce valid results."""
        props = CanopyProperties.evergreen()
        result = solve_canopy_eb_two_leaf(props=props, **two_leaf_inputs)
        valid = ~torch.isnan(result['t_leaf'])
        assert valid.any()
        assert result['t_leaf'][valid].mean().item() > 273.15
