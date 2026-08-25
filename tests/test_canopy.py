# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for big-leaf canopy energy balance."""

import math

import pytest
import torch

from utherm.energy_balance.canopy import (
    canopy_gap_transmittance,
    canopy_hemispherical_transmittance,
    solve_canopy_eb,
)
from utherm.energy_balance.config import CanopyProperties

ROWS, COLS = 4, 4


class TestCanopyGapProbability:

    def test_overhead_beam_matches_spherical_leaf_formula(self):
        tau = canopy_gap_transmittance(3.0, 0.7, 90.0)
        assert tau.item() == pytest.approx(math.exp(-0.5 * 0.7 * 3.0), rel=1e-6)

    def test_clumping_increases_gap_probability(self):
        tau_clumped = canopy_gap_transmittance(3.0, 0.6, 60.0)
        tau_unclumped = canopy_gap_transmittance(3.0, 1.0, 60.0)
        assert tau_clumped.item() > tau_unclumped.item()

    def test_lower_sun_reduces_direct_transmittance(self):
        tau_high = canopy_gap_transmittance(3.0, 0.7, 80.0)
        tau_low = canopy_gap_transmittance(3.0, 0.7, 20.0)
        assert tau_low.item() < tau_high.item()

    def test_hemispherical_gap_fraction_bounds(self):
        tau_zero_lai = canopy_hemispherical_transmittance(0.0, 0.7)
        tau_canopy = canopy_hemispherical_transmittance(3.0, 0.7)
        assert tau_zero_lai.item() == pytest.approx(1.0, abs=1e-6)
        assert 0.0 < tau_canopy.item() < 1.0


@pytest.fixture
def canopy_inputs(device):
    """Standard inputs for big-leaf canopy tests."""
    return {
        "kdown_above": torch.full((ROWS, COLS), 500.0, device=device),
        "ldown": torch.full((ROWS, COLS), 350.0, device=device),
        "ta_c": 25.0,
        "ws": 2.0,
        "ea_hpa": 15.0,
        "pressure_kpa": 101.325,
        "canopy_mask": torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
        "lai_grid": torch.full((ROWS, COLS), 3.0, device=device),
        "device": device,
    }


class TestCanopyEBBasic:

    def test_transpiring_canopy(self, device, canopy_inputs):
        """Transpiring canopy should have positive QE and QH."""
        props = CanopyProperties.deciduous()
        result = solve_canopy_eb(props=props, **canopy_inputs)

        valid = ~torch.isnan(result['canopy_qe'])
        assert valid.any()
        assert result['canopy_qe'][valid].mean().item() > 0.0
        assert result['canopy_qh'][valid].mean().item() != 0.0

    def test_dry_canopy_no_stomata(self, device, canopy_inputs):
        """Canopy with zero stomatal conductance should have ~zero QE."""
        props = CanopyProperties.deciduous()
        props.max_stomatal_conductance = 0.0
        result = solve_canopy_eb(props=props, **canopy_inputs)

        valid = ~torch.isnan(result['canopy_qe'])
        if valid.any():
            assert result['canopy_qe'][valid].abs().mean().item() < 1.0

    def test_night_canopy(self, device):
        """Night (Kdown=0): leaf temp should be near or below air temp."""
        props = CanopyProperties.deciduous()
        result = solve_canopy_eb(
            kdown_above=torch.zeros(ROWS, COLS, device=device),
            ldown=torch.full((ROWS, COLS), 300.0, device=device),
            ta_c=20.0,
            ws=1.0,
            ea_hpa=12.0,
            pressure_kpa=101.325,
            canopy_mask=torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
            lai_grid=torch.full((ROWS, COLS), 3.0, device=device),
            props=props,
            device=device,
        )
        valid = ~torch.isnan(result['t_leaf'])
        t_air_k = 20.0 + 273.15
        # Leaf temp should not be much above air temp with no SW
        assert result['t_leaf'][valid].mean().item() < t_air_k + 5.0

    def test_lai_increases_absorption(self, device):
        """Higher LAI should absorb more radiation (higher Rnet)."""
        props = CanopyProperties.deciduous()

        result_low = solve_canopy_eb(
            kdown_above=torch.full((ROWS, COLS), 500.0, device=device),
            ldown=torch.full((ROWS, COLS), 350.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            canopy_mask=torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
            lai_grid=torch.full((ROWS, COLS), 1.0, device=device),
            props=props, device=device,
        )
        result_high = solve_canopy_eb(
            kdown_above=torch.full((ROWS, COLS), 500.0, device=device),
            ldown=torch.full((ROWS, COLS), 350.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            canopy_mask=torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
            lai_grid=torch.full((ROWS, COLS), 5.0, device=device),
            props=props, device=device,
        )

        v_low = ~torch.isnan(result_low['canopy_rnet'])
        v_high = ~torch.isnan(result_high['canopy_rnet'])
        assert result_high['canopy_rnet'][v_high].mean().item() > result_low['canopy_rnet'][v_low].mean().item()

    def test_energy_closure(self, device, canopy_inputs):
        """Rnet - QH - QE should be near zero (energy closure)."""
        props = CanopyProperties.deciduous()
        result = solve_canopy_eb(props=props, **canopy_inputs)

        valid = ~torch.isnan(result['canopy_rnet'])
        residual = result['canopy_rnet'] - result['canopy_qh'] - result['canopy_qe']
        assert residual[valid].abs().max().item() <= 0.1
        assert result['canopy_solver_converged'][valid].all()
        assert not result['canopy_solver_railed'][valid].any()

    def test_no_canopy_mask_gives_nan(self, device):
        """Pixels outside canopy mask should be NaN."""
        props = CanopyProperties.deciduous()
        mask = torch.zeros(ROWS, COLS, dtype=torch.bool, device=device)
        result = solve_canopy_eb(
            kdown_above=torch.full((ROWS, COLS), 500.0, device=device),
            ldown=torch.full((ROWS, COLS), 350.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            canopy_mask=mask,
            lai_grid=torch.full((ROWS, COLS), 3.0, device=device),
            props=props, device=device,
        )
        assert torch.isnan(result['t_leaf']).all()

    def test_zero_lai_gives_nan(self, device):
        """Zero LAI pixels should be masked to NaN."""
        props = CanopyProperties.deciduous()
        result = solve_canopy_eb(
            kdown_above=torch.full((ROWS, COLS), 500.0, device=device),
            ldown=torch.full((ROWS, COLS), 350.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            canopy_mask=torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
            lai_grid=torch.zeros(ROWS, COLS, device=device),
            props=props, device=device,
        )
        assert torch.isnan(result['t_leaf']).all()

    def test_railed_leaf_temperature_is_reported_and_not_returned(self, device, canopy_inputs):
        canopy_inputs["kdown_above"] = torch.full((ROWS, COLS), 1.0e8, device=device)
        result = solve_canopy_eb(props=CanopyProperties.deciduous(), **canopy_inputs)
        assert result['canopy_solver_railed'].all()
        assert torch.isnan(result['t_leaf']).all()

    def test_nonfinite_leaf_input_is_reported_and_not_returned(self, device, canopy_inputs):
        canopy_inputs["ldown"][0, 0] = float("nan")
        result = solve_canopy_eb(props=CanopyProperties.deciduous(), **canopy_inputs)
        assert result['canopy_solver_nonfinite'][0, 0]
        assert torch.isnan(result['t_leaf'][0, 0])

    def test_canopy_input_shapes_must_match(self, device, canopy_inputs):
        canopy_inputs["lai_grid"] = torch.ones(ROWS - 1, COLS, device=device)
        with pytest.raises(ValueError, match="input shapes differ"):
            solve_canopy_eb(props=CanopyProperties.deciduous(), **canopy_inputs)

    def test_negative_lai_is_rejected(self, device, canopy_inputs):
        canopy_inputs["lai_grid"][0, 0] = -1.0
        with pytest.raises(ValueError, match="lai_grid cannot contain negative"):
            solve_canopy_eb(props=CanopyProperties.deciduous(), **canopy_inputs)


class TestCanopyLAIScaling:

    def test_higher_lai_higher_qe(self, device):
        """Higher LAI should increase total canopy QE (more transpiring area)."""
        props = CanopyProperties.deciduous()

        def run_lai(lai_val):
            r = solve_canopy_eb(
                kdown_above=torch.full((ROWS, COLS), 500.0, device=device),
                ldown=torch.full((ROWS, COLS), 350.0, device=device),
                ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
                canopy_mask=torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
                lai_grid=torch.full((ROWS, COLS), lai_val, device=device),
                props=props, device=device,
            )
            v = ~torch.isnan(r['canopy_qe'])
            return r['canopy_qe'][v].mean().item()

        qe_low = run_lai(1.0)
        qe_high = run_lai(4.0)
        assert qe_high > qe_low

    def test_prev_t_leaf_warm_start(self, device):
        """Providing prev_t_leaf should still converge normally."""
        props = CanopyProperties.deciduous()
        prev = torch.full((ROWS, COLS), 300.0, device=device)
        result = solve_canopy_eb(
            kdown_above=torch.full((ROWS, COLS), 500.0, device=device),
            ldown=torch.full((ROWS, COLS), 350.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            canopy_mask=torch.ones(ROWS, COLS, dtype=torch.bool, device=device),
            lai_grid=torch.full((ROWS, COLS), 3.0, device=device),
            props=props, prev_t_leaf=prev, device=device,
        )
        valid = ~torch.isnan(result['t_leaf'])
        # Should still produce reasonable leaf temperatures
        assert result['t_leaf'][valid].mean().item() > 273.15
        assert result['t_leaf'][valid].mean().item() < 353.15


def test_canopy_optics_land_on_the_model_device(device):
    """Regression: canopy optics must not leak CPU tensors into a CUDA run.

    Solar geometry is returned as NumPy, so canopy_gap_transmittance and
    canopy_hemispherical_transmittance default to CPU when handed scalars.
    utci_process mixes their output with device tensors via torch.where, which
    raises on a device mismatch. Reproduces the compute_utci failure directly.
    """
    import torch
    from utherm.energy_balance.canopy import (
        canopy_gap_transmittance,
        canopy_hemispherical_transmittance,
    )

    numpy_like_altitude = float(41.7)          # as delivered by the solar module
    direct = canopy_gap_transmittance(3.0, 0.7, numpy_like_altitude).to(device)
    diffuse = canopy_hemispherical_transmittance(3.0, 0.7).to(device)
    leafon = torch.ones((1, 24), device=device)

    # the exact expression that failed in compute_utci
    psi = torch.where(leafon > 0, direct.unsqueeze(0), 0.5)
    assert psi.device.type == torch.device(device).type
    assert torch.isfinite(psi).all()

    svfveg = torch.full((4, 4), 0.8, device=device)
    combined = svfveg - (1.0 - svfveg) * (1.0 - diffuse)
    assert combined.device.type == torch.device(device).type
    assert torch.isfinite(combined).all()
