# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for core physics kernels: NR solver, Thomas algorithm, HTTC, vapor pressure, g_coupling_beta."""

import math

import pytest
import torch

from utherm.energy_balance.physics import (
    e_sat_kpa,
    delta_e_sat,
    psychrometric_constant,
    calculate_httc_ground,
    calculate_httc_wall,
    g_coupling_beta,
    solve_tsfc_newton_raphson,
    solve_conduction_thomas_batched,
    AIR_DENSITY,
    AIR_SPECIFIC_HEAT,
)

ROWS, COLS = 4, 4


# ── Vapor Pressure ──────────────────────────────────────────────

class TestVaporPressure:

    def test_e_sat_kpa_at_20c(self, device):
        """e_sat at 20C should be ~2.338 kPa (Magnus formula)."""
        t_k = torch.tensor([293.15], device=device)
        es = e_sat_kpa(t_k)
        assert es.item() == pytest.approx(2.338, abs=0.01)

    def test_e_sat_kpa_at_0c(self, device):
        """e_sat at 0C should be ~0.611 kPa."""
        t_k = torch.tensor([273.15], device=device)
        es = e_sat_kpa(t_k)
        assert es.item() == pytest.approx(0.6108, abs=0.01)

    def test_e_sat_monotonic(self, device):
        """e_sat must increase with temperature."""
        temps = torch.tensor([263.15, 273.15, 283.15, 293.15, 303.15], device=device)
        es = e_sat_kpa(temps)
        for i in range(len(es) - 1):
            assert es[i] < es[i + 1]

    def test_delta_e_sat_at_20c(self, device):
        """Slope of saturation VP curve at 20C should be ~0.145 kPa/K."""
        t_k = torch.tensor([293.15], device=device)
        d = delta_e_sat(t_k)
        assert d.item() == pytest.approx(0.1447, abs=0.01)

    def test_psychrometric_constant_standard(self):
        """Psychrometric constant at 101.325 kPa should be ~0.0665 kPa/K."""
        gamma = psychrometric_constant(101.325)
        assert gamma == pytest.approx(0.0654, abs=0.005)


# ── HTTC ────────────────────────────────────────────────────────

class TestHTTC:

    def test_httc_unstable(self, device):
        """Unstable conditions (Tsfc > Tair) should produce higher HTTC."""
        u = torch.full((ROWS, COLS), 2.0, device=device)
        t_air = torch.full((ROWS, COLS), 293.15, device=device)
        t_hot = torch.full((ROWS, COLS), 310.0, device=device)
        t_cold = torch.full((ROWS, COLS), 280.0, device=device)

        h_unstable = calculate_httc_ground(u, t_air, t_hot, 2.0, 0.01)
        h_stable = calculate_httc_ground(u, t_air, t_cold, 2.0, 0.01)
        # Unstable should give higher convective transfer
        assert h_unstable.mean().item() > h_stable.mean().item()

    def test_httc_neutral_matches_mascart(self, device):
        """At Ri=0 the coefficient matches the Mascart neutral limit."""
        u = torch.full((ROWS, COLS), 0.1, device=device)
        t_air = torch.full((ROWS, COLS), 293.15, device=device)
        t_sfc = t_air + 9.81 / AIR_SPECIFIC_HEAT * 2.0
        h = calculate_httc_ground(u, t_air, t_sfc, 2.0, 0.01)
        ln_m = math.log(2.0 / 0.01)
        ln_h = math.log(2.0 / 0.001)
        expected = AIR_DENSITY * AIR_SPECIFIC_HEAT * 0.1 * (0.4 / ln_m) ** 2 / 0.74 * (ln_m / ln_h)
        assert h.mean().item() == pytest.approx(expected, rel=1.0e-4)

    def test_httc_wall_scalar(self):
        """Wall HTTC at 2 m/s should be reasonable."""
        h = calculate_httc_wall(2.0)
        assert 5.0 <= h <= 100.0

    def test_httc_wall_zero_wind(self):
        """Wall HTTC at zero wind uses u_eff = 0.1 m/s minimum."""
        h = calculate_httc_wall(0.0)
        assert h >= 5.0


# ── G Coupling Beta ─────────────────────────────────────────────

class TestGCouplingBeta:

    def test_beta_typical_values(self, device):
        """Beta should be between 0.2 and 1.0 for typical hourly timestep."""
        hc = torch.full((ROWS, COLS), 1.9e6, device=device)
        d = torch.full((ROWS, COLS), 0.01, device=device)
        k = torch.full((ROWS, COLS), 0.75, device=device)
        beta = g_coupling_beta(hc, d, k, 3600.0)
        assert beta.min().item() >= 0.2
        assert beta.max().item() <= 1.0

    def test_beta_floor(self, device):
        """Beta must not drop below 0.2."""
        hc = torch.full((ROWS, COLS), 1e5, device=device)
        d = torch.full((ROWS, COLS), 0.001, device=device)
        k = torch.full((ROWS, COLS), 10.0, device=device)
        beta = g_coupling_beta(hc, d, k, 3600.0)
        assert beta.min().item() >= 0.2 - 1e-6


# ── Newton-Raphson ──────────────────────────────────────────────

class TestNewtonRaphson:

    @staticmethod
    def _inputs(device, rnet_value=400.0):
        t_air_k = 298.15
        rnet = torch.full((ROWS, COLS), rnet_value, device=device)
        httc = torch.full((ROWS, COLS), 15.0, device=device)
        t_air = torch.full((ROWS, COLS), t_air_k, device=device)
        t_layer = torch.full((ROWS, COLS), t_air_k, device=device)
        emiss = torch.full((ROWS, COLS), 0.95, device=device)
        lam_eff = torch.full((ROWS, COLS), 0.75, device=device)
        d = torch.full((ROWS, COLS), 0.01, device=device)
        t_guess = torch.full((ROWS, COLS), t_air_k, device=device)
        ra = AIR_DENSITY * AIR_SPECIFIC_HEAT / httc
        rs = torch.full((ROWS, COLS), 1e10, device=device)
        return rnet, httc, t_air, t_layer, emiss, lam_eff, d, t_guess, ra, rs

    def test_diagnostics_report_residual_and_iterations(self, device):
        inputs = self._inputs(device)
        result = solve_tsfc_newton_raphson(
            *inputs[:8], ea_kpa=1.0, gamma=0.065, ra=inputs[8], rs=inputs[9],
            return_diagnostics=True,
        )
        assert result.converged.all()
        assert not result.railed.any()
        assert not result.nonfinite.any()
        assert result.residual.abs().max().item() <= 0.1
        assert (result.iterations > 0).all()

    def test_nan_is_failure_not_convergence(self, device):
        inputs = list(self._inputs(device))
        inputs[0][0, 0] = float("nan")
        result = solve_tsfc_newton_raphson(
            *inputs[:8], ea_kpa=1.0, gamma=0.065, ra=inputs[8], rs=inputs[9],
            return_diagnostics=True,
        )
        assert result.nonfinite[0, 0]
        assert not result.converged[0, 0]
        assert result.converged[1:, 1:].all()

    def test_out_of_range_root_is_railed_not_converged(self, device):
        inputs = self._inputs(device, rnet_value=1.0e8)
        result = solve_tsfc_newton_raphson(
            *inputs[:8], ea_kpa=1.0, gamma=0.065, ra=inputs[8], rs=inputs[9],
            return_diagnostics=True,
        )
        assert result.railed.all()
        assert not result.converged.any()

    def test_convergence_no_le(self, device):
        """NR should converge for impervious surface (no latent heat).

        With Rnet=400 W/m2, Tsfc should be warmer than Tair.
        """
        t_air_k = 298.15
        rnet = torch.full((ROWS, COLS), 400.0, device=device)
        httc = torch.full((ROWS, COLS), 15.0, device=device)
        t_air = torch.full((ROWS, COLS), t_air_k, device=device)
        t_layer = torch.full((ROWS, COLS), t_air_k, device=device)
        emiss = torch.full((ROWS, COLS), 0.95, device=device)
        lam_eff = torch.full((ROWS, COLS), 0.75, device=device)
        d = torch.full((ROWS, COLS), 0.01, device=device)
        t_guess = torch.full((ROWS, COLS), t_air_k, device=device)
        ra = AIR_DENSITY * AIR_SPECIFIC_HEAT / httc
        rs = torch.full((ROWS, COLS), 1e10, device=device)  # impervious

        tsfc = solve_tsfc_newton_raphson(
            rnet, httc, t_air, t_layer, emiss, lam_eff, d, t_guess,
            ea_kpa=1.0, gamma=0.065, ra=ra, rs=rs,
        )
        # Surface should be warmer than air
        assert (tsfc > t_air_k).all()
        # Should be within reasonable range
        assert tsfc.max().item() < 373.15

    def test_nr_with_le_cooler(self, device):
        """NR with latent heat (vegetated) should produce cooler Tsfc than impervious."""
        t_air_k = 298.15
        rnet = torch.full((ROWS, COLS), 400.0, device=device)
        httc = torch.full((ROWS, COLS), 15.0, device=device)
        t_air = torch.full((ROWS, COLS), t_air_k, device=device)
        t_layer = torch.full((ROWS, COLS), t_air_k, device=device)
        emiss = torch.full((ROWS, COLS), 0.95, device=device)
        lam_eff = torch.full((ROWS, COLS), 0.75, device=device)
        d = torch.full((ROWS, COLS), 0.01, device=device)
        t_guess = torch.full((ROWS, COLS), t_air_k, device=device)
        ra = AIR_DENSITY * AIR_SPECIFIC_HEAT / httc

        # Impervious
        rs_imp = torch.full((ROWS, COLS), 1e10, device=device)
        tsfc_imp = solve_tsfc_newton_raphson(
            rnet, httc, t_air, t_layer, emiss, lam_eff, d, t_guess,
            ea_kpa=1.0, gamma=0.065, ra=ra, rs=rs_imp,
        )
        # Vegetated (moderate stomatal resistance)
        rs_veg = torch.full((ROWS, COLS), 100.0, device=device)
        tsfc_veg = solve_tsfc_newton_raphson(
            rnet, httc, t_air, t_layer, emiss, lam_eff, d, t_guess,
            ea_kpa=1.0, gamma=0.065, ra=ra, rs=rs_veg,
        )
        # Evaporation should cool the surface
        assert (tsfc_veg < tsfc_imp).all()

    def test_nr_zero_rnet_near_air_temp(self, device):
        """With zero Rnet, surface temp should stay near air temp."""
        t_air_k = 293.15
        rnet = torch.zeros(ROWS, COLS, device=device)
        httc = torch.full((ROWS, COLS), 15.0, device=device)
        t_air = torch.full((ROWS, COLS), t_air_k, device=device)
        t_layer = torch.full((ROWS, COLS), t_air_k, device=device)
        emiss = torch.full((ROWS, COLS), 0.95, device=device)
        lam_eff = torch.full((ROWS, COLS), 0.75, device=device)
        d = torch.full((ROWS, COLS), 0.01, device=device)
        t_guess = torch.full((ROWS, COLS), t_air_k, device=device)
        ra = AIR_DENSITY * AIR_SPECIFIC_HEAT / httc
        rs = torch.full((ROWS, COLS), 1e10, device=device)

        tsfc = solve_tsfc_newton_raphson(
            rnet, httc, t_air, t_layer, emiss, lam_eff, d, t_guess,
            ea_kpa=1.0, gamma=0.065, ra=ra, rs=rs,
        )
        # Should be close to air temp (within a few K)
        diff = (tsfc - t_air_k).abs()
        assert diff.max().item() < 5.0


# ── Thomas Algorithm ────────────────────────────────────────────

class TestThomasAlgorithm:

    def test_steady_state_linear_profile(self, device):
        """Steady state with constant BCs should produce linear temperature profile."""
        N = 16  # 4x4 grid flattened
        n_layers = 4
        t_sfc = torch.full((N,), 310.0, device=device)
        t_deep = torch.full((N,), 290.0, device=device)
        # Start from uniform mid-point
        t_layers = torch.full((N, n_layers), 300.0, device=device)
        thick = torch.full((N, n_layers), 0.05, device=device)
        cond = torch.full((N, n_layers), 1.0, device=device)
        hcap = torch.full((N, n_layers), 2.0e6, device=device)

        # Run many steps to approach steady state
        dt = 3600.0
        for _ in range(200):
            t_layers = solve_conduction_thomas_batched(
                t_layers, t_sfc, t_deep, thick, cond, hcap, dt,
            )

        # Check near-linear profile (monotonic decrease from surface to deep)
        for k in range(n_layers - 1):
            assert (t_layers[:, k] >= t_layers[:, k + 1] - 0.1).all()

    def test_single_layer_forward_euler(self, device):
        """Single-layer implicit solve should preserve a symmetric equilibrium."""
        N = 4
        t_layers = torch.full((N, 1), 300.0, device=device)
        t_sfc = torch.full((N,), 310.0, device=device)
        t_deep = torch.full((N,), 290.0, device=device)
        thick = torch.full((N, 1), 0.05, device=device)
        cond = torch.full((N, 1), 1.0, device=device)
        hcap = torch.full((N, 1), 2.0e6, device=device)

        new = solve_conduction_thomas_batched(
            t_layers, t_sfc, t_deep, thick, cond, hcap, 3600.0,
        )
        # Mid-layer should warm (Tsfc > T_layer > T_deep mean)
        assert new.shape == (N, 1)
        # Temperature should move toward average of BCs
        bc_avg = (310.0 + 290.0) / 2.0
        assert (new[:, 0] - bc_avg).abs().mean().item() < (t_layers[:, 0] - bc_avg).abs().mean().item() + 1.0

    def test_thomas_preserves_shape(self, device):
        """Output shape must match input shape."""
        N, n_layers = 16, 4
        t_layers = torch.full((N, n_layers), 293.15, device=device)
        t_sfc = torch.full((N,), 295.0, device=device)
        t_deep = torch.full((N,), 288.15, device=device)
        thick = torch.full((N, n_layers), 0.05, device=device)
        cond = torch.full((N, n_layers), 1.0, device=device)
        hcap = torch.full((N, n_layers), 2e6, device=device)

        result = solve_conduction_thomas_batched(
            t_layers, t_sfc, t_deep, thick, cond, hcap, 3600.0,
        )
        assert result.shape == (N, n_layers)

    def test_heterogeneous_layers_match_series_resistance(self, device):
        """Steady temperatures must carry one flux through unequal layers."""
        t_sfc = torch.tensor([310.0], device=device)
        t_deep = torch.tensor([290.0], device=device)
        thick = torch.tensor([[0.02, 0.08, 0.20]], device=device)
        cond = torch.tensor([[0.20, 1.60, 0.50]], device=device)
        hcap = torch.tensor([[1.0e6, 2.0e6, 1.5e6]], device=device)
        layers = torch.full((1, 3), 300.0, device=device)

        for _ in range(2000):
            layers = solve_conduction_thomas_batched(
                layers, t_sfc, t_deep, thick, cond, hcap, 3600.0,
            )

        total_r = torch.sum(thick[0] / cond[0])
        flux = (t_sfc[0] - t_deep[0]) / total_r
        centre_r = torch.cumsum(thick[0] / cond[0], dim=0) - thick[0] / (2.0 * cond[0])
        expected = t_sfc[0] - flux * centre_r
        assert torch.allclose(layers[0], expected, atol=0.03)

    def test_conduction_step_conserves_heat(self, device):
        """Layer storage equals top influx minus bottom outflux."""
        old = torch.tensor([[295.0, 298.0, 301.0]], device=device)
        t_sfc = torch.tensor([312.0], device=device)
        t_deep = torch.tensor([289.0], device=device)
        thick = torch.tensor([[0.03, 0.11, 0.25]], device=device)
        cond = torch.tensor([[0.3, 1.8, 0.6]], device=device)
        hcap = torch.tensor([[1.1e6, 2.2e6, 1.4e6]], device=device)
        dt = 1800.0
        new = solve_conduction_thomas_batched(
            old, t_sfc, t_deep, thick, cond, hcap, dt,
        )
        storage = torch.sum(hcap * thick * (new - old), dim=1) / dt
        q_top = (2.0 * cond[:, 0] / thick[:, 0]) * (t_sfc - new[:, 0])
        q_bottom = (2.0 * cond[:, -1] / thick[:, -1]) * (new[:, -1] - t_deep)
        assert torch.allclose(storage, q_top - q_bottom, atol=2.0e-4)
