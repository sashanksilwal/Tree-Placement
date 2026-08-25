# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for net radiation calculation."""

import pytest
import torch

from utherm.energy_balance.net_radiation import calculate_net_radiation
from utherm.energy_balance.physics import STEFAN_BOLTZMANN
from utherm.radiation import _ground_temperature_for_longwave

ROWS, COLS = 4, 4


class TestNetRadiation:

    def test_empirical_water_uses_documented_class_and_not_eb(self, device):
        ground_delta = torch.tensor([[1.0, 2.0, 3.0]], device=device)
        landcover = torch.tensor([[3, 7, 1]], device=device)
        empirical = _ground_temperature_for_longwave(
            ground_delta, landcover, 20.0, 15.0, True
        )
        assert empirical.tolist() == [[1.0, -5.0, 3.0]]
        assert ground_delta.tolist() == [[1.0, 2.0, 3.0]]
        physical = _ground_temperature_for_longwave(
            ground_delta, landcover, 20.0, 15.0, False
        )
        assert physical is ground_delta

    def test_rnet_grid_computation(self, device):
        """Rnet = (1-a)*K + e*L - e*s*T^4 should produce correct values."""
        ldown = torch.full((ROWS, COLS), 350.0, device=device)
        kdown = torch.full((ROWS, COLS), 500.0, device=device)
        alb = torch.full((ROWS, COLS), 0.2, device=device)
        emis = torch.full((ROWS, COLS), 0.95, device=device)
        t_air_k = 298.15

        rnet = calculate_net_radiation(ldown, kdown, alb, emis, t_air_k)

        expected_k_abs = 0.8 * 500.0
        expected_l_in = 0.95 * 350.0
        expected_l_out = 0.95 * STEFAN_BOLTZMANN * (298.15 ** 4)
        expected = expected_k_abs + expected_l_in - expected_l_out

        assert rnet[0, 0].item() == pytest.approx(expected, rel=1e-4)

    def test_higher_kdown_higher_rnet(self, device):
        """Increasing Kdown should increase Rnet."""
        ldown = torch.full((ROWS, COLS), 300.0, device=device)
        alb = torch.full((ROWS, COLS), 0.2, device=device)
        emis = torch.full((ROWS, COLS), 0.95, device=device)
        t_air_k = 293.15

        kdown_low = torch.full((ROWS, COLS), 200.0, device=device)
        kdown_high = torch.full((ROWS, COLS), 800.0, device=device)

        rnet_low = calculate_net_radiation(ldown, kdown_low, alb, emis, t_air_k)
        rnet_high = calculate_net_radiation(ldown, kdown_high, alb, emis, t_air_k)

        assert (rnet_high > rnet_low).all()

    def test_rnet_per_pixel_albedo(self, device):
        """Pixels with higher albedo should have lower Rnet."""
        ldown = torch.full((ROWS, COLS), 300.0, device=device)
        kdown = torch.full((ROWS, COLS), 500.0, device=device)
        emis = torch.full((ROWS, COLS), 0.95, device=device)
        t_air_k = 293.15

        alb_low = torch.full((ROWS, COLS), 0.1, device=device)
        alb_high = torch.full((ROWS, COLS), 0.5, device=device)

        rnet_dark = calculate_net_radiation(ldown, kdown, alb_low, emis, t_air_k)
        rnet_bright = calculate_net_radiation(ldown, kdown, alb_high, emis, t_air_k)

        assert (rnet_dark > rnet_bright).all()
