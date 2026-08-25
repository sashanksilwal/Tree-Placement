# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Jarvis-Stewart surface conductance."""

import pytest
import torch

from utherm.energy_balance.physics import jarvis_stewart_conductance

ROWS, COLS = 4, 4


class TestJarvisStewart:

    def test_impervious_zero_conductance(self, device):
        """Impervious surface (max_gs=0) must have zero conductance."""
        gs = jarvis_stewart_conductance(
            max_gs_mm_s=torch.zeros(ROWS, COLS, device=device),
            kdown=torch.full((ROWS, COLS), 500.0, device=device),
            vpd_kpa=1.0,
            ta_c=25.0,
            g1=torch.full((ROWS, COLS), 100.0, device=device),
            g3=torch.full((ROWS, COLS), 0.5, device=device),
            g4=torch.full((ROWS, COLS), 0.0016, device=device),
            temp_opt=torch.full((ROWS, COLS), 25.0, device=device),
        )
        assert (gs == 0.0).all()

    def test_grass_positive_conductance(self, device):
        """Grass (max_gs=12 mm/s) under sunlight should have positive conductance."""
        gs = jarvis_stewart_conductance(
            max_gs_mm_s=torch.full((ROWS, COLS), 12.0, device=device),
            kdown=torch.full((ROWS, COLS), 500.0, device=device),
            vpd_kpa=1.0,
            ta_c=25.0,
            g1=torch.full((ROWS, COLS), 100.0, device=device),
            g3=torch.full((ROWS, COLS), 0.5, device=device),
            g4=torch.full((ROWS, COLS), 0.0016, device=device),
            temp_opt=torch.full((ROWS, COLS), 25.0, device=device),
        )
        assert (gs > 0.0).all()
        # Should be less than max_gs in m/s (12 mm/s = 0.012 m/s)
        assert gs.max().item() <= 0.012 + 1e-6

    def test_water_free_evaporation(self, device):
        """Water (max_gs=999 mm/s) should use free evaporation (bypass stress)."""
        gs = jarvis_stewart_conductance(
            max_gs_mm_s=torch.full((ROWS, COLS), 999.0, device=device),
            kdown=torch.full((ROWS, COLS), 0.0, device=device),  # even at night
            vpd_kpa=0.0,
            ta_c=25.0,
            g1=torch.zeros(ROWS, COLS, device=device),
            g3=torch.zeros(ROWS, COLS, device=device),
            g4=torch.zeros(ROWS, COLS, device=device),
            temp_opt=torch.full((ROWS, COLS), 25.0, device=device),
        )
        assert gs.mean().item() == pytest.approx(0.999, abs=0.001)
