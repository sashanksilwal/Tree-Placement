# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for canyon air temperature solver."""

from utherm.energy_balance.canyon import solve_canyon_air_temperature
from utherm.energy_balance.config import CanyonAirTempConfig


class TestCanyonAirTemperature:

    def test_warming_with_positive_qh(self):
        """Positive sensible heat should warm canyon air."""
        config = CanyonAirTempConfig(dt=3600.0)
        ta_k = 298.15
        tcan = solve_canyon_air_temperature(
            tcan_prev=ta_k,
            ta_k=ta_k,
            ws=2.0,
            mean_sensible_heat=200.0,
            config=config,
        )
        assert tcan > ta_k

    def test_cooling_with_negative_qh(self):
        """Negative sensible heat should cool canyon air."""
        config = CanyonAirTempConfig(dt=3600.0)
        ta_k = 298.15
        tcan = solve_canyon_air_temperature(
            tcan_prev=ta_k,
            ta_k=ta_k,
            ws=2.0,
            mean_sensible_heat=-100.0,
            config=config,
        )
        assert tcan < ta_k

    def test_zero_qh_stays_near_ta(self):
        """Zero QH should keep canyon temp close to air temp."""
        config = CanyonAirTempConfig(dt=3600.0)
        ta_k = 298.15
        tcan = solve_canyon_air_temperature(
            tcan_prev=ta_k,
            ta_k=ta_k,
            ws=2.0,
            mean_sensible_heat=0.0,
            config=config,
        )
        assert abs(tcan - ta_k) < 1.0
