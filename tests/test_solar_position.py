# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference and seasonal tests for solar geometry."""

import numpy as np
import torch

from utherm.radiation import daylen
from utherm.sun_position import Solweig_2015a_metdata_noload, sun_position


def test_solar_position_matches_nrel_spa_reference_case():
    result = sun_position(
        {
            "year": 2003,
            "month": 10,
            "day": 17,
            "hour": 12,
            "min": 30,
            "sec": 30,
            "UTC": -7,
        },
        {
            "latitude": 39.742476,
            "longitude": -105.1786,
            "altitude": 1830.14,
        },
    )
    np.testing.assert_allclose(result["zenith"], 50.11162, atol=0.01)
    np.testing.assert_allclose(result["azimuth"], 194.34024, atol=0.01)


def test_day_length_has_expected_solstice_ordering(device):
    latitude = torch.tensor(30.27, device=device)
    summer = daylen(torch.tensor(172.0, device=device), latitude)[0]
    equinox = daylen(torch.tensor(80.0, device=device), latitude)[0]
    winter = daylen(torch.tensor(355.0, device=device), latitude)[0]
    assert summer > equinox > winter
    assert abs(equinox.item() - 12.0) < 0.5


def test_deciduous_leaf_season_turns_off_in_winter():
    met = np.zeros((2, 24), dtype=float)
    met[:, 0] = 2023
    met[:, 1] = (50, 200)
    met[:, 2] = 12
    location = {"latitude": 40.0, "longitude": -86.0, "altitude": 200.0}
    *_, leaf_on, _, _ = Solweig_2015a_metdata_noload(met, location, -5.0)
    np.testing.assert_array_equal(leaf_on, [[0.0, 1.0]])
