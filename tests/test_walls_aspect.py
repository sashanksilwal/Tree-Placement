# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for wall-height and wall-aspect preprocessing."""

import numpy as np
import pytest

from utherm.walls_aspect import (
    cart2pol,
    findwalls,
    get_ders,
    process_file_parallel,
    run_parallel_processing,
)


def test_findwalls_marks_the_exterior_of_a_raised_cell():
    dsm = np.zeros((5, 5), dtype=float)
    dsm[2, 2] = 10.0
    walls = findwalls(dsm, 3.0)
    expected = np.zeros_like(dsm)
    expected[1, 2] = expected[2, 1] = expected[2, 3] = expected[3, 2] = 10.0
    np.testing.assert_array_equal(walls, expected)


def test_findwalls_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="two-dimensional"):
        findwalls(np.zeros(4), 3.0)
    with pytest.raises(ValueError, match="non-finite"):
        findwalls(np.full((3, 3), np.nan), 3.0)
    with pytest.raises(ValueError, match="non-negative"):
        findwalls(np.zeros((3, 3)), -1.0)


def test_cart2pol_known_axes_and_units():
    theta, radius = cart2pol(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    np.testing.assert_allclose(theta, [0.0, 90.0], atol=1.0e-12)
    np.testing.assert_allclose(radius, [1.0, 1.0], atol=1.0e-12)
    with pytest.raises(ValueError, match="units"):
        cart2pol(1.0, 0.0, units="gradians")


def test_get_ders_matches_a_planar_surface():
    rows, columns = np.mgrid[:5, :6]
    dsm = 2.0 * rows + 3.0 * columns
    gradient, aspect = get_ders(dsm, scale=1.0)
    np.testing.assert_allclose(gradient, np.arctan(np.sqrt(13.0)), atol=1.0e-12)
    expected_aspect = (-np.arctan2(3.0, 2.0)) % (2.0 * np.pi)
    np.testing.assert_allclose(aspect, expected_aspect, atol=1.0e-12)


def test_wall_worker_propagates_missing_input(tmp_path):
    arguments = ("Building_DSM_0_0.tif", tmp_path, tmp_path, tmp_path)
    with pytest.raises(RuntimeError, match="failed to process"):
        process_file_parallel(arguments)


def test_parallel_wall_processing_requires_tiles(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(FileNotFoundError, match="no GeoTIFF tiles"):
        run_parallel_processing(source, tmp_path / "walls", tmp_path / "aspect")
