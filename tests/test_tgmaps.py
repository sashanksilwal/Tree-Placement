import numpy as np
import pytest

from utherm.Tgmaps_v1 import Tgmaps_v1


LC_CLASS = np.array([
    [1, 0.18, 0.95, 0.58, -9.78, 15.0],
    [5, 0.16, 0.94, 0.21, -3.38, 14.0],
    [6, 0.25, 0.94, 0.33, -3.01, 14.0],
    [7, 0.05, 0.98, 0.00, 0.00, 12.0],
    [99, 0.20, 0.90, 0.58, -3.41, 15.0],
])


def test_integer_landcover_preserves_float_properties():
    grid = np.array([[1, 5], [6, 7]], dtype=np.int64)

    tgk, tstart, albedo, emissivity, tgk_wall, tstart_wall, tmax, tmax_wall = Tgmaps_v1(
        grid, LC_CLASS
    )

    np.testing.assert_allclose(albedo, [[0.18, 0.16], [0.25, 0.05]])
    np.testing.assert_allclose(emissivity, [[0.95, 0.94], [0.94, 0.98]])
    np.testing.assert_allclose(tgk, [[0.58, 0.21], [0.33, 0.00]])
    np.testing.assert_allclose(tstart, [[-9.78, -3.38], [-3.01, 0.00]])
    np.testing.assert_allclose(tmax, [[15.0, 14.0], [14.0, 12.0]])
    np.testing.assert_allclose([tgk_wall[0], tstart_wall[0], tmax_wall[0]], [0.58, -3.41, 15.0])


def test_unknown_landcover_code_is_rejected():
    with pytest.raises(ValueError, match="class 4"):
        Tgmaps_v1(np.array([[4]], dtype=np.int64), LC_CLASS)
