import numpy as np
import pytest

from study_code.geospatial_preprocessing.derive_ground_cover import derive_ground_cover


def test_canopy_classes_are_filled_only_from_observed_ground():
    landcover = np.array(
        [
            [1, 1, 2, 5, 5],
            [1, 4, 4, 4, 5],
            [1, 4, 3, 4, 5],
            [6, 4, 4, 4, 7],
        ],
        dtype=np.int16,
    )
    derived, distance = derive_ground_cover(landcover, 1.0)
    vegetation = np.isin(landcover, (3, 4))
    assert set(np.unique(derived[vegetation])).issubset({1, 5, 6, 7})
    np.testing.assert_array_equal(derived[~vegetation], landcover[~vegetation])
    assert np.all(distance[vegetation] > 0.0)
    assert np.all(distance[~vegetation] == 0.0)


def test_canopy_fill_requires_an_observed_ground_class():
    with pytest.raises(ValueError, match="no observed ground"):
        derive_ground_cover(np.array([[2, 4], [4, 2]], dtype=np.int16), 1.0)
