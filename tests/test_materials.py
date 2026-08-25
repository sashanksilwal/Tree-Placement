# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for material presets and canopy properties."""

import pytest

from utherm.energy_balance.materials import MaterialProperties
from utherm.energy_balance.config import CanopyProperties, EnergyBalanceConfig


class TestRoofMaterial:

    def test_roof_4_layers(self):
        m = MaterialProperties.roof()
        assert m.n_layers == 4
        assert len(m.thickness) == 4
        assert len(m.conductivity) == 4
        assert len(m.heat_capacity) == 4

    def test_roof_impervious(self):
        """Standard roof should have zero stomatal conductance."""
        m = MaterialProperties.roof()
        assert m.max_conductance == 0.0

    def test_roof_albedo(self):
        m = MaterialProperties.roof()
        assert m.albedo == 0.25

    def test_roof_xps_insulation(self):
        """Layer 3 (XPS insulation) should have low conductivity."""
        m = MaterialProperties.roof()
        assert m.conductivity[2] == pytest.approx(0.04, abs=0.01)


class TestGreenRoofMaterial:

    def test_green_roof_vegetated(self):
        """Green roof should have positive stomatal conductance."""
        m = MaterialProperties.green_roof()
        assert m.max_conductance > 0.0

    def test_green_roof_substrate_top(self):
        """First layer should be soil substrate."""
        m = MaterialProperties.green_roof()
        assert m.thickness[0] == 0.05
        assert m.conductivity[0] == pytest.approx(0.5, abs=0.1)


class TestAsphaltMaterial:

    def test_asphalt_impervious(self):
        m = MaterialProperties.asphalt()
        assert m.max_conductance == 0.0
        assert m.albedo == 0.12

    def test_asphalt_emissivity(self):
        m = MaterialProperties.asphalt()
        assert m.emissivity == 0.95


class TestCanopyProperties:

    def test_deciduous_defaults(self):
        p = CanopyProperties.deciduous()
        assert p.albedo_leaf == 0.20
        assert p.max_stomatal_conductance == 10.0
        assert p.g1 == 100.0
        assert not p.allow_dew

    def test_evergreen_smaller_leaves(self):
        """Evergreen should have smaller leaf size than deciduous."""
        dec = CanopyProperties.deciduous()
        evg = CanopyProperties.evergreen()
        assert evg.d_leaf < dec.d_leaf

    def test_evergreen_lower_albedo(self):
        dec = CanopyProperties.deciduous()
        evg = CanopyProperties.evergreen()
        assert evg.albedo_leaf < dec.albedo_leaf


class TestPropertyValidation:

    def test_layer_lengths_must_match(self):
        with pytest.raises(ValueError, match="thickness"):
            MaterialProperties(n_layers=2, thickness=[0.1])

    def test_thermal_properties_must_be_positive(self):
        with pytest.raises(ValueError, match="conductivity"):
            MaterialProperties(conductivity=[0.75, 0.0, 0.75, 0.75])

    def test_optical_properties_are_bounded(self):
        with pytest.raises(ValueError, match="albedo"):
            MaterialProperties(albedo=1.1)

    def test_energy_balance_wetness_is_bounded(self):
        with pytest.raises(ValueError, match="ground_wetness"):
            EnergyBalanceConfig(ground_wetness=1.1)

    def test_canopy_leaf_size_must_be_positive(self):
        with pytest.raises(ValueError, match="d_leaf"):
            CanopyProperties(d_leaf=0.0)
