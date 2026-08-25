# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for EBBridge integration layer."""

import pytest
import torch

from utherm.eb_bridge import EBBridge
from utherm.energy_balance.materials import MaterialProperties
from utherm.energy_balance.config import CanopyProperties, CanyonAirTempConfig

ROWS, COLS = 4, 4


@pytest.fixture
def bridge(device):
    """Create an EBBridge with all-ground, asphalt."""
    buildings = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)
    alb = torch.full((ROWS, COLS), 0.12, dtype=torch.float32, device=device)
    emis = torch.full((ROWS, COLS), 0.95, dtype=torch.float32, device=device)
    svf = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)

    return EBBridge(
        ground_material=MaterialProperties.asphalt(),
        buildings=buildings,
        alb_grid=alb,
        emis_grid=emis,
        svf=svf,
        svfveg=svf,
        svfaveg=svf,
        device=device,
    )


class TestBridgeInit:

    def test_bridge_creates_solver(self, bridge):
        """Bridge should create an internal EBSolver."""
        assert bridge.solver is not None

    def test_bridge_initial_state_none(self, bridge):
        """Initial state should be None (lazy init)."""
        assert bridge.prev_layer_temps is None
        assert bridge.prev_tsfc is None


class TestBridgeDaytime:

    def test_compute_tg_returns_tuple(self, device, bridge):
        """compute_tg should return (Tg, qh, qe) tensors."""
        kdown = torch.full((ROWS, COLS), 500.0, device=device)
        shadow = torch.ones(ROWS, COLS, device=device)

        Tg, qh, qe = bridge.compute_tg(
            esky=0.7, Ta=25.0, ws=2.0, RH=50.0, P=101.325,
            radI=400.0, radD=100.0,
            Kdown_gpu=kdown, shadow_gpu=shadow,
            CI=1.0, device=device,
        )
        assert Tg.shape == (ROWS, COLS)
        assert qh.shape == (ROWS, COLS)
        assert qe.shape == (ROWS, COLS)

    def test_tg_positive_daytime(self, device, bridge):
        """Daytime Tg (surface - air delta) should be positive for asphalt."""
        kdown = torch.full((ROWS, COLS), 500.0, device=device)
        shadow = torch.ones(ROWS, COLS, device=device)

        Tg, _, _ = bridge.compute_tg(
            esky=0.7, Ta=25.0, ws=2.0, RH=50.0, P=101.325,
            radI=400.0, radD=100.0,
            Kdown_gpu=kdown, shadow_gpu=shadow,
            CI=1.0, device=device,
        )
        # Asphalt should warm above air
        assert Tg.mean().item() > 0.0


class TestBridgeNight:

    def test_compute_tg_night(self, device, bridge):
        """Nighttime solve should work and produce near-zero or negative Tg."""
        # First do a daytime step to populate state
        kdown = torch.full((ROWS, COLS), 500.0, device=device)
        shadow = torch.ones(ROWS, COLS, device=device)
        bridge.compute_tg(
            esky=0.7, Ta=25.0, ws=2.0, RH=50.0, P=101.325,
            radI=400.0, radD=100.0,
            Kdown_gpu=kdown, shadow_gpu=shadow,
            CI=1.0, device=device,
        )

        Tg, qh, qe = bridge.compute_tg_night(
            Ta=20.0, ws=1.0, RH=60.0, P=101.325, device=device,
        )
        assert Tg.shape == (ROWS, COLS)
        assert torch.isfinite(Tg).all()
        expected = bridge.last_tsfc_ground - 273.15 - 20.0
        torch.testing.assert_close(Tg, expected)


class TestBridgeCanopy:

    def test_canopy_eb_returns_tau_ldown(self, device):
        """compute_canopy_eb should return (tau, ldown_emit) tensors."""
        buildings = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)
        alb = torch.full((ROWS, COLS), 0.12, dtype=torch.float32, device=device)
        emis = torch.full((ROWS, COLS), 0.95, dtype=torch.float32, device=device)
        svf = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)
        mask = torch.ones(ROWS, COLS, dtype=torch.bool, device=device)
        lai = torch.full((ROWS, COLS), 3.0, device=device)

        bridge = EBBridge(
            ground_material=MaterialProperties.asphalt(),
            buildings=buildings, alb_grid=alb, emis_grid=emis,
            svf=svf, svfveg=svf, svfaveg=svf,
            canopy_properties=CanopyProperties.deciduous(),
            canopy_mask=mask,
            lai_grid=lai,
            device=device,
        )

        kdown = torch.full((ROWS, COLS), 500.0, device=device)
        prescribed_tau = torch.full((ROWS, COLS), 0.35, device=device)
        result = bridge.compute_canopy_eb(
            kdown, Ta=25.0, ws=2.0, RH=50.0, P=101.325,
            altitude_deg=45.0,
            shortwave_transmittance_gpu=prescribed_tau,
        )
        assert result is not None
        tau, ldown_emit = result
        assert tau.shape == (ROWS, COLS)
        assert torch.allclose(tau, prescribed_tau)
        assert (ldown_emit >= 0.0).all()


class TestBridgeCanyon:

    def test_canyon_air_temp_update(self, device):
        """update_canyon_air_temp should modify tcan_prev."""
        buildings = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)
        alb = torch.full((ROWS, COLS), 0.12, dtype=torch.float32, device=device)
        emis = torch.full((ROWS, COLS), 0.95, dtype=torch.float32, device=device)
        svf = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)

        bridge = EBBridge(
            ground_material=MaterialProperties.asphalt(),
            buildings=buildings, alb_grid=alb, emis_grid=emis,
            svf=svf, svfveg=svf, svfaveg=svf,
            canyon_config=CanyonAirTempConfig(),
            device=device,
        )

        qh = torch.full((ROWS, COLS), 100.0, device=device)
        bridge.last_tsfc_ground = torch.full(
            (ROWS, COLS), 303.15, device=device
        )
        bridge.update_canyon_air_temp(qh, Ta=25.0, ws=2.0)

        assert bridge.tcan_prev is not None
        # With positive QH, canyon should be warmer than air
        assert bridge.tcan_prev > 298.15


class TestBridgeStatePersistence:

    def test_state_persists_across_steps(self, device, bridge):
        """Layer temps and Tsfc should persist between solve calls."""
        kdown = torch.full((ROWS, COLS), 500.0, device=device)
        shadow = torch.ones(ROWS, COLS, device=device)

        bridge.compute_tg(
            esky=0.7, Ta=25.0, ws=2.0, RH=50.0, P=101.325,
            radI=400.0, radD=100.0,
            Kdown_gpu=kdown, shadow_gpu=shadow,
            CI=1.0, device=device,
        )

        # State should now be populated
        assert bridge.prev_layer_temps is not None
        assert bridge.prev_tsfc is not None

        # Save state
        tsfc_step1 = bridge.prev_tsfc.clone()

        # Second step
        bridge.compute_tg(
            esky=0.7, Ta=25.0, ws=2.0, RH=50.0, P=101.325,
            radI=400.0, radD=100.0,
            Kdown_gpu=kdown, shadow_gpu=shadow,
            CI=1.0, device=device,
        )

        # State should have changed (thermal mass evolves)
        tsfc_step2 = bridge.prev_tsfc
        assert not torch.equal(tsfc_step1, tsfc_step2)
        assert bridge.prev_layer_temps is not None
