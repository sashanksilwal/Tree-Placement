# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for EBSolver integration (ground + roof)."""

import pytest
import torch

from utherm.energy_balance.solver import EBSolver
from utherm.energy_balance.materials import MaterialProperties
from utherm.energy_balance.config import EnergyBalanceConfig
from utherm.energy_balance.net_radiation import calculate_net_radiation

ROWS, COLS = 4, 4


@pytest.fixture
def solver_setup(device):
    """Create an EBSolver with all-ground, uniform asphalt."""
    config = EnergyBalanceConfig()
    mat = MaterialProperties.asphalt()
    buildings = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)
    alb = torch.full((ROWS, COLS), mat.albedo, dtype=torch.float32, device=device)
    emis = torch.full((ROWS, COLS), mat.emissivity, dtype=torch.float32, device=device)
    svf = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)
    svfveg = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)
    svfaveg = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)

    solver = EBSolver(
        config=config,
        ground_material=mat,
        buildings=buildings,
        alb_grid=alb,
        emis_grid=emis,
        svf=svf,
        svfveg=svfveg,
        svfaveg=svfaveg,
        device=device,
    )
    return solver


@pytest.fixture
def rnet_daytime(device):
    """Positive net radiation for daytime conditions."""
    ldown = torch.full((ROWS, COLS), 350.0, device=device)
    kdown = torch.full((ROWS, COLS), 500.0, device=device)
    alb = torch.full((ROWS, COLS), 0.12, device=device)
    emis = torch.full((ROWS, COLS), 0.95, device=device)
    rnet = calculate_net_radiation(ldown, kdown, alb, emis, 298.15)
    return rnet


class TestEBSolverGround:

    def test_constructor_rejects_mismatched_grids(self, device):
        base = torch.ones(ROWS, COLS, device=device)
        with pytest.raises(ValueError, match="grid shapes differ"):
            EBSolver(
                EnergyBalanceConfig(), MaterialProperties.asphalt(),
                buildings=base, alb_grid=base[:-1], emis_grid=base,
                svf=base, svfveg=base, svfaveg=base, device=device,
            )

    def test_constructor_rejects_invalid_material_id(self, device):
        base = torch.ones(ROWS, COLS, device=device)
        ids = torch.zeros(ROWS, COLS, dtype=torch.long, device=device)
        ids[0, 0] = 2
        with pytest.raises(ValueError, match="material_ids must be between"):
            EBSolver(
                EnergyBalanceConfig(), MaterialProperties.asphalt(),
                buildings=base, alb_grid=base * 0.2, emis_grid=base * 0.95,
                svf=base, svfveg=base, svfaveg=base,
                material_ids=ids,
                material_table=[MaterialProperties.asphalt(), MaterialProperties.grass()],
                device=device,
            )

    def test_roof_model_requires_roof_material(self, device):
        base = torch.ones(ROWS, COLS, device=device)
        with pytest.raises(ValueError, match="roof_material is required"):
            EBSolver(
                EnergyBalanceConfig(), MaterialProperties.asphalt(),
                buildings=base, alb_grid=base * 0.2, emis_grid=base * 0.95,
                svf=base, svfveg=base, svfaveg=base,
                enable_roof_eb=True, device=device,
            )

    def test_solve_rejects_invalid_wind(self, device, solver_setup, rnet_daytime):
        with pytest.raises(ValueError, match="ws must be finite and nonnegative"):
            solver_setup.solve(
                rnet=rnet_daytime,
                shadow=torch.ones(ROWS, COLS, device=device),
                kdown=torch.full((ROWS, COLS), 500.0, device=device),
                ta_c=25.0, ws=-1.0, ea_hpa=15.0, pressure_kpa=101.325,
                rad_i=400.0, rad_d=100.0,
            )

    def test_solve_rejects_state_shape_mismatch(self, device, solver_setup, rnet_daytime):
        with pytest.raises(ValueError, match="prev_layer_temps shape"):
            solver_setup.solve(
                rnet=rnet_daytime,
                shadow=torch.ones(ROWS, COLS, device=device),
                kdown=torch.full((ROWS, COLS), 500.0, device=device),
                ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
                rad_i=400.0, rad_d=100.0,
                prev_layer_temps=torch.zeros(ROWS, COLS, 3, device=device),
            )

    def test_ground_solve_returns_all_keys(self, device, solver_setup, rnet_daytime):
        """Solve should return all expected output keys."""
        result = solver_setup.solve(
            rnet=rnet_daytime,
            shadow=torch.ones(ROWS, COLS, device=device),
            kdown=torch.full((ROWS, COLS), 500.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            rad_i=400.0, rad_d=100.0,
        )
        expected_keys = {
            'tsfc_ground', 'ground_layer_temps', 'sensible_heat',
            'ground_heat', 'latent_heat', 'tsfc_roof', 'roof_layer_temps',
            'roof_sensible_heat', 'roof_ground_heat', 'roof_latent_heat',
            'surface_solver_residual', 'surface_solver_converged',
            'surface_solver_railed', 'surface_solver_nonfinite',
            'surface_solver_iterations', 'roof_surface_solver_residual',
            'roof_surface_solver_converged', 'roof_surface_solver_railed',
            'roof_surface_solver_nonfinite', 'roof_surface_solver_iterations',
            'roof_coupling_converged',
        }
        assert set(result.keys()) == expected_keys

    def test_ground_solver_exposes_convergence_diagnostics(self, device, solver_setup, rnet_daytime):
        result = solver_setup.solve(
            rnet=rnet_daytime,
            shadow=torch.ones(ROWS, COLS, device=device),
            kdown=torch.full((ROWS, COLS), 500.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            rad_i=400.0, rad_d=100.0,
        )
        assert result['surface_solver_converged'].all()
        assert not result['surface_solver_railed'].any()
        assert not result['surface_solver_nonfinite'].any()
        assert result['surface_solver_residual'].abs().max().item() <= 0.1

    def test_nonfinite_forcing_is_not_returned_as_temperature(self, device, solver_setup, rnet_daytime):
        rnet_daytime[0, 0] = float("nan")
        result = solver_setup.solve(
            rnet=rnet_daytime,
            shadow=torch.ones(ROWS, COLS, device=device),
            kdown=torch.full((ROWS, COLS), 500.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            rad_i=400.0, rad_d=100.0,
        )
        assert result['surface_solver_nonfinite'][0, 0]
        assert not result['surface_solver_converged'][0, 0]
        assert torch.isnan(result['tsfc_ground'][0, 0])

    def test_ground_tsfc_warmer_than_air(self, device, solver_setup, rnet_daytime):
        """Surface should be warmer than air under positive Rnet."""
        result = solver_setup.solve(
            rnet=rnet_daytime,
            shadow=torch.ones(ROWS, COLS, device=device),
            kdown=torch.full((ROWS, COLS), 500.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            rad_i=400.0, rad_d=100.0,
        )
        t_air_k = 298.15
        valid = ~torch.isnan(result['tsfc_ground'])
        assert (result['tsfc_ground'][valid] > t_air_k).all()

    def test_surface_energy_balance_closes(self, device, solver_setup, rnet_daytime):
        result = solver_setup.solve(
            rnet=rnet_daytime,
            shadow=torch.ones(ROWS, COLS, device=device),
            kdown=torch.full((ROWS, COLS), 500.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            rad_i=400.0, rad_d=100.0,
        )
        tsfc = result['tsfc_ground']
        thermal_lw = 0.95 * 5.67051e-8 * (tsfc ** 4 - 298.15 ** 4)
        residual = (
            rnet_daytime - thermal_lw - result['sensible_heat']
            - result['ground_heat'] - result['latent_heat']
        )
        valid = torch.isfinite(residual)
        assert residual[valid].abs().max().item() < 0.1


class TestImperviousSurface:

    def test_le_zero_for_asphalt(self, device, solver_setup, rnet_daytime):
        """Asphalt (impervious) should have zero latent heat."""
        result = solver_setup.solve(
            rnet=rnet_daytime,
            shadow=torch.ones(ROWS, COLS, device=device),
            kdown=torch.full((ROWS, COLS), 500.0, device=device),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            rad_i=400.0, rad_d=100.0,
        )
        valid = ~torch.isnan(result['latent_heat'])
        assert result['latent_heat'][valid].abs().max().item() < 0.1


class TestGreenRoofCooling:

    def test_roof_rejects_nonfinite_wind(self, device):
        config = EnergyBalanceConfig()
        buildings = torch.zeros(ROWS, COLS, dtype=torch.float32, device=device)
        grid = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)
        solver = EBSolver(
            config=config,
            ground_material=MaterialProperties.asphalt(),
            buildings=buildings,
            alb_grid=grid * 0.2,
            emis_grid=grid * 0.95,
            svf=grid,
            svfveg=grid,
            svfaveg=grid,
            enable_roof_eb=True,
            roof_material=MaterialProperties.roof(),
            device=device,
        )
        with pytest.raises(ValueError, match="ws must be finite"):
            solver.solve(
                rnet=grid * 300.0,
                shadow=grid,
                kdown=grid * 500.0,
                ta_c=25.0,
                ws=float("nan"),
                ea_hpa=15.0,
                pressure_kpa=101.325,
                rad_i=400.0,
                rad_d=100.0,
            )

    def test_green_roof_le_cooling(self, device):
        """Green roof should have latent heat > 0 and cooler than standard roof."""
        config = EnergyBalanceConfig()
        buildings = torch.zeros(ROWS, COLS, dtype=torch.float32, device=device)  # all buildings
        alb = torch.full((ROWS, COLS), 0.2, dtype=torch.float32, device=device)
        emis = torch.full((ROWS, COLS), 0.95, dtype=torch.float32, device=device)
        svf = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)

        # Standard roof
        solver_std = EBSolver(
            config=config,
            ground_material=MaterialProperties.asphalt(),
            buildings=buildings, alb_grid=alb, emis_grid=emis,
            svf=svf, svfveg=svf, svfaveg=svf,
            enable_roof_eb=True,
            roof_material=MaterialProperties.roof(),
            device=device,
        )
        # Green roof
        solver_grn = EBSolver(
            config=config,
            ground_material=MaterialProperties.asphalt(),
            buildings=buildings, alb_grid=alb, emis_grid=emis,
            svf=svf, svfveg=svf, svfaveg=svf,
            enable_roof_eb=True,
            roof_material=MaterialProperties.green_roof(),
            device=device,
        )

        # Compute Rnet for roof
        ldown = torch.full((ROWS, COLS), 350.0, device=device)
        kdown = torch.full((ROWS, COLS), 500.0, device=device)
        rnet = calculate_net_radiation(ldown, kdown, alb, emis, 298.15)

        solve_kwargs = dict(
            rnet=rnet,
            shadow=torch.ones(ROWS, COLS, device=device),
            kdown=kdown,
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            rad_i=400.0, rad_d=100.0,
        )
        result_std = solver_std.solve(**solve_kwargs)
        result_grn = solver_grn.solve(**solve_kwargs)

        # Green roof should have positive LE at building pixels
        bldg = buildings < 0.5
        le_grn = result_grn['roof_latent_heat'][bldg]
        valid = ~torch.isnan(le_grn)
        assert le_grn[valid].mean().item() > 0.0

        # Green roof should be cooler than standard roof
        tsfc_std = result_std['tsfc_roof'][bldg]
        tsfc_grn = result_grn['tsfc_roof'][bldg]
        v_std = ~torch.isnan(tsfc_std)
        v_grn = ~torch.isnan(tsfc_grn)
        assert tsfc_grn[v_grn].mean().item() < tsfc_std[v_std].mean().item()

    def test_roof_direct_beam_is_projected_to_horizontal(self, device):
        """A horizontal roof receives DNI multiplied by sine of elevation."""
        config = EnergyBalanceConfig()
        buildings = torch.zeros(ROWS, COLS, dtype=torch.float32, device=device)
        grid = torch.full((ROWS, COLS), 0.2, dtype=torch.float32, device=device)
        svf = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)
        solver = EBSolver(
            config=config, ground_material=MaterialProperties.asphalt(),
            buildings=buildings, alb_grid=grid, emis_grid=torch.full_like(grid, 0.95),
            svf=svf, svfveg=svf, svfaveg=svf, enable_roof_eb=True,
            roof_material=MaterialProperties.roof(), device=device,
        )
        kwargs = dict(
            rnet=torch.zeros_like(grid), shadow=torch.ones_like(grid), kdown=torch.zeros_like(grid),
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            rad_i=800.0, rad_d=0.0,
        )
        low = solver.solve(**kwargs, solar_elev_deg=10.0)['tsfc_roof'].nanmean()
        solver = EBSolver(
            config=config, ground_material=MaterialProperties.asphalt(),
            buildings=buildings, alb_grid=grid, emis_grid=torch.full_like(grid, 0.95),
            svf=svf, svfveg=svf, svfaveg=svf, enable_roof_eb=True,
            roof_material=MaterialProperties.roof(), device=device,
        )
        high = solver.solve(**kwargs, solar_elev_deg=80.0)['tsfc_roof'].nanmean()
        assert high > low


class TestBuildingMasking:

    def test_buildings_masked_to_nan(self, device):
        """Building pixels should be NaN in ground outputs."""
        config = EnergyBalanceConfig()
        mat = MaterialProperties.asphalt()
        buildings = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)
        buildings[0, 0] = 0.0  # one building pixel
        alb = torch.full((ROWS, COLS), 0.12, dtype=torch.float32, device=device)
        emis = torch.full((ROWS, COLS), 0.95, dtype=torch.float32, device=device)
        svf = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)

        solver = EBSolver(
            config=config, ground_material=mat,
            buildings=buildings, alb_grid=alb, emis_grid=emis,
            svf=svf, svfveg=svf, svfaveg=svf, device=device,
        )

        ldown = torch.full((ROWS, COLS), 350.0, device=device)
        kdown = torch.full((ROWS, COLS), 500.0, device=device)
        rnet = calculate_net_radiation(ldown, kdown, alb, emis, 298.15)

        result = solver.solve(
            rnet=rnet,
            shadow=torch.ones(ROWS, COLS, device=device),
            kdown=kdown,
            ta_c=25.0, ws=2.0, ea_hpa=15.0, pressure_kpa=101.325,
            rad_i=400.0, rad_d=100.0,
        )

        # Building pixel (0,0) should be NaN
        assert torch.isnan(result['tsfc_ground'][0, 0])
        assert torch.isnan(result['sensible_heat'][0, 0])
        # Ground pixel should NOT be NaN
        assert not torch.isnan(result['tsfc_ground'][1, 1])
