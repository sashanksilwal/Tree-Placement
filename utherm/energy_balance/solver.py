# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Main ground + roof energy balance solver — all GPU.

EBSolver pre-expands per-pixel material properties at init via tensor
advanced indexing, then solves the full domain in a single vectorized pass.
"""

import math
from typing import Dict, List, Optional

import torch

from .config import EnergyBalanceConfig
from .materials import MaterialProperties
from .physics import (
    STEFAN_BOLTZMANN, AIR_DENSITY, AIR_SPECIFIC_HEAT,
    e_sat_kpa, psychrometric_constant,
    jarvis_stewart_conductance, calculate_httc_ground,
    solve_tsfc_newton_raphson,
    solve_conduction_substepped,
)


class EBSolver:
    """GPU-resident energy balance solver for ground and roof surfaces.

    At init, material properties are expanded to per-pixel (rows, cols) tensors
    using advanced indexing on the material_ids grid. This is done once and
    avoids per-pixel branching during the solve.
    """

    def __init__(
        self,
        config: EnergyBalanceConfig,
        ground_material: MaterialProperties,
        buildings: torch.Tensor,          # (rows, cols) UMEP: 0=building, 1=ground
        alb_grid: torch.Tensor,           # (rows, cols)
        emis_grid: torch.Tensor,          # (rows, cols)
        svf: torch.Tensor,               # (rows, cols)
        svfveg: torch.Tensor,            # (rows, cols)
        svfaveg: torch.Tensor,           # (rows, cols)
        ewall: float = 0.9,
        material_ids: Optional[torch.Tensor] = None,   # (rows, cols) uint8/long
        material_table: Optional[List[MaterialProperties]] = None,
        enable_roof_eb: bool = False,
        roof_material: Optional[MaterialProperties] = None,
        device: torch.device = None,
    ):
        grids = {
            "buildings": buildings,
            "alb_grid": alb_grid,
            "emis_grid": emis_grid,
            "svf": svf,
            "svfveg": svfveg,
            "svfaveg": svfaveg,
        }
        shapes = {name: tuple(grid.shape) for name, grid in grids.items()}
        if len(shapes["buildings"]) != 2:
            raise ValueError(f"buildings must be two-dimensional, got {shapes['buildings']}")
        if any(shape != shapes["buildings"] for shape in shapes.values()):
            raise ValueError(f"energy-balance grid shapes differ: {shapes}")
        for name, grid in grids.items():
            if not bool(torch.isfinite(grid).all()):
                raise ValueError(f"{name} contains non-finite values")
        for name in ("buildings", "alb_grid", "emis_grid", "svf", "svfveg", "svfaveg"):
            grid = grids[name]
            if bool(((grid < 0.0) | (grid > 1.0)).any()):
                raise ValueError(f"{name} must contain values between 0 and 1")
        if not math.isfinite(ewall) or not 0.0 <= ewall <= 1.0:
            raise ValueError("ewall must be between 0 and 1")
        if not isinstance(enable_roof_eb, bool):
            raise ValueError("enable_roof_eb must be boolean")
        if enable_roof_eb and roof_material is None:
            raise ValueError("roof_material is required when enable_roof_eb=True")
        if (material_ids is None) != (material_table is None):
            raise ValueError("material_ids and material_table must be supplied together")

        self.config = config
        self.device = device or buildings.device
        self.rows, self.cols = buildings.shape
        self.ewall = ewall
        self.enable_roof_eb = enable_roof_eb

        # Store SVF grids on device
        self.svf = svf.to(self.device)
        self.svfveg = svfveg.to(self.device)
        self.svfaveg = svfaveg.to(self.device)
        self.alb_grid = alb_grid.to(self.device)
        self.emis_grid = emis_grid.to(self.device)

        # Building mask: True = building pixel (UMEP: buildings < 0.5)
        self.buildings = buildings.to(self.device)
        self.building_mask = self.buildings < 0.5  # True for buildings
        self.ground_mask = ~self.building_mask  # True for ground

        # Pre-expand material properties to per-pixel tensors
        self._expand_material_properties(
            ground_material, material_ids, material_table
        )

        # Pre-expand roof material (if enabled)
        if enable_roof_eb and roof_material is not None:
            self._expand_roof_properties(roof_material)
        else:
            self.roof_thickness = None

        # Wind height correction ratio (computed once)
        self._wind_ratio = 1.0
        if config.z_wind is not None:
            z_w = config.z_wind
            z_r = config.z_ref
            z0 = config.z0
            if z_w > z_r and z0 > 0 and z_w > z0 and z_r > z0:
                self._wind_ratio = math.log(z_r / z0) / math.log(z_w / z0)

    def _expand_material_properties(
        self,
        ground_material: MaterialProperties,
        material_ids: Optional[torch.Tensor],
        material_table: Optional[List[MaterialProperties]],
    ):
        """Build per-pixel (rows, cols, n_layers) property tensors."""
        R, C = self.rows, self.cols
        n = ground_material.n_layers
        self.n_layers = n
        dev = self.device

        if material_table is not None and material_ids is not None:
            n_mats = len(material_table)
            if n_mats == 0:
                raise ValueError("material_table cannot be empty")
            if any(material.n_layers != n for material in material_table):
                raise ValueError("all materials must have the same number of layers")
            if tuple(material_ids.shape) != (R, C):
                raise ValueError(
                    f"material_ids shape {tuple(material_ids.shape)} does not match {(R, C)}"
                )
            if material_ids.dtype not in {
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            }:
                raise ValueError("material_ids must use an integer dtype")
            ids = material_ids.long().to(dev)
            if bool(((ids < 0) | (ids >= n_mats)).any()):
                raise ValueError(f"material_ids must be between 0 and {n_mats - 1}")
            # Build lookup tables: (n_mats, n_layers) for layer props
            thick_lut = torch.tensor(
                [m.thickness for m in material_table], dtype=torch.float32, device=dev
            )  # (n_mats, n_layers)
            cond_lut = torch.tensor(
                [m.conductivity for m in material_table], dtype=torch.float32, device=dev
            )
            hcap_lut = torch.tensor(
                [m.heat_capacity for m in material_table], dtype=torch.float32, device=dev
            )

            # Scalar property LUTs: (n_mats,)
            alb_lut = torch.tensor([m.albedo for m in material_table], dtype=torch.float32, device=dev)
            emis_lut = torch.tensor([m.emissivity for m in material_table], dtype=torch.float32, device=dev)
            maxgs_lut = torch.tensor([m.max_conductance for m in material_table], dtype=torch.float32, device=dev)
            g1_lut = torch.tensor([m.g1_radiation for m in material_table], dtype=torch.float32, device=dev)
            g3_lut = torch.tensor([m.g3_vpd for m in material_table], dtype=torch.float32, device=dev)
            g4_lut = torch.tensor([m.g4_temp for m in material_table], dtype=torch.float32, device=dev)
            topt_lut = torch.tensor([m.temp_optimal for m in material_table], dtype=torch.float32, device=dev)

            # Index into LUTs using material_ids
            flat_ids = ids.reshape(-1)

            self.thickness = thick_lut[flat_ids].reshape(R, C, n)
            self.conductivity = cond_lut[flat_ids].reshape(R, C, n)
            self.heat_capacity = hcap_lut[flat_ids].reshape(R, C, n)

            self.mat_albedo = alb_lut[flat_ids].reshape(R, C)
            self.mat_emissivity = emis_lut[flat_ids].reshape(R, C)
            self.max_conductance = maxgs_lut[flat_ids].reshape(R, C)
            self.g1 = g1_lut[flat_ids].reshape(R, C)
            self.g3 = g3_lut[flat_ids].reshape(R, C)
            self.g4 = g4_lut[flat_ids].reshape(R, C)
            self.temp_optimal = topt_lut[flat_ids].reshape(R, C)
        else:
            # Uniform material for all pixels
            m = ground_material
            self.thickness = torch.tensor(m.thickness, dtype=torch.float32, device=dev).expand(R, C, n).contiguous()
            self.conductivity = torch.tensor(m.conductivity, dtype=torch.float32, device=dev).expand(R, C, n).contiguous()
            self.heat_capacity = torch.tensor(m.heat_capacity, dtype=torch.float32, device=dev).expand(R, C, n).contiguous()

            self.mat_albedo = torch.full((R, C), m.albedo, dtype=torch.float32, device=dev)
            self.mat_emissivity = torch.full((R, C), m.emissivity, dtype=torch.float32, device=dev)
            self.max_conductance = torch.full((R, C), m.max_conductance, dtype=torch.float32, device=dev)
            self.g1 = torch.full((R, C), m.g1_radiation, dtype=torch.float32, device=dev)
            self.g3 = torch.full((R, C), m.g3_vpd, dtype=torch.float32, device=dev)
            self.g4 = torch.full((R, C), m.g4_temp, dtype=torch.float32, device=dev)
            self.temp_optimal = torch.full((R, C), m.temp_optimal, dtype=torch.float32, device=dev)

    def _expand_roof_properties(self, roof_material: MaterialProperties):
        """Build per-pixel roof property tensors (only used at building pixels)."""
        R, C = self.rows, self.cols
        n = roof_material.n_layers
        self.roof_n_layers = n
        dev = self.device
        m = roof_material

        self.roof_thickness = torch.tensor(m.thickness, dtype=torch.float32, device=dev).expand(R, C, n).contiguous()
        self.roof_conductivity = torch.tensor(m.conductivity, dtype=torch.float32, device=dev).expand(R, C, n).contiguous()
        self.roof_heat_capacity = torch.tensor(m.heat_capacity, dtype=torch.float32, device=dev).expand(R, C, n).contiguous()
        self.roof_albedo = m.albedo
        self.roof_emissivity = m.emissivity
        self.roof_max_conductance = m.max_conductance
        self.roof_g1 = m.g1_radiation
        self.roof_g3 = m.g3_vpd
        self.roof_g4 = m.g4_temp
        self.roof_temp_optimal = m.temp_optimal

    def correct_wind(self, ws: float) -> float:
        """Apply neutral log-profile wind correction."""
        return ws * self._wind_ratio

    def solve(
        self,
        rnet: torch.Tensor,           # (rows, cols) net radiation
        shadow: torch.Tensor,         # (rows, cols) shadow factor 0-1
        kdown: torch.Tensor,          # (rows, cols) shortwave
        ta_c: float,                   # Air temperature (°C)
        ws: float,                     # Wind speed (m/s)
        ea_hpa: float,                 # Actual vapor pressure (hPa)
        pressure_kpa: float,           # Atmospheric pressure (kPa; UMEP met convention)
        rad_i: float,                  # Direct radiation (W/m²)
        rad_d: float,                  # Diffuse radiation (W/m²)
        prev_layer_temps: Optional[torch.Tensor] = None,  # (R,C,n_layers)
        prev_tsfc: Optional[torch.Tensor] = None,         # (R,C)
        prev_roof_temps: Optional[torch.Tensor] = None,   # (R,C,roof_n_layers)
        prev_tsfc_roof: Optional[torch.Tensor] = None,    # (R,C)
        solar_elev_deg: float = 90.0,
    ) -> Dict[str, torch.Tensor]:
        """Solve ground + roof energy balance for all pixels.

        Returns dict of GPU tensors:
            tsfc_ground, ground_layer_temps, sensible_heat, ground_heat, latent_heat,
            tsfc_roof, roof_layer_temps, roof_sensible_heat, roof_ground_heat, roof_latent_heat
        """
        R, C = self.rows, self.cols
        dev = self.device
        cfg = self.config
        n = self.n_layers
        expected = (R, C)
        for name, value in (("rnet", rnet), ("shadow", shadow), ("kdown", kdown)):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} shape {tuple(value.shape)} does not match {expected}")
        if bool(torch.isinf(shadow).any()) or bool(
            (torch.isfinite(shadow) & ((shadow < 0.0) | (shadow > 1.0))).any()
        ):
            raise ValueError("shadow must contain values between 0 and 1 or NaN")
        if not bool(torch.isfinite(kdown).all()) or bool((kdown < 0.0).any()):
            raise ValueError("kdown must contain finite nonnegative values")
        if isinstance(ws, torch.Tensor):
            if ws.ndim == 0:
                ws_value = float(ws.item())
                if not math.isfinite(ws_value) or ws_value < 0.0:
                    raise ValueError("ws must be finite and nonnegative")
            elif ws.ndim == 2:
                if tuple(ws.shape) != expected:
                    raise ValueError(f"ws shape {tuple(ws.shape)} does not match {expected}")
                if not bool(torch.isfinite(ws).all()) or bool((ws < 0.0).any()):
                    raise ValueError("ws must contain finite nonnegative values")
            else:
                raise ValueError("ws must be a scalar or a two-dimensional tensor")
        elif not math.isfinite(float(ws)) or float(ws) < 0.0:
            raise ValueError("ws must be finite and nonnegative")
        scalar_bounds = {
            "ta_c": (ta_c, None, None),
            "ea_hpa": (ea_hpa, 0.0, None),
            "pressure_kpa": (pressure_kpa, 0.0, None),
            "rad_i": (rad_i, 0.0, None),
            "rad_d": (rad_d, 0.0, None),
            "solar_elev_deg": (solar_elev_deg, -90.0, 90.0),
        }
        for name, (value, lower, upper) in scalar_bounds.items():
            value = float(value)
            if (
                not math.isfinite(value)
                or (lower is not None and value < lower)
                or (upper is not None and value > upper)
            ):
                raise ValueError(f"invalid {name}: {value}")
        state_shapes = {
            "prev_layer_temps": (R, C, n),
            "prev_tsfc": expected,
            "prev_roof_temps": (
                (R, C, self.roof_n_layers) if self.roof_thickness is not None else None
            ),
            "prev_tsfc_roof": expected if self.roof_thickness is not None else None,
        }
        state_values = {
            "prev_layer_temps": prev_layer_temps,
            "prev_tsfc": prev_tsfc,
            "prev_roof_temps": prev_roof_temps,
            "prev_tsfc_roof": prev_tsfc_roof,
        }
        for name, value in state_values.items():
            if value is not None and tuple(value.shape) != state_shapes[name]:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} does not match {state_shapes[name]}"
                )
        t_air_k = ta_c + 273.15
        nan = float('nan')

        # Wind correction — supports scalar or 2D tensor ws
        if isinstance(ws, torch.Tensor) and ws.dim() >= 2:
            ws_tensor = ws * self._wind_ratio
        else:
            ws_corr = self.correct_wind(float(ws) if isinstance(ws, torch.Tensor) else ws)
            ws_tensor = torch.full((R, C), ws_corr, dtype=torch.float32, device=dev)

        # Convert vapor pressure to kPa
        ea_kpa = ea_hpa * 0.1
        p_kpa = pressure_kpa
        gamma = psychrometric_constant(p_kpa)
        vpd_air = max(e_sat_kpa(torch.tensor(t_air_k)).item() - ea_kpa, 0.0)

        # Ground wetness
        ground_wetness = 1.0 if cfg.ground_wetness is None else max(0.0, min(cfg.ground_wetness, 1.0))

        # ── Initialize previous state ──
        t_air_tensor = torch.full((R, C), t_air_k, dtype=torch.float32, device=dev)

        if prev_layer_temps is None:
            prev_layer_temps = torch.full((R, C, n), t_air_k, dtype=torch.float32, device=dev)
        if prev_tsfc is None:
            prev_tsfc = t_air_tensor.clone()

        # ── Compute httc from previous Tsfc ──
        httc = calculate_httc_ground(
            ws_tensor,
            t_air_tensor,
            prev_tsfc,
            cfg.z_ref,
            cfg.z0,
        )
        httc = torch.clamp(httc, min=1e-6)

        # ── Jarvis-Stewart surface conductance ──
        gs = jarvis_stewart_conductance(
            self.max_conductance, kdown, vpd_air, ta_c,
            self.g1, self.g3, self.g4, self.temp_optimal,
        ) * ground_wetness
        rs = torch.where(gs > 1e-8, 1.0 / gs, torch.full_like(gs, 1e10))

        # Conductance from the surface to the centre of the first layer.
        d = self.thickness[:, :, 0] / 2.0
        lam = self.conductivity[:, :, 0]

        # ── Emissivity (use grid emissivity to match Rnet computation) ──
        emiss = self.emis_grid

        # Aerodynamic resistance
        ra = AIR_DENSITY * AIR_SPECIFIC_HEAT / httc

        # The implicit layer solution is affine in surface temperature:
        # T1_new = intercept + response*Ts. Evaluate that response once so the
        # surface Newton solve includes the new layer state exactly.
        N = R * C
        layers_flat = prev_layer_temps.reshape(N, n)
        t_deep_scalar = torch.full((N,), cfg.t_deep, dtype=torch.float32, device=dev)
        thick_flat = self.thickness.reshape(N, n)
        cond_flat = self.conductivity.reshape(N, n)
        hcap_flat = self.heat_capacity.reshape(N, n)
        ref_layers = solve_conduction_substepped(
            layers_flat, prev_tsfc.reshape(N), t_deep_scalar,
            thick_flat, cond_flat, hcap_flat, cfg.dt,
        ).reshape(R, C, n)
        plus_layers = solve_conduction_substepped(
            layers_flat, (prev_tsfc + 1.0).reshape(N), t_deep_scalar,
            thick_flat, cond_flat, hcap_flat, cfg.dt,
        ).reshape(R, C, n)
        response = plus_layers[:, :, 0] - ref_layers[:, :, 0]
        one_minus_response = torch.clamp(1.0 - response, min=1.0e-6)
        intercept = ref_layers[:, :, 0] - response * prev_tsfc
        layer_equivalent = intercept / one_minus_response
        interface_conductance = lam / torch.clamp(d, min=1.0e-6)
        effective_conductivity = interface_conductance * one_minus_response * d

        surface_solution = solve_tsfc_newton_raphson(
            rnet, httc, t_air_tensor, layer_equivalent, emiss,
            effective_conductivity, d, prev_tsfc, ea_kpa, gamma, ra, rs,
            n_iterations=cfg.max_iterations,
            residual_tolerance=cfg.surface_residual_tolerance,
            valid_mask=self.ground_mask,
            return_diagnostics=True,
        )
        tsfc = surface_solution.temperature
        ground_failed = self.ground_mask & ~surface_solution.converged
        tsfc = torch.where(ground_failed, torch.full_like(tsfc, nan), tsfc)
        new_layers_flat = solve_conduction_substepped(
            layers_flat, tsfc.reshape(N), t_deep_scalar,
            thick_flat, cond_flat, hcap_flat, cfg.dt,
        )
        new_layer_temps = new_layers_flat.reshape(R, C, n)

        # ── Compute fluxes ──
        qh = httc * (tsfc - t_air_k)
        qg = lam / torch.clamp(d, min=1e-6) * (tsfc - new_layer_temps[:, :, 0])

        es_sfc = e_sat_kpa(tsfc)
        le_coeff = torch.where(
            rs < 1e6,
            AIR_DENSITY * AIR_SPECIFIC_HEAT / (gamma * (ra + rs)),
            torch.zeros_like(ra),
        )
        qe = torch.clamp(le_coeff * (es_sfc - ea_kpa), min=0.0)

        # ── Mask buildings to NaN ──
        bldg = self.building_mask
        tsfc = torch.where(bldg, torch.full_like(tsfc, nan), tsfc)
        qh = torch.where(bldg, torch.full_like(qh, nan), qh)
        qg = torch.where(bldg, torch.full_like(qg, nan), qg)
        qe = torch.where(bldg, torch.full_like(qe, nan), qe)
        new_layer_temps = torch.where(
            bldg.unsqueeze(-1).expand_as(new_layer_temps),
            torch.full_like(new_layer_temps, nan),
            new_layer_temps,
        )

        result = {
            'tsfc_ground': tsfc,
            'ground_layer_temps': new_layer_temps,
            'sensible_heat': qh,
            'ground_heat': qg,
            'latent_heat': qe,
            'surface_solver_residual': surface_solution.residual,
            'surface_solver_converged': surface_solution.converged,
            'surface_solver_railed': surface_solution.railed,
            'surface_solver_nonfinite': surface_solution.nonfinite,
            'surface_solver_iterations': surface_solution.iterations,
        }

        # ── Roof energy balance ──
        if self.enable_roof_eb and self.roof_thickness is not None:
            roof_result = self._solve_roof(
                shadow, ta_c, ws_tensor, ea_kpa, gamma, vpd_air,
                rad_i, rad_d, t_air_k,
                prev_roof_temps, prev_tsfc_roof,
                solar_elev_deg,
            )
            result.update(roof_result)
        else:
            result['tsfc_roof'] = torch.full((R, C), nan, dtype=torch.float32, device=dev)
            result['roof_layer_temps'] = torch.full((R, C, n), nan, dtype=torch.float32, device=dev)
            result['roof_sensible_heat'] = torch.full((R, C), nan, dtype=torch.float32, device=dev)
            result['roof_ground_heat'] = torch.full((R, C), nan, dtype=torch.float32, device=dev)
            result['roof_latent_heat'] = torch.full((R, C), nan, dtype=torch.float32, device=dev)
            result['roof_surface_solver_residual'] = torch.full((R, C), nan, dtype=torch.float32, device=dev)
            result['roof_surface_solver_converged'] = torch.zeros((R, C), dtype=torch.bool, device=dev)
            result['roof_surface_solver_railed'] = torch.zeros((R, C), dtype=torch.bool, device=dev)
            result['roof_surface_solver_nonfinite'] = torch.zeros((R, C), dtype=torch.bool, device=dev)
            result['roof_surface_solver_iterations'] = torch.zeros((R, C), dtype=torch.int32, device=dev)
            result['roof_coupling_converged'] = torch.zeros((R, C), dtype=torch.bool, device=dev)

        return result

    def _solve_roof(
        self,
        shadow: torch.Tensor,
        ta_c: float,
        ws: float,
        ea_kpa: float,
        gamma: float,
        vpd_air: float,
        rad_i: float,
        rad_d: float,
        t_air_k: float,
        prev_roof_temps: Optional[torch.Tensor],
        prev_tsfc_roof: Optional[torch.Tensor],
        solar_elev_deg: float,
    ) -> Dict[str, torch.Tensor]:
        """Solve roof energy balance at building pixels."""
        R, C = self.rows, self.cols
        dev = self.device
        cfg = self.config
        rn = self.roof_n_layers
        nan = float('nan')

        t_air_tensor = torch.full((R, C), t_air_k, dtype=torch.float32, device=dev)

        if prev_roof_temps is None:
            prev_roof_temps = torch.full((R, C, rn), t_air_k, dtype=torch.float32, device=dev)
        if prev_tsfc_roof is None:
            prev_tsfc_roof = t_air_tensor.clone()

        # Roof Rnet: SVF ≈ 1, Prata 1996 esky
        ea_hpa_val = ea_kpa * 10.0
        msteg = 46.5 * (ea_hpa_val / t_air_k)
        esky = 1.0 - (1.0 + msteg) * math.exp(-math.sqrt(1.2 + 3.0 * msteg))

        # Shadow on roof: use existing shadow, treat NaN as sunlit
        shadow_roof = torch.where(torch.isnan(shadow), torch.ones_like(shadow), shadow)
        sin_beta = max(math.sin(math.radians(solar_elev_deg)), 0.0)
        kdown_roof = rad_i * sin_beta * shadow_roof + rad_d

        ta4 = t_air_k ** 4
        rnet_roof = ((1.0 - self.roof_albedo) * kdown_roof
                     + self.roof_emissivity * esky * STEFAN_BOLTZMANN * ta4
                     - self.roof_emissivity * STEFAN_BOLTZMANN * ta4)

        # Roof wetness
        roof_wetness = 1.0 if cfg.roof_wetness is None else max(0.0, min(cfg.roof_wetness, 1.0))

        # Jarvis-Stewart for roof (uniform material)
        kdown_roof_clamped = torch.clamp(kdown_roof, min=0.0)
        gs_r = self._roof_jarvis(kdown_roof_clamped, vpd_air, ta_c) * roof_wetness
        rs_r = torch.where(gs_r > 1e-8, 1.0 / gs_r, torch.full_like(gs_r, 1e10))

        d_r = self.roof_thickness[:, :, 0] / 2.0
        lam_r = self.roof_conductivity[:, :, 0]

        # Roof deep temperature
        roof_deep = t_air_k if cfg.roof_t_deep is None else cfg.roof_t_deep

        # Iterate surface temperature, convective transfer and layer conduction.
        if isinstance(ws, torch.Tensor) and ws.dim() >= 2:
            ws_tensor = ws
        else:
            ws_tensor = torch.full((R, C), float(ws), dtype=torch.float32, device=dev)
        httc_r = torch.clamp(
            calculate_httc_ground(ws_tensor, t_air_tensor, prev_tsfc_roof, cfg.z_ref, cfg.z0),
            min=1e-6,
        )
        tsfc_r = prev_tsfc_roof.clone()

        N = R * C
        roof_deep_tensor = torch.full((N,), roof_deep, dtype=torch.float32, device=dev)
        old_roof_layers = prev_roof_temps.reshape(N, rn)
        roof_thickness_flat = self.roof_thickness.reshape(N, rn)
        roof_conductivity_flat = self.roof_conductivity.reshape(N, rn)
        roof_capacity_flat = self.roof_heat_capacity.reshape(N, rn)
        roof_ref_layers = solve_conduction_substepped(
            old_roof_layers, prev_tsfc_roof.reshape(N), roof_deep_tensor,
            roof_thickness_flat, roof_conductivity_flat, roof_capacity_flat, cfg.dt,
        ).reshape(R, C, rn)
        roof_plus_layers = solve_conduction_substepped(
            old_roof_layers, (prev_tsfc_roof + 1.0).reshape(N), roof_deep_tensor,
            roof_thickness_flat, roof_conductivity_flat, roof_capacity_flat, cfg.dt,
        ).reshape(R, C, rn)
        roof_response = roof_plus_layers[:, :, 0] - roof_ref_layers[:, :, 0]
        roof_one_minus_response = torch.clamp(1.0 - roof_response, min=1.0e-6)
        roof_intercept = roof_ref_layers[:, :, 0] - roof_response * prev_tsfc_roof
        roof_layer_equivalent = roof_intercept / roof_one_minus_response
        roof_interface_conductance = lam_r / torch.clamp(d_r, min=1.0e-6)
        roof_effective_conductivity = roof_interface_conductance * roof_one_minus_response * d_r

        roof_coupling_converged = torch.zeros((R, C), dtype=torch.bool, device=dev)
        for _ in range(cfg.max_iterations):
            old_tsfc_r = tsfc_r
            ra_r = AIR_DENSITY * AIR_SPECIFIC_HEAT / httc_r
            roof_iteration = solve_tsfc_newton_raphson(
                rnet_roof, httc_r, t_air_tensor, roof_layer_equivalent,
                torch.full((R, C), self.roof_emissivity, dtype=torch.float32, device=dev),
                roof_effective_conductivity, d_r,
                tsfc_r, ea_kpa, gamma, ra_r, rs_r,
                n_iterations=cfg.max_iterations,
                residual_tolerance=cfg.surface_residual_tolerance,
                valid_mask=self.building_mask,
                return_diagnostics=True,
            )
            tsfc_r = roof_iteration.temperature
            httc_new = torch.clamp(
                calculate_httc_ground(ws_tensor, t_air_tensor, tsfc_r, cfg.z_ref, cfg.z0),
                min=1e-6,
            )
            rel_change = (httc_new - httc_r).abs() / torch.clamp(httc_r, min=1.0)
            temp_change = (tsfc_r - old_tsfc_r).abs()
            httc_r = httc_new
            finite_change = torch.isfinite(rel_change) & torch.isfinite(temp_change)
            roof_coupling_converged = (
                self.building_mask & roof_iteration.converged & finite_change
                & (rel_change < 0.01) & (temp_change < cfg.convergence_threshold)
            )
            if bool(roof_coupling_converged[self.building_mask].all()):
                break

        # One final solve uses the converged convective coefficient.
        ra_r = AIR_DENSITY * AIR_SPECIFIC_HEAT / httc_r
        roof_solution = solve_tsfc_newton_raphson(
            rnet_roof, httc_r, t_air_tensor, roof_layer_equivalent,
            torch.full((R, C), self.roof_emissivity, dtype=torch.float32, device=dev),
            roof_effective_conductivity, d_r, tsfc_r, ea_kpa, gamma, ra_r, rs_r,
            n_iterations=cfg.max_iterations,
            residual_tolerance=cfg.surface_residual_tolerance,
            valid_mask=self.building_mask,
            return_diagnostics=True,
        )
        tsfc_r = roof_solution.temperature
        roof_failed = self.building_mask & (
            ~roof_solution.converged | ~roof_coupling_converged
        )
        tsfc_r = torch.where(roof_failed, torch.full_like(tsfc_r, nan), tsfc_r)
        new_roof_layers = solve_conduction_substepped(
            old_roof_layers, tsfc_r.reshape(N), roof_deep_tensor,
            roof_thickness_flat, roof_conductivity_flat, roof_capacity_flat, cfg.dt,
        ).reshape(R, C, rn)

        # Roof fluxes
        h_roof = httc_r * (tsfc_r - t_air_k)
        g_roof = lam_r / torch.clamp(d_r, min=1e-6) * (tsfc_r - new_roof_layers[:, :, 0])
        ra_r = AIR_DENSITY * AIR_SPECIFIC_HEAT / httc_r
        le_coeff_r = torch.where(
            rs_r < 1e6,
            AIR_DENSITY * AIR_SPECIFIC_HEAT / (gamma * (ra_r + rs_r)),
            torch.zeros_like(ra_r),
        )
        le_roof = torch.clamp(le_coeff_r * (e_sat_kpa(tsfc_r) - ea_kpa), min=0.0)

        # Mask ground pixels to NaN (roof only at buildings)
        gnd = self.ground_mask
        tsfc_r = torch.where(gnd, torch.full_like(tsfc_r, nan), tsfc_r)
        h_roof = torch.where(gnd, torch.full_like(h_roof, nan), h_roof)
        g_roof = torch.where(gnd, torch.full_like(g_roof, nan), g_roof)
        le_roof = torch.where(gnd, torch.full_like(le_roof, nan), le_roof)
        new_roof_layers = torch.where(
            gnd.unsqueeze(-1).expand_as(new_roof_layers),
            torch.full_like(new_roof_layers, nan),
            new_roof_layers,
        )

        return {
            'tsfc_roof': tsfc_r,
            'roof_layer_temps': new_roof_layers,
            'roof_sensible_heat': h_roof,
            'roof_ground_heat': g_roof,
            'roof_latent_heat': le_roof,
            'roof_surface_solver_residual': roof_solution.residual,
            'roof_surface_solver_converged': roof_solution.converged,
            'roof_surface_solver_railed': roof_solution.railed,
            'roof_surface_solver_nonfinite': roof_solution.nonfinite,
            'roof_surface_solver_iterations': roof_solution.iterations,
            'roof_coupling_converged': roof_coupling_converged,
        }

    def _roof_jarvis(self, kdown: torch.Tensor, vpd_air: float, ta_c: float) -> torch.Tensor:
        """Jarvis-Stewart for uniform roof material."""
        max_gs = self.roof_max_conductance
        if max_gs <= 0:
            return torch.zeros_like(kdown)
        if max_gs > 900:
            return torch.full_like(kdown, max_gs * 0.001)

        kdown_pos = torch.clamp(kdown, min=0.0)
        f_rad = torch.clamp(kdown_pos / torch.clamp(kdown_pos + self.roof_g1, min=0.01), 0.0, 1.0)
        f_vpd = 1.0 / (1.0 + self.roof_g3 * max(vpd_air, 0.0))
        dt = ta_c - self.roof_temp_optimal
        f_temp = max(0.01, 1.0 - self.roof_g4 * dt * dt)

        return max_gs * 0.001 * f_rad * f_vpd * f_temp
