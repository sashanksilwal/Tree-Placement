# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""PyTorch bridge between the radiation and energy-balance calculations.

Raster state remains on the selected device between timesteps. A few scalar
reductions are copied to the host for the canopy and canyon diagnostics.
"""

import math
from typing import Optional, Tuple

import torch

from .energy_balance import (
    MaterialProperties,
    EnergyBalanceConfig,
    CanyonAirTempConfig,
    CanopyProperties,
    EBSolver,
    solve_canopy_eb,
    solve_canopy_eb_two_leaf,
    solve_canyon_air_temperature_coupled,
    calculate_net_radiation,
    STEFAN_BOLTZMANN,
)
from .energy_balance.canopy import canopy_hemispherical_transmittance


class EBBridge:
    """Bridge between radiation model and PyTorch energy balance solver.

    Raster inputs, outputs and state use the selected PyTorch device.
    """

    def __init__(
        self,
        ground_material: MaterialProperties,
        buildings: torch.Tensor,      # (rows, cols) GPU tensor, UMEP: 0=bldg, 1=ground
        alb_grid: torch.Tensor,       # (rows, cols) GPU
        emis_grid: torch.Tensor,      # (rows, cols) GPU
        svf: torch.Tensor,            # (rows, cols) GPU
        svfveg: torch.Tensor,         # (rows, cols) GPU
        svfaveg: torch.Tensor,        # (rows, cols) GPU
        z0: float = 0.01,
        z_wind: Optional[float] = None,
        dt: float = 3600.0,
        ewall: float = 0.9,
        material_ids: Optional[torch.Tensor] = None,   # (rows, cols) GPU long
        material_table: Optional[list] = None,
        # Roof EB
        enable_roof_eb: bool = False,
        roof_material: Optional[MaterialProperties] = None,
        # Canopy EB
        canopy_properties: Optional[CanopyProperties] = None,
        canopy_mask: Optional[torch.Tensor] = None,    # (rows, cols) GPU bool
        lai_grid: Optional[torch.Tensor] = None,       # (rows, cols) GPU float32
        canopy_model: str = "big_leaf",
        # Canyon air temp
        canyon_config: Optional[CanyonAirTempConfig] = None,
        ground_wetness: Optional[float] = None,   # 0–1 soil-moisture multiplier on ground stomatal conductance; None→1.0
        device: torch.device = None,
    ):
        self.device = device or buildings.device
        dev = self.device
        self.ewall = ewall

        # Config
        self.config = EnergyBalanceConfig(
            z0=z0, dt=dt, z_wind=z_wind, ground_wetness=ground_wetness,
        )

        # Store GPU grids
        self.buildings = buildings.to(dev, dtype=torch.float32)
        self.svf = svf.to(dev, dtype=torch.float32)
        self.svfveg = svfveg.to(dev, dtype=torch.float32)
        self.svfaveg = svfaveg.to(dev, dtype=torch.float32)
        self.rows, self.cols = buildings.shape

        # Create solver
        self.solver = EBSolver(
            config=self.config,
            ground_material=ground_material,
            buildings=buildings,
            alb_grid=alb_grid,
            emis_grid=emis_grid,
            svf=svf,
            svfveg=svfveg,
            svfaveg=svfaveg,
            ewall=ewall,
            material_ids=material_ids,
            material_table=material_table,
            enable_roof_eb=enable_roof_eb,
            roof_material=roof_material,
            device=dev,
        )

        # Building mask validation
        bmean = float(buildings.float().mean().item())
        if bmean > 0.99:
            print("WARNING: building mask mean > 0.99 (nearly all ground). "
                  "UMEP convention: 0=building, 1=ground. Mask may be correct for open terrain.")
        elif bmean < 0.01:
            print("WARNING: building mask mean < 0.01 (nearly all building). "
                  "UMEP convention: 0=building, 1=ground. Check if mask is inverted.")

        # Ground state (GPU tensors, None until first timestep)
        self.prev_layer_temps = None   # (R, C, n_layers)
        self.prev_tsfc = None          # (R, C)

        # Diagnostics (GPU tensors)
        self.last_tsfc_ground = None
        self.last_qh = None
        self.last_qe = None

        # Roof EB
        self.enable_roof_eb = enable_roof_eb
        self.roof_material = roof_material
        self.prev_roof_temps = None
        self.prev_tsfc_roof = None
        self.last_tsfc_roof = None
        self.last_roof_qh = None
        self.last_roof_qe = None

        # Canopy EB
        self.canopy_properties = canopy_properties
        self.canopy_mask = canopy_mask.to(dev, dtype=torch.bool) if canopy_mask is not None else None
        self.lai_grid = lai_grid.to(dev, dtype=torch.float32) if lai_grid is not None else None
        self.canopy_model = canopy_model
        self.canopy_longwave_tau = (
            canopy_hemispherical_transmittance(
                self.lai_grid, canopy_properties.clumping_factor,
            )
            if self.lai_grid is not None and canopy_properties is not None
            else None
        )
        self.prev_t_leaf = None
        self.last_t_leaf = None
        self.last_canopy_qh = None
        self.last_canopy_qe = None
        self.last_canopy_rnet = None
        self.last_canopy_solver_residual = None
        self.last_canopy_solver_converged = None
        self.last_canopy_solver_railed = None
        self.last_canopy_solver_nonfinite = None
        self.last_canopy_solver_iterations = None

        # Canyon air temp
        self.canyon_config = canyon_config
        self.tcan_prev = None
        self.last_tair_2m = None

        # Prognostic advection-diffusion air-temperature field (lazily created)
        self.advdiff = None

        # Wind correction
        self._wind_ratio = 1.0
        if z_wind is not None:
            z_r = self.config.z_ref
            if z_wind > z_r and z0 > 0 and z_wind > z0 and z_r > z0:
                self._wind_ratio = math.log(z_r / z0) / math.log(z_wind / z0)

    @staticmethod
    def _to_float(v) -> float:
        return float(v) if isinstance(v, torch.Tensor) else float(v)

    def _correct_wind(self, ws: float) -> float:
        return ws * self._wind_ratio

    # ── SVF-weighted Ldown for EB (Jonsson et al. 2006) ──────────

    def _compute_ldown_eb(self, esky: float, Ta: float, CI: float) -> torch.Tensor:
        """SVF-weighted Ldown for energy balance — all GPU tensors."""
        Ta_K = Ta + 273.15

        # Mean ground/wall temperature offset from previous step
        Tgwall_eb = 0.0
        if self.prev_tsfc is not None:
            valid_mask = ~torch.isnan(self.prev_tsfc)
            if valid_mask.any():
                Tgwall_eb = float(self.prev_tsfc[valid_mask].mean().item()) - Ta - 273.15
                if math.isnan(Tgwall_eb):
                    Tgwall_eb = 0.0

        svf = self.svf
        svfveg = self.svfveg
        svfaveg = self.svfaveg
        ewall = self.ewall
        SBC = STEFAN_BOLTZMANN

        TaK4 = Ta_K ** 4
        TgK4 = (Ta_K + Tgwall_eb) ** 4

        Ldown_eb = (
            (svf + svfveg - 1) * esky * SBC * TaK4
            + (2 - svfveg - svfaveg) * ewall * SBC * TaK4
            + (svfaveg - svf) * ewall * SBC * TgK4
            + (2 - svf - svfveg) * (1 - ewall) * esky * SBC * TaK4
        )

        if CI < 0.95:
            c = 1 - CI
            Ldown_cloudy = (
                (svf + svfveg - 1) * SBC * TaK4
                + (2 - svfveg - svfaveg) * ewall * SBC * TaK4
                + (svfaveg - svf) * ewall * SBC * TgK4
                + (2 - svf - svfveg) * (1 - ewall) * SBC * TaK4
            )
            Ldown_eb = Ldown_eb * (1 - c) + c * Ldown_cloudy

        return Ldown_eb

    # ── Canopy energy balance ─────────────────────────────────────

    def compute_canopy_eb(
        self,
        Kdown_gpu: torch.Tensor,
        Ta, ws, RH, P, altitude_deg,
        Kdown_beam_gpu: Optional[torch.Tensor] = None,
        Kdown_diffuse_gpu: Optional[torch.Tensor] = None,
        shortwave_transmittance_gpu: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute canopy energy balance for vegetated pixels — all GPU.

        Args:
            Kdown_gpu: Incident direct-plus-diffuse shortwave (GPU tensor, rows x cols).
            Kdown_beam_gpu: Optional incident direct beam on the horizontal plane.
            Kdown_diffuse_gpu: Optional incident diffuse irradiance.
            shortwave_transmittance_gpu: Beam/diffuse-weighted gap fraction already
                used by the radiation solver. Supplying it keeps canopy absorption
                consistent with the single ground-level attenuation pass.
            Ta: Air temperature (scalar or tensor).
            ws: Wind speed — scalar or (rows, cols) tensor.
            RH, P, altitude_deg: Meteorological scalars.

        Returns:
            Diagnostic ``(shortwave_tau, canopy_ldown_emit)`` or ``None``.
            These are not additional multipliers for SOLWEIG radiation.
        """
        if (self.canopy_properties is None or self.canopy_mask is None
                or self.lai_grid is None):
            return None

        Ta_f = self._to_float(Ta)
        # ws can be scalar or 2D tensor; canopy EB uses a scalar representative value
        if isinstance(ws, torch.Tensor) and ws.dim() >= 2:
            # Use mean wind over canopy pixels for canopy EB
            if self.canopy_mask.any():
                ws_f = float(ws[self.canopy_mask].mean().item())
            else:
                ws_f = float(ws.mean().item())
            ws_f = self._correct_wind(ws_f)
        else:
            ws_f = self._correct_wind(self._to_float(ws))
        RH_f = self._to_float(RH)
        P_f = self._to_float(P)
        alt_f = self._to_float(altitude_deg)

        # Compute esky (Prata 1996)
        ea_hpa = 6.107 * 10 ** ((7.5 * Ta_f) / (237.3 + Ta_f)) * (RH_f / 100.0)
        msteg = 46.5 * (ea_hpa / (Ta_f + 273.15))
        esky = 1 - (1 + msteg) * math.exp(-math.sqrt(1.2 + 3.0 * msteg))

        # Uniform sky Ldown for canopy
        Ta_K = Ta_f + 273.15
        Ldown_canopy = float(esky) * STEFAN_BOLTZMANN * (Ta_K ** 4)
        Ldown_2d = torch.full(
            (self.rows, self.cols), Ldown_canopy,
            dtype=torch.float32, device=self.device,
        )

        props = self.canopy_properties
        tau_lw = self.canopy_longwave_tau

        if self.canopy_model == "two_big_leaf":
            beam = Kdown_beam_gpu
            diffuse = Kdown_diffuse_gpu
            if beam is None or diffuse is None:
                beam = torch.zeros_like(Kdown_gpu)
                diffuse = Kdown_gpu
            result = solve_canopy_eb_two_leaf(
                beam, diffuse,
                Ldown_2d, Ta_f, ws_f, ea_hpa, P_f, alt_f,
                self.canopy_mask, self.lai_grid, props,
                self.prev_t_leaf, self.device,
            )
        else:
            result = solve_canopy_eb(
                Kdown_gpu, Ldown_2d, Ta_f, ws_f, ea_hpa, P_f,
                self.canopy_mask, self.lai_grid, props,
                self.prev_t_leaf, self.device,
                shortwave_transmittance=shortwave_transmittance_gpu,
                longwave_transmittance=tau_lw,
            )

        # Update state
        self.prev_t_leaf = result['t_leaf'].clone()
        self.last_t_leaf = result['t_leaf']
        self.last_canopy_qh = result['canopy_qh']
        self.last_canopy_qe = result['canopy_qe']
        self.last_canopy_rnet = result['canopy_rnet']
        self.last_canopy_solver_residual = result['canopy_solver_residual']
        self.last_canopy_solver_converged = result['canopy_solver_converged']
        self.last_canopy_solver_railed = result['canopy_solver_railed']
        self.last_canopy_solver_nonfinite = result['canopy_solver_nonfinite']
        self.last_canopy_solver_iterations = result['canopy_solver_iterations']

        valid = self.canopy_mask & (self.lai_grid > 0)
        if shortwave_transmittance_gpu is None:
            tau = tau_lw
        else:
            tau = torch.as_tensor(
                shortwave_transmittance_gpu,
                dtype=self.lai_grid.dtype,
                device=self.device,
            ).expand_as(self.lai_grid).clamp(0.0, 1.0)
        tau = torch.where(valid, tau, torch.ones_like(self.lai_grid))

        # Diagnostic hemispheric canopy LW emission. SOLWEIG already represents
        # vegetation in its directional longwave calculation, so this value must
        # not be applied as a second post-hoc attenuation/emission layer.
        t_leaf = result['t_leaf']
        ldown_emit = torch.where(
            valid & ~torch.isnan(t_leaf),
            props.emissivity_leaf * STEFAN_BOLTZMANN * t_leaf ** 4 * (1.0 - tau_lw),
            torch.zeros_like(t_leaf),
        )

        return tau, ldown_emit

    # ── Ground EB (daytime) ───────────────────────────────────────

    def compute_tg(
        self,
        esky, Ta, ws, RH, P, radI, radD,
        Kdown_gpu: torch.Tensor,
        shadow_gpu: torch.Tensor,
        CI, device, solar_elev_deg: float = 90.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Physics-based surface temperature — all GPU.

        Args:
            ws: Wind speed — scalar float or (rows, cols) GPU tensor.

        Returns:
            (Tg_gpu, qh_gpu, qe_gpu) — all GPU tensors on `device`.
        """
        esky_f = self._to_float(esky)
        Ta_f = self._to_float(Ta)
        # ws passes through as scalar or tensor to solver
        if isinstance(ws, torch.Tensor) and ws.dim() >= 2:
            ws_val = ws  # 2D tensor — solver handles it
        else:
            ws_val = self._to_float(ws)
        RH_f = self._to_float(RH)
        P_f = self._to_float(P)
        radI_f = self._to_float(radI)
        radD_f = self._to_float(radD)
        CI_f = self._to_float(CI)

        # SVF-weighted Ldown
        Ldown_eb = self._compute_ldown_eb(esky_f, Ta_f, CI_f)

        # Net radiation (GPU)
        t_air_k = Ta_f + 273.15
        rnet = calculate_net_radiation(
            Ldown_eb, Kdown_gpu,
            self.solver.alb_grid, self.solver.emis_grid, t_air_k,
        )

        # Vapor pressure
        ea_hpa = 6.107 * 10 ** ((7.5 * Ta_f) / (237.3 + Ta_f)) * (RH_f / 100.0)

        # Solve ground + roof EB (all GPU)
        result = self.solver.solve(
            rnet, shadow_gpu, Kdown_gpu,
            Ta_f, ws_val, ea_hpa, P_f, radI_f, radD_f,
            self.prev_layer_temps, self.prev_tsfc,
            self.prev_roof_temps, self.prev_tsfc_roof,
            solar_elev_deg,
        )

        # Update ground state
        self.prev_layer_temps = result['ground_layer_temps'].clone()
        self.prev_tsfc = result['tsfc_ground'].clone()
        self.last_tsfc_ground = result['tsfc_ground']

        # Update roof state
        if self.enable_roof_eb:
            self.prev_roof_temps = result['roof_layer_temps'].clone()
            self.prev_tsfc_roof = result['tsfc_roof'].clone()
            self.last_tsfc_roof = result['tsfc_roof']
            self.last_roof_qh = result['roof_sensible_heat']
            self.last_roof_qe = result['roof_latent_heat']

        qh = result['sensible_heat']
        qe = result['latent_heat']
        self.last_qh = qh
        self.last_qe = qe

        # Convert tsfc to Tg delta format: Tg = Tsfc - 273.15 - Ta_C
        Tg = result['tsfc_ground'] - 273.15 - Ta_f
        Tg = torch.where(self.solver.building_mask, torch.zeros_like(Tg), Tg)

        return Tg, qh, qe

    # ── Ground EB (nighttime) ─────────────────────────────────────

    def compute_tg_night(
        self, Ta, ws, RH, P, device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Nighttime Tg using energy balance — all GPU.

        Args:
            ws: Wind speed — scalar float or (rows, cols) GPU tensor.

        Returns:
            (Tg_gpu, qh_gpu, qe_gpu) — all GPU tensors.
        """
        R, C = self.rows, self.cols
        dev = self.device
        zeros = torch.zeros((R, C), dtype=torch.float32, device=dev)

        Ta_f = self._to_float(Ta)
        if isinstance(ws, torch.Tensor) and ws.dim() >= 2:
            ws_val = ws
        else:
            ws_val = self._to_float(ws)
        RH_f = self._to_float(RH)
        P_f = self._to_float(P)

        ea_hpa = 6.107 * 10 ** ((7.5 * Ta_f) / (237.3 + Ta_f)) * (RH_f / 100.0)
        msteg = 46.5 * (ea_hpa / (Ta_f + 273.15))
        esky = 1 - (1 + msteg) * math.exp(-math.sqrt(1.2 + 3.0 * msteg))

        Ldown_eb = self._compute_ldown_eb(esky, Ta_f, 1.0)
        t_air_k = Ta_f + 273.15

        rnet = calculate_net_radiation(
            Ldown_eb, zeros, self.solver.alb_grid, self.solver.emis_grid, t_air_k,
        )

        result = self.solver.solve(
            rnet, zeros, zeros,
            Ta_f, ws_val, ea_hpa, P_f, 0.0, 0.0,
            self.prev_layer_temps, self.prev_tsfc,
            self.prev_roof_temps, self.prev_tsfc_roof,
        )

        # Update state
        self.prev_layer_temps = result['ground_layer_temps'].clone()
        self.prev_tsfc = result['tsfc_ground'].clone()
        self.last_tsfc_ground = result['tsfc_ground']

        if self.enable_roof_eb:
            self.prev_roof_temps = result['roof_layer_temps'].clone()
            self.prev_tsfc_roof = result['tsfc_roof'].clone()
            self.last_tsfc_roof = result['tsfc_roof']
            self.last_roof_qh = result['roof_sensible_heat']
            self.last_roof_qe = result['roof_latent_heat']

        qh = result['sensible_heat']
        qe = result['latent_heat']
        self.last_qh = qh
        self.last_qe = qe

        Tg = result['tsfc_ground'] - 273.15 - Ta_f
        Tg = torch.where(self.solver.building_mask, torch.zeros_like(Tg), Tg)

        return Tg, qh, qe

    # ── Canyon air temperature ────────────────────────────────────

    def update_canyon_air_temp(
        self,
        qh: torch.Tensor,
        Ta,
        ws,
    ):
        """Update the VTUF-3D-inspired implicit canyon box diagnostic.

        Builds (weight, httc, Tsfc) facet tuples for open ground and canopy and solves
        Tcan from ``Qh_i = httc_i (Tsfc_i - Tcan)`` summed against the canyon-top
        exchange (Nice 2018, eq. 4.2.21/4.2.24). Because every facet flux responds to
        Tcan, the solve is a conductance-weighted average -> unconditionally stable.
        This replaces the prior fixed-mean-QH source, whose lack of a Tcan feedback let a
        strongly negative transpiring-canopy QH drive the box to an unphysical cold runaway.

        Args:
            qh: Ground sensible heat (GPU tensor, rows x cols, W/m^2).
            Ta: Air temperature (°C, scalar).
            ws: Wind speed (m/s, scalar or 2D tensor).
        """
        if self.canyon_config is None:
            return
        from .energy_balance.physics import calculate_httc_ground

        Ta_f = self._to_float(Ta)
        Ta_k = Ta_f + 273.15
        # Canyon model uses a scalar ws — take ground-pixel mean if tensor
        if isinstance(ws, torch.Tensor) and ws.dim() >= 2:
            gm = self.buildings > 0.5
            valid_ws = ws[gm]
            ws_f = float(valid_ws.mean().item()) if valid_ws.numel() > 0 else 1.5
        else:
            ws_f = self._to_float(ws)

        if self.tcan_prev is None:
            self.tcan_prev = Ta_k

        ground_mask = self.buildings > 0.5
        valid = ground_mask & ~torch.isnan(qh)
        if not valid.any():
            return

        cfg = self.config
        facets = []   # (weight=pixel count, httc W/m^2/K, Tsfc K)

        # ── Open-ground facet: Qh = httc_ground (Tsfc_ground - Tcan) ──
        open_ground = (valid & ~self.canopy_mask) if self.canopy_mask is not None else valid
        n_open = 0
        if self.last_tsfc_ground is not None:
            og = open_ground & ~torch.isnan(self.last_tsfc_ground)
            n_open = int(og.sum().item())
            if n_open > 0:
                tsfc_g = self.last_tsfc_ground[og]                       # K
                ws_t = torch.full_like(tsfc_g, ws_f)
                ta_t = torch.full_like(tsfc_g, Ta_k)
                httc_g = calculate_httc_ground(ws_t, ta_t, tsfc_g, cfg.z_ref, cfg.z0)
                facets.append((n_open, float(httc_g.mean().item()), float(tsfc_g.mean().item())))

        # ── Canopy facet: Qh = httc_leaf (T_leaf - Tcan); reconstruct httc from stored QH ──
        # canopy_qh == httc_leaf (T_leaf - Ta_k), so httc_leaf = canopy_qh / (T_leaf - Ta_k);
        # clamp to the physical ground-httc range and fall back to 30 W/m^2/K near T_leaf≈Ta.
        n_can = 0
        if (self.canopy_mask is not None and self.last_t_leaf is not None
                and self.last_canopy_qh is not None):
            cg = (valid & self.canopy_mask
                  & ~torch.isnan(self.last_t_leaf) & ~torch.isnan(self.last_canopy_qh))
            n_can = int(cg.sum().item())
            if n_can > 0:
                tl = self.last_t_leaf[cg]                                # K
                qc = self.last_canopy_qh[cg]                             # W/m^2
                dT = tl - Ta_k
                httc_c = torch.where(dT.abs() > 1.0, qc / dT, torch.full_like(qc, 30.0))
                httc_c = torch.clamp(httc_c, 5.0, 200.0)
                facets.append((n_can, float(httc_c.mean().item()), float(tl.mean().item())))

        ntot = n_open + n_can
        if ntot == 0 or not facets:
            return
        facets = [(w / ntot, h, t) for (w, h, t) in facets]

        tcan_new = solve_canyon_air_temperature_coupled(
            self.tcan_prev, Ta_k, ws_f, facets, self.canyon_config,
        )
        self.tcan_prev = float(tcan_new)

    # ── Per-pixel 2m air temperature ──────────────────────────────

    def compute_tair_2m(self, ta_c: float) -> torch.Tensor:
        """Compute per-pixel 2m air temperature using log-profile interpolation.

        Uses MOST log-profile between surface temperature and reference air
        temperature to estimate air temperature at pedestrian height (2m).

        Args:
            ta_c: Air temperature from met file (deg C).

        Returns:
            (rows, cols) GPU tensor of 2m air temperature (K). NaN on buildings.
        """
        z_ped = 2.0
        z0h = max(self.config.z0 / 10.0, 1e-4)  # thermal roughness length

        # Reference temperature and height
        if self.tcan_prev is not None:
            t_ref = self.tcan_prev  # K (scalar)
            z_ref = self.canyon_config.canyon_air_height
        else:
            t_ref = ta_c + 273.15  # K
            z_ref = self.config.z_ref

        # Log-profile blending factor
        if z_ref > z_ped and z_ped > z0h:
            f = math.log(z_ped / z0h) / math.log(z_ref / z0h)
        else:
            f = 1.0  # degenerate: T_2m = T_ref

        # Per-pixel: T_2m = Tsfc + f * (T_ref - Tsfc)
        tair_2m = torch.full((self.rows, self.cols), float('nan'),
                             dtype=torch.float32, device=self.device)

        # Ground pixels
        if self.last_tsfc_ground is not None:
            ground = self.buildings > 0.5
            tsfc_g = self.last_tsfc_ground
            valid = ground & ~torch.isnan(tsfc_g)
            tair_2m = torch.where(valid, tsfc_g + f * (t_ref - tsfc_g), tair_2m)

        # Roof pixels
        if self.last_tsfc_roof is not None:
            roof = self.buildings < 0.5
            tsfc_r = self.last_tsfc_roof
            valid_r = roof & ~torch.isnan(tsfc_r)
            tair_2m = torch.where(valid_r, tsfc_r + f * (t_ref - tsfc_r), tair_2m)

        self.last_tair_2m = tair_2m
        return tair_2m

    # ── Prognostic advection-diffusion spatial air temperature ────

    def compute_tair_advdiff(self, ta_c: float, speed, direction, scale: float,
                             mixing_depth: float = None, eddy_diffusivity: float = 2.0):
        """Per-pixel 2 m air temperature via advection–diffusion of the canopy air layer.

        Assembles the per-pixel facet source (open-ground Tsfc + canopy leaf temperature,
        with their convective conductances) and the SVF-weighted canyon-top venting
        conductance, then advances the persistent air-temperature field one step under the
        URock wind. See ``energy_balance/advection_diffusion.py`` for the numerics.

        Args:
            ta_c: Free-stream (met) air temperature (°C).
            speed: Wind speed — (rows, cols) tensor (URock) or scalar (uniform met wind).
            direction: Meteorological wind direction (deg from N) — tensor or scalar.
            scale: Pixels per metre (dx = 1/scale).
            mixing_depth: Air-column depth h (m); defaults to canyon air height or 6 m.
            eddy_diffusivity: Horizontal K (m²/s).

        Returns:
            (rows, cols) GPU tensor of 2 m air temperature (K); NaN on building pixels.
        """
        from .energy_balance.advection_diffusion import (
            AdvectionDiffusionAirTemp, wind_to_raster_velocity,
        )
        from .energy_balance.canyon import _calculate_httc_canyon_top
        from .energy_balance.physics import calculate_httc_ground

        Ta_k = ta_c + 273.15
        R, C = self.rows, self.cols
        dev = self.device
        valid = self.buildings > 0.5  # ground/air pixels

        # Lazily create the persistent field solver.
        if self.advdiff is None:
            if mixing_depth is None:
                mixing_depth = (self.canyon_config.canyon_air_height
                                if self.canyon_config is not None else 6.0)
            self.advdiff = AdvectionDiffusionAirTemp(
                R, C, dx=1.0 / max(scale, 1e-6), device=dev,
                mixing_depth=mixing_depth, eddy_diffusivity=eddy_diffusivity,
                dt=self.config.dt,
            )
            self.advdiff.reset(Ta_k)

        # Degenerate (no surface temperature yet): return uniform free-stream.
        if self.last_tsfc_ground is None:
            return torch.where(valid, torch.full((R, C), Ta_k, device=dev),
                              torch.full((R, C), float("nan"), device=dev))

        # ── Raster-frame wind velocity ──
        vel_row, vel_col = wind_to_raster_velocity(speed, direction, dev, R, C)

        # ── Per-pixel facet target temperature (K): ground Tsfc, canopy → leaf temp ──
        tsfc_pix = self.last_tsfc_ground.clone()
        if self.canopy_mask is not None and self.last_t_leaf is not None:
            leaf_ok = self.canopy_mask & ~torch.isnan(self.last_t_leaf)
            tsfc_pix = torch.where(leaf_ok, self.last_t_leaf, tsfc_pix)

        # ── Per-pixel facet conductance (W/m²/K) ──
        if isinstance(speed, torch.Tensor):
            ws_t = torch.nan_to_num(speed, nan=1.0).clamp(min=0.1)
        else:
            ws_t = torch.full((R, C), max(float(speed), 0.1), device=dev)
        ta_field = torch.full((R, C), Ta_k, device=dev)
        httc_pix = calculate_httc_ground(
            ws_t, ta_field, torch.nan_to_num(tsfc_pix, nan=Ta_k),
            self.config.z_ref, self.config.z0,
        )
        # Canopy conductance reconstructed from stored leaf QH (as in the VTUF coupling).
        if (self.canopy_mask is not None and self.last_t_leaf is not None
                and self.last_canopy_qh is not None):
            cg = (self.canopy_mask & ~torch.isnan(self.last_t_leaf)
                  & ~torch.isnan(self.last_canopy_qh))
            if cg.any():
                dT = self.last_t_leaf - Ta_k
                httc_leaf = torch.where(dT.abs() > 1.0,
                                        self.last_canopy_qh / dT,
                                        torch.full_like(dT, 30.0)).clamp(5.0, 200.0)
                httc_pix = torch.where(cg, httc_leaf, httc_pix)

        # ── Canyon-top venting conductance to the free-stream, SVF-weighted ──
        if self.canyon_config is not None:
            cc = self.canyon_config
            dz = max(cc.z_ref - cc.z_h + cc.z0_roof, 0.1)
            ws_mean = float(ws_t[valid].mean().item()) if valid.any() else 1.5
            httc_top = _calculate_httc_canyon_top(0.0, ws_mean, dz, cc.z0_roof)  # m/s
        else:
            httc_top = 0.02
        rho_cp = 1.225 * 1005.0
        vent_cond = rho_cp * httc_top * self.svf.clamp(0.0, 1.0)

        tair_k = self.advdiff.step(Ta_k, vel_row, vel_col, tsfc_pix, httc_pix,
                                   vent_cond, valid)
        self.last_tair_2m = tair_k
        return tair_k
