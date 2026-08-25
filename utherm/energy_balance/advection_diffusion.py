# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prognostic canopy-layer air-temperature field — advection–diffusion solver.

This is a spatial numerical extension of a well-mixed canyon box
(``canyon.solve_canyon_air_temperature_coupled``). Where VTUF integrates a single
scalar ``Tcan`` for the whole domain, and TARGET (in ``utci_process``) diagnoses each
pixel independently as an algebraic blend of its own surface temperature and the
forcing, this module evolves a 2-D air-temperature field ``T(x, y)`` that is
**transported** by the raster wind diagnostic:

    ∂T/∂t + u·∂T/∂x + v·∂T/∂y = K·∇²T + [ Σ_i httc_i (Tsfc_i − T) + c_top (Ta − T) ] / (ρ cp h)
       │        │                  │              │                       │
     storage  advection         turbulent    surface facet source     canyon-top venting
                (raster u,v)     diffusion    (ground + canopy leaf)   to free-stream Ta

The advection term is the whole point: it is the only mechanism by which a park's cool
air is carried *downwind* into a plume. TARGET and VTUF have no ``u·∇T`` and therefore
cannot produce one. The source term reuses exactly the per-pixel conductances and
surface/leaf temperatures the energy balance already computes, so this introduces a new
transport *operator*, not new surface physics.

Numerics (per outer meteorological step, sub-cycled internally):
  1. Semi-Lagrangian advection — unconditionally stable, so the sub-step is set by
     *accuracy* (a Courant target) and by the explicit-diffusion limit, not by a CFL
     blow-up condition. Implemented as a single ``grid_sample`` back-trace on GPU.
  2. Gaussian-kernel diffusion approximating horizontal mixing with an eddy
     diffusivity ``K``.
  3. Implicit per-pixel source relaxation — unconditionally stable and bounded: the
     updated pixel is a conductance-weighted average of ``T``, the facet ``Tsfc`` and the
     free-stream ``Ta`` (the same implicit trick that makes VTUF non-runaway).

Inflow boundary condition: the outermost ring is pinned to the free-stream ``Ta`` each
sub-step, i.e. air entering the domain is at background temperature. The production
domain carries a 200 m buffer, so pinning the edge is harmless to the analysis area.

The field ``self.T`` persists between outer steps — this is the prognostic *memory*
that neither TARGET (memoryless) fully has.
"""

import math

import torch
import torch.nn.functional as F

from .physics import AIR_DENSITY, AIR_SPECIFIC_HEAT


def wind_to_raster_velocity(speed, direction, device, rows, cols):
    """Convert (speed, meteorological-direction) to raster-frame (vel_row, vel_col) in m/s.

    Met convention: ``direction`` is the compass bearing the wind blows *from*
    (0 = from North, 90 = from East). Wind *travel* is the opposite bearing. In raster
    coordinates row+ points South and col+ points East, giving:

        vel_row = +speed · cos(dir)      (North-source wind → travels South → +row)
        vel_col = −speed · sin(dir)      (West-source  wind → travels East  → +col)

    Accepts tensor fields (per-pixel URock wind) or Python scalars (uniform met wind);
    scalars are broadcast to a full ``(rows, cols)`` field. NaNs (building pixels) → 0.
    """
    if isinstance(direction, torch.Tensor):
        dir_rad = torch.deg2rad(torch.nan_to_num(direction, nan=0.0))
        sp = speed if isinstance(speed, torch.Tensor) else torch.full((rows, cols), float(speed), device=device)
        sp = torch.nan_to_num(sp, nan=0.0)
        vel_row = sp * torch.cos(dir_rad)
        vel_col = -sp * torch.sin(dir_rad)
    else:
        dir_rad = math.radians(float(direction))
        sp = float(speed) if not isinstance(speed, torch.Tensor) else float(speed.mean().item())
        vel_row = torch.full((rows, cols), sp * math.cos(dir_rad), device=device)
        vel_col = torch.full((rows, cols), -sp * math.sin(dir_rad), device=device)
    return vel_row, vel_col


class AdvectionDiffusionAirTemp:
    """Semi-Lagrangian advection + diffusion + implicit-source air-temperature field.

    All state is a GPU tensor and persists between outer timesteps. One instance per tile.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        dx: float,
        device,
        mixing_depth: float = 6.0,       # h — effective canopy air-column depth (m)
        eddy_diffusivity: float = 2.0,   # K — horizontal turbulent diffusivity (m²/s)
        dt: float = 3600.0,              # outer meteorological step (s)
        courant: float = 1.0,            # advection sub-step Courant target (accuracy)
        diff_courant: float = 0.5,       # diffusion sub-step target (Gaussian is stable)
        max_substeps: int = 2000,        # accuracy cap: back-trace jump ≈ dt_sub·|u|/dx cells
        clamp_delta: float = 15.0,       # hard bound |T − Ta| (K) — physical safety net
    ):
        self.rows = rows
        self.cols = cols
        self.dx = max(float(dx), 1e-6)
        self.device = device
        self.h = max(float(mixing_depth), 0.5)
        self.K = max(float(eddy_diffusivity), 0.0)
        self.dt = float(dt)
        self.courant = float(courant)
        self.diff_courant = float(diff_courant)
        self.max_substeps = int(max_substeps)
        self.clamp_delta = float(clamp_delta)
        self.rho_cp = AIR_DENSITY * AIR_SPECIFIC_HEAT

        self.T = None  # (rows, cols) K — persistent field state

        # Normalised base grid for grid_sample back-trace (x=col, y=row in [-1, 1]).
        ys = torch.arange(rows, device=device, dtype=torch.float32)
        xs = torch.arange(cols, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self._grid_row = gy  # (rows, cols) row index
        self._grid_col = gx  # (rows, cols) col index

    def reset(self, ta_k: float):
        """Initialise the field to a uniform free-stream temperature (K)."""
        self.T = torch.full((self.rows, self.cols), float(ta_k),
                            dtype=torch.float32, device=self.device)

    def _n_substeps(self, max_speed: float) -> int:
        """Sub-step count from advection + diffusion *accuracy* (not stability).

        Semi-Lagrangian advection and Gaussian diffusion are both unconditionally stable,
        so the sub-step is chosen only to resolve (a) the wind's Courant number and (b) the
        interleaving of source pickup with transport — never a blow-up limit.
        """
        dt_adv = self.courant * self.dx / max(max_speed, 1e-3)
        dt_diff = self.diff_courant * self.dx * self.dx / max(self.K, 1e-6) if self.K > 0 else self.dt
        dt_sub = min(dt_adv, dt_diff, self.dt)
        n = int(math.ceil(self.dt / max(dt_sub, 1e-6)))
        return max(1, min(n, self.max_substeps))

    def _gaussian_kernel(self, sigma_cells: float) -> torch.Tensor:
        """1-D separable Gaussian kernel for a diffusion length of `sigma_cells` pixels."""
        radius = int(min(math.ceil(3.0 * sigma_cells), 40))
        x = torch.arange(-radius, radius + 1, device=self.device, dtype=torch.float32)
        k = torch.exp(-x * x / (2.0 * sigma_cells * sigma_cells))
        return (k / k.sum()), radius

    def _advect(self, field: torch.Tensor, vel_row, vel_col, dt_sub: float) -> torch.Tensor:
        """One semi-Lagrangian back-trace step (bilinear, replicate at edges)."""
        # Departure point = current position − velocity·dt (in cells).
        dep_row = self._grid_row - vel_row * dt_sub / self.dx
        dep_col = self._grid_col - vel_col * dt_sub / self.dx
        # Normalise to grid_sample's [-1, 1] (align_corners=True).
        gx = 2.0 * dep_col / max(self.cols - 1, 1) - 1.0
        gy = 2.0 * dep_row / max(self.rows - 1, 1) - 1.0
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)  # (1, R, C, 2)
        inp = field.unsqueeze(0).unsqueeze(0)              # (1, 1, R, C)
        out = F.grid_sample(inp, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.squeeze(0).squeeze(0)

    def _diffuse(self, field: torch.Tensor, dt_sub: float) -> torch.Tensor:
        """Approximate linear diffusion with a separable Gaussian blur.

        The Green's function of ∂T/∂t = K∇²T is convolution with a Gaussian of standard
        deviation σ = sqrt(2·K·dt_sub). The finite kernel and replicated edge values
        approximate that convolution without an explicit-Laplacian CFL limit.
        """
        if self.K <= 0.0:
            return field
        sigma_cells = math.sqrt(2.0 * self.K * dt_sub) / self.dx
        if sigma_cells < 0.15:  # sub-pixel: negligible, skip
            return field
        kern, r = self._gaussian_kernel(sigma_cells)
        inp = field.unsqueeze(0).unsqueeze(0)
        inp = F.pad(inp, (r, r, 0, 0), mode="replicate")
        inp = F.conv2d(inp, kern.view(1, 1, 1, -1))
        inp = F.pad(inp, (0, 0, r, r), mode="replicate")
        inp = F.conv2d(inp, kern.view(1, 1, -1, 1))
        return inp.squeeze(0).squeeze(0)

    def step(
        self,
        ta_k: float,
        vel_row: torch.Tensor,
        vel_col: torch.Tensor,
        tsfc_pix: torch.Tensor,   # (R, C) K — per-pixel facet surface/leaf temperature
        httc_pix: torch.Tensor,   # (R, C) W/m²/K — per-pixel facet conductance
        vent_cond: torch.Tensor,  # (R, C) W/m²/K — canyon-top venting conductance to Ta
        valid_mask: torch.Tensor, # (R, C) bool — True on air (ground) pixels
    ) -> torch.Tensor:
        """Advance the air-temperature field one outer meteorological step.

        Returns the updated ``(R, C)`` field in Kelvin (NaN on invalid pixels).
        """
        if self.T is None:
            self.reset(ta_k)

        # Zero velocity at building cells. Bilinear back-tracing and Gaussian
        # diffusion do not implement exact impermeable obstacle boundaries.
        vr = torch.where(valid_mask, vel_row, torch.zeros_like(vel_row))
        vc = torch.where(valid_mask, vel_col, torch.zeros_like(vel_col))
        vr = torch.nan_to_num(vr, nan=0.0)
        vc = torch.nan_to_num(vc, nan=0.0)

        max_speed = float(torch.sqrt(vr * vr + vc * vc).max().item()) if valid_mask.any() else 0.0
        n = self._n_substeps(max_speed)
        dt_sub = self.dt / n

        # Sanitise source inputs on invalid pixels so they never inject NaNs.
        tsfc = torch.nan_to_num(tsfc_pix, nan=ta_k)
        httc = torch.nan_to_num(httc_pix, nan=0.0).clamp(0.0, 500.0)
        vent = torch.nan_to_num(vent_cond, nan=0.0).clamp(0.0, 500.0)
        a = dt_sub / (self.rho_cp * self.h)

        T = self.T
        lo, hi = ta_k - self.clamp_delta, ta_k + self.clamp_delta
        for _ in range(n):
            T = self._advect(T, vr, vc, dt_sub)
            T = self._diffuse(T, dt_sub)
            # Implicit source relaxation toward conductance-weighted (Tsfc, Ta).
            T = (T + a * (httc * tsfc + vent * ta_k)) / (1.0 + a * (httc + vent))
            # Inflow boundary: outermost ring carries the free-stream.
            T[0, :] = ta_k
            T[-1, :] = ta_k
            T[:, 0] = ta_k
            T[:, -1] = ta_k
            T = T.clamp(lo, hi)

        self.T = T
        out = torch.where(valid_mask, T, torch.full_like(T, float("nan")))
        return out
