# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""URock-inspired raster diagnostic wind field model (GPU-accelerated).

This is a reduced raster adaptation, not a reimplementation of URock. It uses
connected raster footprints and simplified zone geometry rather than URock's
vector facade construction, database intersections and uniform solver grid.
The zone equations are adapted from the cited parameterizations. A numerical
Poisson correction reduces, but does not guarantee elimination of, divergence.

Zone types (7 from URock, plus vegetation):
    1. Street canyon           — helical U,V,W flow (Soulhac et al. 2008)
    2. Cavity                  — reverse recirculation (Roeckle 1990)
    3. Displacement vortex     — sinusoidal V-W vortex at upwind face
    3. Rooftop recirculation   — reversed flow above building (perpendicular)
    3. Rooftop recirculation   — reversed flow above building (corner)
    4. Displacement            — C_DZ*(z/H)^P_DZ sheltering upwind
    5. Wake                    — z-dependent power-law recovery
    -. Vegetation attenuation  — exp(-alpha*LAI) drag

Poisson solver uses finite differences and a Pardyjak & Brown (2003)-style
correction with per-cell obstacle coefficients.

References:
    Bernard et al. (2023). URock 2023a. GMD 16, 5703-5727.
    Roeckle (1990). Bestimmung der Stroemungsverhaeltnisse ...
    Bagal et al. (2004). Improved upwind cavity parameterization.
    Pardyjak & Brown (2003). QUIC-URB fast response urban wind model.
    Soulhac et al. (2008). Flow in a street canyon ... BLM 126.
    Pol et al. (2006). Recirculation zone parameterization ...
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F


# ── Vertical grid levels (metres above ground) ────────────────────────────
Z_LEVELS = [0.0, 1.5, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 40.0, 60.0]
NZ = len(Z_LEVELS)
PED_LEVEL = 1  # index of z=1.5m in Z_LEVELS

# Zone priority constants (lower = higher priority, matching URock)
ZONE_OPEN = 200
ZONE_WAKE = 5
ZONE_DISPLACEMENT = 4
ZONE_ROOFTOP_CORNER = 3
ZONE_ROOFTOP_PERP = 3
ZONE_DISP_VORTEX = 3
ZONE_CAVITY = 2
ZONE_CANYON = 1
ZONE_BUILDING = 99

# URock constants (from GlobalVariables.py)
C_DZ = 0.4       # displacement zone coefficient
P_DZ = 0.16      # displacement zone power-law exponent
P_RTP = 0.16     # rooftop recirculation exponent
PERP_THRESHOLD = 15.0  # degrees — facade within this of perpendicular activates vortex
CORNER_ANGLE_MIN = 30.0  # degrees — corner recirculation min angle from perpendicular
CORNER_ANGLE_MAX = 70.0  # degrees — corner recirculation max angle from perpendicular


@dataclass
class WindFieldConfig:
    """Configuration for the URock-style diagnostic wind field model."""

    # Reference heights
    z_ref: float = 1.5        # pedestrian output height (m)
    z_wind: float = 10.0      # met station wind measurement height (m)
    z0_open: float = 0.03     # roughness length for open terrain (m)

    # Clamping
    min_wind_factor: float = 0.05
    max_wind_factor: float = 1.5

    # Vegetation attenuation
    veg_extinction: float = 1.0

    # 3D Poisson solver
    poisson_iterations: int = 500
    poisson_tol: float = 1e-4

    # Recompute when the wind direction changes by this many degrees.
    direction_cache_tol: float = 5.0

    # Street canyon detection: gap < canyon_ratio * mean(H1, H2)
    canyon_ratio: float = 1.0

    # Minimum building area (pixels) and height (m) to include
    min_building_area: int = 4
    min_building_height: float = 2.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        positive = {
            "z_ref": self.z_ref,
            "z_wind": self.z_wind,
            "poisson_tol": self.poisson_tol,
            "canyon_ratio": self.canyon_ratio,
            "min_building_height": self.min_building_height,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.z0_open) or self.z0_open < 0:
            raise ValueError("z0_open must be finite and nonnegative")
        if (
            not math.isfinite(self.min_wind_factor)
            or not math.isfinite(self.max_wind_factor)
            or not 0 <= self.min_wind_factor <= self.max_wind_factor
        ):
            raise ValueError("wind-factor bounds must satisfy 0 <= min <= max")
        if not math.isfinite(self.veg_extinction) or self.veg_extinction < 0:
            raise ValueError("veg_extinction must be finite and nonnegative")
        if (
            not isinstance(self.poisson_iterations, int)
            or isinstance(self.poisson_iterations, bool)
            or self.poisson_iterations < 1
        ):
            raise ValueError("poisson_iterations must be a positive integer")
        if not math.isfinite(self.direction_cache_tol) or not 0 <= self.direction_cache_tol <= 180:
            raise ValueError("direction_cache_tol must be between 0 and 180 degrees")
        if (
            not isinstance(self.min_building_area, int)
            or isinstance(self.min_building_area, bool)
            or self.min_building_area < 1
        ):
            raise ValueError("min_building_area must be a positive integer")


class WindFieldModel:
    """GPU-resident URock-inspired raster wind diagnostic around buildings.

    Precomputes static building geometry once at init; per-timestep
    cost is dominated by zone assignment + 3D SOR solve.
    """

    def __init__(
        self,
        building_dsm: torch.Tensor,   # (R, C) buildings + terrain
        dem: torch.Tensor,             # (R, C) bare terrain
        vegdsm: torch.Tensor,          # (R, C) tree canopy height (0=no tree)
        config: WindFieldConfig,
        device: torch.device,
    ):
        config.validate()
        shapes = {
            "building_dsm": tuple(building_dsm.shape),
            "dem": tuple(dem.shape),
            "vegdsm": tuple(vegdsm.shape),
        }
        if any(len(shape) != 2 for shape in shapes.values()):
            raise ValueError(f"wind-field inputs must be two-dimensional: {shapes}")
        if len(set(shapes.values())) != 1:
            raise ValueError(f"wind-field input shapes differ: {shapes}")
        self.config = config
        self.device = device
        self.rows, self.cols = building_dsm.shape

        # Power-law exponent: p = 0.12*z0 + 0.18 (Matzarakis et al. 2009)
        self._power_exp = 0.12 * config.z0_open + 0.18

        # Building mask and heights
        building_dsm = building_dsm.to(device, dtype=torch.float32)
        dem = dem.to(device, dtype=torch.float32)
        veg = vegdsm.to(device, dtype=torch.float32)
        if not (
            torch.isfinite(building_dsm).all()
            and torch.isfinite(dem).all()
            and torch.isfinite(veg).all()
        ):
            raise ValueError("wind-field inputs must contain only finite values")
        building_heights = building_dsm - dem
        self.building_mask = building_heights > config.min_building_height
        self.building_heights = torch.where(
            self.building_mask, building_heights, torch.zeros_like(building_heights)
        )

        # Vegetation
        self.veg_height = veg.clamp(min=0.0)
        self.veg_mask = veg > 0.0
        self.lai_proxy = torch.where(self.veg_mask, veg / 3.0, torch.zeros_like(veg))

        # Power-law height correction (reference → pedestrian)
        z_w = config.z_wind
        z_r = config.z_ref
        self._height_ratio = (z_r / z_w) ** self._power_exp

        # Vertical grid
        self._z_levels = torch.tensor(Z_LEVELS, dtype=torch.float32, device=device)

        # Detect buildings (CPU, one-time)
        self._detect_buildings()

        # Build 3D building mask
        self._build_3d_building_mask()

        # Cache
        self._cached_speed: Optional[torch.Tensor] = None
        self._cached_direction: Optional[torch.Tensor] = None
        self._cached_wd: Optional[float] = None
        self._cached_ws: Optional[float] = None
        self.last_poisson_converged: Optional[bool] = None
        self.last_poisson_iterations = 0
        self.last_poisson_max_change = float("nan")

    # ── Public API ────────────────────────────────────────────────

    def compute_wind_field(self, ws_ref: float, wd_deg: float) -> torch.Tensor:
        """Compute 2D wind speed at pedestrian height. Returns (R, C)."""
        speed, _ = self.compute_wind_field_uv(ws_ref, wd_deg)
        return speed

    def compute_wind_field_uv(
        self, ws_ref: float, wd_deg: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (speed, direction) at pedestrian height. Returns ((R,C), (R,C))."""
        if not math.isfinite(ws_ref) or ws_ref < 0:
            raise ValueError("ws_ref must be finite and nonnegative")
        if not math.isfinite(wd_deg):
            raise ValueError("wd_deg must be finite")
        wd_deg = wd_deg % 360.0
        ws_ped = ws_ref * self._height_ratio
        direction_change = None
        if self._cached_wd is not None:
            direction_change = abs((wd_deg - self._cached_wd + 180.0) % 360.0 - 180.0)

        # Direction caching
        if (self._cached_speed is not None
                and self._cached_wd is not None
                and direction_change is not None
                and direction_change < self.config.direction_cache_tol
                and self._cached_ws is not None
                and self._cached_ws > 0):
            ratio = ws_ped / self._cached_ws
            return self._cached_speed * ratio, self._cached_direction.clone()

        speed, direction = self._compute_full(ws_ped, wd_deg)
        self._cached_speed = speed.clone()
        self._cached_direction = direction.clone()
        self._cached_wd = wd_deg
        self._cached_ws = ws_ped
        return speed, direction

    # ── Building detection (CPU, one-time) ────────────────────────

    def _detect_buildings(self):
        """Connected-component labeling to identify individual buildings."""
        # Use Python lists at this one-time boundary so NumPy ABI differences in
        # third-party PyTorch builds cannot disable building detection.
        bm_np = np.asarray(self.building_mask.detach().cpu().tolist(), dtype=bool)
        labeled, n_buildings = scipy.ndimage.label(bm_np)

        buildings = []
        heights_np = np.asarray(
            self.building_heights.detach().cpu().tolist(), dtype=np.float32
        )
        for bid in range(1, n_buildings + 1):
            mask = labeled == bid
            ys, xs = np.where(mask)
            h = float(heights_np[mask].max())
            if h < self.config.min_building_height or len(ys) < self.config.min_building_area:
                continue

            # Bounding box area vs actual footprint area ratio (URock scaling)
            r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
            bbox_area = max((r1 - r0 + 1) * (c1 - c0 + 1), 1)
            footprint_area = len(ys)
            area_ratio = footprint_area / bbox_area

            buildings.append({
                'id': bid,
                'height': h,
                'bbox': (r0, r1, c0, c1),
                'centroid': (float(ys.mean()), float(xs.mean())),
                'area': footprint_area,
                'area_ratio': area_ratio,
            })

        self._labeled = torch.tensor(
            labeled.tolist(), dtype=torch.int32, device=self.device
        )
        self._building_list = buildings

    def _build_3d_building_mask(self):
        """Build (NZ, R, C) boolean mask: True where building occupies cell."""
        R, C = self.rows, self.cols
        bld3d = torch.zeros((NZ, R, C), dtype=torch.bool, device=self.device)
        for k in range(NZ):
            bld3d[k] = self.building_heights > Z_LEVELS[k]
        self._building_3d = bld3d

    # ── Per-building dimensions relative to wind ──────────────────

    def _compute_building_dimensions(self, wind_dir_rad: float):
        """Compute W_eff and L_eff for each building relative to wind.

        URock: projects bounding box onto wind axes then scales by
        area_ratio = footprint_area / bounding_box_area.
        """
        cos_wd = math.cos(wind_dir_rad)
        sin_wd = math.sin(wind_dir_rad)

        for bld in self._building_list:
            r0, r1, c0, c1 = bld['bbox']
            dy = r1 - r0 + 1
            dx = c1 - c0 + 1
            ar = bld['area_ratio']

            # Along-wind (L) and cross-wind (W), scaled by area ratio
            L_raw = abs(dy * cos_wd) + abs(dx * sin_wd)
            W_raw = abs(dy * sin_wd) + abs(dx * cos_wd)
            bld['L_eff'] = max(L_raw * ar, 1.0)
            bld['W_eff'] = max(W_raw * ar, 1.0)

            H = bld['height']
            W = bld['W_eff']
            L = bld['L_eff']

            # ── Bagal et al. (2004) zone lengths ──
            # Cavity / recirculation length (Kaplan & Dinar 1996, eq. 3)
            Lr = 1.8 * W / max((L / H) ** 0.3 * (1.0 + 0.24 * W / H), 0.01)
            Lw = 3.0 * Lr  # wake length

            # Displacement length
            Lf = 1.5 * W / max(1.0 + 0.8 * W / H, 0.01)
            # Displacement vortex length
            Lfv = 0.6 * W / max(1.0 + 0.8 * W / H, 0.01)

            # Rooftop recirculation (Pol et al. 2006)
            Hm = 0.67 * min(H, W) + 0.33 * max(H, W)
            Hcm = 0.22 * Hm   # rooftop recirculation height (perp)

            # Rooftop corner height
            Hcc = 0.22 * Hm * 0.5   # corner is typically smaller

            # Corner wind factor (C1) — Bagal et al. 2004
            C1 = 1.0 + 0.05 * W / H

            bld['Lr'] = Lr
            bld['Lw'] = Lw
            bld['Lf'] = Lf
            bld['Lfv'] = Lfv
            bld['Hcm'] = Hcm
            bld['Hcc'] = Hcc
            bld['C1'] = C1

    # ── Full wind field computation ────────────────────────────────

    def _compute_full(
        self, ws_ped: float, wd_deg: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zone assignment → 3D solve → extract z=1.5m."""
        cfg = self.config
        R, C = self.rows, self.cols
        dev = self.device
        wind_dir_rad = math.radians(wd_deg)

        self._compute_building_dimensions(wind_dir_rad)

        # Wind direction vectors (met: 0=from N, 90=from E)
        sin_wd = math.sin(wind_dir_rad)
        cos_wd = math.cos(wind_dir_rad)
        # Unit vector of wind TRAVEL (opposite of "from"):
        u_dir = -sin_wd   # eastward component
        v_dir = -cos_wd   # northward component
        # In raster coords (row+ = south, col+ = east):
        dr_dir = cos_wd    # row component of wind travel
        dc_dir = -sin_wd   # col component of wind travel

        # Reference speed at z_wind
        ws_ref = ws_ped / self._height_ratio

        # Initialize 3D velocity field with power-law profile
        u3d = torch.zeros((NZ, R, C), dtype=torch.float32, device=dev)
        v3d = torch.zeros((NZ, R, C), dtype=torch.float32, device=dev)
        w3d = torch.zeros((NZ, R, C), dtype=torch.float32, device=dev)
        zone_priority = torch.full((NZ, R, C), 255, dtype=torch.uint8, device=dev)

        p = self._power_exp
        for k in range(NZ):
            z = max(Z_LEVELS[k], 0.1)
            speed_at_z = ws_ref * (z / cfg.z_wind) ** p
            u3d[k] = speed_at_z * u_dir
            v3d[k] = speed_at_z * v_dir

        zone_priority[:] = ZONE_OPEN
        u3d[self._building_3d] = 0.0
        v3d[self._building_3d] = 0.0
        zone_priority[self._building_3d] = ZONE_BUILDING

        # Assign zones per building
        self._assign_building_zones(
            u3d, v3d, w3d, zone_priority,
            ws_ref, wind_dir_rad, u_dir, v_dir, dr_dir, dc_dir,
        )

        # Street canyon helical flow
        self._apply_street_canyons(
            u3d, v3d, w3d, zone_priority,
            ws_ref, wind_dir_rad, u_dir, v_dir, dr_dir, dc_dir,
        )
        self._apply_vegetation_attenuation(u3d, v3d)
        del zone_priority  # free ~10MB

        # Mass conservation: 3D Poisson solver (modifies u3d, v3d, w3d in-place)
        u3d, v3d, w3d = self._solve_3d_poisson(u3d, v3d, w3d)

        # Extract z=1.5m and free 3D arrays
        u_ped = u3d[PED_LEVEL].clone()
        v_ped = v3d[PED_LEVEL].clone()
        del u3d, v3d, w3d
        speed = torch.sqrt(u_ped ** 2 + v_ped ** 2)
        direction = torch.atan2(-u_ped, -v_ped) * (180.0 / math.pi) % 360.0

        speed = torch.clamp(speed, cfg.min_wind_factor * ws_ped, cfg.max_wind_factor * ws_ped)
        speed[self.building_mask] = 0.0
        direction[self.building_mask] = float('nan')

        return speed, direction

    def _apply_vegetation_attenuation(
        self, u3d: torch.Tensor, v3d: torch.Tensor,
    ) -> None:
        """Attenuate horizontal flow only at levels inside each canopy."""
        if not self.veg_mask.any():
            return
        attenuation = torch.exp(-self.config.veg_extinction * self.lai_proxy)
        open_vegetation = self.veg_mask & (~self.building_mask)
        for level, height in enumerate(Z_LEVELS):
            inside_canopy = open_vegetation & (self.veg_height >= height)
            if inside_canopy.any():
                u3d[level][inside_canopy] *= attenuation[inside_canopy]
                v3d[level][inside_canopy] *= attenuation[inside_canopy]

    # ── Zone assignment per building (URock formulas) ──────────────

    def _assign_building_zones(
        self,
        u3d: torch.Tensor, v3d: torch.Tensor, w3d: torch.Tensor,
        zone_priority: torch.Tensor,
        ws_ref: float, wind_dir_rad: float,
        u_dir: float, v_dir: float,
        dr_dir: float, dc_dir: float,
    ):
        """Assign all Roeckle zones per building with URock velocity formulas."""
        R, C = self.rows, self.cols
        dev = self.device
        z_wind = self.config.z_wind
        p = self._power_exp

        row_coords = torch.arange(R, dtype=torch.float32, device=dev).unsqueeze(1).expand(R, C)
        col_coords = torch.arange(C, dtype=torch.float32, device=dev).unsqueeze(0).expand(R, C)

        for bld in self._building_list:
            H = bld['height']
            Lr = bld['Lr']
            Lw = bld['Lw']
            Lf = bld['Lf']
            Lfv = bld['Lfv']
            Hcm = bld['Hcm']
            Hcc = bld['Hcc']
            C1 = bld['C1']

            cy, cx = bld['centroid']
            half_W = bld['W_eff'] / 2.0
            half_L = bld['L_eff'] / 2.0

            bld_mask = self._labeled == bld['id']

            # Signed distances relative to building centroid
            d_along = (row_coords - cy) * dr_dir + (col_coords - cx) * dc_dir
            d_cross = (row_coords - cy) * (-dc_dir) + (col_coords - cx) * dr_dir
            d_cross_abs = torch.abs(d_cross)

            # Distances from building faces
            d_downwind = d_along - half_L   # positive = beyond downwind face
            d_upwind = -d_along - half_L    # positive = beyond upwind face

            # Relative position within downwind zones: 0 at building, 1 at zone end
            # For cavity: d_downwind / Lr
            cavity_rel = torch.clamp(d_downwind / max(Lr, 0.1), 0.0, 1.0)
            # For displacement: d_upwind / Lf
            # Cross-wind extent filter (generous margin)
            margin = max(Lf, Lr, Lw) * 0.5
            cross_ok = d_cross_abs < (half_W + margin)

            for k in range(NZ):
                z = max(Z_LEVELS[k], 0.01)
                z_H = min(z / H, 0.999)  # z/H clamped below 1

                if z < H:
                    # ── DISPLACEMENT zone: upwind, [0, Lf] ──
                    # URock: V = C_DZ * (z/H)^P_DZ, U = C_DZ * (z/H)^P_DZ
                    disp_factor = C_DZ * (z_H ** P_DZ)
                    disp_mask = (
                        cross_ok
                        & (d_upwind > 0) & (d_upwind < Lf)
                        & (d_cross_abs < half_W * 1.2)
                        & (~bld_mask) & (~self._building_3d[k])
                        & (zone_priority[k] > ZONE_DISPLACEMENT)
                    )
                    if disp_mask.any():
                        speed_at_z = ws_ref * (z / z_wind) ** p
                        u3d[k][disp_mask] = disp_factor * speed_at_z * u_dir
                        v3d[k][disp_mask] = disp_factor * speed_at_z * v_dir
                        zone_priority[k][disp_mask] = ZONE_DISPLACEMENT

                    # ── DISPLACEMENT VORTEX: upwind, [0, Lfv], facade ~ perpendicular ──
                    # URock: V = -(0.6*cos(pi*z/(0.5*H)) + 0.05) * 0.6*sin(pi*rel_pos)
                    #         W = -0.1*cos(pi*rel_pos) - 0.05
                    vortex_mask = (
                        cross_ok
                        & (d_upwind > 0) & (d_upwind < Lfv)
                        & (d_cross_abs < half_W * 0.8)
                        & (~bld_mask) & (~self._building_3d[k])
                        & (zone_priority[k] > ZONE_DISP_VORTEX)
                    )
                    if vortex_mask.any():
                        rel_pos = torch.clamp(d_upwind[vortex_mask] / max(Lfv, 0.1), 0.0, 1.0)
                        # Along-wind velocity (reversed near wall)
                        v_factor = -(0.6 * math.cos(math.pi * z / (0.5 * H)) + 0.05) * \
                                   0.6 * torch.sin(math.pi * rel_pos)
                        speed_at_z = ws_ref * (z / z_wind) ** p
                        # Apply along wind direction
                        u3d[k][vortex_mask] = v_factor * speed_at_z * u_dir
                        v3d[k][vortex_mask] = v_factor * speed_at_z * v_dir
                        # Vertical component
                        w_factor = -0.1 * torch.cos(math.pi * rel_pos) - 0.05
                        w3d[k][vortex_mask] = w_factor * speed_at_z
                        zone_priority[k][vortex_mask] = ZONE_DISP_VORTEX

                    # ── CAVITY zone: downwind, [0, Lr] ──
                    # URock: V = -(1 - d_rel / sqrt(1 - (z/H)^2))^2  (reverse flow!)
                    cavity_mask = (
                        cross_ok
                        & (d_downwind > 0) & (d_downwind < Lr)
                        & (d_cross_abs < half_W * 1.0)
                        & (~bld_mask) & (~self._building_3d[k])
                        & (zone_priority[k] > ZONE_CAVITY)
                    )
                    if cavity_mask.any():
                        d_rel = cavity_rel[cavity_mask]  # 0 at building, 1 at Lr
                        sqrt_term = math.sqrt(max(1.0 - z_H ** 2, 0.01))
                        # Recirculation factor: negative (reverse flow)
                        v_factor = -((1.0 - d_rel / sqrt_term).clamp(min=0.0)) ** 2
                        speed_at_z = ws_ref * (z / z_wind) ** p
                        u3d[k][cavity_mask] = v_factor * speed_at_z * u_dir
                        v3d[k][cavity_mask] = v_factor * speed_at_z * v_dir
                        zone_priority[k][cavity_mask] = ZONE_CAVITY

                    # ── WAKE zone: downwind, [Lr, Lw] ──
                    # URock: weight = 1 - (D0_C/d)^1.5 * sqrt(1-(z/H)^2)^1.5
                    wake_mask = (
                        cross_ok
                        & (d_downwind >= Lr) & (d_downwind < Lw)
                        & (d_cross_abs < half_W * 1.5)
                        & (~bld_mask) & (~self._building_3d[k])
                        & (zone_priority[k] > ZONE_WAKE)
                    )
                    if wake_mask.any():
                        d_wake = d_downwind[wake_mask]
                        # wake_rel = (Lr / d)^1.5 — decays from 1 at Lr to ~0 at Lw
                        wake_rel = (Lr / d_wake.clamp(min=Lr * 0.5)) ** 1.5
                        sqrt_term = math.sqrt(max(1.0 - z_H ** 2, 0.01))
                        # Factor: transitions from ~0 at Lr to ~1 at Lw
                        recovery = 1.0 - wake_rel * (sqrt_term ** 1.5)
                        recovery = recovery.clamp(0.0, 1.0)
                        speed_at_z = ws_ref * (z / z_wind) ** p
                        u3d[k][wake_mask] = recovery * speed_at_z * u_dir
                        v3d[k][wake_mask] = recovery * speed_at_z * v_dir
                        zone_priority[k][wake_mask] = ZONE_WAKE

                else:
                    # ── ROOFTOP PERPENDICULAR RECIRCULATION ──
                    # URock: V = -((H + Hr - z) / z_ref)^P_RTP * |H + Hr - z| / Hr
                    # where Hr = Hcm (rooftop recirculation height)
                    z_above = z - H
                    if z_above < Hcm and Hcm > 0.01:
                        rooftop_mask = (
                            bld_mask
                            & (~self._building_3d[k])
                            & (zone_priority[k] > ZONE_ROOFTOP_PERP)
                        )
                        if rooftop_mask.any():
                            h_plus_hr_minus_z = H + Hcm - z
                            if h_plus_hr_minus_z > 0:
                                v_factor = -(
                                    (h_plus_hr_minus_z / z_wind) ** P_RTP
                                    * abs(h_plus_hr_minus_z) / Hcm
                                )
                                speed_at_z = ws_ref * (z / z_wind) ** p
                                u3d[k][rooftop_mask] = v_factor * speed_at_z * u_dir
                                v3d[k][rooftop_mask] = v_factor * speed_at_z * v_dir
                                zone_priority[k][rooftop_mask] = ZONE_ROOFTOP_PERP

                    # ── ROOFTOP CORNER RECIRCULATION ──
                    # Applied to area beyond footprint but near corners, above H
                    # URock: U = -C1*sin(2*theta)*((H+Hcc-z)/z_ref)^P_RTP*|H+Hcc-z|/Hcc
                    #         V = -C1*sin(theta)^2*((H+Hcc-z)/z_ref)^P_RTP*|H+Hcc-z|/Hcc
                    if z_above < Hcc and Hcc > 0.01:
                        # Corner region: dilated footprint minus footprint
                        bld_float = bld_mask.float().unsqueeze(0).unsqueeze(0)
                        kern = torch.ones(1, 1, 5, 5, device=dev)
                        dilated = (F.conv2d(bld_float, kern, padding=2) > 0.5).squeeze()
                        corner_mask = (
                            dilated & (~bld_mask) & (~self.building_mask)
                            & (~self._building_3d[k])
                            & (zone_priority[k] > ZONE_ROOFTOP_CORNER)
                        )
                        if corner_mask.any():
                            h_plus_hcc_z = H + Hcc - z
                            if h_plus_hcc_z > 0:
                                # Approximate theta from wind-facade angle
                                # Use 45 degrees as typical corner angle
                                theta = math.pi / 4.0
                                base = (h_plus_hcc_z / z_wind) ** P_RTP * abs(h_plus_hcc_z) / Hcc
                                # Cross-wind component
                                u_factor = -C1 * math.sin(2.0 * theta) * base
                                # Along-wind component
                                v_factor = -C1 * (math.sin(theta) ** 2) * base
                                speed_at_z = ws_ref * (z / z_wind) ** p
                                # Decompose into raster u,v
                                # Cross-wind unit vector: perpendicular to wind direction
                                u_cross = -v_dir  # perpendicular: rotate 90 degrees
                                v_cross = u_dir
                                u3d[k][corner_mask] = speed_at_z * (v_factor * u_dir + u_factor * u_cross)
                                v3d[k][corner_mask] = speed_at_z * (v_factor * v_dir + u_factor * v_cross)
                                zone_priority[k][corner_mask] = ZONE_ROOFTOP_CORNER

    # ── Street canyon: simplified helical flow (URock/Soulhac) ───

    def _apply_street_canyons(
        self,
        u3d: torch.Tensor, v3d: torch.Tensor, w3d: torch.Tensor,
        zone_priority: torch.Tensor,
        ws_ref: float, wind_dir_rad: float,
        u_dir: float, v_dir: float,
        dr_dir: float, dc_dir: float,
    ):
        """Detect canyons between building pairs and apply helical flow.

        URock canyon formulas (from InitWindField.py):
            U = sin(2*(theta-pi/2)) * parabolic * asymmetry * aspect_corr
            V = (1 - cos(theta-pi/2)^2 * parabolic) * asymmetry * aspect_corr
            W = -|0.5*(1-y_rel/(0.5*D0))| * (1-(D0-y_rel)/(0.5*D0)) * asymmetry * aspect_corr
        """
        cfg = self.config
        R, C = self.rows, self.cols
        dev = self.device

        if len(self._building_list) < 2:
            return

        max_gap = max(b['height'] for b in self._building_list) * cfg.canyon_ratio
        z_wind = cfg.z_wind
        p = self._power_exp

        row_coords = torch.arange(R, dtype=torch.float32, device=dev).unsqueeze(1).expand(R, C)
        col_coords = torch.arange(C, dtype=torch.float32, device=dev).unsqueeze(0).expand(R, C)

        for i, bld_a in enumerate(self._building_list):
            for j, bld_b in enumerate(self._building_list):
                if j <= i:
                    continue

                ra0, ra1, ca0, ca1 = bld_a['bbox']
                rb0, rb1, cb0, cb1 = bld_b['bbox']

                row_gap = max(0, max(ra0, rb0) - min(ra1, rb1))
                col_gap = max(0, max(ca0, cb0) - min(ca1, cb1))
                gap = math.sqrt(row_gap ** 2 + col_gap ** 2)

                if gap > max_gap or gap < 1:
                    continue

                Ha, Hb = bld_a['height'], bld_b['height']
                Hc = min(Ha, Hb)  # canyon height = shorter building
                mean_h = (Ha + Hb) / 2.0
                if gap >= cfg.canyon_ratio * mean_h:
                    continue

                # Determine gap region
                row_overlap = (max(ra0, rb0), min(ra1, rb1))
                col_overlap = (max(ca0, cb0), min(ca1, cb1))
                if row_overlap[0] > row_overlap[1] and col_overlap[0] > col_overlap[1]:
                    continue

                if row_gap > 0 and col_overlap[0] <= col_overlap[1]:
                    r_lo = min(ra1, rb1) + 1
                    r_hi = max(ra0, rb0)
                    c_lo, c_hi = col_overlap[0], col_overlap[1] + 1
                    # Canyon axis is along columns (E-W)
                    canyon_dr, canyon_dc = 0.0, 1.0
                elif col_gap > 0 and row_overlap[0] <= row_overlap[1]:
                    r_lo, r_hi = row_overlap[0], row_overlap[1] + 1
                    c_lo = min(ca1, cb1) + 1
                    c_hi = max(ca0, cb0)
                    # Canyon axis is along rows (N-S)
                    canyon_dr, canyon_dc = 1.0, 0.0
                else:
                    continue

                r_lo, r_hi = max(0, r_lo), min(R, r_hi)
                c_lo, c_hi = max(0, c_lo), min(C, c_hi)
                if r_lo >= r_hi or c_lo >= c_hi:
                    continue

                D0_S = float(gap)  # canyon width
                deltaH = Hb - Ha   # height difference (positive = B taller)

                # Theta: angle between wind direction and canyon axis
                # Canyon axis in world coords: (canyon_dc, -canyon_dr) for eastward, northward
                dot = u_dir * canyon_dc + v_dir * (-canyon_dr)
                theta = math.acos(max(-1.0, min(1.0, dot)))

                # Asymmetry correction for unequal building heights
                asym = 1.0 + (0.6 * deltaH) / (Hc + abs(0.6 * deltaH)) if Hc > 0 else 1.0

                # Aspect ratio correction: canyon too wide → less helical
                aspect_sq = D0_S ** 2 / max(Hc ** 2, 1.0)
                aspect_corr = aspect_sq / (1.0 + aspect_sq)

                for k in range(NZ):
                    z = max(Z_LEVELS[k], 0.01)
                    if z >= Hc:
                        break

                    region_slice = (
                        (~self._building_3d[k, r_lo:r_hi, c_lo:c_hi])
                        & (zone_priority[k, r_lo:r_hi, c_lo:c_hi] > ZONE_CANYON)
                    )
                    if not region_slice.any():
                        continue

                    # y_rel: position across canyon (0 at one wall, D0_S at other)
                    # Use cross-canyon coordinate
                    sub_rows = row_coords[r_lo:r_hi, c_lo:c_hi]
                    sub_cols = col_coords[r_lo:r_hi, c_lo:c_hi]
                    if canyon_dr != 0:
                        y_wall = sub_cols - c_lo
                    else:
                        y_wall = sub_rows - r_lo
                    y_rel = y_wall / max(D0_S, 1.0)
                    y_rel = y_rel.clamp(0.0, 1.0)

                    # Parabolic profile across canyon
                    parabolic = 0.5 + y_rel * (1.0 - y_rel) / 0.5

                    speed_at_z = ws_ref * (z / z_wind) ** p

                    # URock U (cross-canyon): sin(2*(theta-pi/2)) * parabolic * corrections
                    U_canyon = (
                        math.sin(2.0 * (theta - math.pi / 2.0))
                        * parabolic * asym * aspect_corr
                    )
                    # URock V (along-canyon): (1 - cos^2(theta-pi/2) * parabolic) * corrections
                    V_canyon = (
                        (1.0 - (math.cos(theta - math.pi / 2.0) ** 2) * parabolic)
                        * asym * aspect_corr
                    )
                    # URock W (vertical): vortex component
                    W_canyon = (
                        -torch.abs(0.5 * (1.0 - y_rel / 0.5))
                        * (1.0 - (1.0 - y_rel) / 0.5)
                        * asym * aspect_corr
                    )

                    # Decompose U (cross-canyon) and V (along-canyon) into raster coords
                    # Along-canyon unit vector in raster: (canyon_dr, canyon_dc)
                    # Cross-canyon unit vector in raster: perpendicular
                    cross_r, cross_c = -canyon_dc, canyon_dr

                    u_east = speed_at_z * (V_canyon * canyon_dc + U_canyon * cross_c)
                    v_north = speed_at_z * (V_canyon * (-canyon_dr) + U_canyon * (-cross_r))

                    # Convert to raster: u3d = east component, v3d = north component
                    u3d[k, r_lo:r_hi, c_lo:c_hi][region_slice] = u_east[region_slice]
                    v3d[k, r_lo:r_hi, c_lo:c_hi][region_slice] = v_north[region_slice]
                    w3d[k, r_lo:r_hi, c_lo:c_hi][region_slice] = (
                        speed_at_z * W_canyon[region_slice]
                    )
                    zone_priority[k, r_lo:r_hi, c_lo:c_hi][region_slice] = ZONE_CANYON

    # ── 3D Poisson solver (Pardyjak & Brown 2003) ─────────────────

    def _solve_3d_poisson(
        self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Red-Black SOR with Neumann BC at buildings.

        Following URock / QUIC-URB (Pardyjak & Brown 2003):
        - Forward differences for divergence
        - Neumann BC at building surfaces (air mask zeroes building neighbors)
        - Red-Black ordering for GPU-parallel convergence
        - Velocity correction: u += 0.5 * (lam[i+1] - lam[i]) / dx

        Performance: uses element-wise float masks and in-place ops instead
        of masked scatter indexing.  All per-iteration temporaries are
        pre-allocated and reused.
        """
        cfg = self.config
        dev = self.device
        NZ_l, R, C = u.shape
        bld3d = self._building_3d
        air = (~bld3d).float()  # 1.0 = air, 0.0 = building

        dx = 1.0
        dy = 1.0

        # Variable dz per level
        dz_levels = []
        for k in range(NZ_l - 1):
            dz_levels.append(Z_LEVELS[k + 1] - Z_LEVELS[k])
        dz_levels.append(dz_levels[-1])
        dz_t = torch.tensor(dz_levels, dtype=torch.float32, device=dev)
        B_bc = ((dx ** 2) / dz_t.clamp(min=0.1) ** 2).view(NZ_l, 1, 1)

        # Compute SOR omega (spectral radius estimate)
        nx, ny = C, R
        Xi_val = ((math.cos(math.pi / max(nx, 2)) + math.cos(math.pi / max(ny, 2)))
                  / 2.0) ** 2
        omega = 2.0 * (1.0 - math.sqrt(max(1.0 - Xi_val, 1e-10))) / max(Xi_val, 1e-10)
        if omega < 1.0 or omega > 2.0:
            omega = 1.78

        # Forward-difference divergence. The SOR update below solves
        # Laplacian(lambda) = -rhs, while velocity is corrected by
        # +0.5*grad(lambda), so rhs must be +2*divergence.
        rhs = torch.zeros_like(u)
        rhs[:, :, :-1] += (u[:, :, 1:] - u[:, :, :-1]) / dx
        # v is northward, while raster row indices increase southward.
        rhs[:, :-1, :] -= (v[:, 1:, :] - v[:, :-1, :]) / dy
        for k in range(NZ_l - 1):
            dz_k = max(dz_levels[k], 0.1)
            rhs[k] += (w[k + 1] - w[k]) / dz_k
        rhs *= 2.0

        # Pre-compute static diagonal (sum of air-neighbor weights)
        diag = torch.zeros((NZ_l, R, C), dtype=torch.float32, device=dev)
        # x-axis neighbors
        diag[:, :, 1:-1] += air[:, :, 2:] + air[:, :, :-2]
        diag[:, :, 0] += air[:, :, 1]
        diag[:, :, -1] += air[:, :, -2]
        # y-axis neighbors
        py_sum = torch.zeros_like(diag)
        py_sum[:, 1:-1, :] += air[:, 2:, :] + air[:, :-2, :]
        py_sum[:, 0, :] += air[:, 1, :]
        py_sum[:, -1, :] += air[:, -2, :]
        diag += py_sum
        del py_sum
        # z-axis neighbors (weighted by B_bc = dx^2/dz^2)
        qz_sum = torch.zeros_like(diag)
        qz_sum[1:-1] += air[2:] + air[:-2]
        qz_sum[0] += air[1]
        qz_sum[-1] += air[-2]
        diag += B_bc * qz_sum
        del qz_sum
        diag = diag.clamp(min=1e-10)
        omega_over_diag = omega / diag
        del diag

        # Pre-compute static z-neighbor air*B_bc products (avoids per-iter temps)
        air_B_zp = B_bc[:-1] * air[1:]    # for k → k+1 neighbor
        air_B_zm = B_bc[1:] * air[:-1]    # for k → k-1 neighbor

        # Red-Black float masks (element-wise multiply, no scatter)
        solvable = torch.zeros((NZ_l, R, C), dtype=torch.bool, device=dev)
        solvable[1:-1, 1:-1, 1:-1] = True
        solvable &= ~bld3d
        i_idx = torch.arange(NZ_l, device=dev).view(NZ_l, 1, 1)
        j_idx = torch.arange(R, device=dev).view(1, R, 1)
        k_idx = torch.arange(C, device=dev).view(1, 1, C)
        parity = (i_idx + j_idx + k_idx) % 2
        red_f = (solvable & (parity == 0)).float()
        black_f = (solvable & (parity == 1)).float()
        del i_idx, j_idx, k_idx, parity, solvable

        # Pre-allocate working buffers (reused every iteration)
        lam = torch.zeros_like(u)
        nb = torch.zeros_like(lam)
        delta = torch.empty_like(lam)

        converged = False
        completed_iterations = 0
        max_change = float("inf")
        for iteration in range(cfg.poisson_iterations):
            check_convergence = (
                (iteration + 1) % 25 == 0
                or iteration + 1 == cfg.poisson_iterations
            )
            # ── Red cells update (element-wise, no scatter) ──
            nb.zero_()
            nb[:, :, :-1].addcmul_(lam[:, :, 1:], air[:, :, 1:])
            nb[:, :, 1:].addcmul_(lam[:, :, :-1], air[:, :, :-1])
            nb[:, :-1, :].addcmul_(lam[:, 1:, :], air[:, 1:, :])
            nb[:, 1:, :].addcmul_(lam[:, :-1, :], air[:, :-1, :])
            nb[:-1].addcmul_(air_B_zp, lam[1:])
            nb[1:].addcmul_(air_B_zm, lam[:-1])

            # delta = omega/diag * (rhs + nb) - omega * lam
            # At red cells: lam += delta; at others: delta = 0 (via red_f mask)
            torch.add(rhs, nb, out=delta)
            delta.mul_(omega_over_diag)
            delta.sub_(lam, alpha=omega)
            delta.mul_(red_f)
            if check_convergence:
                red_max_change = delta.abs().max().item()
            lam.add_(delta)

            # ── Black cells update (using updated red values) ──
            nb.zero_()
            nb[:, :, :-1].addcmul_(lam[:, :, 1:], air[:, :, 1:])
            nb[:, :, 1:].addcmul_(lam[:, :, :-1], air[:, :, :-1])
            nb[:, :-1, :].addcmul_(lam[:, 1:, :], air[:, 1:, :])
            nb[:, 1:, :].addcmul_(lam[:, :-1, :], air[:, :-1, :])
            nb[:-1].addcmul_(air_B_zp, lam[1:])
            nb[1:].addcmul_(air_B_zm, lam[:-1])

            torch.add(rhs, nb, out=delta)
            delta.mul_(omega_over_diag)
            delta.sub_(lam, alpha=omega)
            delta.mul_(black_f)

            completed_iterations = iteration + 1
            if check_convergence:
                black_max_change = delta.abs().max().item()
                max_change = max(red_max_change, black_max_change)
            lam.add_(delta)
            if check_convergence and math.isfinite(max_change) and max_change < cfg.poisson_tol:
                converged = True
                break

        self.last_poisson_converged = converged
        self.last_poisson_iterations = completed_iterations
        self.last_poisson_max_change = max_change

        # Free solver temporaries
        del nb, delta, rhs, omega_over_diag
        del red_f, black_f, air, B_bc, air_B_zp, air_B_zm

        # Velocity correction (Pardyjak & Brown 2003):
        u[:, :, 1:] += 0.5 * (lam[:, :, 1:] - lam[:, :, :-1]) / dx
        v[:, 1:, :] -= 0.5 * (lam[:, 1:, :] - lam[:, :-1, :]) / dy
        for k in range(1, NZ_l):
            dz_k = max(dz_levels[k - 1], 0.1)
            w[k] += 0.5 * (lam[k] - lam[k - 1]) / dz_k
        del lam

        # Zero inside buildings
        u[bld3d] = 0.0
        v[bld3d] = 0.0
        w[bld3d] = 0.0

        return u, v, w
