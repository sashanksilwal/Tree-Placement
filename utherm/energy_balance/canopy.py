# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canopy energy balance — big-leaf and two-big-leaf models on GPU.

Port of Rust solve_canopy_energy_balance / solve_canopy_energy_balance_two_leaf.
All operations are vectorized PyTorch tensor ops.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .config import CanopyProperties
from .physics import (
    STEFAN_BOLTZMANN, AIR_DENSITY, AIR_SPECIFIC_HEAT,
    e_sat_kpa, delta_e_sat, psychrometric_constant,
)


@dataclass
class LeafSolveResult:
    """Per-pixel canopy leaf-temperature result and diagnostics."""

    temperature: torch.Tensor
    sensible_heat: torch.Tensor
    latent_heat: torch.Tensor
    net_radiation: torch.Tensor
    residual: torch.Tensor
    converged: torch.Tensor
    railed: torch.Tensor
    nonfinite: torch.Tensor
    iterations: torch.Tensor


def canopy_gap_transmittance(
    lai,
    clumping_factor: float,
    elevation_deg,
) -> torch.Tensor:
    """Unintercepted directional beam fraction for a spherical leaf-angle distribution.

    Uses ``tau = exp[-0.5 * Omega * LAI / sin(beta)]``, where beta is
    elevation above the horizon. Values at or below the horizon are zero.
    ``lai`` and ``elevation_deg`` may be scalars or broadcastable tensors.
    """
    reference = lai if isinstance(lai, torch.Tensor) else elevation_deg
    if isinstance(reference, torch.Tensor):
        device = reference.device
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
    else:
        device = None
        dtype = torch.float32

    lai_t = torch.as_tensor(lai, dtype=dtype, device=device).clamp(min=0.0)
    elevation_t = torch.as_tensor(elevation_deg, dtype=dtype, device=device)
    mu = torch.sin(torch.deg2rad(elevation_t))
    mu_safe = mu.clamp(min=1.0e-4)
    omega = max(float(clumping_factor), 0.0)
    tau = torch.exp(-0.5 * omega * lai_t / mu_safe)
    return torch.where(mu > 0.0, tau, torch.zeros_like(tau)).clamp(0.0, 1.0)


def canopy_hemispherical_transmittance(
    lai,
    clumping_factor: float,
    quadrature_points: int = 32,
) -> torch.Tensor:
    """Isotropic hemispherical gap fraction for diffuse shortwave or longwave.

    Numerically evaluates ``2 integral_0^1 mu exp(-0.5*Omega*LAI/mu) dmu``
    using midpoint quadrature. Leaves are treated as opaque along intercepted
    paths; scattering is handled separately through leaf albedo.
    """
    if quadrature_points < 1:
        raise ValueError("quadrature_points must be positive")

    if isinstance(lai, torch.Tensor):
        device = lai.device
        dtype = lai.dtype if lai.is_floating_point() else torch.float32
    else:
        device = None
        dtype = torch.float32

    lai_t = torch.as_tensor(lai, dtype=dtype, device=device).clamp(min=0.0)
    omega = max(float(clumping_factor), 0.0)
    result = torch.zeros_like(lai_t)
    # Accumulate 2-D terms rather than constructing an (..., n) tensor, which
    # would be prohibitively large for city-scale raster tiles.
    for point in range(quadrature_points):
        mu = torch.as_tensor((point + 0.5) / quadrature_points, dtype=dtype, device=device)
        weight = 2.0 * mu / quadrature_points
        result = result + weight * torch.exp(-0.5 * omega * lai_t / mu)
    return result.clamp(0.0, 1.0)


def _validate_canopy_inputs(
    named_tensors: Dict[str, torch.Tensor],
    props: CanopyProperties,
    ta_c: float,
    ws: float,
    ea_hpa: float,
    pressure_kpa: float,
) -> Tuple[int, int]:
    props.validate()
    shapes = {name: tuple(value.shape) for name, value in named_tensors.items()}
    if any(len(shape) != 2 for shape in shapes.values()):
        raise ValueError(f"canopy inputs must be two-dimensional: {shapes}")
    if len(set(shapes.values())) != 1:
        raise ValueError(f"canopy input shapes differ: {shapes}")
    devices = {name: value.device for name, value in named_tensors.items()}
    if len(set(devices.values())) != 1:
        raise ValueError(f"canopy input devices differ: {devices}")
    lai_grid = named_tensors.get("lai_grid")
    if lai_grid is not None and bool((torch.isfinite(lai_grid) & (lai_grid < 0.0)).any()):
        raise ValueError("lai_grid cannot contain negative values")
    scalars = {
        "ta_c": ta_c,
        "ws": ws,
        "ea_hpa": ea_hpa,
        "pressure_kpa": pressure_kpa,
    }
    for name, value in scalars.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if ta_c <= -273.15:
        raise ValueError("ta_c must be above absolute zero")
    if ws < 0.0:
        raise ValueError("ws must be nonnegative")
    if ea_hpa < 0.0:
        raise ValueError("ea_hpa must be nonnegative")
    if pressure_kpa <= 0.0:
        raise ValueError("pressure_kpa must be positive")
    return next(iter(shapes.values()))


def _solve_leaf_temperature_batched(
    kabs: torch.Tensor,        # Absorbed shortwave per ground area (W/m²)
    ldown: torch.Tensor,       # Downwelling LW from sky (W/m²)
    t_air_k: float,            # Air temperature (K)
    r_b: torch.Tensor,         # Leaf boundary layer resistance (s/m)
    r_s: torch.Tensor,         # Stomatal resistance (s/m), large = no LE
    ea_kpa: float,             # Actual vapor pressure (kPa)
    gamma: float,              # Psychrometric constant (kPa/K)
    emiss: float,              # Leaf emissivity
    lai: torch.Tensor,         # Leaf area index
    longwave_tau: torch.Tensor,  # Hemispherical longwave gap fraction
    allow_dew: bool,           # Allow negative QE
    t_guess: torch.Tensor,     # Initial guess (K)
    longwave_absorb_fraction: Optional[torch.Tensor] = None,
    n_iterations: int = 40,
    residual_tolerance: float = 0.1,
    valid_mask: Optional[torch.Tensor] = None,
    return_diagnostics: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | LeafSolveResult:
    """Solve leaf temperature with safeguarded Newton iterations.

    Big-leaf: G = 0, so Rnet - QH - QE = 0.

    Newton steps are kept inside a 223.15–353.15 K bracket and fall back to
    bisection. A root outside the bracket is reported as railed.
    """
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive")

    active = torch.ones_like(kabs, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    rho_cp = AIR_DENSITY * AIR_SPECIFIC_HEAT

    if longwave_absorb_fraction is None:
        longwave_absorb_frac = torch.clamp(1.0 - longwave_tau, min=0.0)
    else:
        longwave_absorb_frac = torch.as_tensor(
            longwave_absorb_fraction, dtype=lai.dtype, device=lai.device,
        ).expand_as(lai).clamp(0.0, 1.0)

    # QH coefficient: LAI * rho*cp / r_b
    h_coeff = lai * rho_cp / torch.clamp(r_b, min=1.0)

    # QE coefficient: LAI * rho*cp / (gamma * (r_b + r_s))
    le_active = r_s < 1e6
    total_r = torch.where(le_active, r_b + r_s, torch.ones_like(r_b))
    le_coeff = torch.where(
        le_active,
        lai * rho_cp / (gamma * total_r),
        torch.zeros_like(r_b),
    )

    # LW from ground ≈ σ * Ta⁴
    ta4 = t_air_k ** 4
    lup_ground = STEFAN_BOLTZMANN * ta4

    def equation(t_leaf: torch.Tensor):
        t3 = t_leaf * t_leaf * t_leaf
        t4 = t3 * t_leaf
        rnet = (kabs
                + emiss * longwave_absorb_frac * (ldown + lup_ground)
                - 2.0 * emiss * longwave_absorb_frac * STEFAN_BOLTZMANN * t4)
        qh = h_coeff * (t_leaf - t_air_k)
        es = e_sat_kpa(t_leaf)
        qe_raw = le_coeff * (es - ea_kpa)
        if allow_dew:
            qe = torch.where(le_active, qe_raw, torch.zeros_like(qe_raw))
            dqe_dt = torch.where(le_active, le_coeff * delta_e_sat(t_leaf), torch.zeros_like(qe_raw))
        else:
            qe = torch.where(le_active & (qe_raw > 0), qe_raw, torch.zeros_like(qe_raw))
            dqe_dt = torch.where(le_active & (qe_raw > 0), le_coeff * delta_e_sat(t_leaf), torch.zeros_like(qe_raw))

        residual = rnet - qh - qe
        derivative = (
            -8.0 * emiss * longwave_absorb_frac * STEFAN_BOLTZMANN * t3
            - h_coeff
            - dqe_dt
        )
        return residual, derivative, qh, qe, rnet

    scalar_inputs_finite = all(
        math.isfinite(float(value))
        for value in (t_air_k, ea_kpa, gamma, emiss)
    ) and gamma > 0.0 and 0.0 <= emiss <= 1.0
    finite_inputs = (
        torch.isfinite(kabs) & torch.isfinite(ldown) & torch.isfinite(r_b)
        & torch.isfinite(r_s) & torch.isfinite(lai) & torch.isfinite(t_guess)
        & torch.isfinite(longwave_absorb_frac) & (r_b > 0.0) & (r_s >= 0.0)
        & (lai >= 0.0)
    )
    if not scalar_inputs_finite:
        finite_inputs = torch.zeros_like(finite_inputs)

    lower = torch.full_like(kabs, 223.15)
    upper = torch.full_like(kabs, 353.15)
    f_lower, _, _, _, _ = equation(lower)
    f_upper, _, _, _, _ = equation(upper)
    bracket_finite = torch.isfinite(f_lower) & torch.isfinite(f_upper)
    railed = active & finite_inputs & bracket_finite & (
        (f_lower < -residual_tolerance) | (f_upper > residual_tolerance)
    )
    nonfinite = active & (~finite_inputs | ~bracket_finite)
    eligible = active & ~railed & ~nonfinite

    t_leaf = torch.clamp(t_guess, 223.15, 353.15)
    residual, derivative, qh, qe, rnet = equation(t_leaf)
    failed = eligible & (
        ~torch.isfinite(residual) | ~torch.isfinite(derivative)
    )
    nonfinite = nonfinite | failed
    eligible = eligible & ~failed
    converged = eligible & (residual.abs() <= residual_tolerance)
    iterations = torch.zeros_like(kabs, dtype=torch.int32)

    for iteration in range(1, n_iterations + 1):
        working = eligible & ~converged
        if not bool(working.any()):
            break
        lower = torch.where(working & (residual > 0.0), t_leaf, lower)
        upper = torch.where(working & (residual <= 0.0), t_leaf, upper)
        safe_derivative = torch.where(
            derivative.abs() < 1.0e-10,
            torch.full_like(derivative, -1.0e-10),
            derivative,
        )
        newton = t_leaf - residual / safe_derivative
        use_bisection = (~torch.isfinite(newton)) | (newton <= lower) | (newton >= upper)
        candidate = torch.where(use_bisection, 0.5 * (lower + upper), newton)
        t_leaf = torch.where(working, candidate, t_leaf)
        residual, derivative, qh, qe, rnet = equation(t_leaf)
        failed = working & (
            ~torch.isfinite(residual) | ~torch.isfinite(derivative)
        )
        nonfinite = nonfinite | failed
        eligible = eligible & ~failed
        newly_converged = working & ~failed & (residual.abs() <= residual_tolerance)
        iterations = torch.where(newly_converged & ~converged, iteration, iterations)
        converged = converged | newly_converged

    iterations = torch.where(eligible & ~converged, n_iterations, iterations)
    residual = torch.where(active, residual, torch.full_like(residual, float("nan")))
    result = LeafSolveResult(
        temperature=t_leaf,
        sensible_heat=qh,
        latent_heat=qe,
        net_radiation=rnet,
        residual=residual,
        converged=converged,
        railed=railed,
        nonfinite=nonfinite,
        iterations=iterations,
    )
    if return_diagnostics:
        return result
    return t_leaf, qh, qe, rnet


def solve_canopy_eb(
    kdown_above: torch.Tensor,    # Shortwave above canopy (W/m²)
    ldown: torch.Tensor,          # Longwave from sky (W/m²)
    ta_c: float,                   # Air temperature (°C)
    ws: float,                     # Wind speed (m/s)
    ea_hpa: float,                 # Vapor pressure (hPa)
    pressure_kpa: float,           # Pressure (kPa; UMEP met convention)
    canopy_mask: torch.Tensor,     # Bool, True where canopy exists
    lai_grid: torch.Tensor,        # Per-pixel LAI
    props: CanopyProperties,
    prev_t_leaf: Optional[torch.Tensor] = None,
    device: torch.device = None,
    shortwave_transmittance: Optional[torch.Tensor] = None,
    longwave_transmittance: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Solve big-leaf canopy energy balance for all vegetated pixels.

    Returns dict with t_leaf, canopy_qh, canopy_qe, canopy_rnet — all GPU tensors.
    NaN where no canopy.
    """
    dev = device or kdown_above.device
    R, C = _validate_canopy_inputs(
        {
            "kdown_above": kdown_above,
            "ldown": ldown,
            "canopy_mask": canopy_mask,
            "lai_grid": lai_grid,
        },
        props,
        ta_c,
        ws,
        ea_hpa,
        pressure_kpa,
    )
    if prev_t_leaf is not None and tuple(prev_t_leaf.shape) != (R, C):
        raise ValueError("prev_t_leaf shape does not match canopy inputs")
    nan = float('nan')
    t_air_k = ta_c + 273.15

    ea_kpa = ea_hpa * 0.1
    p_kpa = pressure_kpa
    gamma = psychrometric_constant(p_kpa)
    vpd_air = max(e_sat_kpa(torch.tensor(t_air_k)).item() - ea_kpa, 0.0)

    lai = torch.clamp(lai_grid, min=0.0)
    kdown = torch.clamp(kdown_above, min=0.0)

    # Directionally integrated gap fractions. The radiation solver can supply
    # a beam/diffuse-weighted shortwave value; otherwise assume isotropic flux.
    tau_default = None
    if shortwave_transmittance is None or longwave_transmittance is None:
        tau_default = canopy_hemispherical_transmittance(lai, props.clumping_factor)
    tau_sw = tau_default if shortwave_transmittance is None else torch.as_tensor(
        shortwave_transmittance, dtype=lai.dtype, device=dev,
    ).expand_as(lai).clamp(0.0, 1.0)
    tau_lw = tau_default if longwave_transmittance is None else torch.as_tensor(
        longwave_transmittance, dtype=lai.dtype, device=dev,
    ).expand_as(lai).clamp(0.0, 1.0)

    # Absorbed shortwave per ground area
    kabs = kdown * (1.0 - props.albedo_leaf) * (1.0 - tau_sw)

    # Below-canopy wind and boundary layer resistance
    u_canopy = torch.clamp(ws * torch.exp(-props.wind_extinction * lai), min=0.1)
    r_b = 200.0 * torch.sqrt(props.d_leaf / u_canopy)

    # Depth-averaged Kdown for Jarvis radiation stress. Recover the effective
    # optical depth from the supplied total transmittance so direct and diffuse
    # energy use exactly the same single attenuation applied by the solver.
    optical_depth = -torch.log(tau_sw.clamp(min=1.0e-6))
    mean_flux_factor = torch.where(
        optical_depth > 1.0e-6,
        (1.0 - tau_sw) / optical_depth,
        torch.ones_like(optical_depth),
    )
    kdown_leaf_mean = kdown * mean_flux_factor

    # Jarvis-Stewart stomatal conductance
    gs = _jarvis_canopy(props, kdown_leaf_mean, vpd_air, ta_c)
    r_s = torch.where(gs > 1e-8, 1.0 / gs, torch.full_like(gs, 1e10))

    # Initial guess
    if prev_t_leaf is not None:
        t_guess = torch.where(torch.isnan(prev_t_leaf), torch.full_like(prev_t_leaf, t_air_k), prev_t_leaf)
    else:
        t_guess = torch.full((R, C), t_air_k, dtype=torch.float32, device=dev)

    valid = canopy_mask.bool() & (lai_grid > 0)
    solved = _solve_leaf_temperature_batched(
        kabs, ldown, t_air_k, r_b, r_s, ea_kpa, gamma,
        props.emissivity_leaf, lai, tau_lw, props.allow_dew, t_guess,
        n_iterations=40,
        residual_tolerance=0.1,
        valid_mask=valid,
        return_diagnostics=True,
    )
    t_leaf = solved.temperature
    qh = solved.sensible_heat
    qe = solved.latent_heat
    rnet = solved.net_radiation

    successful = valid & solved.converged & ~solved.railed & ~solved.nonfinite
    nan_fill = torch.full_like(t_leaf, nan)
    t_leaf = torch.where(successful, t_leaf, nan_fill)
    qh = torch.where(successful, qh, nan_fill)
    qe = torch.where(successful, qe, nan_fill)
    rnet = torch.where(successful, rnet, nan_fill)

    return {
        't_leaf': t_leaf,
        'canopy_qh': qh,
        'canopy_qe': qe,
        'canopy_rnet': rnet,
        'canopy_solver_residual': solved.residual,
        'canopy_solver_converged': solved.converged,
        'canopy_solver_railed': solved.railed,
        'canopy_solver_nonfinite': solved.nonfinite,
        'canopy_solver_iterations': solved.iterations,
    }


def solve_canopy_eb_two_leaf(
    kdown_beam: torch.Tensor,      # Direct beam on horizontal (W/m²)
    kdown_diffuse: torch.Tensor,   # Diffuse irradiance (W/m²)
    ldown: torch.Tensor,           # LW from sky (W/m²)
    ta_c: float,
    ws: float,
    ea_hpa: float,
    pressure_kpa: float,
    solar_elev_deg: float,
    canopy_mask: torch.Tensor,
    lai_grid: torch.Tensor,
    props: CanopyProperties,
    prev_t_leaf: Optional[torch.Tensor] = None,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """Two-big-leaf canopy energy balance (de Pury & Farquhar 1997).

    Splits canopy into sunlit/shaded fractions with separate NR solves.
    Falls back to big-leaf when beam < 0.1 W/m² or solar_elev <= 0.
    """
    dev = device or kdown_beam.device
    R, C = _validate_canopy_inputs(
        {
            "kdown_beam": kdown_beam,
            "kdown_diffuse": kdown_diffuse,
            "ldown": ldown,
            "canopy_mask": canopy_mask,
            "lai_grid": lai_grid,
        },
        props,
        ta_c,
        ws,
        ea_hpa,
        pressure_kpa,
    )
    if prev_t_leaf is not None and tuple(prev_t_leaf.shape) != (R, C):
        raise ValueError("prev_t_leaf shape does not match canopy inputs")
    if not math.isfinite(solar_elev_deg) or not -90.0 <= solar_elev_deg <= 90.0:
        raise ValueError("solar_elev_deg must be between -90 and 90")

    # Fallback to big-leaf at night or low beam
    if solar_elev_deg <= 0.0 or kdown_beam.max().item() < 0.1:
        kdown_total = kdown_beam + kdown_diffuse
        # Compute uniform Ldown for big-leaf (esky * σ * Ta⁴)
        return solve_canopy_eb(
            kdown_total, ldown, ta_c, ws, ea_hpa, pressure_kpa,
            canopy_mask, lai_grid, props, prev_t_leaf, dev,
        )

    nan = float('nan')
    t_air_k = ta_c + 273.15
    ea_kpa = ea_hpa * 0.1
    p_kpa = pressure_kpa
    gamma = psychrometric_constant(p_kpa)
    vpd_air = max(e_sat_kpa(torch.tensor(t_air_k)).item() - ea_kpa, 0.0)

    sin_beta = max(math.sin(solar_elev_deg * math.pi / 180.0), 0.035)
    k_b = 0.5 * max(props.clumping_factor, 0.0) / sin_beta

    lai = torch.clamp(lai_grid, min=0.0)
    beam = torch.clamp(kdown_beam, min=0.0)
    diffuse = torch.clamp(kdown_diffuse, min=0.0)
    valid = canopy_mask.bool() & (lai_grid > 0)

    # LAI partition
    lai_sun = torch.clamp((1.0 - torch.exp(-k_b * lai)) / k_b, max=lai)
    lai_sha = torch.clamp(lai - lai_sun, min=0.0)

    # Transmittances
    tau_total = canopy_hemispherical_transmittance(lai, props.clumping_factor)
    tau_sun = torch.exp(-k_b * lai)

    # Absorbed shortwave per ground area
    lai_safe = torch.clamp(lai, min=0.01)
    kabs_sun = (beam * (1.0 - props.albedo_leaf) * (1.0 - tau_sun)
                + diffuse * (1.0 - props.albedo_leaf) * (1.0 - tau_total) * (lai_sun / lai_safe))
    kabs_sha = diffuse * (1.0 - props.albedo_leaf) * (1.0 - tau_total) * (lai_sha / lai_safe)

    # Wind and boundary layer
    u_canopy = torch.clamp(ws * torch.exp(-props.wind_extinction * lai), min=0.1)
    r_b = 200.0 * torch.sqrt(props.d_leaf / u_canopy)

    # Initial guess
    if prev_t_leaf is not None:
        t_guess = torch.where(torch.isnan(prev_t_leaf), torch.full_like(prev_t_leaf, t_air_k), prev_t_leaf)
    else:
        t_guess = torch.full((R, C), t_air_k, dtype=torch.float32, device=dev)

    # ── Sunlit leaf class ──
    tau_sun_class = canopy_hemispherical_transmittance(lai_sun, props.clumping_factor)
    optical_depth_sun = -torch.log(tau_sun_class.clamp(min=1.0e-6))
    kdown_sun_mean = torch.where(
        lai_sun > 0.01,
        (beam * (1.0 - tau_sun) + diffuse * (1.0 - tau_total) * (lai_sun / lai_safe))
        / optical_depth_sun.clamp(min=1.0e-6),
        beam + diffuse,
    )
    gs_sun = _jarvis_canopy(props, kdown_sun_mean, vpd_air, ta_c)
    rs_sun = torch.where(gs_sun > 1e-8, 1.0 / gs_sun, torch.full_like(gs_sun, 1e10))
    tau_sun_class = torch.where(lai_sun > 0.01, tau_sun_class, torch.ones_like(lai_sun))

    f_sun = lai_sun / lai_safe
    f_sha = lai_sha / lai_safe
    longwave_intercepted = 1.0 - tau_total

    active_sun = valid & (lai_sun > 0.0)
    solved_sun = _solve_leaf_temperature_batched(
        kabs_sun, ldown, t_air_k, r_b, rs_sun, ea_kpa, gamma,
        props.emissivity_leaf, lai_sun, tau_sun_class, props.allow_dew, t_guess,
        longwave_absorb_fraction=longwave_intercepted * f_sun,
        valid_mask=active_sun,
        return_diagnostics=True,
    )
    tl_sun = solved_sun.temperature
    qh_sun = solved_sun.sensible_heat
    qe_sun = solved_sun.latent_heat
    rn_sun = solved_sun.net_radiation

    # ── Shaded leaf class ──
    tau_sha_class = canopy_hemispherical_transmittance(lai_sha, props.clumping_factor)
    optical_depth_sha = -torch.log(tau_sha_class.clamp(min=1.0e-6))
    kdown_sha_mean = torch.where(
        lai_sha > 0.01,
        diffuse * (1.0 - tau_total) * (lai_sha / lai_safe)
        / optical_depth_sha.clamp(min=1.0e-6),
        diffuse * 0.5,
    )
    gs_sha = _jarvis_canopy(props, kdown_sha_mean, vpd_air, ta_c)
    rs_sha = torch.where(gs_sha > 1e-8, 1.0 / gs_sha, torch.full_like(gs_sha, 1e10))
    tau_sha_class = torch.where(lai_sha > 0.01, tau_sha_class, torch.ones_like(lai_sha))

    active_sha = valid & (lai_sha >= 0.01)
    solved_sha = _solve_leaf_temperature_batched(
        kabs_sha, ldown, t_air_k, r_b, rs_sha, ea_kpa, gamma,
        props.emissivity_leaf, lai_sha, tau_sha_class, props.allow_dew, t_guess,
        longwave_absorb_fraction=longwave_intercepted * f_sha,
        valid_mask=active_sha,
        return_diagnostics=True,
    )
    tl_sha = solved_sha.temperature
    qh_sha = solved_sha.sensible_heat
    qe_sha = solved_sha.latent_heat
    rn_sha = solved_sha.net_radiation

    # ── LAI-weighted aggregation ──
    t_leaf = tl_sun * f_sun + tl_sha * f_sha
    qh = qh_sun + qh_sha  # additive (both per ground area)
    qe = qe_sun + qe_sha
    rnet = rn_sun + rn_sha

    # For pixels where all leaves are sunlit (lai_sha < 0.01)
    all_sunlit = lai_sha < 0.01
    t_leaf = torch.where(all_sunlit, tl_sun, t_leaf)
    qh = torch.where(all_sunlit, qh_sun, qh)
    qe = torch.where(all_sunlit, qe_sun, qe)
    rnet = torch.where(all_sunlit, rn_sun, rnet)

    converged = solved_sun.converged & (~active_sha | solved_sha.converged)
    railed = (active_sun & solved_sun.railed) | (active_sha & solved_sha.railed)
    nonfinite = (
        (active_sun & solved_sun.nonfinite) | (active_sha & solved_sha.nonfinite)
    )
    successful = valid & converged & ~railed & ~nonfinite
    residual = torch.where(
        active_sun, solved_sun.residual, torch.zeros_like(solved_sun.residual)
    ) + torch.where(
        active_sha, solved_sha.residual, torch.zeros_like(solved_sha.residual)
    )
    residual = torch.where(valid, residual, torch.full_like(residual, float("nan")))
    iterations = torch.maximum(solved_sun.iterations, solved_sha.iterations)
    nan_fill = torch.full_like(t_leaf, nan)
    t_leaf = torch.where(successful, t_leaf, nan_fill)
    qh = torch.where(successful, qh, nan_fill)
    qe = torch.where(successful, qe, nan_fill)
    rnet = torch.where(successful, rnet, nan_fill)

    return {
        't_leaf': t_leaf,
        'canopy_qh': qh,
        'canopy_qe': qe,
        'canopy_rnet': rnet,
        'canopy_solver_residual': residual,
        'canopy_solver_converged': converged,
        'canopy_solver_railed': railed,
        'canopy_solver_nonfinite': nonfinite,
        'canopy_solver_iterations': iterations,
    }


def _jarvis_canopy(
    props: CanopyProperties,
    kdown_mean: torch.Tensor,
    vpd_air: float,
    ta_c: float,
) -> torch.Tensor:
    """Jarvis-Stewart conductance for canopy with scalar properties."""
    max_gs = props.max_stomatal_conductance
    if max_gs <= 0:
        return torch.zeros_like(kdown_mean)
    if max_gs > 900:
        return torch.full_like(kdown_mean, max_gs * 0.001)

    kdown_pos = torch.clamp(kdown_mean, min=0.0)
    f_rad = torch.clamp(kdown_pos / torch.clamp(kdown_pos + props.g1, min=0.01), 0.0, 1.0)
    f_vpd = 1.0 / (1.0 + props.g3 * max(vpd_air, 0.0))
    dt = ta_c - props.temp_optimal
    f_temp = max(0.01, 1.0 - props.g4 * dt * dt)

    return max_gs * 0.001 * f_rad * f_vpd * f_temp
