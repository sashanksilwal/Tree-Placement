# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Vectorized physics kernels for energy balance on GPU.

All functions operate on PyTorch tensors of shape (rows, cols) or (N,)
unless otherwise noted. Scalar inputs are broadcast automatically.

Port of Rust energy_balance.rs physics functions to PyTorch tensor ops.
"""

import math
from dataclasses import dataclass

import torch

# Physical constants
STEFAN_BOLTZMANN: float = 5.67051e-8  # W/m²/K⁴
GRAVITY: float = 9.81                  # m/s²
AIR_DENSITY: float = 1.225            # kg/m³ at sea level
AIR_SPECIFIC_HEAT: float = 1005.0     # J/(kg·K)
VON_KARMAN: float = 0.4
LATENT_HEAT_VAP: float = 2.501e6      # J/kg
EPSILON_RATIO: float = 0.622          # Mv/Md


# ── Vapor pressure utilities ─────────────────────────────────────

def e_sat_kpa(t_k: torch.Tensor) -> torch.Tensor:
    """Saturation vapor pressure (kPa) from temperature (K) — Magnus formula."""
    t_c = t_k - 273.15
    return 0.6108 * torch.exp(17.27 * t_c / (t_c + 237.3))


def delta_e_sat(t_k: torch.Tensor) -> torch.Tensor:
    """Slope of saturation vapor pressure curve (kPa/K) — Clausius-Clapeyron."""
    es = e_sat_kpa(t_k)
    t_c = t_k - 273.15
    denom = t_c + 237.3
    return 4098.0 * es / (denom * denom)


def psychrometric_constant(p_kpa: float) -> float:
    """Psychrometric constant γ (kPa/K) from atmospheric pressure."""
    return p_kpa * AIR_SPECIFIC_HEAT / (EPSILON_RATIO * LATENT_HEAT_VAP)


# ── Jarvis-Stewart surface conductance ───────────────────────────

def jarvis_stewart_conductance(
    max_gs_mm_s: torch.Tensor,
    kdown: torch.Tensor,
    vpd_kpa: float,
    ta_c: float,
    g1: torch.Tensor,
    g3: torch.Tensor,
    g4: torch.Tensor,
    temp_opt: torch.Tensor,
) -> torch.Tensor:
    """Jarvis-Stewart surface conductance (m/s) — vectorized per-pixel.

    Uses torch.where for impervious/water/vegetation branches.
    All tensor inputs should be (rows, cols) or broadcastable.

    Returns:
        gs (m/s): surface conductance tensor, same shape as max_gs_mm_s.
    """
    kdown_pos = torch.clamp(kdown, min=0.0)
    denom = torch.clamp(kdown_pos + g1, min=0.01)
    f_rad = torch.clamp(kdown_pos / denom, 0.0, 1.0)
    f_vpd = 1.0 / (1.0 + g3 * max(vpd_kpa, 0.0))
    dt = ta_c - temp_opt
    f_temp = torch.clamp(1.0 - g4 * dt * dt, min=0.01)

    gs_vegetation = max_gs_mm_s * 0.001 * f_rad * f_vpd * f_temp

    # Water: free evaporation (max_gs > 900 mm/s)
    gs_water = max_gs_mm_s * 0.001

    # Impervious: no evaporation (max_gs <= 0)
    gs = torch.where(max_gs_mm_s <= 0.0, torch.zeros_like(max_gs_mm_s),
         torch.where(max_gs_mm_s > 900.0, gs_water, gs_vegetation))

    return gs


# ── Convective heat transfer coefficients ────────────────────────

def calculate_httc_ground(
    u: torch.Tensor,
    t_air: torch.Tensor,
    t_sfc: torch.Tensor,
    z_ref: float,
    z0: float,
) -> torch.Tensor:
    """Mascart (1995) ground heat-transfer coefficient (W/m²/K).

    Uses the VTUF-3D implementation with ``z0h = z0m/10`` and a bulk
    Richardson number corrected for the dry adiabatic lapse rate.
    """
    u_eff = torch.clamp(u, min=0.1)
    z = max(float(z_ref), float(z0) * 1.001)
    z0m = max(float(z0), 1.0e-6)
    z0h = max(z0m / 10.0, z0m / 200.0)
    t_corr_air = t_air + GRAVITY / AIR_SPECIFIC_HEAT * z
    t_mean = torch.clamp((t_air + t_sfc) * 0.5, min=1.0)
    ri = GRAVITY * z * (t_corr_air - t_sfc) / (t_mean * u_eff * u_eff)

    mu = max(0.0, math.log(z0m / z0h))
    cstar_h = 3.2165 + 4.3431 * mu + 0.5360 * mu ** 2 - 0.0781 * mu ** 3
    p_h = 0.5802 - 0.1571 * mu + 0.0327 * mu ** 2 - 0.0026 * mu ** 3
    ln_m = math.log(z / z0m)
    ln_h = math.log(z / z0h)
    aa = (VON_KARMAN / ln_m) ** 2
    c_h = cstar_h * aa * 9.4 * (ln_m / ln_h) * (z / z0h) ** p_h
    stable = (ln_m / ln_h) * torch.pow(1.0 + 4.7 * ri, -2.0)
    unstable = (ln_m / ln_h) * (
        1.0 - 9.4 * ri / (1.0 + c_h * torch.sqrt(torch.abs(ri)))
    )
    f_h = torch.where(ri > 0.0, stable, unstable)
    httc = AIR_DENSITY * AIR_SPECIFIC_HEAT * u_eff * aa / 0.74 * f_h
    return torch.clamp(httc, 0.1, 500.0)


def calculate_httc_wall(u: float) -> float:
    """Convective heat transfer coefficient for walls (W/m²/K). Scalar."""
    u_eff = max(u, 0.1)
    httc = AIR_DENSITY * AIR_SPECIFIC_HEAT * (11.8 + 4.2 * u_eff) / 1000.0 - 4.0
    return max(5.0, min(httc, 100.0))


# ── G coupling beta (damping factor) ─────────────────────────────

def g_coupling_beta(
    heat_capacity: torch.Tensor,
    thickness_half: torch.Tensor,
    conductivity: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Time-averaged ground heat flux damping factor β.

    Prevents G over-swing from operator splitting at hourly timesteps.
    β = (τ/dt) * (1 - exp(-dt/τ)), floored at 0.2.
    """
    tau = heat_capacity * thickness_half * thickness_half / torch.clamp(conductivity, min=1e-10)
    ratio = dt / torch.clamp(tau, min=1e-6)

    beta = (tau / dt) * (1.0 - torch.exp(-ratio))

    # For very small ratio (ratio < 0.01) → beta ≈ 1
    beta = torch.where(ratio < 0.01, torch.ones_like(beta), beta)
    # For dt <= 0 or invalid → 1.0
    if dt <= 0.0:
        return torch.ones_like(beta)

    return torch.clamp(beta, min=0.2)


# ── Surface temperature solver ──────────────────────────────


@dataclass(frozen=True)
class SurfaceSolveResult:
    """Per-pixel result and diagnostics from the surface solve."""

    temperature: torch.Tensor
    residual: torch.Tensor
    converged: torch.Tensor
    railed: torch.Tensor
    nonfinite: torch.Tensor
    iterations: torch.Tensor

def solve_tsfc_newton_raphson(
    rnet: torch.Tensor,        # Net radiation (W/m²)
    httc: torch.Tensor,        # Heat transfer coefficient (W/m²/K)
    t_air: torch.Tensor,       # Air temperature (K), scalar broadcast OK
    t_layer: torch.Tensor,     # First subsurface layer temperature (K)
    emiss: torch.Tensor,       # Surface emissivity
    lambda_eff: torch.Tensor,  # Effective thermal conductivity (W/m/K)
    d: torch.Tensor,           # Thickness to first layer center (m)
    t_guess: torch.Tensor,     # Initial guess (K)
    ea_kpa: float,             # Actual vapor pressure (kPa)
    gamma: float,              # Psychrometric constant (kPa/K)
    ra: torch.Tensor,          # Aerodynamic resistance (s/m)
    rs: torch.Tensor,          # Surface resistance (s/m), large = no LE
    n_iterations: int = 40,
    residual_tolerance: float = 0.1,
    valid_mask: torch.Tensor | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | SurfaceSolveResult:
    """Solve surface temperature with safeguarded Newton iterations.

    Solves: εσ(T⁴ - Ta⁴) + (h + λ/d)·T + LE - Rnet - h·Ta - (λ/d)·T_layer = 0

    The equation is monotonic over the model's parameter range. Newton steps
    are kept inside a physical bracket and fall back to bisection when needed.
    A root outside 173.15–373.15 K is reported as railed instead of silently
    returned as a boundary temperature.

    Returns:
        Surface temperature, or ``SurfaceSolveResult`` when diagnostics are
        requested. Residuals are in W m-2.
    """
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive")

    active = torch.ones_like(rnet, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    lambda_over_d = lambda_eff / torch.clamp(d, min=1e-6)

    # Pre-compute εσTa⁴ correction (constant w.r.t. Tsfc)
    ta4 = t_air * t_air * t_air * t_air
    lw_air_correction = emiss * STEFAN_BOLTZMANN * ta4

    # LE coefficient: ρ*cp / (γ*(ra + rs)), zero for impervious (rs >= 1e6)
    le_active = rs < 1e6
    total_resistance = torch.where(le_active, ra + rs, torch.ones_like(ra))
    le_coeff = torch.where(
        le_active,
        AIR_DENSITY * AIR_SPECIFIC_HEAT / (gamma * total_resistance),
        torch.zeros_like(ra),
    )

    def equation(temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t3 = temperature * temperature * temperature
        t4 = t3 * temperature
        es = e_sat_kpa(temperature)
        le_raw = le_coeff * (es - ea_kpa)
        le = torch.where(le_active & (le_raw > 0), le_raw, torch.zeros_like(le_raw))
        delta = delta_e_sat(temperature)
        dle_dt = torch.where(le_active & (le_raw > 0), le_coeff * delta, torch.zeros_like(delta))
        residual = (emiss * STEFAN_BOLTZMANN * t4 - lw_air_correction
             + (httc + lambda_over_d) * temperature
             + le
             - rnet
             - httc * t_air
             - lambda_over_d * t_layer)
        derivative = 4.0 * emiss * STEFAN_BOLTZMANN * t3 + httc + lambda_over_d + dle_dt
        return residual, derivative

    scalar_inputs_finite = math.isfinite(float(ea_kpa)) and math.isfinite(float(gamma)) and gamma > 0.0
    finite_inputs = (
        torch.isfinite(rnet) & torch.isfinite(httc) & torch.isfinite(t_air)
        & torch.isfinite(t_layer) & torch.isfinite(emiss)
        & torch.isfinite(lambda_eff) & torch.isfinite(d)
        & torch.isfinite(t_guess) & torch.isfinite(ra) & torch.isfinite(rs)
        & (d > 0.0) & (lambda_eff >= 0.0) & (httc >= 0.0)
    )
    if not scalar_inputs_finite:
        finite_inputs = torch.zeros_like(finite_inputs)

    lower = torch.full_like(rnet, 173.15)
    upper = torch.full_like(rnet, 373.15)
    f_lower, _ = equation(lower)
    f_upper, _ = equation(upper)
    bracket_finite = torch.isfinite(f_lower) & torch.isfinite(f_upper)
    railed = active & finite_inputs & bracket_finite & (
        (f_lower > residual_tolerance) | (f_upper < -residual_tolerance)
    )
    nonfinite = active & (~finite_inputs | ~bracket_finite)
    eligible = active & ~railed & ~nonfinite

    temperature = torch.clamp(t_guess, 173.15, 373.15)
    residual, derivative = equation(temperature)
    nonfinite = nonfinite | (eligible & (~torch.isfinite(residual) | ~torch.isfinite(derivative)))
    eligible = eligible & ~nonfinite
    converged = eligible & (residual.abs() <= residual_tolerance)
    iterations = torch.zeros_like(rnet, dtype=torch.int32)

    for iteration in range(1, n_iterations + 1):
        working = eligible & ~converged
        if not bool(working.any()):
            break

        lower = torch.where(working & (residual < 0.0), temperature, lower)
        upper = torch.where(working & (residual >= 0.0), temperature, upper)
        newton = temperature - residual / torch.clamp(derivative, min=1.0e-10)
        use_bisection = (~torch.isfinite(newton)) | (newton <= lower) | (newton >= upper)
        candidate = torch.where(use_bisection, 0.5 * (lower + upper), newton)
        temperature = torch.where(working, candidate, temperature)
        residual, derivative = equation(temperature)

        failed_now = working & (~torch.isfinite(residual) | ~torch.isfinite(derivative))
        nonfinite = nonfinite | failed_now
        eligible = eligible & ~failed_now
        newly_converged = working & ~failed_now & (residual.abs() <= residual_tolerance)
        iterations = torch.where(newly_converged & ~converged, iteration, iterations)
        converged = converged | newly_converged

    iterations = torch.where(eligible & ~converged, n_iterations, iterations)
    residual = torch.where(active, residual, torch.full_like(residual, float("nan")))
    result = SurfaceSolveResult(
        temperature=temperature,
        residual=residual,
        converged=converged,
        railed=railed,
        nonfinite=nonfinite,
        iterations=iterations,
    )
    return result if return_diagnostics else result.temperature


# ── Thomas algorithm (batched conduction solver) ──────────────────

def solve_conduction_thomas_batched(
    t_layers: torch.Tensor,     # (N, n_layers) current layer temperatures
    t_sfc: torch.Tensor,        # (N,) surface temperatures
    t_deep: torch.Tensor,       # (N,) or scalar deep temperature
    thickness: torch.Tensor,    # (N, n_layers) layer thicknesses
    conductivity: torch.Tensor, # (N, n_layers) thermal conductivities
    heat_capacity: torch.Tensor,# (N, n_layers) volumetric heat capacities
    dt: float,                  # Timestep (s)
    insulated_deep_boundary: bool = False,
) -> torch.Tensor:
    """Batched Thomas algorithm for 1D heat conduction.

    Solves the tridiagonal system for N pixels simultaneously.
    The 4 sequential layer steps are unavoidable but process all N pixels in parallel.

    Returns:
        new_t_layers: (N, n_layers) updated layer temperatures.
    """
    N, n = t_layers.shape
    device = t_layers.device

    if n == 0:
        return t_layers.clone()

    # Finite-volume conductances at layer faces.  At an internal face the
    # two half layers act as thermal resistances in series.  This harmonic
    # form is required for flux continuity when conductivity or thickness
    # changes between layers.
    layer_capacity = heat_capacity * thickness  # J m-2 K-1
    g_up = torch.zeros_like(thickness)           # W m-2 K-1
    g_down = torch.zeros_like(thickness)
    g_up[:, 0] = 2.0 * conductivity[:, 0] / thickness[:, 0]
    if not insulated_deep_boundary:
        g_down[:, -1] = 2.0 * conductivity[:, -1] / thickness[:, -1]
    for k in range(n - 1):
        g_face = 1.0 / (
            thickness[:, k] / (2.0 * conductivity[:, k])
            + thickness[:, k + 1] / (2.0 * conductivity[:, k + 1])
        )
        g_down[:, k] = g_face
        g_up[:, k + 1] = g_face

    coeff_up = dt * g_up / layer_capacity
    coeff_down = dt * g_down / layer_capacity

    # Build tridiagonal coefficients: a (sub), b (main), c (super), d (RHS)
    dtype = t_layers.dtype
    a = torch.zeros(N, n, dtype=dtype, device=device)
    b = torch.zeros(N, n, dtype=dtype, device=device)
    c_coef = torch.zeros(N, n, dtype=dtype, device=device)
    d_vec = torch.zeros(N, n, dtype=dtype, device=device)

    for k in range(n):
        if n == 1:
            b[:, k] = 1.0 + coeff_up[:, k] + coeff_down[:, k]
            d_vec[:, k] = (
                t_layers[:, k]
                + coeff_up[:, k] * t_sfc
                + coeff_down[:, k] * t_deep
            )
        elif k == 0:
            b[:, k] = 1.0 + coeff_up[:, k] + coeff_down[:, k]
            c_coef[:, k] = -coeff_down[:, k]
            d_vec[:, k] = t_layers[:, k] + coeff_up[:, k] * t_sfc
        elif k == n - 1:
            a[:, k] = -coeff_up[:, k]
            b[:, k] = 1.0 + coeff_up[:, k] + coeff_down[:, k]
            d_vec[:, k] = t_layers[:, k] + coeff_down[:, k] * t_deep
        else:
            a[:, k] = -coeff_up[:, k]
            b[:, k] = 1.0 + coeff_up[:, k] + coeff_down[:, k]
            c_coef[:, k] = -coeff_down[:, k]
            d_vec[:, k] = t_layers[:, k]

    # Thomas forward elimination (sequential over layers, parallel over pixels)
    c_prime = torch.zeros(N, n, dtype=dtype, device=device)
    d_prime = torch.zeros(N, n, dtype=dtype, device=device)

    c_prime[:, 0] = c_coef[:, 0] / b[:, 0]
    d_prime[:, 0] = d_vec[:, 0] / b[:, 0]

    for k in range(1, n):
        denom = b[:, k] - a[:, k] * c_prime[:, k - 1]
        # Avoid division by zero — fallback to unchanged temps
        safe_denom = torch.where(denom.abs() < 1e-10, torch.ones_like(denom), denom)
        c_prime[:, k] = c_coef[:, k] / safe_denom
        d_prime[:, k] = (d_vec[:, k] - a[:, k] * d_prime[:, k - 1]) / safe_denom

    # Back substitution
    result = torch.zeros(N, n, dtype=dtype, device=device)
    result[:, n - 1] = d_prime[:, n - 1]
    for k in range(n - 2, -1, -1):
        result[:, k] = d_prime[:, k] - c_prime[:, k] * result[:, k + 1]

    return result


def compute_max_fourier(
    thickness: torch.Tensor,    # (N, n_layers) or (n_layers,)
    conductivity: torch.Tensor,
    heat_capacity: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Compute maximum Fourier number across all layers.

    Returns scalar (global max across all pixels and layers).
    """
    if thickness.dim() == 1:
        thickness = thickness.unsqueeze(0)
        conductivity = conductivity.unsqueeze(0)
        heat_capacity = heat_capacity.unsqueeze(0)

    n = thickness.shape[-1]
    alpha_k = conductivity / heat_capacity

    max_fo = torch.zeros(thickness.shape[0], device=thickness.device)

    for k in range(n):
        dk = thickness[:, k]
        dx_up = dk / 2.0 if k == 0 else (thickness[:, k - 1] + dk) / 2.0
        dx_down = dk / 2.0 if k == n - 1 else (thickness[:, k + 1] + dk) / 2.0
        fo_up = alpha_k[:, k] * dt / (dx_up * dx_up)
        fo_down = alpha_k[:, k] * dt / (dx_down * dx_down)
        max_fo = torch.maximum(max_fo, torch.maximum(fo_up, fo_down))

    return max_fo.max()


def solve_conduction_substepped(
    t_layers: torch.Tensor,
    t_sfc: torch.Tensor,
    t_deep: torch.Tensor,
    thickness: torch.Tensor,
    conductivity: torch.Tensor,
    heat_capacity: torch.Tensor,
    dt: float,
    insulated_deep_boundary: bool = False,
) -> torch.Tensor:
    """Advance layer temperatures with the implicit finite-volume solver.

    The function name is retained for API compatibility. Backward Euler is
    unconditionally stable, so Fourier-number substeps are not required.
    """
    return solve_conduction_thomas_batched(
        t_layers,
        t_sfc,
        t_deep,
        thickness,
        conductivity,
        heat_capacity,
        dt,
        insulated_deep_boundary,
    )
