# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canyon air temperature solver — VTUF-3D-inspired well-mixed box model.

Scalar computation (single domain-wide value per timestep).
Cannot be vectorized — performance irrelevant (microseconds).
"""

import math

from .config import CanyonAirTempConfig
from .physics import GRAVITY, AIR_DENSITY, AIR_SPECIFIC_HEAT, VON_KARMAN


def _calculate_httc_canyon_top(
    ri: float,
    u_above: float,
    dz: float,
    z0: float,
) -> float:
    """Mascart (1995) heat transfer coefficient at canyon top (m/s)."""
    u_eff = max(u_above, 1e-3)
    z0m = max(z0, 1.0e-6)
    z0h = z0m
    z = max(dz, z0m * 1.001)
    mu = 0.0
    cstar_h = 3.2165 + 4.3431 * mu + 0.5360 * mu ** 2 - 0.0781 * mu ** 3
    p_h = 0.5802 - 0.1571 * mu + 0.0327 * mu ** 2 - 0.0026 * mu ** 3
    ln_m = math.log(z / z0m)
    ln_h = math.log(z / z0h)
    aa = (VON_KARMAN / ln_m) ** 2
    c_h = cstar_h * aa * 9.4 * (ln_m / ln_h) * (z / z0h) ** p_h
    if ri > 0.0:
        f_h = (ln_m / ln_h) * (1.0 + 4.7 * ri) ** -2.0
    else:
        f_h = (ln_m / ln_h) * (1.0 - 9.4 * ri / (1.0 + c_h * math.sqrt(abs(ri))))
    return max(u_eff * aa / 0.74 * f_h, 1.0e-6)


def solve_canyon_air_temperature(
    tcan_prev: float,
    ta_k: float,
    ws: float,
    mean_sensible_heat: float,
    config: CanyonAirTempConfig,
) -> float:
    """Solve the reduced VTUF-3D-inspired well-mixed canyon budget.

    Args:
        tcan_prev: Previous canyon air temperature (K).
        ta_k: Above-canyon air temperature (K).
        ws: Wind speed (m/s).
        mean_sensible_heat: Domain-mean QH from surfaces (W/m²).
        config: Canyon air temperature configuration.

    Returns:
        Updated canyon air temperature (K).
    """
    cairavg = config.canyon_air_height * AIR_DENSITY * AIR_SPECIFIC_HEAT
    if cairavg <= 0.0:
        return ta_k

    u_above = max(ws, 1e-3)
    dz = max(config.z_ref - config.z_h + config.z0_roof, 0.1)

    tcan = tcan_prev
    remaining_dt = config.dt
    substep_dt = config.dt
    steps = 0

    while remaining_dt > 0.01 and steps < config.max_substeps:
        # Bulk Richardson number at canyon top
        t_corrhi = ta_k + 0.0098 * dz
        t_mean = (ta_k + tcan) / 2.0
        if abs(t_mean) > 1.0:
            ri = GRAVITY * dz * (t_corrhi - tcan) / t_mean / (u_above * u_above)
        else:
            ri = 0.0

        httc_top = _calculate_httc_canyon_top(ri, u_above, dz, config.z0_roof)
        qhtop = AIR_DENSITY * AIR_SPECIFIC_HEAT * httc_top * (tcan - ta_k)
        net_flux = mean_sensible_heat - qhtop
        dtcan = substep_dt / cairavg * net_flux

        # Adaptive sub-stepping
        if abs(dtcan) > config.max_dtcan and substep_dt > 1.0:
            substep_dt *= 5.0 / 8.0
            continue

        tcan += dtcan
        tcan = max(ta_k - 20.0, min(tcan, ta_k + 20.0))

        remaining_dt -= substep_dt
        steps += 1

        if abs(dtcan) < config.max_dtcan * 0.25:
            substep_dt = min(substep_dt * 1.5, remaining_dt)

    return tcan


def solve_canyon_air_temperature_coupled(
    tcan_prev: float,
    ta_k: float,
    ws: float,
    facets,
    config: CanyonAirTempConfig,
) -> float:
    """Implicit well-mixed canyon balance inspired by VTUF-3D.

    Unlike ``solve_canyon_air_temperature`` (which integrates a *fixed* surface QH
    source and can run away when that source is strongly negative under canopy
    transpiration), here EVERY facet's sensible heat is computed against ``Tcan``
    itself (``Qh_i = httc_i (Tsfc_i - Tcan)``). One
    implicit-Euler step of the well-mixed budget,

        Cair/dt (Tcan - Tcan_prev) = Σ_i w_i httc_i (Tsfc_i - Tcan)
                                       - rho_cp httc_top (Tcan - Ta)

    solves to a *conductance-weighted average* of the facet temps, ``Tcan_prev`` and
    ``Ta`` — always bounded between them, hence **unconditionally stable** (no runaway).
    The Ri-dependent canyon-top conductance ``httc_top`` is iterated to convergence.

    Args:
        tcan_prev: Previous canyon air temperature (K).
        ta_k: Above-canyon air temperature (K).
        ws: Wind speed (m/s).
        facets: iterable of (weight, httc, tsfc_k) — weight is the facet plan-area
            fraction, httc its convective coefficient (W/m^2/K), tsfc_k its surface
            temperature (K). Vegetated cells pass T_leaf; open ground passes ground Tsfc.
        config: Canyon air temperature configuration.

    Returns:
        Updated canyon air temperature (K).
    """
    rho_cp = AIR_DENSITY * AIR_SPECIFIC_HEAT
    cairavg = config.canyon_air_height * rho_cp
    if cairavg <= 0.0:
        return ta_k
    cair_dt = cairavg / max(config.dt, 1e-6)

    u_above = max(ws, 1e-3)
    dz = max(config.z_ref - config.z_h + config.z0_roof, 0.1)

    sum_w_httc = 0.0
    sum_w_httc_tsfc = 0.0
    for (w, h, tsfc) in facets:
        sum_w_httc += w * h
        sum_w_httc_tsfc += w * h * tsfc
    if sum_w_httc <= 0.0:
        return tcan_prev

    tcan = tcan_prev
    for _ in range(20):  # iterate the Ri-dependent top conductance to convergence
        t_corrhi = ta_k + 0.0098 * dz
        t_mean = (ta_k + tcan) / 2.0
        if abs(t_mean) > 1.0:
            ri = GRAVITY * dz * (t_corrhi - tcan) / t_mean / (u_above * u_above)
        else:
            ri = 0.0
        httc_top = _calculate_httc_canyon_top(ri, u_above, dz, config.z0_roof)
        c_top = rho_cp * httc_top
        denom = cair_dt + sum_w_httc + c_top
        if denom <= 0.0:
            break
        tcan_new = (cair_dt * tcan_prev + sum_w_httc_tsfc + c_top * ta_k) / denom
        if abs(tcan_new - tcan) < config.max_dtcan * 0.01:
            tcan = tcan_new
            break
        tcan = tcan_new

    return tcan
