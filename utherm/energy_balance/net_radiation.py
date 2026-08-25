# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU tensor net radiation calculation.

Replaces Rust calculate_net_radiation with pure PyTorch tensor ops.
"""

import torch
from .physics import STEFAN_BOLTZMANN


def calculate_net_radiation(
    ldown: torch.Tensor,       # Downwelling longwave (W/m²), (rows, cols)
    kdown: torch.Tensor,       # Per-pixel shortwave (W/m²), (rows, cols)
    alb_grid: torch.Tensor,    # Surface albedo, (rows, cols)
    emis_grid: torch.Tensor,   # Surface emissivity, (rows, cols)
    t_air_k: float,            # Air temperature (K)
) -> torch.Tensor:
    """Compute net radiation for each pixel.

    Rnet = (1 - α) * Kdown + ε * Ldown - ε * σ * Ta⁴

    All inputs are GPU tensors of shape (rows, cols).

    Returns:
        rnet: (rows, cols) net radiation (W/m²).
    """
    k_abs = (1.0 - alb_grid) * kdown
    l_in = emis_grid * ldown
    l_out = emis_grid * STEFAN_BOLTZMANN * (t_air_k ** 4)
    return k_abs + l_in - l_out
