# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Domain and device tests for the operational UTCI approximation."""

import torch

from utherm.calculate_utci import utci_calculator


def test_utci_accepts_published_domain(device):
    ta = torch.tensor([25.0], device=device)
    result = utci_calculator(
        ta, torch.tensor([50.0], device=device),
        torch.tensor([30.0], device=device), torch.tensor([1.0], device=device),
    )
    assert torch.isfinite(result).all()
    assert result.item() != -999.0


def test_utci_rejects_wind_below_half_metre_per_second(device):
    result = utci_calculator(
        torch.tensor([25.0], device=device), torch.tensor([50.0], device=device),
        torch.tensor([30.0], device=device), torch.tensor([0.49], device=device),
    )
    assert result.item() == -999.0


def test_utci_rejects_radiant_temperature_difference_outside_domain(device):
    result = utci_calculator(
        torch.tensor([25.0], device=device), torch.tensor([50.0], device=device),
        torch.tensor([96.0], device=device), torch.tensor([1.0], device=device),
    )
    assert result.item() == -999.0
