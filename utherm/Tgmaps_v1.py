#UTherm: GPU-accelerated urban microclimate model
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
import numpy as np

def Tgmaps_v1(lc_grid, lc_class):
    """
    Populate surface property grids from land cover classification.
    
    Maps land cover classes to their corresponding thermal and optical properties
    for ground temperature wave calculations.
    
    Args:
        lc_grid (np.ndarray): Land cover classification grid
        lc_class (np.ndarray): Land cover lookup table with columns:
            [class_id, albedo, emissivity, TgK, Tstart, TmaxLST]
    
    Returns:
        tuple: (TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, 
                TmaxLST, TmaxLST_wall) - Surface property grids and wall parameters
    """

    lc_grid = np.asarray(lc_grid)
    lc_class = np.asarray(lc_class, dtype=float)

    TgK = np.empty(lc_grid.shape, dtype=float)
    Tstart = np.empty(lc_grid.shape, dtype=float)
    alb_grid = np.empty(lc_grid.shape, dtype=float)
    emis_grid = np.empty(lc_grid.shape, dtype=float)
    TmaxLST = np.empty(lc_grid.shape, dtype=float)
    mapped = np.zeros(lc_grid.shape, dtype=bool)

    for code in np.unique(lc_grid):
        rows = lc_class[lc_class[:, 0] == code]
        if rows.shape[0] != 1:
            raise ValueError(f"land-cover class {code:g} has no unique property row")
        mask = lc_grid == code
        alb_grid[mask] = rows[0, 1]
        emis_grid[mask] = rows[0, 2]
        TgK[mask] = rows[0, 3]
        Tstart[mask] = rows[0, 4]
        TmaxLST[mask] = rows[0, 5]
        mapped[mask] = True

    if not mapped.all():
        raise ValueError("not all land-cover cells were mapped")

    wall_rows = lc_class[lc_class[:, 0] == 99]
    if wall_rows.shape[0] != 1:
        raise ValueError("land-cover table must contain one wall row (code 99)")

    TgK_wall = np.array([wall_rows[0, 3]])
    Tstart_wall = np.array([wall_rows[0, 4]])
    TmaxLST_wall = np.array([wall_rows[0, 5]])

    return TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall
