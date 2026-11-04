"""Solid occupancy silhouettes from calibrated pixel-ray voxel traversal.

This renders occupied voxel boxes, not sparse projected centre points. A ray
exactly on a voxel boundary uses the half-open cell convention. The image is
sampled once at each pixel centre; no anti-aliasing or amodal completion.
"""
from __future__ import annotations

import numpy as np
from .cameras import Camera, camera_rays
from .visual_hull import VoxelGrid


def _raycast_batch(grid, origin, directions):
    bounds = grid.bounds_xyz
    zero = np.abs(directions) < 1e-15
    parallel_outside = np.any(zero & ((origin < bounds[:, 0]) | (origin >= bounds[:, 1])), axis=1)
    inverse = np.zeros_like(directions)
    np.divide(1., directions, out=inverse, where=~zero)
    t0 = (bounds[:, 0] - origin)*inverse
    t1 = (bounds[:, 1] - origin)*inverse
    low = np.where(zero, -np.inf, np.minimum(t0, t1))
    high = np.where(zero, np.inf, np.maximum(t0, t1))
    enter = np.maximum(np.max(low, axis=1), 0)
    leave = np.min(high, axis=1)
    active = (~parallel_outside) & (leave > enter)
    hits = np.zeros(len(directions), dtype=bool)
    # Shift the entry position by one representable coordinate towards the ray
    # interior. nextafter(t=0) alone is too small to move a nonzero coordinate,
    # and could falsely visit a zero-length cell when starting on a boundary.
    positions = origin + enter[:, None]*directions
    positions = np.where(directions > 0, np.nextafter(positions, np.inf),
                         np.where(directions < 0, np.nextafter(positions, -np.inf), positions))
    cells = np.floor((positions - grid.origin_xyz)/grid.voxel_size_xyz).astype(np.int64)
    shape = np.array(grid.occupancy.shape)
    # Floating roundoff at the outer boundary may leave one index exactly N.
    cells = np.clip(cells, 0, shape-1)
    step = np.sign(directions).astype(np.int64)
    boundary = grid.origin_xyz + (cells + (step > 0))*grid.voxel_size_xyz
    next_t = np.full_like(directions, np.inf)
    np.divide(boundary - origin, directions, out=next_t, where=~zero)
    delta = np.full_like(directions, np.inf)
    np.divide(grid.voxel_size_xyz, np.abs(directions), out=delta, where=~zero)
    for _ in range(int(shape.sum()) + 3):
        rows = np.flatnonzero(active)
        if not len(rows):
            return hits
        indices = cells[rows]
        occupied = grid.occupancy[indices[:, 0], indices[:, 1], indices[:, 2]]
        hits[rows[occupied]] = True
        active[rows[occupied]] = False
        rows = rows[~occupied]
        if not len(rows):
            continue
        crossing = np.min(next_t[rows], axis=1)
        # Step all exactly tied axes, avoiding false silhouette hits from
        # zero-length visits to cells touched only at an edge/corner.
        axes = next_t[rows] <= np.nextafter(crossing[:, None], np.inf)
        cells[rows] += step[rows]*axes
        next_t[rows] = np.where(axes, next_t[rows] + delta[rows], next_t[rows])
        inside = np.all((cells[rows] >= 0) & (cells[rows] < shape), axis=1)
        active[rows] = inside & (crossing < leave[rows])
    raise RuntimeError("DDA traversal exceeded geometric step bound")


def render_occupancy(grid: VoxelGrid, camera: Camera, *, batch_size=16384):
    if grid.coordinate_frame != camera.coordinate_frame or grid.units != camera.units:
        raise ValueError("Grid and rendering camera must share declared coordinate frame and units")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be positive integer")
    width, height = camera.image_size
    silhouette = np.zeros(width*height, dtype=bool)
    for start in range(0, len(silhouette), batch_size):
        stop = min(start + batch_size, len(silhouette))
        ids = np.arange(start, stop)
        pixels = np.column_stack((ids % width, ids // width))
        origin, directions = camera_rays(camera, pixels)
        silhouette[start:stop] = _raycast_batch(grid, origin, directions)
    return silhouette.reshape(height, width)
