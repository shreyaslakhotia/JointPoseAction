"""Voxel-centre visual hull using explicitly selected binary silhouettes."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import numpy as np
from .cameras import Camera, _array, project


@dataclass(frozen=True)
class VoxelGrid:
    origin_xyz: np.ndarray
    voxel_size_xyz: np.ndarray
    occupancy: np.ndarray
    coordinate_frame: str = "world"
    units: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        origin = _array(self.origin_xyz, (3,), "origin_xyz")
        size = _array(self.voxel_size_xyz, (3,), "voxel_size_xyz")
        if np.any(size <= 0):
            raise ValueError("voxel_size_xyz must be positive")
        occupied = np.array(self.occupancy, copy=True)
        if occupied.ndim != 3 or any(n == 0 for n in occupied.shape) or occupied.dtype != np.bool_:
            raise ValueError("occupancy must be nonempty-shape bool NxNyNz")
        if any(not isinstance(s, str) or not s for s in (self.coordinate_frame, self.units)):
            raise ValueError("coordinate_frame and units must be explicit strings")
        upper = origin + size*np.array(occupied.shape)
        if not np.isfinite(upper).all() or np.any(upper <= origin):
            raise ValueError("Grid bounds must be finite and numerically distinguishable")
        occupied.setflags(write=False)
        object.__setattr__(self, "origin_xyz", origin)
        object.__setattr__(self, "voxel_size_xyz", size)
        object.__setattr__(self, "occupancy", occupied)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def bounds_xyz(self):
        return np.column_stack((self.origin_xyz, self.origin_xyz + self.voxel_size_xyz*np.array(self.occupancy.shape)))

    def centers(self):
        return self.origin_xyz + (np.argwhere(self.occupancy) + .5)*self.voxel_size_xyz


def _binary_mask(mask, camera):
    array = np.asarray(mask)
    width, height = camera.image_size
    if array.shape != (height, width):
        raise ValueError(f"Mask dimensions for {camera.camera_id}: expected {(height, width)}, got {array.shape}")
    if array.dtype != np.bool_:
        if array.dtype.kind not in "ifu" or not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
            raise ValueError("Masks must be bool or explicit binary 0/1; convert 0/255 deliberately")
    return array.astype(bool, copy=False)


def carve_voxels(cameras: dict[str, Camera], masks: dict, bounds_xyz, resolution,
                 camera_ids, missing_policy="error", *, chunk_size=262144):
    """Intersect camera silhouette cones at voxel centres, with no real-data gate.

    This low-level routine supports synthetic tests. Real observations require an
    externally reviewed calibration, synchronization and mask readiness manifest.
    Out-of-image and behind-camera voxel centres are excluded. Missing views are
    either fatal or explicitly skipped and recorded; an empty effective set fails.
    """
    bounds = _array(bounds_xyz, (3, 2), "bounds_xyz")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Each bound maximum must exceed its minimum")
    if isinstance(resolution, (int, np.integer)) and not isinstance(resolution, (bool, np.bool_)):
        resolution = (int(resolution),)*3
    if not isinstance(resolution, (tuple, list, np.ndarray)) or len(resolution) != 3 or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0 for n in resolution):
        raise ValueError("resolution must be a positive integer or three positive integers")
    shape = tuple(int(n) for n in resolution)
    if missing_policy not in ("error", "skip"):
        raise ValueError("missing_policy must be 'error' or 'skip'")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be positive integer")
    ids = list(camera_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("camera_ids must be a nonempty ordered set without duplicates")
    unknown = [key for key in ids if key not in cameras]
    if unknown:
        raise ValueError(f"Unknown selected cameras: {unknown}")
    missing = [key for key in ids if key not in masks or masks[key] is None]
    if missing and missing_policy == "error":
        raise ValueError(f"Missing masks: {missing}")
    retained = [key for key in ids if key not in missing]
    if not retained:
        raise ValueError("No usable selected camera masks")
    first = cameras[retained[0]]
    validated = {}
    for key in retained:
        camera = cameras[key]
        if key != camera.camera_id:
            raise ValueError("Camera dictionary key disagrees with camera_id")
        if camera.coordinate_frame != first.coordinate_frame or camera.units != first.units:
            raise ValueError("Selected cameras must share declared coordinate frame and units")
        validated[key] = _binary_mask(masks[key], camera)
    voxel_size = (bounds[:, 1] - bounds[:, 0])/np.array(shape)
    total = int(np.prod(shape, dtype=object))
    occupancy = np.zeros(total, dtype=bool)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        indices = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
        points = bounds[:, 0] + (indices + .5)*voxel_size
        keep = np.ones(len(points), dtype=bool)
        for key in retained:
            active = np.flatnonzero(keep)
            if not len(active):
                break
            camera = cameras[key]
            pixels, depth = project(camera, points[active])
            width, height = camera.image_size
            visible = ((depth > 0) & np.isfinite(pixels).all(axis=1)
                       & (pixels[:, 0] >= -.5) & (pixels[:, 0] < width-.5)
                       & (pixels[:, 1] >= -.5) & (pixels[:, 1] < height-.5))
            sampled = np.zeros(len(active), dtype=bool)
            xy = np.floor(pixels[visible] + .5).astype(np.int64)
            sampled[visible] = validated[key][xy[:, 1], xy[:, 0]]
            keep[active] = sampled
        occupancy[start:stop] = keep
    return VoxelGrid(bounds[:, 0], voxel_size, occupancy.reshape(shape),
                     first.coordinate_frame, first.units,
                     {"algorithm": "voxel_centre_binary_silhouette_intersection",
                      "requested_camera_ids": ids, "retained_camera_ids": retained,
                      "missing_camera_ids": missing, "missing_policy": missing_policy,
                      "cameras": {key: cameras[key].to_dict() for key in retained},
                      "input_binary_mask_sha256": {
                          key: hashlib.sha256(np.ascontiguousarray(validated[key]).tobytes()).hexdigest()
                          for key in retained},
                      "mask_hash_encoding": "row-major bool bytes; dimensions from each camera image_size",
                      "resolution_xyz": list(shape), "bounds_xyz": bounds.tolist(),
                      "input_mask_semantics": "unspecified_by_geometry_caller",
                      "real_data_readiness_verified": False,
                      "voxel_samples": "centres; nearest pixel centre with half-open image bounds"})
