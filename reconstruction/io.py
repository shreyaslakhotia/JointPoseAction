"""Lightweight ASCII PLY export with explicit coordinate/scale provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from .visual_hull import VoxelGrid


def write_point_cloud_ply(points, path, *, coordinate_frame, units, metadata=None, overwrite=False):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("PLY points must be finite Nx3")
    if not isinstance(coordinate_frame, str) or not coordinate_frame or not isinstance(units, str) or not units:
        raise ValueError("coordinate_frame and units must be explicit")
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if not overwrite and (path.exists() or sidecar.exists()):
        raise FileExistsError("PLY or metadata sidecar already exists")
    provenance = dict(metadata or {})
    provenance.update({"representation": "point_cloud", "coordinate_frame": coordinate_frame,
                       "units": units, "point_count": len(points),
                       "bounds_xyz": np.column_stack((points.min(0), points.max(0))).tolist() if len(points) else None,
                       "metric_scale_verified": False})
    # ASCII JSON is escaped so user metadata cannot inject PLY header lines.
    compact = json.dumps(provenance, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\ncomment metadata_json " + compact + "\n")
        handle.write(f"element vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n")
        np.savetxt(handle, points, fmt="%.17g")
    provenance["ply_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance["ply_bytes"] = path.stat().st_size
    sidecar.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return provenance


def export_ply(grid: VoxelGrid, path, *, metadata=None, overwrite=False):
    details = dict(grid.metadata)
    details.update(metadata or {})
    details.update({"sample_representation": "occupied_voxel_centres",
                    "voxel_origin_xyz": grid.origin_xyz.tolist(),
                    "voxel_size_xyz": grid.voxel_size_xyz.tolist(),
                    "grid_shape_xyz": list(grid.occupancy.shape),
                    "grid_bounds_xyz": grid.bounds_xyz.tolist(),
                    "solid_rendering_requires": "original occupancy; point centres alone are not a solid silhouette"})
    return write_point_cloud_ply(grid.centers(), path, coordinate_frame=grid.coordinate_frame,
                                 units=grid.units, metadata=details, overwrite=overwrite)


def save_occupancy(grid: VoxelGrid, path, *, overwrite=False):
    """Retain the solid representation needed for exact voxel-box rendering."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Occupancy path must end in .npz")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, occupancy=grid.occupancy, origin_xyz=grid.origin_xyz,
                        voxel_size_xyz=grid.voxel_size_xyz,
                        coordinate_frame=np.array(grid.coordinate_frame), units=np.array(grid.units),
                        metadata_json=np.array(json.dumps(grid.metadata, allow_nan=False)))


def load_occupancy(path):
    with np.load(path, allow_pickle=False) as data:
        return VoxelGrid(data["origin_xyz"], data["voxel_size_xyz"], data["occupancy"],
                         str(data["coordinate_frame"].item()), str(data["units"].item()),
                         json.loads(data["metadata_json"].item()))
