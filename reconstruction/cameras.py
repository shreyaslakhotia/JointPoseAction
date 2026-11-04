"""Explicit right/down/forward calibrated pinhole cameras with Brown distortion.

Integer image coordinates are pixel centres. Extrinsics map declared world
coordinates into camera coordinates. Units are retained, never inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


def _array(value, shape, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Camera:
    camera_id: str
    K: np.ndarray
    world_to_camera: np.ndarray
    image_size: tuple[int, int]
    distortion: tuple[float, float, float, float, float] = (0., 0., 0., 0., 0.)
    coordinate_frame: str = "world"
    units: str = "unknown"

    def __post_init__(self):
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id must be nonempty")
        for name in ("coordinate_frame", "units"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be explicit; use 'unknown' for unverified units")
        K = _array(self.K, (3, 3), "K")
        T = _array(self.world_to_camera, (4, 4), "world_to_camera")
        if not np.allclose(K[2], [0, 0, 1], atol=1e-12, rtol=0) or abs(K[1, 0]) > 1e-12:
            raise ValueError("K must use the upper-triangular pinhole convention")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("focal lengths must be positive")
        if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-12, rtol=0):
            raise ValueError("world_to_camera must be homogeneous")
        R = T[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(np.linalg.det(R), 1, atol=1e-7):
            raise ValueError("extrinsic rotation must be proper orthonormal, without scale")
        if len(self.image_size) != 2 or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0 for n in self.image_size):
            raise ValueError("image_size must be positive integer (width, height)")
        coefficients = _array(self.distortion, (5,), "Brown distortion k1,k2,p1,p2,k3")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "world_to_camera", T)
        object.__setattr__(self, "image_size", tuple(int(n) for n in self.image_size))
        object.__setattr__(self, "distortion", tuple(float(x) for x in coefficients))

    def to_dict(self):
        return {"camera_id": self.camera_id, "K": self.K.tolist(),
                "world_to_camera": self.world_to_camera.tolist(),
                "image_size": list(self.image_size), "distortion": list(self.distortion),
                "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
                "coordinate_frame": self.coordinate_frame, "units": self.units,
                "camera_axes": "right_down_forward", "pixel_convention": "integer_centres"}

    @classmethod
    def from_dict(cls, row):
        """Load the canonical schema without guessing axes/coefficient order/units."""
        if row.get("camera_axes") != "right_down_forward" or row.get("pixel_convention") != "integer_centres":
            raise ValueError("Camera axes and integer pixel-centre convention must be declared")
        if row.get("distortion_order") != ["k1", "k2", "p1", "p2", "k3"]:
            raise ValueError("Named distortion order must be k1,k2,p1,p2,k3")
        required = ("camera_id", "K", "world_to_camera", "image_size", "distortion", "coordinate_frame", "units")
        if any(key not in row for key in required):
            raise ValueError("Canonical camera schema is missing required calibration metadata")
        return cls(**{key: row[key] for key in required})


def load_camera_json(path):
    """Load {'schema_version':1,'cameras':[Camera.to_dict(), ...]} explicitly."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("cameras"), list):
        raise ValueError("Expected canonical camera schema_version 1 and camera list")
    cameras = {}
    for row in payload["cameras"]:
        camera = Camera.from_dict(row)
        if camera.camera_id in cameras:
            raise ValueError("Duplicate camera ID")
        cameras[camera.camera_id] = camera
    if not cameras:
        raise ValueError("Camera list is empty")
    return cameras


def distort_normalized(xy, coefficients):
    xy = np.asarray(xy, dtype=np.float64)
    k1, k2, p1, p2, k3 = coefficients
    x, y = xy[..., 0], xy[..., 1]
    r2 = x*x + y*y
    radial = 1 + r2*(k1 + r2*(k2 + r2*k3))
    return np.stack((x*radial + 2*p1*x*y + p2*(r2 + 2*x*x),
                     y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y), axis=-1)


def undistort_normalized(distorted, coefficients, *, tolerance=1e-10, max_iterations=40):
    """Invert named five-coefficient Brown distortion with checked Newton solves.

    Extreme/non-invertible models fail explicitly instead of emitting dubious rays.
    """
    target = np.asarray(distorted, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 2 or not np.isfinite(target).all():
        raise ValueError("distorted coordinates must be finite Nx2")
    xy = target.copy()
    k1, k2, p1, p2, k3 = coefficients
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(max_iterations):
            residual = distort_normalized(xy, coefficients) - target
            if not np.isfinite(residual).all():
                raise ValueError("distortion inversion diverged")
            active = np.max(np.abs(residual), axis=1) > tolerance
            if not active.any():
                return xy
            x, y = xy[active].T
            r2 = x*x + y*y
            radial = 1 + r2*(k1 + r2*(k2 + r2*k3))
            slope = k1 + 2*k2*r2 + 3*k3*r2*r2
            a = radial + 2*x*x*slope + 2*p1*y + 6*p2*x
            b = 2*x*y*slope + 2*p1*x + 2*p2*y
            d = radial + 2*y*y*slope + 6*p1*y + 2*p2*x
            det = a*d - b*b
            if not np.isfinite(det).all() or np.any(np.abs(det) < 1e-14):
                raise ValueError("distortion inversion is singular")
            rx, ry = residual[active].T
            xy[active, 0] -= (d*rx - b*ry)/det
            xy[active, 1] -= (a*ry - b*rx)/det
        residual = distort_normalized(xy, coefficients) - target
    if not np.isfinite(xy).all() or not np.isfinite(residual).all() or np.max(np.abs(residual), initial=0) > tolerance:
        raise ValueError("distortion inversion did not converge")
    return xy


def project(camera: Camera, points):
    """Return Nx2 distorted pixel centres and signed camera z; z<=0 gives NaN pixels."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite Nx3")
    cam = points @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
    z = cam[:, 2]
    pixels = np.full((len(points), 2), np.nan)
    front = z > 0
    normalized = cam[front, :2]/z[front, None]
    distorted = distort_normalized(normalized, camera.distortion)
    pixels[front] = distorted @ camera.K[:2, :2].T + camera.K[:2, 2]
    return pixels, z


def camera_rays(camera: Camera, pixels=None):
    """Return common world origin (3,) and world directions Nx3.

    Directions have camera z=1, are not unit length. If pixels is omitted, rows
    cover the entire HxW image in row-major order. Distortion is inverted.
    """
    if pixels is None:
        width, height = camera.image_size
        yy, xx = np.indices((height, width))
        pixels = np.column_stack((xx.ravel(), yy.ravel()))
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError("pixels must be finite Nx2")
    normalized = (pixels - camera.K[:2, 2]) @ np.linalg.inv(camera.K[:2, :2]).T
    xy = undistort_normalized(normalized, camera.distortion)
    R, t = camera.world_to_camera[:3, :3], camera.world_to_camera[:3, 3]
    return -R.T @ t, np.column_stack((xy, np.ones(len(xy)))) @ R
