"""CPU reconstruction preparation; real-data scientific readiness is external."""

from .cameras import Camera, camera_rays, project
from .visual_hull import VoxelGrid, carve_voxels
from .rendering import render_occupancy

__all__ = ["Camera", "camera_rays", "project", "VoxelGrid", "carve_voxels", "render_occupancy"]
