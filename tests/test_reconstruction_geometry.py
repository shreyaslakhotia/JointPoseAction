"""Independent analytic synthetic checks; no observed data or GPU required."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from reconstruction.cameras import Camera, camera_rays, project, load_camera_json
from reconstruction.io import export_ply, load_occupancy, save_occupancy
from reconstruction.rendering import render_occupancy
from reconstruction.visual_hull import VoxelGrid, carve_voxels


def virtual_camera(name="front", position=(0, 0, -4), size=65, focal=48., distortion=(0,)*5):
    position = np.asarray(position, dtype=float)
    forward = -position/np.linalg.norm(position)
    up = np.array([0., -1, 0])
    if abs(up @ forward) > .99:
        up = np.array([0., 0, 1])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, :3] = np.stack((right, down, forward))
    T[:3, 3] = -T[:3, :3] @ position
    K = np.array([[focal, 0, (size-1)/2], [0, focal, (size-1)/2], [0, 0, 1.]])
    return Camera(name, K, T, (size, size), distortion, "synthetic_world", "synthetic_units")


def analytic_box_mask(camera, low, high):
    """Independent ray/AABB slab reference, without using occupancy renderer."""
    origin, directions = camera_rays(camera)
    hits = []
    for direction in directions:
        lower, upper = 0., np.inf
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if not low[axis] <= origin[axis] < high[axis]:
                    upper = -1.
                    break
            else:
                first = (low[axis] - origin[axis])/direction[axis]
                second = (high[axis] - origin[axis])/direction[axis]
                lower = max(lower, min(first, second))
                upper = min(upper, max(first, second))
        hits.append(upper > lower)
    return np.array(hits).reshape(camera.image_size[::-1])


class CameraTests(unittest.TestCase):
    def test_known_projection_and_behind_camera(self):
        camera = virtual_camera()
        pixels, depth = project(camera, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -5]])
        np.testing.assert_allclose(pixels[:3], [[32, 32], [44, 32], [32, 44]])
        np.testing.assert_allclose(depth, [4, 4, 4, -1])
        self.assertTrue(np.isnan(pixels[3]).all())

    def test_nonidentity_rotation_translation_and_distortion(self):
        # World +x is forward, world -z is image-right; nontrivial extrinsic.
        camera = virtual_camera(position=(-4, 0, 0), distortion=(.1, -.02, .003, -.002, .001))
        origin, rays = camera_rays(camera, [[0, 0], [32, 32], [50, 48], [64, 64]])
        np.testing.assert_allclose(origin, [-4, 0, 0], atol=1e-12)
        points = origin + rays*3.25
        pixels, depth = project(camera, points)
        np.testing.assert_allclose(pixels, [[0, 0], [32, 32], [50, 48], [64, 64]], atol=1e-7)
        np.testing.assert_allclose(depth, 3.25)
        # Independent Brown formula for x=.2,y=.1.
        cam = virtual_camera(distortion=(.1, 0, .01, -.02, 0))
        got, _ = project(cam, [[.8, .4, 0]])
        expected = [32+48*(.2*1.005 + 2*.01*.2*.1 - .02*(.05+2*.04)),
                    32+48*(.1*1.005 + .01*(.05+2*.01) - 2*.02*.2*.1)]
        np.testing.assert_allclose(got[0], expected)

    def test_invalid_calibration_and_explicit_json_schema(self):
        camera = virtual_camera()
        bad = camera.world_to_camera.copy()
        bad[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "rotation"):
            Camera("bad", camera.K, bad, camera.image_size)
        with self.assertRaises(ValueError):
            Camera("bad", np.diag([-2, 2, 1]), np.eye(4), (10, 10))
        with self.assertRaises(ValueError):
            Camera("bad", np.eye(3), np.eye(4), (10.5, 10))
        row = camera.to_dict()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"cameras.json"
            path.write_text(json.dumps({"schema_version": 1, "cameras": [row]}))
            parsed = load_camera_json(path)[camera.camera_id]
            np.testing.assert_array_equal(parsed.K, camera.K)
            del row["units"]
            path.write_text(json.dumps({"schema_version": 1, "cameras": [row]}))
            with self.assertRaises(ValueError):
                load_camera_json(path)


class SolidRenderingTests(unittest.TestCase):
    def test_voxel_box_matches_independent_analytic_silhouette(self):
        camera = virtual_camera()
        grid = VoxelGrid([-1, -1, -1], [2, 2, 2], np.ones((1, 1, 1), bool), camera.coordinate_frame, camera.units)
        rendered = render_occupancy(grid, camera, batch_size=17)
        expected = analytic_box_mask(camera, np.array([-1]*3), np.array([1]*3))
        np.testing.assert_array_equal(rendered, expected)
        self.assertGreater(rendered.sum(), 500)  # One centre projection would hit one pixel.

    def test_rotated_distorted_camera_and_sparse_occupancy(self):
        camera = virtual_camera(position=(-4, 1, -2), distortion=(.08, -.01, .002, -.003, .001))
        occupied = np.zeros((3, 2, 2), bool)
        occupied[0, 0, 1] = True
        occupied[2, 1, 0] = True
        grid = VoxelGrid([-1.5, -1, -1], [1, 1, 1], occupied, camera.coordinate_frame, camera.units)
        expected = np.zeros(camera.image_size[::-1], bool)
        for index in np.argwhere(occupied):
            low = grid.origin_xyz + index*grid.voxel_size_xyz
            expected |= analytic_box_mask(camera, low, low+grid.voxel_size_xyz)
        np.testing.assert_array_equal(render_occupancy(grid, camera), expected)

    def test_empty_grid_behind_camera_and_inside_origin(self):
        camera = virtual_camera()
        grid = VoxelGrid([-1, -1, -1], [1, 1, 1], np.zeros((2, 2, 2), bool), camera.coordinate_frame, camera.units)
        self.assertFalse(render_occupancy(grid, camera).any())
        behind = VoxelGrid([-1, -1, -8], [2, 2, 2], np.ones((1, 1, 1), bool), camera.coordinate_frame, camera.units)
        self.assertFalse(render_occupancy(behind, camera).any())
        inside = VoxelGrid([-1, -1, -5], [2, 2, 2], np.ones((1, 1, 1), bool), camera.coordinate_frame, camera.units)
        self.assertTrue(render_occupancy(inside, camera).all())

    def test_frame_or_units_mismatch_fails(self):
        camera = virtual_camera()
        grid = VoxelGrid([0]*3, [1]*3, np.ones((1, 1, 1), bool), "other", "unknown")
        with self.assertRaises(ValueError):
            render_occupancy(grid, camera)

    def test_internal_boundary_does_not_hit_zero_length_cell(self):
        # Camera centre x=0 lies on the split between occupied +x and empty -x.
        camera = virtual_camera(position=(0, 0, -4))
        T = np.eye(4)
        T[:3, :3] = [[0,0,1], [0,1,0], [-1,0,0]]
        # Camera looks along world -x from the grid's internal x=1 plane.
        T[:3, 3] = -T[:3, :3] @ np.array([1.,.5,.5])
        camera = Camera("boundary", np.eye(3), T, (1,1), coordinate_frame="world", units="unknown")
        occupied = np.zeros((2,1,1), bool)
        occupied[1,0,0] = True
        grid = VoxelGrid([0,0,0], [1,1,1], occupied)
        self.assertFalse(render_occupancy(grid, camera)[0,0])


class CarvingTests(unittest.TestCase):
    def setUp(self):
        self.camera = virtual_camera()
        self.cameras = {"front": self.camera}
        self.mask = analytic_box_mask(self.camera, np.array([-1]*3), np.array([1]*3))

    def test_known_cone_constraint_and_resolution_sensitivity(self):
        bounds = [[-2, 2], [-2, 2], [-1, 1]]
        volumes = []
        # Tangential rays at16/48 touch the cube only at a zero-length edge;
        # the solid analytic silhouette spans inclusive pixels17..47.
        # Nearest-centre carving therefore uses continuous16.5..47.5.
        # Cone half-width a(z)=(15.5/48)*(z+4); exact integral 4*a(z)^2 dz.
        a = 15.5/48
        analytic_volume = 4*a*a*((5**3-3**3)/3)
        for resolution in (12, 48):
            grid = carve_voxels(self.cameras, {"front": self.mask}, bounds, resolution, ["front"], chunk_size=211)
            points = grid.centers()
            self.assertGreater(len(points), 0)
            pixels, _ = project(self.camera, points)
            self.assertTrue(np.all((pixels >= 16.5) & (pixels < 47.5)))
            volumes.append(grid.occupancy.sum()*np.prod(grid.voxel_size_xyz))
        self.assertLess(abs(volumes[1]-analytic_volume), abs(volumes[0]-analytic_volume))
        self.assertLess(abs(volumes[1]-analytic_volume)/analytic_volume, .02)

    def test_sphere_masks_multiview_and_no_hidden_surface_claim(self):
        cameras = {str(i): virtual_camera(str(i), position=position) for i, position in enumerate(
            [(0,0,-4), (0,0,4), (-4,0,0), (4,0,0), (0,-4,0), (0,4,0)])}
        yy, xx = np.indices((65, 65))
        sphere_mask = ((xx-32)**2+(yy-32)**2) <= (48/np.sqrt(15))**2
        masks = {key: sphere_mask for key in cameras}
        grid = carve_voxels(cameras, masks, [[-1.3,1.3]]*3, 32, list(cameras))
        points = grid.centers()
        self.assertTrue(np.any(np.linalg.norm(points, axis=1) < .2))
        self.assertTrue(np.all(np.linalg.norm(points, axis=1) < 1.4))
        for camera in cameras.values():
            rendered = render_occupancy(grid, camera)
            iou = (rendered & sphere_mask).sum()/(rendered | sphere_mask).sum()
            self.assertGreater(iou, .85)
        self.assertEqual(grid.metadata["retained_camera_ids"], list(cameras))
        self.assertFalse(grid.metadata["real_data_readiness_verified"])

    def test_camera_subset_missing_empty_and_invalid_dimensions(self):
        cameras = dict(self.cameras, side=virtual_camera("side", position=(4,0,0)))
        with self.assertRaisesRegex(ValueError, "Missing"):
            carve_voxels(cameras, {"front": self.mask}, [[-1,1]]*3, 4, ["front", "side"])
        grid = carve_voxels(cameras, {"front": self.mask}, [[-1,1]]*3, (3,4,5), ["front", "side"], "skip")
        self.assertEqual(grid.occupancy.shape, (3,4,5))
        self.assertEqual(grid.metadata["missing_camera_ids"], ["side"])
        self.assertEqual(grid.metadata["retained_camera_ids"], ["front"])
        with self.assertRaisesRegex(ValueError, "No usable"):
            carve_voxels(cameras, {}, [[-1,1]]*3, 4, ["front"], "skip")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            carve_voxels(self.cameras, {"front": np.zeros((4,4), bool)}, [[-1,1]]*3, 4, ["front"])
        with self.assertRaisesRegex(ValueError, "binary"):
            carve_voxels(self.cameras, {"front": self.mask.astype(np.uint8)*255}, [[-1,1]]*3, 4, ["front"])
        for resolution in (True, 0, 3.2, (2,3,0)):
            with self.assertRaises(ValueError):
                carve_voxels(self.cameras, {"front": self.mask}, [[-1,1]]*3, resolution, ["front"])

    def test_empty_silhouette_is_valid_explicit_empty_reconstruction(self):
        grid = carve_voxels(self.cameras, {"front": np.zeros((65,65), bool)}, [[-1,1]]*3, 4, ["front"])
        self.assertEqual(grid.centers().shape, (0,3))

    def test_ply_and_occupancy_roundtrip_preserve_unknown_scale(self):
        grid = VoxelGrid([1,2,3], [.1,.2,.3], np.ones((2,2,2), bool), "declared_world", "unknown")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"points.ply"
            report = export_ply(grid, path)
            self.assertEqual(report["units"], "unknown")
            self.assertFalse(report["metric_scale_verified"])
            text = path.read_text()
            coordinates = np.loadtxt(text.split("end_header\n", 1)[1].splitlines())
            np.testing.assert_allclose(coordinates, grid.centers())
            self.assertTrue(path.with_suffix(".ply.metadata.json").is_file())
            with self.assertRaises(FileExistsError):
                export_ply(grid, path)
            occupancy = Path(temp)/"occupancy.npz"
            save_occupancy(grid, occupancy)
            restored = load_occupancy(occupancy)
            np.testing.assert_array_equal(restored.occupancy, grid.occupancy)
            self.assertEqual(restored.units, "unknown")


if __name__ == "__main__":
    unittest.main()
