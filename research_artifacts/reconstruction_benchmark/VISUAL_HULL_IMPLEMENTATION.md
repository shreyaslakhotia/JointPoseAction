# Visual-hull implementation and synthetic validation

Status: CPU implementation and synthetic validation only. No Assembly101 object
was reconstructed. Calibration accuracy, temporal correspondence and native SAM 3
mask review remain prerequisites for a real experiment. The low-level geometry
functions are deliberately usable by synthetic tests; they do not authorize real
data or certify a dataset. The benchmark entrypoint must enforce readiness.

## Camera and scale contract

`reconstruction.cameras.Camera` holds an immutable copied intrinsic matrix,
`world_to_camera` rigid 4x4 transform, `(width, height)`, camera ID, coordinate-frame
name, unit name and five named Brown coefficients in **k1, k2, p1, p2, k3** order.
World coordinates transform as `X_camera = R @ X_world + t`. Camera axes point
right, down and forward. Positive camera z is in front of the camera. Integer
image coordinates identify pixel centres; intrinsics may include x skew.
`project(camera, points)` returns distorted pixel coordinates and signed camera
depth; behind-camera pixel coordinates are NaN.

Validation rejects nonfinite values, invalid matrix shapes, nonpositive focal
lengths, noninteger image sizes, a nonhomogeneous last transform row, reflections,
extrinsic scale and nonorthonormal rotations. These checks establish a coherent
mathematical convention. They **do not establish image-to-camera association,
calibration accuracy, timestamp correspondence or physical scale**.

The canonical JSON loader requires `schema_version: 1` and a `cameras` list of
`Camera.to_dict()` records. Axes, pixel convention, coefficient order, frame and
units must be declared explicitly; missing fields fail. No filename, calibration
translation magnitude or toy dimensions are used to guess units. Use
`units="unknown"` for unverified scale. All selected cameras and the rendered grid
must share their declared frame and units. Matching labels alone are not evidence
that those labels are correct. Original distorted-video calibration must never
be silently used with a separately rectified image release.

`camera_rays(camera, pixels=None)` inversely projects distorted pixel centres.
It uses checked Newton inversion of the five-coefficient Brown model and returns
one world origin plus world directions whose camera z is 1. Directions are not
unit-length vectors. Singular, divergent or nonconvergent inversion fails.
Convergence is a local numerical check, not a proof of global distortion-model
invertibility or vendor calibration accuracy.

## Silhouette carving

`carve_voxels(cameras, masks, bounds_xyz, resolution, camera_ids,
missing_policy="error")` accepts ordered, predetermined camera IDs. `bounds_xyz`
has shape `(3,2)` with minimum/maximum rows for x, y and z. Resolution is one
positive integer or three positive integers. The output `VoxelGrid` stores
occupancy in **x, y, z** axis order, the lower-corner origin, per-axis voxel sizes,
frame, units and a metadata record of requested/retained/missing cameras, actual
calibration matrices, binary-mask SHA-256 values, bounds and resolution.

For every voxel centre, the algorithm projects it through every retained camera
and intersects the corresponding silhouette-cone membership tests. A projected
centre uses `floor(pixel + 0.5)` nearest-pixel sampling with half-open image
support `[-0.5, width-0.5) × [-0.5, height-0.5)`. Behind-camera or out-of-image
centres are excluded. Chunked projection limits temporary coordinate arrays;
occupancy storage still grows with the total number of voxels. This is a
voxel-centre approximation, not conservative voxel-corner carving.

Masks must be boolean or numeric binary 0/1 of exactly `(height, width)`.
Numeric 0/255 masks require a deliberate conversion. Missing masks fail by
default; `missing_policy="skip"` records omitted cameras and fails if no usable
camera remains. An available empty mask is valid evidence yielding an empty
intersection, rather than being silently treated as missing. Duplicate/unknown
camera IDs, inconsistent frame/units and invalid bounds/resolutions fail.

The hull is the intersection of silhouette cones, clipped to the supplied volume.
It generally **fills silhouette-invisible concavities**. Occlusion-corrupted masks,
incorrect instances and calibration errors can remove valid object volume or
inflate the hull. A plausible point cloud therefore does not imply correct
surface, dimensions or metric hand–object distances. Predicted SAM 3 silhouettes
are inputs for agreement analysis, not independent segmentation ground truth.

## Solid rendering versus point coverage

`render_occupancy(grid, camera)` casts one calibrated ray through each output
pixel centre and uses slab intersection plus 3D DDA traversal to test intersection
with occupied **voxel boxes**. A positive-length intersection produces a foreground
pixel. It steps tied voxel boundaries together, avoiding false foreground from
zero-length edge/corner contacts. Parallel rays, cameras inside the grid, objects
behind the camera and empty occupancy are handled explicitly. Brown distortion
is inverted for the rays, so output masks remain in the native distorted image
domain. The renderer is a CPU reference, with image rays processed in batches.

The rendered mask is a solid occupancy silhouette, not a scatter of projected
voxel centres. Pixel-centre sampling has no antialiasing; binary edge pixels may
change with voxel resolution. The synthetic benchmark reports resolution
sensitivity. Point-cloud coverage can be reported separately and must never be
substituted for solid silhouette IoU. DDA runtime scales approximately with image
pixels times traversed grid cells; high-resolution native images are not promised
to render interactively with this CPU reference.

## Artifacts and provenance

`export_ply(grid, path)` writes occupied voxel centres as double-precision ASCII
PLY, with frame/units and configuration metadata in a JSON header comment plus a
`.ply.metadata.json` sidecar containing the PLY SHA-256 and byte count. Unknown
units remain unknown. Metric-scale verification is false in this preparation
implementation. Original image, checkpoint, prompt and source identifiers must
be attached by the benchmark's provenance record; binary-mask hashes alone do
not identify their image/model history.

`save_occupancy` / `load_occupancy` retain the full solid voxel representation in
compressed NPZ with `allow_pickle=False` on load. PLY centre points alone are
insufficient to recover an equivalent rendered silhouette unless voxel sizes,
origin and occupancy semantics are also retained. Existing output files are
protected against accidental overwrite by default. Optional mesh extraction is
deferred; this implementation prioritizes the point-cloud/occupancy interface.
It does not imply a watertight smooth mesh or a learned surface representation.

## CPU tests

Run from the repository root:

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s tests -p test_reconstruction_geometry.py -v
```

The 13 geometry tests passed locally on 2026-11-05 (0.817 seconds for this
targeted run). They check:

- Analytically known projected coordinates and behind-camera treatment.
- A nonidentity rotation/translation, inverse rays, named radial/tangential
  distortion against an independent scalar formula, and explicit JSON loading.
- Invalid calibration, dimensions, binary encodings, resolution and missing-view
  handling, plus deliberate empty reconstructions.
- Solid ray-cast silhouettes against an independent ray/AABB slab reference for
  one occupied cube and separated occupied boxes under a rotated distorted camera.
- Camera-inside-grid, entirely behind-camera, empty occupancy and an internal
  boundary with an occupied cell of zero ray-intersection length.
- A known silhouette-cone volume at resolutions 12 and 48; the finer grid is
  closer to the analytic integral, with less than 2% relative volume error.
- Six analytically generated sphere silhouettes, nonempty central occupancy and
  solid-rendered per-view agreement above 0.85. Finite-view hulls are not asserted
  to recover hidden sphere geometry exactly.
- PLY coordinate precision and metadata, overwrite protection, and occupancy
  round-trip while preserving unknown units.

These synthetic checks establish implementation behavior for declared virtual
cameras. They provide no measured real-object reconstruction accuracy. See the
benchmark's synthetic validation report for the fixed eight-camera input/holdout
experiment and its measured resolution sensitivity.
