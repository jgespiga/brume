# Brume

A LiDAR point cloud viewer written in Rust, built to render tens of millions
of points interactively. Brume opens LAS/LAZ tiles, reports their attributes,
and renders a shared scene with an orbit camera and per-frame level of detail.

The implementation demonstrates:

- [Background loading with cooperative cancellation](src/app.rs).
- [A shared f64 origin with f32 point offsets](src/cloud.rs).
- [An octree with grid sampling and breadth-first point storage](src/octree.rs).
- [Per-frame level of detail with frustum culling and a point budget](src/lod.rs).
- [Instanced rendering on wgpu with packed attributes](src/render.rs) and
  [screen-space point squares in WGSL](src/points.wgsl).

<!-- Screenshot: add docs/screenshot.png here -->

## Run

```sh
# Open an empty viewer
cargo run --release

# Open one tile and print its report
cargo run --release -- data/tile.laz

# Open several neighbouring tiles as one scene
cargo run --release -- data/tile-a.laz data/tile-b.laz

# Disable vsync for frame-time measurements
cargo run --release -- --no-vsync data/tile.laz
```

Drag in the view to orbit and scroll over it to zoom. Drop files or use
*Open…* to replace the scene. Use *Add tiles…* or hold Shift while dropping
to append tiles. The current scene stays visible during loading, and *Cancel*
requests a stop at the next batch or octree node boundary.

The side panel contains tile visibility, classification counts, colour modes
and point size. Its performance controls set the point budget, minimum cell
size and maximum octree depth. *Freeze LOD* keeps the selection while the
camera moves; *Show octree nodes* draws the selected cubes and frozen frustum.

## Data

Brume is developed with Spain's national LiDAR survey, PNOA-LiDAR (third
coverage, 2022–2025): 1 × 1 km tiles in LAZ, with RGB, near infrared and
classification.

1. Register at the [CNIG download centre](https://centrodedescargas.cnig.es/CentroDescargas/lidar-tercera-cobertura).
2. Search by municipality and download a tile (prefer NPC02 or NPC03).
3. Save it in `data/` (ignored by git).

Data: © Instituto Geográfico Nacional, PNOA-LiDAR, CC BY 4.0.

## Internal structure

Start with the crate overview in [src/main.rs](src/main.rs), then follow
[Brume in src/app.rs](src/app.rs) through loading and rendering. The source
docs link between types and functions and explain the layouts below.

```text
LAS/LAZ headers -> load plan -> background point conversion
                                        |
                               octree + reordered columns
                                        |
                                  shared scene
                                        |
camera + controls -> LOD selection -> merged instance ranges
                                        |
                              GPU buffers + WGSL -> pixels
```

### [Scene coordinates and tile storage](src/cloud.rs)

UTM northings near Madrid are ~4.46 million metres, where consecutive `f32`
values are 0.5 m apart. Brume subtracts a per-scene origin in `f64` and stores
`f32` offsets, which keeps ~0.03 mm precision within a 1 km tile.

```text
global position, f64 -> subtract shared origin, f64 -> offset, f64
                                                         |
                                                    cast to f32
                                                         |
                                                local GPU position
```

All tiles use the centre of the union of their header bounds, rounded to
whole metres, as their shared origin. Added tiles reuse that origin, so
loaded tiles never move. The spacing between consecutive `f32` values grows
with distance from it, about 1 mm at 10 km and 8 mm at 100 km.

Positions, colours, classes and intensity live in separate vectors, a
structure of arrays. Each index identifies the same point across columns.
The renderer uploads positions and colours directly and packs the other
attributes with octree depth.

```text
              point i     point i+1
positions  [ xyz        | xyz        | ... ]
colours    [ rgba       | rgba       | ... ]
classes    [ code       | code       | ... ]
intensity  [ value      | value      | ... ]
```

A tile owns these columns, its octree and its statistics. Scenes share
immutable tiles through `Arc` when adding files.

### [Planning and loading a scene](src/scene.rs)

The UI reads headers before starting the loader. A scene may span at most
50 km in X and in Y. This keeps nearby tiles within ~4 mm horizontal
precision and rejects wider sets before reading points. A wide extent can
indicate mixed coordinate systems, but this geometric check does not verify
the files' coordinate-system metadata. Tiles must already use one coordinate
system; Brume does not reproject tiles from different UTM zones.

The plan also checks each tile against the GPU's buffer-size limit before
point decoding.

```text
headers -> union bounds + point counts -> extent and buffer checks
                                                   |
                                                LoadPlan
                                                   |
                               existing tiles + newly loaded tiles
                                                   |
                                              LoadedScene
```

[The loader](src/loader.rs) decodes batches of ~1 M points into a reused
buffer, converts each batch into columns, then joins and reorders them for
the octree. Batching bounds the decoded buffer; converted points still
accumulate in memory, and reordering needs another copy of the columns.

The LAS spec asks for 16-bit colour, but many files store 0–255. Brume
checks the first batch before converting to 8 bits. A channel above 255
selects the 16-bit conversion for the whole file. Legacy point formats
pack flags into the class byte; Brume masks them out and keeps the full
byte for LAS 1.4 extended formats.

### [Background loading and the UI](src/app.rs)

Opening files builds a plan on the UI thread, then starts a `brume-loader`
thread. That thread loads tiles one after another and sends the completed
scene through an mpsc channel. The UI polls with `try_recv` so it can keep
handling input.

```text
UI thread                           loader thread
LoadPlan --------------------------> read and convert
                                            |
progress <------- shared atomics ---- build octrees and reports
                                            |
poll_pending <---- result channel --- LoadedScene

drop PendingLoad -- cancel flag ---> next batch or node check
```

Dropping the pending load requests cancellation and drops its receiver.
The loader stops cooperatively; the UI does not wait for its thread.
New scenes replace the GPU resources. Appending tiles checks shared tile
identity with `Arc::ptr_eq`, then uploads only the new tiles when the
existing prefix still matches.

### [Octree sampling](src/octree.rs)

Each node keeps the first point in each sampling-grid cell and sends the
remaining points to child octants. This follows Potree's modifiable nested
octree sampling, described by Scheiblauer and Wimmer in 2011. Small nodes
and nodes at the depth limit keep all remaining points.

Each point belongs to one node. Drawing children adds detail to the parent
sample. Nodes and their contiguous point ranges follow breadth-first order,
so coarse samples come first.

```text
       root              Arena: [root, A, B, C, D]
       /  \
      A    B
     / \
    C   D

Point columns
| root |  A   |  B   |  C   |  D   |
<------>                            depth 0
<-------------------->              through depth 1
<----------------------------------> all
```

### [Level of detail](src/lod.rs)

Every frame, Brume chooses nodes across all visible tiles. Frustum culling
extracts six planes from the camera's view-projection matrix. If a node's
box lies entirely outside a plane, selection skips it and its children.
The box test is conservative, so it can retain some boxes outside the view.

One priority queue ranks nodes by projected cube size in pixels. Larger
cubes leave the queue first, so they receive detail before smaller ones.
Selection stops when the next node would exceed the point budget,
1.5 M points by default. A first root that exceeds the budget leaves the
selection empty.

A node's children become candidates only while its sampling cells,
node size / 128, look larger than the minimum cell size in pixels, 1 by
default. This stops refinement when the sample spacing is small on screen,
even if the budget has room left.

```text
visible roots -> shared heap -> largest cube -> budget check
                    ^                              |
                    |                         select node
                    |                              |
                    +--- eligible children <--- cell-size check

selected node indices -> sort per tile -> merge touching ranges -> draw
```

Selected nodes are drawn with one draw call per run of nodes that are
contiguous in the tile's buffers. For the whole-block view with the default
settings, one tile selects 51 nodes in 2 draw calls and three tiles 46 nodes
in 3, and the selection takes about 9 µs of CPU per frame.

| View | Budget | Points drawn | Nodes | Avg frame (ms) |
| --- | --- | --- | --- | --- |
| | | | | |

### [Camera and projection](src/camera.rs)

The orbit camera uses Z as up, with yaw, pitch and distance around a target.
Fitting places the target at the scene's centre and fits its bounding sphere
to the vertical field of view. The projection is left-handed with a
WebGPU/DirectX depth range.

```text
local position -> view -> projection -> clip coordinates
                                              |
                                         divide by w
                                              |
                                   normalized coordinates
                                              |
                                     viewport position
```

The GPU projects point centres. The CPU uses the same matrix to place
diagnostic wireframes through the egui painter.

### [Point rendering](src/render.rs)

Each tile has one GPU buffer per attribute group. Positions take 12 bytes
per point, colours take 4, and packed class, octree depth and intensity
take 4. Each buffer must fit the device's `max_buffer_size`. With wgpu's
default of 256 MiB that is 22.3 M points per tile; the header check rejects
larger tiles before reading points. This is a per-buffer check, not a
guarantee that all tiles fit in GPU memory together.

```text
Packed u32
31                    16 15             8 7              0
+-----------------------+----------------+----------------+
| intensity, 16 bits    | depth, 8 bits  | class, 8 bits  |
+-----------------------+----------------+----------------+
```

The callback binds each selected tile once and issues one instanced draw
per merged point range. [The shader](src/points.wgsl) expands each point
into a screen-space square using a four-vertex triangle strip. Its colour
modes use file RGB, height on viridis, the classification palette, intensity
as grey, or the cyclic octree-depth palette.

```text
point instance -> projected centre -> screen-space square -> depth test
                                             |
                                        point colour
```

### [Reports and frame timing](src/report.rs)

Class rows use counts gathered during loading. Height and intensity colour
ranges use scene-wide percentiles, selected without sorting all values.
The shader clamps outliers to the range endpoints. The same class and depth
palettes feed rendering and UI legends.

```text
tile statistics -> merged counts -> class rows
tile columns    -> percentiles  -> shader colour ranges
UI intervals    -> ring buffer  -> average and worst frame
```

[FrameTimes](src/perf.rs) keeps recent CPU frame intervals in a bounded
ring buffer. Measurements run during continuous repaint and skip the first
interval after enabling it, which would include idle time.

## Performance

Tile `PNOA-2024-MAD-419-4463`: 7,533,341 points (7.5 pts/m²).

| Mode | Load time | Speedup |
| --- | --- | --- |
| Sequential LAZ decompression | 5.5 s | 1.0× |
| Parallel LAZ decompression | 2.0 s | 2.75× |

On 4 cores the speedup is bounded by the serial part of loading (~15 % by
Amdahl's law): mostly single-threaded conversion. Phase 2 introduced background
loading. The current loader decodes and converts batches in sequence on that
background thread.

Hardware: <CPU model>, <RAM>, <OS>.

### Rendering

| Points | Load time (s) | Avg frame (ms) | Worst frame (ms) | Resolution | GPU | Vsync |
| --- | --- | --- | --- | --- | --- | --- |
| 7,533,341 | 2.15 | 67 | 195 | 2258×1355 | GTX 750 Ti (Vulkan, NVIDIA 550.163.01) | off |
| 7,533,341 (1 tile) | 2.07 | 65 | 131 | 2133×1355 | GTX 750 Ti (Vulkan, NVIDIA 550.163.01) | off |
| 17,932,793 (2 tiles) | 5.52 | 152 | 451 | 2125×1355 | GTX 750 Ti (Vulkan, NVIDIA 550.163.01) | off |
| 26,945,741 (3 tiles) | 8.45 | 227 | 473 | 2125×1355 | GTX 750 Ti (Vulkan, NVIDIA 550.163.01) | off |

Frame times are measured with `--no-vsync` and *Continuous repaint* on, and
are CPU-side frame intervals (the time between consecutive `ui` calls), not
GPU execution time. The tile rows have the whole block in view.

In this baseline, every point was drawn every frame, so frame time grew with the
point count: about 8.5 ms per million points. Four tiles (34.2 M points, ~650
MiB of vertex buffers) don't fit in this 2 GB card next to the desktop; at that
rate they would take ~290 ms per frame. These are the "before" numbers for phase
3, recorded before level of detail.

## Checks

```sh
cargo fmt --check
cargo build --release
cargo test
cargo clippy --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
```

## Roadmap

- [x] Phase 0: load LAS/LAZ, local coordinates, column storage, statistics
- [x] Phase 1: egui window, wgpu point rendering, orbit camera, colour modes
- [x] Phase 2: background loading, multiple tiles
- [ ] Phase 3: octree with level of detail and a point budget
