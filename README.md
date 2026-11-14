# Brume

A LiDAR point cloud viewer written in Rust, built to render tens of millions
of points interactively.

> **Status:** phase 0 of 4. Brume loads LAS/LAZ files into a GPU-ready
> layout and reports what they contain. The 3D viewer (egui + wgpu) is next.

## Data

Brume is developed with Spain's national LiDAR survey, PNOA-LiDAR (third
coverage, 2022–2025): 1 × 1 km tiles in LAZ, with RGB, near infrared and
classification.

1. Register at the [CNIG download centre](https://centrodedescargas.cnig.es/CentroDescargas/lidar-tercera-cobertura).
2. Search by municipality and download a tile (prefer NPC02 or NPC03).
3. Save it in `data/` (ignored by git).

Data: © Instituto Geográfico Nacional, PNOA-LiDAR, CC BY 4.0.

## Usage

```sh
# Load and print a report
cargo run --release -- data/tile.laz

# Several neighbouring tiles as one scene
cargo run --release -- data/tile-a.laz data/tile-b.laz
```

## Design

- **Local coordinates.** UTM northings near Madrid are ~4.46 million metres,
  where consecutive `f32` values are 0.5 m apart. Brume subtracts a per-scene
  origin in `f64` and stores `f32` offsets, which keeps ~0.03 mm precision
  within a 1 km tile.
- **Column storage (SoA).** Positions, colours, classes and intensity live in
  separate vectors, so each can be uploaded to its own GPU buffer as is.
- **Batched reading.** Points are decoded in batches of ~1 M into a reused
  buffer, keeping memory bounded and producing chunks for streaming.
- **Colour depth detection.** The LAS spec asks for 16-bit colour, but many
  files store 0–255. Brume checks the data before converting to 8 bits.
- **Classification mask.** Legacy point formats pack flags into the class
  byte; Brume masks them out, and keeps the full byte for LAS 1.4 formats.

### Multiple tiles

Several neighbouring tiles can be loaded into one scene: select or drop several
files at once, or add to the scene on screen with *Add tiles…* or by holding
Shift while dropping.

- **Shared origin.** All tiles of a scene use one `f64` origin, the centre of
  the union of their header bounds rounded to whole metres, and store `f32`
  offsets from it. Tiles added later reuse it, so loaded tiles never move.
- **Precision.** The spacing between consecutive `f32` values grows with the
  distance from the origin: about 1 mm at 10 km and 8 mm at 100 km.
- **50 km check.** A scene may span at most 50 km in X and in Y. Neighbouring
  tiles fit easily, offsets stay within ~4 mm precision, and a wider set almost
  certainly mixes coordinate systems: it is rejected from the headers, before
  any point is read.
- **One coordinate system.** Tiles from different UTM zones would need
  reprojecting to a common zone, which Brume does not support.

## Performance

Tile `PNOA-2024-MAD-419-4463`: 7,533,341 points (7.5 pts/m²).

| Mode | Load time | Speedup |
| --- | --- | --- |
| Sequential LAZ decompression | 5.5 s | 1.0× |
| Parallel LAZ decompression | 2.0 s | 2.75× |

On 4 cores the speedup is bounded by the serial part of loading
(~15 % by Amdahl's law): mostly single-threaded conversion. Phase 2 overlaps
decoding and conversion on separate threads.

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

Every point is drawn every frame, so frame time grows with the point count:
about 8.5 ms per million points. Four tiles (34.2 M points, ~650 MiB of vertex
buffers) don't fit in this 2 GB card next to the desktop; at that rate they
would take ~290 ms per frame. These are the "before" numbers for phase 3.

## Roadmap

- [x] Phase 0: load LAS/LAZ, local coordinates, column storage, statistics
- [ ] Phase 1: egui window, wgpu point rendering, orbit camera, colour modes
- [ ] Phase 2: background loading, multiple tiles
- [ ] Phase 3: octree with level of detail and a point budget
