# HerculeX on an Apple M5 Pro: measured speedups, each gated against stock

[HerculeX](https://github.com/adambetka2/HerculeX) by Adam Betka (stokarz) is a virtual-unwrapping pipeline for the Herculaneum scrolls, built for Apple silicon and released under the MIT licence. It detects papyrus sheets in the CT, solves their winding, flattens them into surfaces and reads ink on them with its Reader v3. Everything here is built on HerculeX at commit `9f59e96`, its current head. This repository, [HerculeX](https://github.com/comoke/HerculeX), is a fork of HerculeX at that commit with the changes on branch `m5`: four commits plus some settings. HerculeX's copyright and permission notice is kept, and the changes are under the same MIT licence. This fork is not part of HerculeX and is not endorsed by its author.

I profiled a full scroll run on a MacBook Pro with an M5 Pro (16-core GPU, 24 GB, macOS 27.0.1), changed the parts that held it back on this machine, and timed each change against stock 9f59e96 on the same Mac with the same inputs. A change counts only if its outputs pass a gate against stock's: byte-identical files, or, where the arithmetic changes, r ≥ 0.9999 on the ink maps.

## Results (one M5 Pro, against stock 9f59e96 on the same Mac)

| stage | stock | changed | speedup | gate against stock |
|---|---|---|---|---|
| Ink reader alone, same captured inputs: a 404-piece unit / the 48 largest pieces of a 6,236-piece unit | 58.7 s / 38.7 s | 36.0 s / 24.5 s | **1.63× / 1.58×** | pooled r 0.99997 / 0.99998 over 296 / 306 Mpx; max abs 0.013 / 0.010 |
| Ink read end to end, a large unit (6,236 pieces, 2,303 cm², PHerc0175B rows 0044–0056) | 1,842 s | 1,488 s | **1.24×** | sheet and layer-0 PNGs byte-identical (12,472 of 12,472); written face identical on 6,236 of 6,236 pieces; ink PNGs r 0.99995, max 8 of 255 grey levels |
| Ink read end to end, a small unit (404 pieces, rows 0055–0063) | 70.0 s | 42.4 s | **1.65×** | as the reader row |
| Pieces stage, the "draws" step (rows 0000–0012) | 215 s | 124 s | **1.73×** | all 30,188 piece files byte-identical (`record.json` aside) |
| Publish, one unit (104,009 tiles) | 956 s | 483 s | **1.98×** | every tile byte-identical |

**Whole scroll, projected:** these stage factors applied to the stage times of a real PHerc0175B run take it from 24,127 s to 19,037 s, **about 1.27×**. That is a projection from measured stages, not a timed end-to-end run. The largest item left on that run is waiting for the CT fetch (3,480 s), which these changes don't touch.

## What changed

**The ink reader** (it touches `ink/model/reader_apple/` and `ink/reading/piece_reading.py` and adds `tensor_stem.metal`):
- **A GPU lane beside the Neural Engine.** Stock sends every 512 px tile's network body to the Neural Engine, which runs one tile at a time at about 19.5 ms on this Mac. The M5's GPU runs the same Core ML body in 15.2 ms, so now a share of tiles runs on the GPU.
  - Each tile's lane is fixed by its position and data, so a rerun gives the same maps whatever order the pieces arrive in.
  - The setting is `HERCULEX_BODY_LANES`; the default is `EEGEG`, and `E` turns the lane off.
- **The depth stem's second convolution on Metal 4 tensor ops.** This replaces the stock kernel's `simdgroup_matrix` loop and keeps the same rounding points. It takes 5.2 ms per tile against 12.7 ms. 99.88 % of feature values are bit-identical to stock's, and the rest differ by one half-precision step.
- **The per-stack normalisation statistics moved onto the same tensor kernels**, about 30 small PyTorch ops in stock: 39 ms to 3.8 ms per stack.
- **More reading in flight:** `HERCULEX_SAMPLERS=8`, an existing setting, and a new `HERCULEX_PIECES_AHEAD=12`.
- `HERCULEX_BODY_LANES=E HERCULEX_STEM=kernel` gives maps bit-identical to stock.

**The pieces stage** (`unwrapping/level/level_pieces.py`):
- The level fit's eight reweighted solves ("draws") were capped by a fixed 3 GB memory budget, so large units ran them on 2 threads and mid-size ones on 4.
- The budget is now `HERCULEX_DRAW_BYTES`, with the default unchanged; 10e9 runs all eight at once. Peak memory was 8.5 GB on the unit timed; on the largest PHerc0175B unit I estimate about 16 GB.
- Each draw is seeded by its own index, so the thread count can't change the result, and the files are byte-identical.

**Publish:** `HERCULEX_COMPOSE_PROCESSES=12`, HerculeX's own setting, in place of its default of 4. No code change.

## About the exactness

Two of the reader changes alter float arithmetic, so the ink maps are not bit-identical to stock's:
- the GPU lane: a GPU body instead of the Neural Engine's float16;
- the tensor-op stem.

They move toward the float32 network, not away from it. On four real tiles:
- stock's Neural Engine output matches the float32 network at r 0.99986;
- the GPU body matches it at r 0.999998.

Over a whole large unit the change is ink PNGs at r 0.99995 against stock, with a max of 8 grey levels. The geometry outputs (sheet and layer-0 images, the written face) stay byte-identical.

The read's digest changes too, so a unit read with stock and one read with these changes shouldn't be mixed in one scroll.

## Why this is about the M5

The GPU lane pays off because the M5's GPU runs the reader's body faster than its Neural Engine. On a chip where the Neural Engine is the faster unit, the split may not help. The tensor-op stem needs Metal 4. I tested only on this M5 Pro. On M1 to M4 Macs the best lane split may be different, and I haven't tested the tensor stem on them. The off switches above give stock behaviour.

## Apply it

```bash
git clone https://github.com/comoke/HerculeX && cd HerculeX && git checkout m5
export HERCULEX_SAMPLERS=8 HERCULEX_PIECES_AHEAD=12 HERCULEX_DRAW_BYTES=10e9 HERCULEX_COMPOSE_PROCESSES=12
# then run HerculeX as usual; HERCULEX_BODY_LANES and HERCULEX_STEM keep their new defaults (EEGEG, tensor)
```

The benchmark and gate scripts are in `bench/`: `profile_reader.py` times the reader and compares its maps with stock's on captured stacks, `run_read.sh` times a unit's read end to end, `compare_faces.py` compares a unit's outputs file by file, and `publish_ab.sh` times the publish step. `bench/env.sh` sets the paths; each needs a stock checkout beside the changed one.

## On an NVIDIA RTX 4070

<!-- SLOT: numbers from the 4070 optimization (branch pc/4070-optimization), only if they land with their gates before the claim-check. Otherwise delete this section and the matching line under "What isn't claimed". Same rules: stock vs changed on the same machine, each row with its gate and a source file in CLAIMS.md. -->

| stage | stock | changed | speedup | gate against stock |
|---|---|---|---|---|
| {{4070 rows}} | | | | |

## What isn't claimed

- The table above is Apple silicon only; it has no CUDA numbers. {{If the 4070 section stays: CUDA numbers are only those in that section.}}
- I tested on one Mac. Other Apple chips are untested.
- Brick read-ahead (`HERCULEX_READ_AHEAD`, `sampling/column_sampling.py`) was byte-identical but gave no measurable gain on a full unit: 1,488 s to 1,455 s, inside the run-to-run spread. It isn't counted.
- A further 1.55× for the whole scroll is possible when the disk has room to prefetch the next unit's CT. That is a matter of disk space, not code, so it isn't in the table.

## Credits and licence

HerculeX is Copyright (c) 2026 Adam Betka and is MIT-licensed; all credit for the pipeline itself is his. The changes in this fork are derivative work under the same licence, with his notice kept in `LICENSE`. The scroll data is the Vesuvius Challenge's PHerc0175B scan, used under its data terms. Every number above has a source file listed in `CLAIMS.md`.
