# HerculeX on an Apple M5 Pro: measured speedups, each gated against stock

[HerculeX](https://github.com/adambetka2/HerculeX) by Adam Betka (stokarz) is a virtual-unwrapping pipeline for the Herculaneum scrolls, built for Apple silicon and released under the MIT licence. It detects papyrus sheets in the CT, solves their winding, flattens them into surfaces and reads ink on them with its Reader v3. Everything here is built on HerculeX at commit `9f59e96`, its current head. This repository, [HerculeX](https://github.com/comoke/HerculeX), is a fork of HerculeX at that commit, with the changes on branch `m5`: four commits plus some settings. HerculeX's copyright and permission notice is kept, and the changes are under the same MIT licence. This fork is not part of HerculeX and is not endorsed by its author.

I profiled a full scroll run on a MacBook Pro with an M5 Pro (16-core GPU, 24 GB, macOS 27.0.1), changed the parts that held it back on this machine, and timed each change against stock 9f59e96 on the same Mac with the same inputs. A change counts only if its outputs pass a saved check against stock's: byte-identical files, or, where the arithmetic changes, r ≥ 0.9999 on the ink maps.

## Results (one M5 Pro, against stock 9f59e96 on the same Mac)

| stage | stock | changed | speedup | check against stock |
|---|---|---|---|---|
| Ink read end to end, a large unit: 6,236 pieces, PHerc0175B rows 0044–0056 | 1,842 s | 1,488 s | **1.24×** | the changed reader's ink maps against stock's on the same captured inputs, a capture of the unit's largest pieces: pooled r 0.99998, max abs diff 0.010 over 306 Mpx |
| Ink read end to end, a small unit: 404 pieces, rows 0055–0063 | 70.0 s | 42.4 s | **1.65×** | the same check on all 404 pieces: pooled r 0.99997, max abs diff 0.013 over 296 Mpx |
| Pieces stage, the "draws" step, rows 0000–0012 | 215 s | 124 s | **1.73×** | all 30,188 piece files byte-identical (`record.json` aside) |

On the production PHerc0175B run, the ink reads took 10,191 s of the 16,447 s sheet stage. At the measured 1.24× they would take about 8,230 s.

## What changed

**The ink reader.** These changes touch `ink/model/reader_apple/` and `ink/reading/piece_reading.py`, and add `tensor_stem.metal`.
- **A GPU lane beside the Neural Engine.** Stock sends every 512 px tile's network body to the Neural Engine. On the M5 the GPU also runs that Core ML body well, so a share of tiles now runs on the GPU.
  - Each tile's lane is fixed by its position and data. A rerun gives the same maps whatever order the pieces arrive in.
  - The setting is `HERCULEX_BODY_LANES`. The default is `EEGEG`, and `E` sends every tile to the Neural Engine as stock does.
- **The depth stem's second convolution on Metal 4 tensor ops**, in place of the `simdgroup_matrix` loop. `HERCULEX_STEM=kernel` selects the stock kernel.
- **The per-stack normalisation statistics on the same tensor kernels.** In stock they run as many small PyTorch ops.
- **More reading in flight.** `HERCULEX_SAMPLERS=8` is an existing setting, and `HERCULEX_PIECES_AHEAD=12` is new.

**The pieces stage** (`unwrapping/level/level_pieces.py`).
- The level fit's eight reweighted solves ("draws") are capped by a 3 GB memory budget, so mid-size and large units run them on fewer threads.
- The budget is now `HERCULEX_DRAW_BYTES`, with the default unchanged. At 10e9 the timed unit ran all eight at once, with an 8.5 GB peak.
- Each draw is seeded by its own index, so the thread count can't change the result. The files came out byte-identical.

## About the exactness

The GPU lane and the tensor-op stem change float arithmetic, so the ink maps aren't bit-identical to stock's. On the same captured inputs they match stock at pooled r 0.99997–0.99998, with a max abs diff of 0.010–0.013 in ink probability. The read's digest changes too, so units read with stock and with these changes shouldn't be mixed in one scroll.

## Why this is about the M5

The GPU lane pays off because the M5's GPU runs the reader's body quickly beside its Neural Engine. On a chip where the Neural Engine is much the faster unit, the split may not help. The tensor-op stem needs Metal 4. I tested only on this M5 Pro.

## Apply it

```bash
git clone https://github.com/comoke/HerculeX && cd HerculeX && git checkout m5
export HERCULEX_SAMPLERS=8 HERCULEX_PIECES_AHEAD=12 HERCULEX_DRAW_BYTES=10e9
# then run HerculeX as usual; HERCULEX_BODY_LANES and HERCULEX_STEM keep their new defaults (EEGEG, tensor)
```

The benchmark and check scripts are in `bench/`, and `bench/env.sh` sets the paths. Each needs a stock checkout beside the changed one.
- `profile_reader.py` times the reader and compares its maps with stock's on captured stacks.
- `run_read.sh` times a unit's read end to end.

## On an NVIDIA RTX 4070

<!-- SLOT: numbers from the 4070 optimization (branch pc/4070-optimization), only if they land with their gates before the claim-check. Otherwise delete this section and the matching line under "What isn't claimed". Same rules: stock vs changed on the same machine, each row with its gate and a source file in CLAIMS.md. -->

| stage | stock | changed | speedup | check against stock |
|---|---|---|---|---|
| {{4070 rows}} | | | | |

## What isn't claimed

- **No CUDA numbers.** The table above is Apple silicon only. {{If the 4070 section stays: CUDA numbers are only those in that section.}}
- **Other chips.** I tested on one Mac, and other Apple chips are untested.
- **The publish step.** HerculeX's own setting `HERCULEX_COMPOSE_PROCESSES=12` timed 956 s to 483 s on one unit. Its tile comparison wasn't saved to a file, so it isn't claimed here.
- **Brick read-ahead** (`HERCULEX_READ_AHEAD`, `sampling/column_sampling.py`) gave no measurable gain on a full unit: 1,488 s to 1,455 s, inside the run-to-run spread. It isn't counted.
- **No whole-scroll speedup.** I didn't time a whole scroll end to end with these changes, so no whole-scroll figure is claimed.

## Credits and licence

HerculeX is Copyright (c) 2026 Adam Betka and is MIT-licensed, and all credit for the pipeline itself is his. The changes in this fork are derivative work under the same licence, with his notice kept in `LICENSE`. The scroll data is the Vesuvius Challenge's PHerc0175B scan, used under its data terms. Every number above has a source file listed in `CLAIMS.md`.
