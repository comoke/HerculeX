# HerculeX on an Apple M5 Pro

This fork of [HerculeX](https://github.com/adambetka2/HerculeX) by Adam Betka (stokarz) changes how HerculeX runs on an Apple M5 Pro. HerculeX is MIT-licensed, and its ink reader is Reader v3 by Domenico Russo. Branch `m5` adds four commits to HerculeX at 9f59e96. HerculeX's licence and copyright notice are kept in `LICENSE`, and the changes are under the same MIT licence. This fork is not part of HerculeX and is not endorsed by its author.

## What changed

**The ink reader** (`ink/model/reader_apple/`, `ink/reading/piece_reading.py`, and a new `tensor_stem.metal`):
- A share of 512 px tiles' network bodies runs on the GPU beside the Neural Engine. Each tile's lane is fixed by its position and data (`lane_of`). The setting is `HERCULEX_BODY_LANES`, default `EEGEG`; `E` sends every tile to the Neural Engine, as stock does.
- The depth stem's second convolution and the per-stack normalisation statistics run on Metal 4 tensor ops. `HERCULEX_STEM=kernel` selects stock's kernel.
- `HERCULEX_PIECES_AHEAD` (new) sets how many pieces are read ahead; `HERCULEX_SAMPLERS` is stock's own setting.

**The pieces stage** (`unwrapping/level/level_pieces.py`): the memory budget that caps how many of the level fit's eight reweighted solves ("draws") run at once is now a setting, `HERCULEX_DRAW_BYTES`, default 3e9 as in stock. Each draw is seeded by its own index, so the thread count can't change the result.

**Sampling** (`sampling/column_sampling.py`): before a piece is sampled, the brick planes its stack reads are announced to the system (`madvise`). `HERCULEX_READ_AHEAD`, on by default on this branch; `0` turns it off.

## Measured on one M5 Pro, against stock 9f59e96 on the same Mac

MacBook Pro, Apple M5 Pro (16-core GPU), 24 GB, macOS 27.0.1. Every number below is in [`CLAIMS.md`](CLAIMS.md) with its saved file in [`bench/evidence/`](bench/evidence/).

- **Reading, like for like.** With stock's own 4 samplers, the initial reader commit (5b1576b) read a 404-piece unit in 59.6 s against stock's 70.0 s, 1.18×. Runs with 8 samplers aren't compared here, because stock wasn't timed with 8.
- **The reader's maps are not bit-identical to stock's.** On the same captured inputs, the current reader matches stock's float ink maps at pooled r 0.99997–0.99998, worst single map r 0.99897, max abs diff 0.010–0.013 in ink probability (a 404-piece unit; the 48 largest pieces of a 6,236-piece unit).
- **Pieces.** With `HERCULEX_DRAW_BYTES=10e9`, a unit's 30,188 piece files were byte-identical to stock's (`record.json` aside). That unit's stock time wasn't saved, so no speedup is claimed.
- **Read-ahead.** 48 of 48 sampled captures were byte-identical to stock's. No speed gain is claimed for it.

## Settings

```bash
git checkout m5
export HERCULEX_SAMPLERS=8 HERCULEX_PIECES_AHEAD=12 HERCULEX_DRAW_BYTES=10e9   # as on the M5 Pro
# on another Apple chip, or to read as stock does:
export HERCULEX_BODY_LANES=E HERCULEX_STEM=kernel HERCULEX_READ_AHEAD=0
```

Keep the settings fixed for a whole scroll, so that all its units are read the same way. The read's digest changes with this branch's code, so units read with stock and with this branch shouldn't be mixed in one scroll.

## What isn't claimed

- I tested on one M5 Pro only. The tensor-op stem needs Metal 4; on a chip where the Neural Engine is much the faster unit, the GPU lane may not help.
- No CUDA numbers, no whole-scroll speedup, no publish speedup and no pieces speedup.

## Benchmarks

`bench/` holds the scripts behind these numbers: `profile_reader.py` times the reader and compares its maps with stock's on captured stacks, and `run_read.sh` times a unit's read end to end. `bench/env.sh` sets the paths; each needs a stock checkout beside this one.

## Credits and licence

HerculeX is Copyright (c) 2026 Adam Betka and MIT-licensed; Reader v3 is by Domenico Russo (MIT). The changes in this fork are derivative work under the same MIT licence, with the notice kept in `LICENSE`. The scan data belongs to the Vesuvius Challenge and is used under its terms.
