# Claims in M5-OPTIMIZATION.md and their files

Files are in `bench/evidence/`, copied from the benchmark runs unchanged except that home-directory paths are shortened to `~/`. `round2_window.excerpt.log` is five lines, verbatim, of a longer run log.

| claim | file | value |
|---|---|---|
| Stock 9f59e96 read the 404-piece unit in 70.0 s with its default 4 samplers | `stock-u55-run1.events.jsonl` (`kind: read`, end − start, `pieces`); `stock-u55-run1.meta.txt` (9f59e96, `env=` empty) | 70.04 s, 404 pieces |
| Commit 5b1576b read it in 59.6 s with 4 samplers, 1.18× | `m5-u55-s4.events.jsonl`; `m5-u55-s4.meta.txt` (5b1576b, `env=` empty) | 59.60 s |
| The current reader against stock's float maps on captured inputs: pooled r, worst map r, max abs diff | `round2_window.excerpt.log`, lines `reader u55`, `reader u44big` | 0.9999736 / 0.9999767; 0.99897 / 0.99978; 0.0131 / 0.0103 |
| Pieces with `HERCULEX_DRAW_BYTES=10e9`: byte-identical to stock's | `round2_window.excerpt.log`, line `pieces files` | 30188 files, 0 differing |
| Read-ahead: sampled captures byte-identical to stock's | `round2_window.excerpt.log`, line `sampler 48 pieces` | 48/48 |
| Defaults and settings | `ink/model/reader_apple/tile_reader.py` (`HERCULEX_BODY_LANES`, `HERCULEX_STEM`), `ink/reading/piece_reading.py` (`HERCULEX_SAMPLERS`, `HERCULEX_PIECES_AHEAD`), `unwrapping/level/level_pieces.py` (`HERCULEX_DRAW_BYTES`), `sampling/column_sampling.py` (`HERCULEX_READ_AHEAD`) | as stated |

The `pieces rows-0000-0012` line also quotes a production stock time typed into the run script. No saved log backs it, so no pieces speedup is claimed. `dirty=` in `meta.txt` counts `git status --short` lines; the checkouts held only untracked benchmark outputs.
