"""Per-op timing of the ink stage's sampler side, one process, stock code paths. Run from a checkout root:
runtime/python/bin/python BENCH/profile_sampler.py UNIT OUT_CACHE [LIMIT]
Also captures every piece's reader inputs (what `opened` returns) so the reader can be profiled alone on identical data."""

import collections
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
unit, cache = sys.argv[1], Path(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.environ["HERCULEX_UNITS"] = unit
os.environ.setdefault("HERCULEX_KEEP_STACKS", "0")

from ink.model import tiles  # noqa: E402
from ink.reading import piece_reading as pr  # noqa: E402
from sampling import column_sampling as sampler  # noqa: E402

T = collections.defaultdict(float)
N = collections.Counter()


def timed(name, fn):
    def inner(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[name] += time.perf_counter() - t
            N[name] += 1

    return inner


sampler.load_surface = timed("load_surface (3 tifs + meta)", sampler.load_surface)
sampler.readable_mesh = timed("readable_mesh (bridge+smooth, scipy)", sampler.readable_mesh)
sampler.grid_normals = timed("grid_normals (EDT+gaussian, scipy)", sampler.grid_normals)
sampler.flatten = timed("flatten (C trilinear from brick mmap)", sampler.flatten)
pr.surface_of = timed("surface_of total", pr.surface_of)
pr.across_sheet = timed("across_sheet (max over layers)", pr.across_sheet)
pr.recto_face = timed("recto_face (fibre test)", pr.recto_face)
pr.scrambled = timed("scrambled control (FFT)", pr.scrambled)
pr.window = timed("window (face layer pick + copy)", pr.window)
tiles.measured = timed("measured (8 norm tiles + percentiles)", tiles.measured)
tiles.prepare_stack = timed("prepare_stack (fill holes)", tiles.prepare_stack)
pr.write_faces = timed("write_faces (2 PNGs)", pr.write_faces)
pr.parcelled = timed("parcelled (copy to shm)", pr.parcelled)
pr.faces_of = timed("faces_of total", pr.faces_of)
pr.sampled = timed("sampled total", pr.sampled)

queue = pr.queue_of(os.environ.get("SCROLL", "PHerc0175B"), os.environ.get("SCAN", "20250521125822"))
((store, jobs),) = queue
jobs = jobs[:limit] if limit else jobs
pr.start_sampler(str(store), None)
cache.mkdir(parents=True, exist_ok=True)
rows, start = [], time.perf_counter()
for k, job in enumerate(jobs):
    t = time.perf_counter()
    piece = pr.sampled_piece(job)
    dt = time.perf_counter() - t
    if "row" in piece:
        rows.append({"piece": job[0].name, "seconds": dt, "skipped": True, **piece["row"]})
        continue
    stacks, sheet = pr.opened(piece)
    rows.append(
        {
            "piece": job[0].name,
            "seconds": dt,
            "px": int(sheet.size),
            "covered": int((sheet > 0).sum()),
            "stacks": len(stacks),
            "shape": list(sheet.shape),
        }
    )
    with open(cache / f"{k:05d}-{job[0].name}.pkl", "wb") as out:
        pickle.dump(
            {
                "stacks": [(tuple(s), m) for s, m in stacks],
                "sheet": sheet,
                "seen": piece["seen"],
                "written": piece["written"],
                "looks": piece["looks"],
                "name": job[0].name,
            },
            out,
            protocol=5,
        )
    if k % 25 == 0:
        print(k, "of", len(jobs), round(time.perf_counter() - start), "s", flush=True)
wall = time.perf_counter() - start
report = {
    "unit": unit,
    "pieces": len(jobs),
    "wall_serial_s": wall,
    "ops": {k: [T[k], N[k]] for k in sorted(T, key=T.get, reverse=True)},
    "rows": rows,
}
(cache / "sampler_profile.json").write_text(json.dumps(report, indent=1))
print(
    f"\nserial sampler wall {wall:.1f} s for {len(jobs)} pieces, {sum(r.get('px', 0) for r in rows) / 1e6:.0f} Mpx bbox, "
    f"{sum(r.get('covered', 0) for r in rows) / 1e6:.0f} Mpx covered"
)
for k in sorted(T, key=T.get, reverse=True):
    print(f"{T[k]:9.2f} s  {100 * T[k] / wall:5.1f} %  n={N[k]:5d}  {k}")
