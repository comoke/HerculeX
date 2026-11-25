#!/usr/bin/env bash
# Publish step (website.sheet_tiles) of one unit with 4 and with 12 processes; time each and compare every tile.
# shellcheck source=/dev/null
. "$(dirname "$0")/env.sh"
L=$M5/bench/logs/publish_ab
mkdir -p "$L"
cd "$M5" || exit 1
export HERCULEX_OUTPUTS=$M5/data/outputs-m5 TMPDIR=$M5/runtime/scratch PYTHONDONTWRITEBYTECODE=1 MallocLargeCache=0
for n in 12 4; do
    s=$(date +%s)
    HERCULEX_COMPOSE_PROCESSES=$n /usr/bin/time -l runtime/python/bin/python -m website.sheet_tiles "$SCROLL" "$SCAN" 0.00864 bench-u44-p$n layer0,sheet,taught --units rows-0044-0056 --model bench --ink-name ink --written-face >"$L/p$n.log" 2>"$L/p$n.time"
    echo "processes=$n rc=$? wall=$(($(date +%s) - s))s $(grep 'level 0' "$L/p$n.log" | tail -1) maxrss=$(awk '/maximum resident/{printf "%.1fGB", $1/2^30}' "$L/p$n.time")"
done
runtime/python/bin/python - <<'PY'
import hashlib
from storage.data_roots import SHEETS
from storage.tile_store import TileStore
def digest(name):
    store = TileStore(SHEETS / name); out = {}
    for layer in ("layer0", "sheet", "taught"):
        for level, col, row, *_ in store.tiles(layer):
            out[(layer, level, col, row)] = hashlib.sha256(store.tile(layer, level, col, row)).hexdigest()
    return out
a, b = digest("bench-u44-p4"), digest("bench-u44-p12")
print("tiles", len(a), len(b), "same keys", a.keys() == b.keys(), "identical tiles", sum(a[k] == b.get(k) for k in a))
PY
