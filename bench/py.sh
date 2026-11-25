#!/usr/bin/env bash
# Run a bench python script inside a checkout with HerculeX's runtime settings. Usage: py.sh CHECKOUT OUTPUTS SCRIPT ARGS...
# shellcheck source=/dev/null
. "$(dirname "$0")/env.sh"
H=$1
OUT=$2
shift 2
D=$OUT/unwrapping/level/$SCROLL/$SCAN
mkdir -p "$D"
[ -e "$D/pieces" ] || ln -s "$SRC_LEVEL/pieces" "$D/pieces"
[ -e "$D/sheet" ] || cp -cRp "$SRC_LEVEL/sheet" "$D/sheet"
cd "$H" || exit 1
export HERCULEX_OUTPUTS="$OUT" TMPDIR="$H/runtime/scratch" PYTHONDONTWRITEBYTECODE=1 MallocLargeCache=0
exec runtime/python/bin/python "$@"
