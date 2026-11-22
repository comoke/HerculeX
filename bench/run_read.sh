#!/usr/bin/env bash
# One ink read of chosen units with a checkout. Usage: run_read.sh CHECKOUT OUTPUTS_DIR UNITS TAG [extra env...]
# Leaves logs in bench/logs/TAG: time, events, util.tsv, read.log. Outputs are wiped first so every piece is read.
set -u
. "$(dirname "$0")/env.sh"
H=$1; OUT=$2; UNITS=$3; TAG=$4; shift 4
L=$M5/bench/logs/$TAG; rm -rf "$L"; mkdir -p "$L"
pmset -g batt | head -1 | grep -q "AC Power" || { echo "on battery: not running" | tee $L/ABORT; exit 9; }
[ "$(df -g ~ | awk 'NR==2{print $4}')" -ge 45 ] || { echo "disk under 45G: not running" | tee $L/ABORT; exit 9; }
D=$OUT/unwrapping/level/$SCROLL/$SCAN
mkdir -p "$D"
[ -e "$D/pieces" ] || ln -s "$SRC_LEVEL/pieces" "$D/pieces"
[ -e "$D/sheet" ] || cp -cRp "$SRC_LEVEL/sheet" "$D/sheet"
rm -rf "$D/faces"
bash $M5/bench/sampler.sh $L/util.tsv & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
cd "$H" || exit 1
echo "checkout $H $(git rev-parse --short HEAD) dirty=$(git status --short | grep -vc metadata_never) units=$UNITS env=$* start=$(date -u +%FT%TZ)" > $L/meta.txt
env HERCULEX_OUTPUTS="$OUT" HERCULEX_UNITS="$UNITS" HERCULEX_KEEP_BRICKS=1 HERCULEX_KEEP_STACKS=0 HERCULEX_EVENTS=$L/events.jsonl "$@" \
  /usr/bin/time -l ./run.sh ink.reading.piece_reading $SCROLL $SCAN > $L/read.log 2> $L/time.txt
echo "rc=$? end=$(date -u +%FT%TZ)" >> $L/meta.txt
