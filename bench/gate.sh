#!/usr/bin/env bash
# gate.sh CHECKOUT CACHE LANES [extra env] : reader on captured stacks vs the stock reference maps, one summary line.
# shellcheck source=/dev/null
. "$(dirname "$0")/env.sh"
H=$1
C=$2
LANES=$3
shift 3
env HX_LANE_LOG=1 HERCULEX_BODY_LANES="$LANES" "$@" "$M5/bench/py.sh" "$H" "$M5/data/outputs-prof" "$M5/bench/profile_reader.py" "$M5/bench/cache/$C" pipelined "$M5/bench/ref/$C" 2>&1 | grep -E "^\{|Error|Traceback" | python3 -c "
import sys, json
for line in sys.stdin:
    try: d = json.loads(line)
    except Exception: print(line[:300]); continue
    lanes = {k[-1]: (v['calls'], v['busy_s'], v['mean_ms']) for k, v in d.items() if k.startswith('lane_')}
    print('$C lanes=$LANES $*: wall %.2f s  max_abs %.5f  r_pooled %.7f  worst_map_r %.6f  lanes(calls,busy_s,mean_ms)=%s' % (d['wall_s'], d.get('max_abs_diff', -1), d.get('pearson_all_pixels', -1), d.get('worst_map_pearson', -1), lanes))
"
