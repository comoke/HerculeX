"""Stage outputs of two reads of the same unit: every piece's ink map (taught PNG), sheet and layer-0 images, and written face.
Usage: python3 compare_faces.py FACES_A/UNIT FACES_B/UNIT"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
a, b = Path(sys.argv[1]), Path(sys.argv[2])
names = sorted(p.name for p in a.glob("piece-*-taught.png"))
s = np.zeros(5)
n = 0
worst = 0
same_px = 0
geo_same = geo = 0
side_same = 0
missing = 0
identical_files = 0
for name in names:
    if not (b / name).exists():
        missing += 1
        continue
    x, y = np.asarray(Image.open(a / name)).astype(np.float64), np.asarray(Image.open(b / name)).astype(np.float64)
    if x.shape != y.shape:
        missing += 1
        continue
    d = np.abs(x - y)
    worst = max(worst, d.max())
    same_px += int((d == 0).sum())
    identical_files += int(d.max() == 0)
    s += [x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()]
    n += x.size
    stem = name[: -len("-taught.png")]
    for other in (f"{stem}.png", f"{stem}-layer0.png"):
        geo += 1
        geo_same += int((a / other).read_bytes() == (b / other).read_bytes())
    ja, jb = (
        json.loads((a / f"{stem}-taught-side.json").read_text()),
        json.loads((b / f"{stem}-taught-side.json").read_text()),
    )
    side_same += int(ja.get("written") == jb.get("written"))
sa, sb, saa, sbb, sab = s
r = (sab - sa * sb / n) / np.sqrt((saa - sa * sa / n) * (sbb - sb * sb / n))
print(
    json.dumps(
        {
            "pieces": len(names),
            "missing_or_shape": missing,
            "ink_png_max_abs_levels": int(worst),
            "ink_png_pearson": float(r),
            "ink_png_same_pixels": same_px / n,
            "ink_png_identical_files": identical_files,
            "sheet_and_layer0_png_byte_identical": f"{geo_same}/{geo}",
            "written_face_same": f"{side_same}/{len(names) - missing}",
            "pixels": int(n),
        }
    )
)
