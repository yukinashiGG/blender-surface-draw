"""Generate the toolbar icon (src/surface_weight_paint/surface_weight_paint_icon.dat).

Blender's toolbar icons are tiny triangle meshes in the "VCO" format that
release/datafiles/blender_icons_geom.py writes (header 'VCO\\0', 255, 255, 0, 0;
then 6 bytes of XY per triangle, then 12 bytes of RGBA per triangle; coordinates
0..255 with y up; later triangles overdraw earlier ones). Blender rasterizes
it at 256 px and box-scales down, and inverts lightness on light themes, so the
palette here is the same two greys the built-in brushes use.

Design: the standard weight-paint brush silhouette (proportions measured from
brush.generic) whose tip sits in a ripple of two flattened rings - the falloff
spreading over the surface. Only the brush is light; the rings are the
mid grey Blender uses for the "context" part of a tool icon.

    python tools/make_icon.py            # writes the .dat
    python tools/make_icon.py --preview  # also writes a PNG mock of the toolbar (needs numpy)
    python tools/make_icon.py --heat     # red/blue variant (weight colours) instead of grey

Any Python 3 works for the .dat; the preview needs numpy (Blender's bundled
python has it).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "src", "surface_weight_paint", "surface_weight_paint_icon.dat")

LIGHT = (229, 229, 229, 255)
MID = (144, 144, 144, 255)
RED = (228, 81, 81, 255)      # brush.particle.weight's colours
BLUE = (1, 161, 225, 255)


# ---------------------------------------------------------------- geometry
def _ccw(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])


def tri(p0, p1, p2, color):
    if _ccw(p0, p1, p2) < 0:
        p1, p2 = p2, p1
    return ((tuple(p0), tuple(p1), tuple(p2)), color)


def quad(p0, p1, p2, p3, color):
    return [tri(p0, p1, p2, color), tri(p0, p2, p3, color)]


def polygon(pts, color):
    """Ear clipping for a simple polygon."""
    pts = [tuple(p) for p in pts]
    if sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
           for i in range(len(pts))) < 0:
        pts.reverse()
    idx = list(range(len(pts)))
    out = []

    def inside(p, a, b, c):
        return _ccw(a, b, p) >= -1e-9 and _ccw(b, c, p) >= -1e-9 and _ccw(c, a, p) >= -1e-9

    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if _ccw(a, b, c) <= 1e-9:
                continue
            if any(inside(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            out.append(tri(a, b, c, color))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        out.append(tri(pts[idx[0]], pts[idx[1]], pts[idx[2]], color))
    return out


def ellipse(cx, cy, rx, ry, color, n=48):
    out = []
    for i in range(n):
        a0 = 2 * math.pi * i / n
        a1 = 2 * math.pi * (i + 1) / n
        out.append(tri((cx, cy), (cx + rx * math.cos(a0), cy + ry * math.sin(a0)),
                       (cx + rx * math.cos(a1), cy + ry * math.sin(a1)), color))
    return out


def ellipse_ring(cx, cy, rx, ry, w, color, n=56):
    out = []
    k = ry / rx
    for i in range(n):
        b0 = 2 * math.pi * i / n
        b1 = 2 * math.pi * (i + 1) / n
        p0 = (cx + (rx - w / 2) * math.cos(b0), cy + (ry - w * k / 2) * math.sin(b0))
        p1 = (cx + (rx + w / 2) * math.cos(b0), cy + (ry + w * k / 2) * math.sin(b0))
        p2 = (cx + (rx + w / 2) * math.cos(b1), cy + (ry + w * k / 2) * math.sin(b1))
        p3 = (cx + (rx - w / 2) * math.cos(b1), cy + (ry - w * k / 2) * math.sin(b1))
        out += quad(p0, p1, p2, p3, color)
    return out


def transform(tris, fn):
    return [tri(*[fn(p) for p in co], color) for co, color in tris]


# half-width along the brush axis (x from the tip), measured from brush.generic
# so the brush has the same proportions as the built-in tools
BRUSH_PROFILE = [(0, 0), (2, 9), (6, 15), (12, 18.5), (20, 21), (30, 22), (40, 21.5),
                 (48, 19.5), (54, 15), (60, 11.5), (66, 11.8), (76, 12.5), (150, 12.8),
                 (162, 11.8), (172, 10.5), (182, 9), (192, 7), (198, 5), (202, 2.5), (204, 0)]


def brush(tip, angle_deg, scale, color=LIGHT):
    upper = [(x * scale, w * scale) for x, w in BRUSH_PROFILE]
    lower = [(x * scale, -w * scale) for x, w in reversed(BRUSH_PROFILE)]
    outline = upper[:-1] + lower[:-1]
    outline = [p for i, p in enumerate(outline) if i == 0 or p != outline[i - 1]]
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return transform(polygon(outline, color),
                     lambda p: (tip[0] + p[0] * ca - p[1] * sa, tip[1] + p[0] * sa + p[1] * ca))


def design(heat=False):
    tip = (92, 86)
    cx, cy = tip[0] + 6, tip[1] - 4
    tris = []
    if heat:
        tris += ellipse(cx, cy, 42, 18, RED)
        tris += ellipse_ring(cx, cy, 76, 33, 15, BLUE)
    else:
        tris += ellipse_ring(cx, cy, 46, 20, 17, MID)
        tris += ellipse_ring(cx, cy, 76, 33, 15, MID)
    tris += brush(tip, 42, 0.92)
    # scale about the centre so the icon sits like its neighbours
    xs = [p[0] for co, _ in tris for p in co]
    ys = [p[1] for co, _ in tris for p in co]
    mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return transform(tris, lambda p: (124 + (p[0] - mx) * 0.9, 126 + (p[1] - my) * 0.9))


# ---------------------------------------------------------------- output
def write_dat(path, tris):
    coords = bytearray()
    colors = bytearray()
    kept = 0
    for co, color in tris:
        q = [(int(round(x)), int(round(y))) for x, y in co]
        if _ccw(*q) <= 0:          # quantised to nothing, like blender_icons_geom.py
            continue
        for x, y in q:
            if not (0 <= x <= 255 and 0 <= y <= 255):
                raise ValueError("out of range: %r" % ((x, y),))
            coords += bytes((x, y))
        colors += bytes(color) * 3
        kept += 1
    with open(path, "wb") as f:
        f.write(b"VCO\x00" + bytes((255, 255, 0, 0)) + coords + colors)
    return kept


def preview(dat_path, png_path):
    """Mock the Blender rasterizer (256 px, box scale) and draw the icon at
    toolbar size next to the built-in weight-paint icons, dark and light theme."""
    import struct
    import zlib
    import numpy as np

    icon_dir = None
    try:
        import bpy
        icon_dir = bpy.utils.system_resource('DATAFILES', path="icons")
    except Exception:
        for root in (r"C:\Program Files\Blender Foundation", "/Applications", "/usr/share/blender"):
            if os.path.isdir(root):
                for dp, dn, fn in os.walk(root):
                    if os.path.basename(dp) == "icons" and "brush.generic.dat" in fn:
                        icon_dir = dp
                        break
                if icon_dir:
                    break

    def read(path):
        b = open(path, "rb").read()[8:]
        n = len(b) // 18
        co = np.frombuffer(b[:n * 6], np.uint8).reshape(n, 3, 2).astype(float)
        col = np.frombuffer(b[n * 6:], np.uint8).reshape(n, 3, 4).astype(float)
        return co, col

    def raster(co, col, size, invert):
        R = 256
        rect = np.zeros((R, R, 4), np.float32)
        ys, xs = np.mgrid[0:R, 0:R]
        px, py = xs + 0.5, ys + 0.5
        if invert:   # BKE_icon_geom_invert_lightness (colours here are grey or saturated)
            rgb = col[..., :3] / 255.0
            mx, mn = rgb.max(-1), rgb.min(-1)
            l = (mx + mn) / 2
            col = col.copy()
            grey = mx == mn
            col[..., :3] = np.where(grey[..., None], (1 - l)[..., None] * 255, col[..., :3])
        for t in range(co.shape[0]):
            (x0, y0), (x1, y1), (x2, y2) = co[t]
            d = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(d) < 1e-9:
                continue
            w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / d
            w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / d
            inside = (w0 >= 0) & (w1 >= 0) & (1 - w0 - w1 >= 0)
            rect[inside] = col[t][0] / 255.0
        f = R // size
        small = rect[:f * size, :f * size].reshape(size, f, size, f, 4).mean(axis=(1, 3))
        return small[::-1]

    names = ["brush.generic", None, "brush.paint_weight.blur", "brush.paint_weight.average",
             "brush.paint_weight.smear", "ops.paint.weight_gradient", "ops.paint.weight_sample"]
    items = [read(dat_path) if n is None else
             (read(os.path.join(icon_dir, n + ".dat")) if icon_dir else None) for n in names]
    items = [i for i in items if i is not None]
    size, k, pad = 32, 4, 12
    rows = []
    for invert, bg in ((False, 0.33), (True, 0.80)):
        cells = []
        for co, col in items:
            img = raster(co, col, size, invert)
            rgb = img[..., :3] + bg * (1 - img[..., 3:4])     # premultiplied, as Blender draws it
            cells.append(np.repeat(np.repeat(rgb, k, 0), k, 1))
        rows.append((cells, bg))
    cw = size * k
    W = pad + len(items) * (cw + pad)
    H = pad + len(rows) * (cw + pad)
    canvas = np.zeros((H, W, 3), np.float32)
    y = pad
    for cells, bg in rows:
        canvas[y - pad // 2:y + cw + pad // 2] = bg
        x = pad
        for c in cells:
            canvas[y:y + cw, x:x + cw] = c
            x += cw + pad
        y += cw + pad
    data = np.clip(canvas * 255 + 0.5, 0, 255).astype(np.uint8)
    raw = b"".join(b"\x00" + data[r].tobytes() for r in range(H))

    def chunk(tag, payload):
        c = tag + payload
        return struct.pack(">I", len(payload)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    with open(png_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    print("preview:", png_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a.startswith("--")]
    out = os.path.abspath(OUT)
    n = write_dat(out, design(heat="--heat" in args))
    print("wrote %s (%d triangles, %d bytes)" % (out, n, os.path.getsize(out)))
    if "--preview" in args:
        preview(out, os.path.join(HERE, "icon_preview.png"))
