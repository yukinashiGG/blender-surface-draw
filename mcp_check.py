"""Verification harness for Surface Draw (geodesic_weight_brush).

Runs in a separate, windowed Blender (the brush needs a 3D viewport for
ray casts and screen-space radii):

    blender --factory-startup --window-geometry 0 0 1600 900 --python mcp_check.py

Loads the package straight from src\\ (or the installed extension when
USE_INSTALLED=1), builds a test scene, drives the brush's dab routine with a
fake event, and checks weights numerically. Results go to check_log.txt next
to this file; screenshots of the header/sidebar in English and Japanese go
there too. Blender quits when done.
"""
import os, sys, time, types, importlib.util, traceback
import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "src", "surface_draw")
# CHECK_OUT lets two Blender versions run at once without sharing files
HERE = os.environ.get("CHECK_OUT", HERE)
LOG = os.path.join(HERE, "check_log.txt")
USE_INSTALLED = os.environ.get("USE_INSTALLED") == "1"
INSTALLED_MODULE = "bl_ext.user_default.surface_draw"

_lines = []
_fails = 0


def log(*a):
    _lines.append(" ".join(str(x) for x in a))
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))


def check(name, cond, detail=""):
    """PASS/FAIL with the measured value printed - a bare PASS hides a test
    that silently skipped."""
    global _fails
    if not cond:
        _fails += 1
    log("  [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
    return bool(cond)


# --------------------------------------------------------------------------
def load_module():
    if USE_INSTALLED:
        import addon_utils
        addon_utils.enable(INSTALLED_MODULE, default_set=True)
        return sys.modules[INSTALLED_MODULE]
    spec = importlib.util.spec_from_file_location(
        "surface_draw", os.path.join(PKG, "__init__.py"),
        submodule_search_locations=[PKG])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["surface_draw"] = mod
    spec.loader.exec_module(mod)
    mod.register()
    return mod


def find_view3d():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return win, area, region
    return None, None, None


class FakeEvent:
    def __init__(self, x, y, pressure=1.0):
        self.mouse_region_x = x
        self.mouse_region_y = y
        self.pressure = pressure


def build_scene():
    """Sphere skinned to an armature: Bone.L / Bone.R / Spine deform, plus a
    non-deform group so auto-normalize has something to ignore."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.object.armature_add(enter_editmode=True)
    arm = bpy.context.active_object
    arm.name = "Armature"
    eb = arm.data.edit_bones
    eb[0].name = "Spine"
    for nm, x in (("Bone.L", 0.5), ("Bone.R", -0.5)):
        b = eb.new(nm)
        b.head = (x, 0, 0)
        b.tail = (x, 0, 1)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
    ob = bpy.context.active_object
    ob.name = "Sphere"
    for nm in ("Bone.L", "Bone.R", "Spine", "Extra"):
        ob.vertex_groups.new(name=nm)
    ob.vertex_groups["Spine"].add(range(len(ob.data.vertices)), 0.5, 'REPLACE')
    mod = ob.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm
    ob.vertex_groups.active_index = ob.vertex_groups["Bone.L"].index
    return ob, arm


def weights(ob, gname):
    gidx = ob.vertex_groups[gname].index
    out = np.zeros(len(ob.data.vertices))
    for v in ob.data.vertices:
        for g in v.groups:
            if g.group == gidx:
                out[v.index] = g.weight
    return out


def make_state(mod, context, ob, invert=False, smooth=False, mirror=False):
    """Everything invoke() prepares, without needing a real mouse event."""
    cls = mod.PAINT_OT_geodesic_weight_brush
    me = ob.data
    ns = types.SimpleNamespace()
    ns.obj, ns.me = ob, me
    ns.vg = ob.vertex_groups.active
    ns.gidx = ns.vg.index
    ns.region, ns.rv3d, ns.area = context.region, context.region_data, context.area
    ns.mirror_on = mirror
    ns.graph, ns.mirror = mod.mesh_graph_cached(me, mirror)
    ns.deform_idx = mod._deform_group_indices(ob)
    ns.locked_idx = mod._locked_group_indices(ob)
    ns.allowed = cls._build_mask(me)
    ns.mirror_vg, ns.mirror_gidx = None, -1
    if mirror:
        fname = mod.flip_name(ns.vg.name)
        mvg = ob.vertex_groups.get(fname) or ob.vertex_groups.new(name=fname)
        ns.mirror_vg, ns.mirror_gidx = mvg, mvg.index
    ns.use_evaluated = False
    ns.invert, ns.smooth = invert, smooth
    ns.last_px = None
    ns.dabs = ns.touched = 0
    ns.radius_scale = 1.0
    for name in ("_get_w", "_set_w", "_normalize", "_dab"):
        setattr(ns, name, types.MethodType(getattr(cls, name), ns))
    return ns


def aim(context, ob, xy=None):
    """Cast the same ray the brush will cast at `xy` (default: region centre)
    and return (index of the hit face's vertex nearest the hit, xy). Aiming
    the other way round - projecting a chosen vertex - drifts when the view
    matrices are mid-update, which made every dab land elsewhere."""
    from bpy_extras import view3d_utils
    from mathutils import Vector
    region, rv3d = context.region, context.region_data
    if xy is None:
        xy = (region.width // 2, region.height // 2)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, xy)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, xy)
    mwi = ob.matrix_world.inverted()
    hit, loc, _n, face = ob.ray_cast(mwi @ origin, (mwi.to_3x3() @ direction).normalized())
    if not hit:
        raise RuntimeError("aim(): ray at %s hits nothing" % (xy,))
    me = ob.data
    vi = min(me.polygons[face].vertices, key=lambda i: (Vector(me.vertices[i].co) - loc).length)
    return int(vi), (int(xy[0]), int(xy[1]))


def set_brush(context, blend='MIX', weight=1.0, strength=1.0, size=120, preset='SMOOTH'):
    ts = context.tool_settings
    ups = ts.unified_paint_settings
    ups.use_unified_weight = ups.use_unified_strength = ups.use_unified_size = False
    brush = ts.weight_paint.brush
    if brush is not None:
        brush.blend, brush.weight, brush.strength, brush.size = blend, weight, strength, size
        brush.curve_preset = preset
        brush.use_space = False
        brush.use_pressure_size = brush.use_pressure_strength = False
    return brush


def dabs(st, context, mod, xy, n=25):
    bv = mod.brush_values(context)
    ev = FakeEvent(*xy)
    # record the surface radius the brush derives from the pixel size
    rec = []
    orig = st.graph.within
    st.graph.within = lambda seeds, radius: (rec.append(radius) or orig(seeds, radius))
    try:
        for _ in range(n):
            st.last_px = None
            st._dab(context, ev, bv)
    finally:
        del st.graph.within
    global LAST_RADIUS
    LAST_RADIUS = rec[0] if rec else None
    log("    (dabs at %s: size=%d px -> surface radius %s)"
        % (xy, bv["size"], "%.3f" % rec[0] if rec else "n/a"))
    return st.dabs


LAST_RADIUS = None


def screen_scale(context):
    """Pixels per world unit at the origin, from the view's own projection."""
    from bpy_extras import view3d_utils
    from mathutils import Vector
    region, rv3d = context.region, context.region_data
    a = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector((0, 0, 0)))
    b = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector((1, 0, 0)))
    return (b - a).length


def tool_ids(context):
    from bl_ui.space_toolsystem_common import ToolSelectPanelHelper
    from bl_ui.space_toolsystem_toolbar import VIEW3D_PT_tools_active as T
    return {it.idname for it in ToolSelectPanelHelper._tools_flatten(
        T.tools_from_context(context, mode='PAINT_WEIGHT')) if it is not None}


# --------------------------------------------------------------------------
G = {}          # scene + window handles shared between the timer steps


def prepare():
    mod = load_module()
    log("Blender", bpy.app.version_string, "| module", mod.__name__,
        "| USE_BRUSHES", mod._USE_BRUSHES, "| installed" if USE_INSTALLED else "| from src")
    ob, arm = build_scene()
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    win, area, region = find_view3d()
    bpy.context.preferences.view.smooth_view = 0     # view ops apply at once
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.view3d.view_axis(type='FRONT')
        # view_selected frames differently in weight paint mode (the sphere
        # came out 57 px wide), so set the zoom explicitly instead
        rv3d = area.spaces[0].region_3d
        rv3d.view_location = (0.0, 0.0, 0.0)
        rv3d.view_distance = 3.68
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    with bpy.context.temp_override(window=win, area=area, region=region, space_data=area.spaces[0]):
        bpy.ops.wm.tool_set_by_id(name=mod.TOOL_IDNAME)
    G.update(mod=mod, ob=ob, arm=arm, win=win, area=area, region=region)


def _screenshot_steps():
    """Sidebar screenshots in ja and en. The sidebar only lays itself out
    between event-loop passes, so each shot needs its own timer tick - this
    generator yields the delay before the next step."""
    area, win, region = G["area"], G["win"], G["region"]
    prefs = bpy.context.preferences.view
    old_lang = prefs.language
    area.spaces[0].show_region_ui = True
    # Whichever tab is active, one of these lands in it.
    tmp = []
    for cat in ("Item", "Tool", "View"):
        cls = type("VIEW3D_PT_geodesic_weight_shot_" + cat, (bpy.types.Panel,), {
            "bl_label": "Surface Draw", "bl_space_type": 'VIEW_3D',
            "bl_region_type": 'UI', "bl_category": cat, "bl_order": -1,   # above Transform
            "draw": lambda self, context: bpy.types.VIEW3D_PT_geodesic_weight.draw(self, context),
        })
        bpy.utils.register_class(cls)
        tmp.append(cls)
    prefs.language = 'ja_JP'
    prefs.use_translate_interface = prefs.use_translate_tooltips = True
    yield 0.6
    # Transform sits above us in the Item tab (bl_order cannot beat a C
    # panel), so scroll the sidebar down to bring our panel into view
    ui_region = next(r for r in area.regions if r.type == 'UI')
    for _ in range(12):        # each call only moves a step; needs a pass between
        with bpy.context.temp_override(window=win, area=area, region=ui_region, space_data=area.spaces[0]):
            bpy.ops.view2d.scroll_down(deltay=100)
        yield 0.15
    yield 0.6
    with bpy.context.temp_override(window=win, area=area, region=region, space_data=area.spaces[0]):
        bpy.ops.screen.screenshot(filepath=os.path.join(HERE, "check_ui_ja.png"))
    prefs.language = 'en_US'
    yield 0.6
    with bpy.context.temp_override(window=win, area=area, region=region, space_data=area.spaces[0]):
        bpy.ops.screen.screenshot(filepath=os.path.join(HERE, "check_ui_en.png"))
    log("screenshots: check_ui_ja.png / check_ui_en.png (sidebar mirrored into the open tab)")
    prefs.language = old_lang
    for cls in tmp:
        bpy.utils.unregister_class(cls)
    area.spaces[0].show_region_ui = False
    yield 0.6


def run():
    global _fails
    if not G:
        prepare()
    mod, ob, arm, win, area, region = (G[k] for k in ("mod", "ob", "arm", "win", "area", "region"))
    # redraw_timer leaves the override without a space; start a fresh one
    with bpy.context.temp_override(window=win, area=area, region=region, space_data=area.spaces[0]):
        ctx = bpy.context
        ts = ctx.tool_settings
        ts.use_auto_normalize = False
        ob.data.use_mirror_x = False

        log("== registration")
        check("tool in weight-paint toolbar", mod.TOOL_IDNAME in tool_ids(ctx))
        from bl_ui.space_toolsystem_common import ToolSelectPanelHelper
        tool = ToolSelectPanelHelper.tool_active_from_context(ctx)
        # 4.2 has no use_brushes at all; there the fallback header is used
        ub = getattr(tool, "use_brushes", None)
        check("tool active + use_brushes matches version",
              tool.idname == mod.TOOL_IDNAME and bool(ub) == mod._USE_BRUSHES,
              "idname=%s use_brushes=%s brush_type=%s" % (tool.idname, ub, getattr(tool, "brush_type", None)))
        check("operator registered", "geodesic_weight_brush" in dir(bpy.ops.paint)
              and "geodesic_weight_spread" in dir(bpy.ops.paint))
        check("panel registered", hasattr(bpy.types, "VIEW3D_PT_geodesic_weight"))
        check("brush available after tool set", ts.weight_paint.brush is not None,
              "brush=%s" % (ts.weight_paint.brush.name if ts.weight_paint.brush else None))
        check("operator poll in weight paint", bpy.ops.paint.geodesic_weight_brush.poll())

        log("== translation")
        prefs = ctx.preferences.view
        old_lang = prefs.language
        try:
            prefs.language = 'ja_JP'
            prefs.use_translate_interface = True
            prefs.use_translate_tooltips = True
            ja = bpy.app.translations.pgettext_iface("Start Resident Mode")
            check("ja_JP label", ja == "常駐モードで開始", repr(ja))
            ja2 = bpy.app.translations.pgettext_iface("Radius %d px") % 7
            check("ja_JP formatted label", ja2 == "半径 7 px", repr(ja2))
            jt = bpy.app.translations.pgettext_tip("Invert the brush (same as Ctrl+drag)")
            check("ja_JP tooltip", jt == "ブラシを反転する（Ctrl+ドラッグ相当）", repr(jt))
            prefs.language = 'en_US'
            en = bpy.app.translations.pgettext_iface("Start Resident Mode")
            check("en_US label unchanged", en == "Start Resident Mode", repr(en))
        finally:
            prefs.language = old_lang

        vi, xy = aim(ctx, ob)
        log("== painting (centre vertex %d at %s, region %dx%d)" % (vi, xy, ctx.region.width, ctx.region.height))

        set_brush(ctx, 'MIX', 1.0, 1.0, 120)
        ob.vertex_groups["Bone.L"].add(range(len(ob.data.vertices)), 0.0, 'REPLACE')
        st = make_state(mod, ctx, ob)
        n = dabs(st, ctx, mod, xy)
        w = weights(ob, "Bone.L")
        scale = screen_scale(ctx)
        check("brush radius matches the view's pixel scale",
              LAST_RADIUS is not None and abs(LAST_RADIUS - 120.0 / scale) < 0.03 * (120.0 / scale),
              "%.1f px/unit -> expected %.3f, brush used %s" % (scale, 120.0 / scale, "%.3f" % LAST_RADIUS if LAST_RADIUS else None))
        check("MIX paints to exactly 1.0 at centre", w[vi] == 1.0, "centre=%r dabs=%d exact1=%d" % (w[vi], n, int((w == 1.0).sum())))
        check("MIX leaves a falloff ring", int(((w > 0) & (w < 1)).sum()) > 0, "mid=%d" % int(((w > 0) & (w < 1)).sum()))
        # a 0.43-unit geodesic cap on a unit sphere is ~4.5% of its area
        check("MIX touches only the cap under the brush", 20 < int((w > 0).sum()) < len(w) * 0.15, "painted=%d/%d" % (int((w > 0).sum()), len(w)))

        st = make_state(mod, ctx, ob, invert=True)
        dabs(st, ctx, mod, xy)
        w = weights(ob, "Bone.L")
        check("MIX+Ctrl reaches exactly 0.0", w[vi] == 0.0 and int((w == 0.0).sum()) > len(w) * 0.5,
              "centre=%r exact0=%d min=%.3g" % (w[vi], int((w == 0.0).sum()), w.min()))

        set_brush(ctx, 'ADD')
        ob.vertex_groups["Bone.L"].add(range(len(ob.data.vertices)), 1.0, 'REPLACE')
        st = make_state(mod, ctx, ob, invert=True)
        dabs(st, ctx, mod, xy)
        check("ADD+Ctrl subtracts to 0", weights(ob, "Bone.L")[vi] == 0.0, "centre=%r" % weights(ob, "Bone.L")[vi])
        set_brush(ctx, 'SUB')
        st = make_state(mod, ctx, ob, invert=True)
        dabs(st, ctx, mod, xy)
        check("SUB+Ctrl adds to 1", weights(ob, "Bone.L")[vi] == 1.0, "centre=%r" % weights(ob, "Bone.L")[vi])

        log("== falloff presets")
        set_brush(ctx, 'MIX')
        for preset in ("SMOOTH", "SMOOTHER", "SPHERE", "ROOT", "SHARP", "LIN", "POW4", "INVSQUARE", "CONSTANT", "CUSTOM"):
            ob.vertex_groups["Bone.L"].add(range(len(ob.data.vertices)), 0.0, 'REPLACE')
            ts.weight_paint.brush.curve_preset = preset
            st = make_state(mod, ctx, ob)
            dabs(st, ctx, mod, xy, n=1)
            w = weights(ob, "Bone.L")
            check("preset %-9s one dab paints" % preset, w.max() > 0.5 and w[vi] > 0.5,
                  "centre=%.3f painted=%d" % (w[vi], int((w > 0).sum())))
        ts.weight_paint.brush.curve_preset = 'SMOOTH'

        log("== blur")
        ob.vertex_groups["Bone.L"].add(range(len(ob.data.vertices)), 0.0, 'REPLACE')
        ob.vertex_groups["Bone.L"].add([vi], 1.0, 'REPLACE')     # one spike
        st = make_state(mod, ctx, ob, smooth=True)
        dabs(st, ctx, mod, xy, n=5)
        w = weights(ob, "Bone.L")
        check("blur lowers the spike and raises neighbours", w[vi] < 1.0 and int((w > 0).sum()) > 1,
              "spike=%.3f nonzero=%d" % (w[vi], int((w > 0).sum())))

        log("== masks")
        set_brush(ctx, 'MIX')
        ob.vertex_groups["Bone.L"].add(range(len(ob.data.vertices)), 0.0, 'REPLACE')
        me = ob.data
        hide = np.zeros(len(me.vertices), dtype=bool)
        hide[vi] = True
        me.vertices.foreach_set("hide", hide)
        st = make_state(mod, ctx, ob)
        dabs(st, ctx, mod, xy)
        w = weights(ob, "Bone.L")
        check("hidden vertex is not painted", w[vi] == 0.0 and w.max() == 1.0, "hidden=%r max=%.3f" % (w[vi], w.max()))
        hide[:] = False
        me.vertices.foreach_set("hide", hide)

        ob.vertex_groups["Bone.L"].add(range(len(ob.data.vertices)), 0.0, 'REPLACE')
        me.use_paint_mask = True
        sel = np.zeros(len(me.polygons), dtype=bool)
        me.polygons.foreach_set("select", sel)                   # nothing selected
        st = make_state(mod, ctx, ob)
        dabs(st, ctx, mod, xy)
        check("face mask with no selection paints nothing", weights(ob, "Bone.L").max() == 0.0,
              "max=%.3f" % weights(ob, "Bone.L").max())
        me.use_paint_mask = False

        log("== X mirror")
        ob.vertex_groups["Bone.L"].add(range(len(ob.data.vertices)), 0.0, 'REPLACE')
        ob.vertex_groups["Bone.R"].add(range(len(ob.data.vertices)), 0.0, 'REPLACE')
        # aim off-centre (+X is screen-right in the front view) so the
        # mirrored side is a different vertex
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        vside, pside = aim(ctx, ob, (xy[0] + ctx.region.width // 8, xy[1]))
        check("mirror aim landed on +X side", co[vside, 0] > 0.2, "x=%.3f" % co[vside, 0])
        set_brush(ctx, 'MIX', size=60)
        st = make_state(mod, ctx, ob, mirror=True)
        dabs(st, ctx, mod, pside)
        wl, wr = weights(ob, "Bone.L"), weights(ob, "Bone.R")
        mirror_idx = int(st.mirror[vside])
        check("mirror writes the flipped group (Bone.R)", wr.max() > 0 and mirror_idx >= 0 and wr[mirror_idx] == wl[vside],
              "L[v]=%.3f R[mirror]=%.3f R.nonzero=%d L.nonzero=%d" % (wl[vside], wr[mirror_idx] if mirror_idx >= 0 else -1, int((wr > 0).sum()), int((wl > 0).sum())))
        check("mirror does not put Bone.L on the other side", int((wl[co[:, 0] < -0.05] > 0).sum()) == 0,
              "L on -X side=%d" % int((wl[co[:, 0] < -0.05] > 0).sum()))
        check("flip_name uses Blender rule", mod.flip_name("Arm.L.001") == "Arm.R.001" and mod.flip_name("Spine") == "Spine",
              "%s %s" % (mod.flip_name("Arm.L.001"), mod.flip_name("Spine")))

        log("== auto normalize + locks")
        ts.use_auto_normalize = True
        for nm in ("Bone.L", "Bone.R", "Extra"):
            ob.vertex_groups[nm].add(range(len(me.vertices)), 0.0, 'REPLACE')
        ob.vertex_groups["Spine"].add(range(len(me.vertices)), 0.3, 'REPLACE')
        ob.vertex_groups["Extra"].add(range(len(me.vertices)), 0.9, 'REPLACE')
        ob.vertex_groups["Spine"].lock_weight = True
        set_brush(ctx, 'MIX')
        st = make_state(mod, ctx, ob)
        dabs(st, ctx, mod, xy)
        wl, ws, wx = weights(ob, "Bone.L"), weights(ob, "Spine"), weights(ob, "Extra")
        # weights are stored as float32, so compare with a tolerance
        check("locked Spine untouched", np.allclose(ws, 0.3, atol=1e-6), "spine min=%.6f max=%.6f" % (ws.min(), ws.max()))
        check("painted group clamped to 1 - locked", abs(wl[vi] - 0.7) < 1e-5, "Bone.L=%.6f" % wl[vi])
        check("non-deform Extra ignored by normalize", abs(wx[vi] - 0.9) < 1e-6, "Extra=%.6f" % wx[vi])
        ob.vertex_groups["Spine"].lock_weight = False
        ts.use_auto_normalize = False

        log("== locked active group refused")
        ob.vertex_groups["Bone.L"].lock_weight = True
        # A Python-invoked operator turns report({'ERROR'}) into RuntimeError
        try:
            res = bpy.ops.paint.geodesic_weight_brush('INVOKE_DEFAULT')
            msg = str(res)
        except RuntimeError as exc:
            res = {'CANCELLED'}
            msg = str(exc).strip()
        check("invoke on locked group is CANCELLED", res == {'CANCELLED'} and "locked" in msg, msg)
        ob.vertex_groups["Bone.L"].lock_weight = False
        ob.vertex_groups.active_index = ob.vertex_groups["Bone.L"].index

        log("== Surface Gradient")
        ob.vertex_groups["Bone.L"].add(range(len(me.vertices)), 0.0, 'REPLACE')
        core = [i for i in range(len(co)) if co[i, 2] > 0.8]
        ob.vertex_groups["Bone.L"].add(core, 1.0, 'REPLACE')
        res = bpy.ops.paint.geodesic_weight_spread(distance=0.5, threshold=0.999)
        w1 = weights(ob, "Bone.L")
        check("gradient ran", res == {'FINISHED'} and int(((w1 > 0) & (w1 < 1)).sum()) > 0,
              "res=%s core=%d ramp=%d" % (res, len(core), int(((w1 > 0) & (w1 < 1)).sum())))
        check("gradient keeps the core at 1", all(w1[i] == 1.0 for i in core))
        # core = rings above 30 deg; 0.5 units along the meridian reaches the
        # rings at 37.5 / 45 / 52.5 deg (z 0.79..0.61) and stops before 60 deg
        ring_a = w1[(co[:, 2] > 0.75) & (co[:, 2] < 0.85)]     # 37.5 deg ring
        ring_b = w1[(co[:, 2] > 0.55) & (co[:, 2] < 0.65)]     # 52.5 deg ring
        below = w1[co[:, 2] < 0.45]
        check("gradient falls off with distance",
              ring_a.mean() > ring_b.mean() > 0.0 and below.max() == 0.0,
              "37.5deg=%.3f 52.5deg=%.3f beyond=%.3f" % (ring_a.mean(), ring_b.mean(), below.max()))
        bpy.ops.paint.geodesic_weight_spread(distance=0.5, threshold=0.999)
        w2 = weights(ob, "Bone.L")
        check("gradient idempotent (keep larger)", np.array_equal(w1, w2), "max diff=%.3g" % np.abs(w1 - w2).max())
        ob.vertex_groups["Bone.R"].add(range(len(me.vertices)), 0.0, 'REPLACE')
        ob.vertex_groups.active_index = ob.vertex_groups["Bone.R"].index
        try:
            res = bpy.ops.paint.geodesic_weight_spread(distance=0.5)
            msg = str(res)
        except RuntimeError as exc:
            res = {'CANCELLED'}
            msg = str(exc).strip()
        check("gradient with no seeds is CANCELLED", res == {'CANCELLED'} and "seeds" in msg, msg)
        ob.vertex_groups.active_index = ob.vertex_groups["Bone.L"].index

        log("== undo")
        ob.vertex_groups["Bone.L"].add(range(len(me.vertices)), 0.25, 'REPLACE')
        bpy.ops.ed.undo_push(message="before")
        st = make_state(mod, ctx, ob)
        dabs(st, ctx, mod, xy)
        bpy.ops.ed.undo_push(message="Surface Draw")
        after = weights(ob, "Bone.L")[vi]
        name = ob.name
        bpy.ops.ed.undo()
        # undo invalidates every reference held so far - ID ones raise, but a
        # stale ToolSettings/Paint pointer crashes Blender outright (seen:
        # EXCEPTION_ACCESS_VIOLATION in BKE_paint_brush). Re-fetch them all.
        ob = bpy.data.objects[name]
        me = ob.data
        ts = ctx.tool_settings
        back = weights(ob, "Bone.L")[vi]
        check("undo restores the weight", after == 1.0 and back == 0.25, "after=%r undone=%r" % (after, back))
        # the resident brush must survive an undo under it: a dab after undo
        # must not crash, and must either paint or end cleanly
        st = make_state(mod, ctx, ob)
        dabs(st, ctx, mod, xy, n=3)
        check("painting after undo works", weights(ob, "Bone.L")[vi] > 0.25, "centre=%.3f" % weights(ob, "Bone.L")[vi])

        log("== cache")
        g1, _ = mod.mesh_graph_cached(me, False)
        g2, _ = mod.mesh_graph_cached(me, False)
        check("graph cached per mesh", g1 is g2)
        me.vertices[0].co.x += 0.01
        g3, _ = mod.mesh_graph_cached(me, False)
        check("graph rebuilt after geometry change", g3 is not g1)

        log("== no brush")
        # Paint.brush is read-only from Python, so simulate a missing brush
        # with a stand-in context and make sure the fallback values are sane
        class _Ctx:
            tool_settings = types.SimpleNamespace(
                weight_paint=types.SimpleNamespace(brush=None),
                unified_paint_settings=types.SimpleNamespace(
                    use_unified_size=False, use_unified_strength=False, use_unified_weight=False,
                    size=33, strength=0.5, weight=0.25))
        bv = mod.brush_values(_Ctx())
        check("brush_values falls back when brush is None",
              bv["brush"] is None and bv["size"] == 33 and bv["strength"] == 0.5 and bv["weight"] == 0.25
              and bv["blend"] == 'MIX' and bv["preset"] == 'SMOOTH',
              "size=%d strength=%.2f weight=%.2f blend=%s" % (bv["size"], bv["strength"], bv["weight"], bv["blend"]))
        with bpy.context.temp_override(window=win, area=area, region=region, space_data=area.spaces[0]):
            bpy.ops.wm.tool_set_by_id(name="builtin_brush.blur")
            bpy.ops.wm.tool_set_by_id(name=mod.TOOL_IDNAME)
        check("brush restored by re-selecting tool", ts.weight_paint.brush is not None,
              "brush=%s" % (ts.weight_paint.brush.name if ts.weight_paint.brush else None))

        log("== register / unregister")
        mod.unregister()
        check("tool gone after unregister", mod.TOOL_IDNAME not in tool_ids(ctx))
        check("operator gone", "geodesic_weight_brush" not in dir(bpy.ops.paint))
        check("panel gone", not hasattr(bpy.types, "VIEW3D_PT_geodesic_weight"))
        mod.register()
        check("re-register works", mod.TOOL_IDNAME in tool_ids(ctx) and "geodesic_weight_brush" in dir(bpy.ops.paint))
        mod.unregister()
        check("second unregister clean", mod.TOOL_IDNAME not in tool_ids(ctx))

    log("== done: %d failure(s)" % _fails)
    return _fails


def _steps():
    prepare()
    yield from _screenshot_steps()
    run()


_gen = None


def _main():
    """Timer callback: each yielded float is the delay before the next tick
    (the sidebar needs event-loop passes between the screenshot steps)."""
    global _gen
    try:
        if _gen is None:
            _gen = _steps()
        return next(_gen)
    except StopIteration:
        pass
    except Exception:
        log("ERROR\n" + traceback.format_exc())
    bpy.ops.wm.quit_blender()
    return None


if __name__ == "__main__":
    # `blender --python mcp_check.py`; a wrapper that imports this module
    # (e.g. after installing the built zip) calls run() itself.
    bpy.context.preferences.view.show_splash = False
    bpy.app.timers.register(_main, first_interval=1.0)
