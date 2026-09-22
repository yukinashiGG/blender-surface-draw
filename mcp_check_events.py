"""Event-driven checks for Surface Weight Paint: real strokes through the
tool keymap, driven with simulated input.

    blender --factory-startup --enable-event-simulate --window-geometry 0 0 1800 950 --python mcp_check_events.py

`--enable-event-simulate` turns on Window.event_simulate(); without it the
script logs that and quits. mcp_check.py covers the maths by calling the
dab routine directly; this one covers what only the real operator path can
show - how Blender invokes the operator, what it carries over between runs,
and what the modal does with modifiers, Esc and undo.

Regression for issue #1: a Shift (blur) stroke sets the operator's `smooth`
property; Blender stores an operator's properties as its last-used values
and hands them to the next invocation that does not set them itself. The
(since removed) resident-mode button inherited them and started in blur
mode. The flags are SKIP_SAVE now; the check here is that an invocation
without keymap properties, right after a Shift stroke, sees smooth=False.

Env: SRC (package folder, default src/surface_weight_paint), CHECK_OUT
(where check_events_log.txt goes, default next to this file).
"""
import os, sys, importlib.util, traceback
import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("SRC", os.path.join(HERE, "src", "surface_weight_paint"))
OUT = os.environ.get("CHECK_OUT", HERE)
LOG = os.path.join(OUT, "check_events_log.txt")
_lines = []
_fails = 0


def log(*a):
    _lines.append(" ".join(str(x) for x in a))
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))


def check(name, cond, detail=""):
    global _fails
    if not cond:
        _fails += 1
    log("  [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
    return bool(cond)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "surface_weight_paint", os.path.join(PKG, "__init__.py"),
        submodule_search_locations=[PKG])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["surface_weight_paint"] = mod
    spec.loader.exec_module(mod)
    # log what the operator sees on entry: keymap item values plus anything
    # Blender carried over from the previous run (value set, is_set False)
    cls = mod.PAINT_OT_geodesic_weight_brush
    orig_invoke = cls.invoke

    def invoke(self, context, event):
        log("      invoke %s/%s shift=%s ctrl=%s | smooth=%s(set=%s) invert=%s(set=%s)"
            % (event.type, event.value, event.shift, event.ctrl,
               self.smooth, self.properties.is_property_set("smooth"),
               self.invert, self.properties.is_property_set("invert")))
        G["last_invoke"] = (bool(self.smooth), self.properties.is_property_set("smooth"))
        return orig_invoke(self, context, event)
    cls.invoke = invoke
    mod.register()
    return mod


def find_view3d():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return win, area, region


def weights(ob, gname):
    gidx = ob.vertex_groups[gname].index
    out = np.zeros(len(ob.data.vertices))
    for v in ob.data.vertices:
        for g in v.groups:
            if g.group == gidx:
                out[v.index] = g.weight
    return out


def hit_vertex(context, ob, rxy):
    from bpy_extras import view3d_utils
    from mathutils import Vector
    region, rv3d = context.region, context.region_data
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, rxy)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, rxy)
    mwi = ob.matrix_world.inverted()
    hit, loc, _n, face = ob.ray_cast(mwi @ origin, (mwi.to_3x3() @ direction).normalized())
    if not hit:
        raise RuntimeError("no hit at %s" % (rxy,))
    me = ob.data
    return int(min(me.polygons[face].vertices, key=lambda i: (Vector(me.vertices[i].co) - loc).length))


G = {}


def override():
    return bpy.context.temp_override(window=G["win"], area=G["area"], region=G["region"],
                                     space_data=G["area"].spaces[0])


def prepare():
    mod = load_module()
    log("Blender", bpy.app.version_string, "| module", mod.__name__, "| package", PKG)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0)
    ob = bpy.context.active_object
    vg = ob.vertex_groups.new(name="Bone.L")
    ob.vertex_groups.new(name="Other")
    ob.vertex_groups.active_index = vg.index
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    win, area, region = find_view3d()
    bpy.context.preferences.view.smooth_view = 0
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.view3d.view_axis(type='FRONT')
        rv3d = area.spaces[0].region_3d
        rv3d.view_location = (0.0, 0.0, 0.0)
        rv3d.view_distance = 3.68
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    G.update(mod=mod, ob=ob, win=win, area=area, region=region)
    with override():
        bpy.ops.wm.tool_set_by_id(name=mod.TOOL_IDNAME)
        ts = bpy.context.tool_settings
        ups = mod.unified_paint_settings(bpy.context)
        ups.use_unified_weight = ups.use_unified_strength = ups.use_unified_size = False
        brush = ts.weight_paint.brush
        brush.blend, brush.weight, brush.strength, brush.size = 'MIX', 1.0, 1.0, 60
        brush.use_pressure_strength = brush.use_pressure_size = False
        ts.use_auto_normalize = False
        ob.data.use_mirror_x = False


def ev(type, value, rxy, shift=False, ctrl=False):
    region = G["region"]
    G["win"].event_simulate(type=type, value=value,
                            x=int(region.x + rxy[0]), y=int(region.y + rxy[1]),
                            shift=shift, ctrl=ctrl)


def stroke(rxy, shift=False, ctrl=False, length=40):
    """Press, a few moves, release. One event per event-loop pass."""
    ev('MOUSEMOVE', 'NOTHING', rxy, shift, ctrl)
    yield 0.05
    ev('LEFTMOUSE', 'PRESS', rxy, shift, ctrl)
    yield 0.1
    for k in range(1, 5):
        ev('MOUSEMOVE', 'NOTHING', (rxy[0] + k * length // 4, rxy[1]), shift, ctrl)
        yield 0.05
    ev('LEFTMOUSE', 'RELEASE', (rxy[0] + length, rxy[1]), shift, ctrl)
    yield 0.2


def key(k, rxy, shift=False, ctrl=False):
    ev(k, 'PRESS', rxy, shift, ctrl)
    yield 0.15
    ev(k, 'RELEASE', rxy, shift, ctrl)
    yield 0.3


def invoke_without_press():
    """Call the operator the way a button or a script would (no mouse press,
    no keymap properties). undo=True keeps Blender's last-used-properties
    bookkeeping on, exactly as for a real button click."""
    with override():
        try:
            res = bpy.ops.paint.geodesic_weight_brush('INVOKE_DEFAULT', True)
            return res, str(res)
        except RuntimeError as exc:
            return {'CANCELLED'}, str(exc).strip()


def stroke_running():
    return any(op.bl_idname == "PAINT_OT_geodesic_weight_brush" for op in G["win"].modal_operators)


def steps():
    prepare()
    yield 0.5
    ob, region, mod = G["ob"], G["region"], G["mod"]
    cx, cy = region.width // 2, region.height // 2
    A, C = (cx - 180, cy), (cx + 180, cy)
    with override():
        vA, vC = hit_vertex(bpy.context, ob, A), hit_vertex(bpy.context, ob, C)

    def reset():
        ob.vertex_groups["Bone.L"].add(range(len(ob.data.vertices)), 0.0, 'REPLACE')

    def w(v):
        return weights(ob, "Bone.L")[v]

    log("== toolbar keymap")
    reset()
    yield from stroke(A)
    check("plain stroke paints", w(vA) == 1.0, "A=%.3f" % w(vA))
    yield from stroke(A, shift=True)
    check("Shift stroke blurs", 0.0 < w(vA) < 1.0, "A=%.3f" % w(vA))
    yield from stroke(C)
    check("plain stroke after Shift stroke paints", w(vC) == 1.0, "C=%.3f" % w(vC))
    check("plain item sets smooth=False explicitly", G["last_invoke"] == (False, True), repr(G["last_invoke"]))
    yield from stroke(C, ctrl=True)
    check("Ctrl stroke erases", w(vC) == 0.0, "C=%.3f" % w(vC))
    yield from stroke(C)
    check("plain stroke after Ctrl stroke paints", w(vC) == 1.0, "C=%.3f" % w(vC))

    check("stroke ends on release (no modal left)", not stroke_running())

    log("== issue #1: invocation without keymap properties after a Shift stroke")
    reset()
    yield from stroke(A, shift=True)
    res, msg = invoke_without_press()
    check("last-used blur flag is not carried over", G["last_invoke"][0] is False, repr(G["last_invoke"]))
    check("invocation without a mouse press is refused with a hint",
          res == {'CANCELLED'} and "toolbar" in msg, msg)
    check("nothing left running", not stroke_running())
    yield 0.2
    yield from stroke(C)
    check("next toolbar stroke paints", w(vC) == 1.0, "C=%.3f" % w(vC))

    log("== Esc in the middle of a Shift stroke")
    reset()
    yield from stroke(A)
    ev('LEFTMOUSE', 'PRESS', A, shift=True)
    yield 0.1
    ev('MOUSEMOVE', 'NOTHING', (A[0] + 20, A[1]), shift=True)
    yield 0.05
    ev('ESC', 'PRESS', (A[0] + 20, A[1]), shift=True)
    yield 0.2
    ev('LEFTMOUSE', 'RELEASE', (A[0] + 20, A[1]), shift=True)
    yield 0.1
    ev('ESC', 'RELEASE', (A[0] + 20, A[1]))
    yield 0.3
    check("stroke ended by Esc", not stroke_running())
    yield from stroke(C)
    check("toolbar stroke afterwards paints", w(vC) == 1.0, "C=%.3f" % w(vC))

    log("== undo between strokes")
    reset()
    yield from stroke(A)
    before = w(vA)
    yield from key('Z', A, ctrl=True)
    after_undo = w(vA)
    yield from stroke(C)
    log("   A before undo=%.3f after=%.3f | C after painting=%.3f" % (before, after_undo, w(vC)))
    check("Ctrl+Z undoes the stroke", after_undo < before, "A %.3f -> %.3f" % (before, after_undo))
    check("painting still works after undo", w(vC) == 1.0, "C=%.3f" % w(vC))

    log("== done: %d failure(s)" % _fails)
    mod.unregister()


_gen = None


def _main():
    global _gen
    try:
        if _gen is None:
            _gen = steps()
        return next(_gen)
    except StopIteration:
        pass
    except Exception:
        log("ERROR\n" + traceback.format_exc())
    bpy.ops.wm.quit_blender()
    return None


if __name__ == "__main__":
    bpy.context.preferences.view.show_splash = False
    try:
        bpy.context.window.event_simulate(type='MOUSEMOVE', value='NOTHING', x=0, y=0)
    except Exception as exc:
        log("event simulation unavailable (start Blender with --enable-event-simulate):", exc)
        bpy.app.timers.register(lambda: bpy.ops.wm.quit_blender() and None, first_interval=0.5)
    else:
        bpy.app.timers.register(_main, first_interval=1.0)
