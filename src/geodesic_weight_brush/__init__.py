# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yukinashi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Surface Draw - a weight paint brush that falls off along the surface.

The built-in brushes fall off by straight-line (or screen) distance, so
painting one thigh bleeds into the other and painting an arm bleeds into the
ribs. This brush measures distance by walking the mesh's edges (geodesic
distance), so faces that are not connected to the one under the cursor are
never painted, however close they sit in space.

Blender's stroke handling lives in C and only the falloff cannot be swapped,
so the brush is a modal operator of its own. Weight, strength, blend mode,
falloff curve, pen pressure and spacing are read from the currently selected
weight paint brush, and the header shows the same controls as the standard
brush, so it behaves like one.

Usage:
    (1) Pick "Surface Draw" in the toolbar (left column) in Weight Paint mode.
        LMB drag paints, Ctrl+LMB inverts, Shift+LMB blurs - same as Draw.
    (2) Sidebar (N) > Surface Draw > "Start Resident Mode" keeps the brush
        running without switching tools; Esc or RMB leaves it.

Packaged as an extension. bl_info is kept only so the build script can read
the version from one more place; blender_manifest.toml must agree with it.

Verified on Blender 4.5 LTS.
"""

bl_info = {
    "name": "Surface Draw (Geodesic Weight Brush)",
    "author": "Yukinashi",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Weight Paint > Toolbar / Sidebar > Surface Draw",
    "description": "Weight paint brush that falls off along the surface, "
                   "so nearby but unconnected parts are never painted",
    "category": "Paint",
}

import math
import heapq

import bpy
import gpu
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

iface_ = bpy.app.translations.pgettext_iface

TOOL_IDNAME = "paint_weight.geodesic_brush"


# ==========================================================================
# Falloff curve - same shapes as Blender's BKE_brush_curve_strength
# ==========================================================================

def falloff_strength(preset, curve_mapping, p):
    """p = distance / radius (0..1) -> strength 0..1."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    q = 1.0 - p
    if preset == 'CONSTANT':
        return 1.0
    if preset == 'SHARP':
        return q * q
    if preset == 'SMOOTH':
        return 3.0 * q * q - 2.0 * q * q * q
    if preset == 'SMOOTHER':
        return q * q * q * (q * (q * 6.0 - 15.0) + 10.0)
    if preset == 'ROOT':
        return math.sqrt(q)
    if preset == 'LIN':
        return q
    if preset == 'SPHERE':
        return math.sqrt(max(0.0, 2.0 * q - q * q))
    if preset == 'POW4':
        return q * q * q * q
    if preset == 'INVSQUARE':
        return q * (2.0 - q)
    if preset == 'CUSTOM' and curve_mapping is not None:
        try:
            return max(0.0, min(1.0, curve_mapping.evaluate(curve_mapping.curves[0], p)))
        except Exception:
            pass
    # Unknown presets fall off smoothly
    return 3.0 * q * q - 2.0 * q * q * q


# ==========================================================================
# Edge graph of the mesh - Dijkstra cut off at the brush radius
# ==========================================================================

class MeshGraph:
    """Vertices as nodes, edges as weighted links."""

    def __init__(self, me):
        n = len(me.vertices)
        self.n = n

        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        self.co = co.reshape(n, 3)

        ne = len(me.edges)
        ev = np.empty(ne * 2, dtype=np.int32)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(ne, 2)

        length = np.linalg.norm(self.co[ev[:, 0]] - self.co[ev[:, 1]], axis=1)

        src = np.concatenate([ev[:, 0], ev[:, 1]])
        dst = np.concatenate([ev[:, 1], ev[:, 0]])
        wei = np.concatenate([length, length])

        order = np.argsort(src, kind="stable")
        src = src[order]
        # CSR-style adjacency. Plain lists beat numpy inside the Dijkstra loop.
        self.dst = dst[order].tolist()
        self.wei = wei[order].tolist()
        self.start = np.searchsorted(src, np.arange(n + 1)).tolist()

    def within(self, seeds, radius):
        """Vertices reachable within `radius` from seeds = [(index, d0), ...].

        Returns {vertex index: geodesic distance}.
        """
        start, dst, wei = self.start, self.dst, self.wei
        best = {}
        heap = []
        for i, d0 in seeds:
            if d0 <= radius and d0 < best.get(i, 1e30):
                best[i] = d0
                heapq.heappush(heap, (d0, i))

        done = {}
        while heap:
            d, v = heapq.heappop(heap)
            if v in done:
                continue
            if d > radius:
                break                      # heap is ordered, nothing closer left
            done[v] = d
            for k in range(start[v], start[v + 1]):
                u = dst[k]
                nd = d + wei[k]
                if nd <= radius and nd < best.get(u, 1e30):
                    best[u] = nd
                    heapq.heappush(heap, (nd, u))
        return done


def _mirror_table(me):
    """For each vertex, the index of its X-mirrored twin (-1 when there is none)."""
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    q = np.round(co, 4)
    table = {}
    for i in range(n):
        table[(q[i, 0], q[i, 1], q[i, 2])] = i
    out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        j = table.get((-q[i, 0], q[i, 1], q[i, 2]), -1)
        if j != i:
            out[i] = j
    return out


# The graph and mirror table only depend on geometry, which does not change
# while weight painting, so they are built once per mesh instead of on every
# stroke. Keyed on the mesh pointer plus a cheap coordinate checksum so an
# edit-mode round trip invalidates them.
_CACHE = {}
_CACHE_MAX = 4


def _mesh_signature(me):
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    return (n, len(me.edges), float(co.sum()), float(np.abs(co).sum()))


def mesh_graph_cached(me, want_mirror):
    key = me.as_pointer()
    sig = _mesh_signature(me)
    entry = _CACHE.get(key)
    if entry is None or entry["sig"] != sig:
        entry = {"sig": sig, "graph": MeshGraph(me), "mirror": None}
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = entry
    if want_mirror and entry["mirror"] is None:
        entry["mirror"] = _mirror_table(me)
    return entry["graph"], (entry["mirror"] if want_mirror else None)


# ==========================================================================
# Brush settings (unified settings aware)
# ==========================================================================

def brush_values(context):
    """Read the active weight paint brush. Falls back to the unified settings
    and plain defaults when no brush is active, so the tool keeps working."""
    ts = context.tool_settings
    wp = ts.weight_paint
    brush = wp.brush if wp else None
    ups = ts.unified_paint_settings

    def pick(flag, attr, default):
        if brush is None or getattr(ups, flag, False):
            return getattr(ups, attr, default)
        return getattr(brush, attr, default)

    size = pick("use_unified_size", "size", 50)
    strength = pick("use_unified_strength", "strength", 1.0)
    weight = pick("use_unified_weight", "weight", 1.0)
    return {
        "brush": brush,
        "size": max(1, int(size)),
        "strength": float(strength),
        "weight": float(weight),
        "blend": getattr(brush, "blend", 'MIX'),
        "preset": getattr(brush, "curve_preset", 'SMOOTH'),
        "curve": getattr(brush, "curve", None),
        "spacing": float(getattr(brush, "spacing", 10)) / 100.0,
        "use_spacing": bool(getattr(brush, "use_space", True)),
        "pressure_strength": bool(getattr(brush, "use_pressure_strength", False)),
        "pressure_size": bool(getattr(brush, "use_pressure_size", False)),
    }


_SIDE_PAIRS = (
    ("L", "R"), ("l", "r"),
    ("Left", "Right"), ("left", "right"), ("LEFT", "RIGHT"),
)
_SEPARATORS = (".", "_", "-", " ")


def flip_name(name):
    """Swap left/right in a name; unchanged when there is no side marker.

    Blender's X mirror does not write the same group to the mirrored vertex,
    it writes the group whose name is flipped. Uses Blender's own rule when
    available (it also handles digit suffixes such as Arm.L.001).
    """
    fn = getattr(bpy.utils, "flip_name", None)
    if fn is not None:
        try:
            return fn(name)
        except Exception:
            pass
    for a, b in _SIDE_PAIRS:
        for sep in _SEPARATORS:
            for x, y in ((a, b), (b, a)):
                tail = sep + x
                if name.endswith(tail):
                    return name[:-len(tail)] + sep + y
                head = x + sep
                if name.startswith(head):
                    return y + sep + name[len(head):]
    return name


# Target-seeking blends never reach exactly 0 or 1 - they leave ~1e-6 behind.
# The overlay paints only an exact 0 black (1e-7 already shows as blue), so
# Ctrl strokes used to stop at blue. Snap to the edge once we are this close.
_SNAP_EPS = 1e-3


def flip_blend(mode, target):
    """Meaning of Ctrl (invert), matching the standard brush (wpaint_blend).

    MIX inverts the target weight; ADD/SUB and LIGHTEN/DARKEN swap places;
    MUL is left alone.
    """
    if mode == 'ADD':
        return 'SUB', target
    if mode == 'SUB':
        return 'ADD', target
    if mode == 'LIGHTEN':
        return 'DARKEN', target
    if mode == 'DARKEN':
        return 'LIGHTEN', target
    if mode == 'MUL':
        return mode, target
    return mode, 1.0 - target


def blend_weight(mode, current, target, alpha):
    """Standard brush blending. alpha = strength * falloff * pressure."""
    if mode == 'ADD':
        return current + target * alpha
    if mode == 'SUB':
        return current - target * alpha
    if mode == 'MUL':
        return current + (current * target - current) * alpha
    if mode == 'LIGHTEN':
        new = max(current, current + (target - current) * alpha)
    elif mode == 'DARKEN':
        new = min(current, current + (target - current) * alpha)
    else:
        # MIX and anything else pulls toward the target
        new = current + (target - current) * alpha
    if target <= 0.0 and new < _SNAP_EPS:
        return 0.0
    if target >= 1.0 and new > 1.0 - _SNAP_EPS:
        return 1.0
    return new


def _deform_group_indices(obj):
    """Groups that auto-normalize spreads weight across (deform bones only)."""
    arm = None
    for mod in obj.modifiers:
        if mod.type == 'ARMATURE' and mod.object:
            arm = mod.object
            break
    if arm is None and obj.parent and obj.parent.type == 'ARMATURE':
        arm = obj.parent
    if arm is None:
        return {vg.index for vg in obj.vertex_groups}
    names = {b.name for b in arm.data.bones if b.use_deform}
    return {vg.index for vg in obj.vertex_groups if vg.name in names}


def _locked_group_indices(obj):
    return {vg.index for vg in obj.vertex_groups if vg.lock_weight}


# ==========================================================================
# The brush (modal operator)
# ==========================================================================

class PAINT_OT_geodesic_weight_brush(bpy.types.Operator):
    bl_idname = "paint.geodesic_weight_brush"
    bl_label = "Surface Draw"
    bl_description = ("Weight brush that falls off along the surface. "
                      "Faces that are not connected are never painted, "
                      "however close they are")
    bl_options = {'REGISTER'}

    radius_scale: FloatProperty(
        name="Radius Scale",
        description="Geodesic radius as a multiple of the brush circle. "
                    "1.0 matches the circle",
        default=1.0, min=0.1, max=4.0,
    )
    use_mirror_x: BoolProperty(
        name="X Mirror",
        description="Paint symmetrically: the mirrored vertex gets the same "
                    "result, written to the left/right-flipped group",
        default=True,
    )
    invert: BoolProperty(
        name="Invert",
        description="Invert the brush (same as Ctrl+drag)",
        default=False,
    )
    smooth: BoolProperty(
        name="Blur",
        description="Blend toward the average of edge-connected neighbours "
                    "(same as Shift+drag). Brush weight and blend mode are ignored",
        default=False,
    )

    # ------------------------------------------------------------------
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == 'MESH'
                and context.mode == 'PAINT_WEIGHT')

    # ------------------------------------------------------------------
    def invoke(self, context, event):
        obj = context.active_object
        me = obj.data

        vg = obj.vertex_groups.active
        if vg is None:
            self.report({'ERROR'}, iface_("No active vertex group"))
            return {'CANCELLED'}
        if vg.lock_weight:
            self.report({'ERROR'}, iface_("Vertex group '%s' is locked") % vg.name)
            return {'CANCELLED'}
        if context.area is None or context.area.type != 'VIEW_3D' \
                or context.region is None or context.region_data is None:
            self.report({'ERROR'}, iface_("Run this in a 3D Viewport"))
            return {'CANCELLED'}

        self.obj = obj
        self.me = me
        self.vg = vg
        self.gidx = vg.index
        self.region = context.region
        self.rv3d = context.region_data
        self.area = context.area

        # X mirror follows the header toggle (mesh.use_mirror_x)
        self.mirror_on = bool(self.use_mirror_x and getattr(me, "use_mirror_x", False))

        # Edge graph and mirror table (cached per mesh)
        self.graph, self.mirror = mesh_graph_cached(me, self.mirror_on)

        # Groups auto-normalize may touch, and groups it must not
        self.deform_idx = _deform_group_indices(obj)
        self.locked_idx = _locked_group_indices(obj)

        # Selection / hide mask
        self.allowed = self._build_mask(me)

        self.mirror_vg = None
        self.mirror_gidx = -1
        if self.mirror_on:
            fname = flip_name(vg.name)
            if fname == vg.name:
                # A centre group such as Spine: mirrored side uses the same group
                self.mirror_vg, self.mirror_gidx = vg, self.gidx
            else:
                mvg = obj.vertex_groups.get(fname)
                if mvg is None:
                    mvg = obj.vertex_groups.new(name=fname)
                    self.report({'INFO'}, iface_("Created vertex group %s") % fname)
                if mvg.lock_weight:
                    self.report({'WARNING'},
                                iface_("Mirror group '%s' is locked, mirror skipped") % mvg.name)
                    self.mirror_on = False
                    self.mirror = None
                else:
                    self.mirror_vg, self.mirror_gidx = mvg, mvg.index

        # Ray-cast against the evaluated mesh when it still matches vertex for
        # vertex (posed armature), otherwise against the base mesh.
        depsgraph = context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(depsgraph)
        self.use_evaluated = (len(ev.data.vertices) == len(me.vertices)
                              and len(ev.data.polygons) == len(me.polygons))

        # Invoked from the toolbar tool: one stroke, ends on release.
        # Invoked from the button: stays resident until Esc.
        self.stroke_mode = (event.type == 'LEFTMOUSE' and event.value == 'PRESS')

        # Defaults from the keymap; in resident mode OR-ed with the modifiers
        self._base_invert = self.invert
        self._base_smooth = self.smooth
        if self.stroke_mode:
            self.invert = self._base_invert or event.ctrl
            self.smooth = self._base_smooth or event.shift

        self.painting = False
        self.last_px = None
        self.mouse = (event.mouse_region_x, event.mouse_region_y)
        self.px_radius = brush_values(context)["size"]
        self.dabs = 0
        self.touched = 0
        self._handle = None

        if self.stroke_mode:
            # The tool's draw_cursor draws the circle, nothing to add here
            self.painting = True
            if not self._safe_dab(context, event, brush_values(context)):
                return {'CANCELLED'}
        else:
            self._handle = bpy.types.SpaceView3D.draw_handler_add(
                self._draw_cursor, (context,), 'WINDOW', 'POST_PIXEL')
            self.area.header_text_set(iface_(
                "Surface Draw - LMB: paint / Ctrl+LMB: invert / "
                "Shift+LMB: blur / Esc, RMB: exit"))
            context.window.cursor_modal_set('PAINT_BRUSH')

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    # ------------------------------------------------------------------
    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            return self._finish(context)

        # Anything that pulls the rug out - mode change, object switched or
        # deleted, file reloaded - ends the brush instead of erroring later.
        try:
            alive = (context.mode == 'PAINT_WEIGHT'
                     and context.active_object == self.obj
                     and self.obj.data == self.me)
        except ReferenceError:
            alive = False
        if not alive:
            return self._finish(context)

        self.mouse = (event.mouse_region_x, event.mouse_region_y)
        bv = brush_values(context)
        self.px_radius = bv["size"]

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.painting = True
                self.invert = self._base_invert or event.ctrl
                self.smooth = self._base_smooth or event.shift
                self.last_px = None
                if not self._safe_dab(context, event, bv):
                    return self._finish(context)
                return {'RUNNING_MODAL'}
            if event.value == 'RELEASE':
                if self.painting:
                    self.painting = False
                    if self.dabs:
                        bpy.ops.ed.undo_push(message="Surface Draw")
                if self.stroke_mode:
                    return self._finish(context)
                return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            if self.painting:
                if not self._safe_dab(context, event, bv):
                    return self._finish(context)
            self.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Navigation and brush shortcuts pass through
        return {'PASS_THROUGH'}

    # ------------------------------------------------------------------
    def _safe_dab(self, context, event, bv):
        """Run one dab; on any error report it and say so instead of leaving
        the modal handler half-dead with the header text and cursor changed."""
        try:
            self._dab(context, event, bv)
            return True
        except Exception as exc:
            self.report({'ERROR'}, iface_("Surface Draw stopped: %s") % exc)
            return False

    def _finish(self, context):
        if self._handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            except Exception:
                pass
            self._handle = None
        if not self.stroke_mode:
            try:
                self.area.header_text_set(None)
            except Exception:
                pass
            context.window.cursor_modal_restore()
            self.report({'INFO'},
                        iface_("Surface Draw finished: %d dabs, %d vertex writes")
                        % (self.dabs, self.touched))
        elif self.painting and self.dabs:
            # Stroke cut short by a mode change etc. - still one undo step
            bpy.ops.ed.undo_push(message="Surface Draw")
        try:
            self.area.tag_redraw()
        except Exception:
            pass
        return {'FINISHED'}

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    @staticmethod
    def _build_mask(me):
        n = len(me.vertices)
        if getattr(me, "use_paint_mask", False):
            sel = np.zeros(len(me.polygons), dtype=bool)
            me.polygons.foreach_get("select", sel)
            allowed = np.zeros(n, dtype=bool)
            for pi in np.nonzero(sel)[0]:
                for vi in me.polygons[pi].vertices:
                    allowed[vi] = True
        elif getattr(me, "use_paint_mask_vertex", False):
            allowed = np.zeros(n, dtype=bool)
            me.vertices.foreach_get("select", allowed)
        else:
            allowed = np.ones(n, dtype=bool)
        # Hidden vertices are never painted, like the standard brush
        hide = np.zeros(n, dtype=bool)
        me.vertices.foreach_get("hide", hide)
        allowed &= ~hide
        return allowed

    # ------------------------------------------------------------------
    # Weight read / write
    # ------------------------------------------------------------------
    def _get_w(self, vi, gidx=None):
        gidx = self.gidx if gidx is None else gidx
        for g in self.me.vertices[vi].groups:
            if g.group == gidx:
                return g.weight
        return 0.0

    def _set_w(self, vi, w, gidx=None):
        gidx = self.gidx if gidx is None else gidx
        if w < 0.0:
            w = 0.0
        elif w > 1.0:
            w = 1.0
        for g in self.me.vertices[vi].groups:
            if g.group == gidx:
                g.weight = w
                return
        # Look the group up by index every time: a VertexGroup reference held
        # across an undo (possible in resident mode) points at freed memory
        # and is not invalidated the way ID references are.
        # VertexGroup.add() rejects numpy.int64 (comes in via the mirror table)
        self.obj.vertex_groups[gidx].add((int(vi),), w, 'REPLACE')

    def _normalize(self, vi, gidx=None):
        """Auto Normalize: scale the other deform groups so the vertex sums to 1.
        Locked groups keep their weight, as in the standard brush."""
        gidx = self.gidx if gidx is None else gidx
        act = None
        others = []
        locked_total = 0.0
        for g in self.me.vertices[vi].groups:
            if g.group == gidx:
                act = g
            elif g.group in self.deform_idx:
                if g.group in self.locked_idx:
                    locked_total += g.weight
                else:
                    others.append(g)
        if act is None:
            return
        # Locked weight is untouchable, so the painted group can only take
        # what is left (the standard brush clamps the same way).
        if act.weight + locked_total > 1.0:
            act.weight = max(0.0, 1.0 - locked_total)
        rest = max(0.0, 1.0 - act.weight - locked_total)
        total = sum(g.weight for g in others)
        if total > 1e-9:
            k = rest / total
            for g in others:
                g.weight = g.weight * k
        elif not others and locked_total <= 0.0:
            act.weight = 1.0

    # ------------------------------------------------------------------
    # One dab
    # ------------------------------------------------------------------
    def _dab(self, context, event, bv):
        region, rv3d = self.region, self.rv3d
        coord = (event.mouse_region_x, event.mouse_region_y)

        px_r = bv["size"]
        if bv["pressure_size"] and event.pressure > 0.0:
            px_r = max(1.0, px_r * event.pressure)

        # Spacing, same idea as the standard brush
        if bv["use_spacing"] and self.last_px is not None:
            step = max(1.0, px_r * 2.0 * bv["spacing"])
            dx = coord[0] - self.last_px[0]
            dy = coord[1] - self.last_px[1]
            if dx * dx + dy * dy < step * step:
                return
        self.last_px = coord

        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

        obj = self.obj
        mw = obj.matrix_world
        mwi = mw.inverted()
        o_l = mwi @ origin
        d_l = (mwi.to_3x3() @ direction).normalized()

        if self.use_evaluated:
            depsgraph = context.evaluated_depsgraph_get()
            ray_obj = obj.evaluated_get(depsgraph)
        else:
            ray_obj = obj
        hit, loc, _nor, face = ray_obj.ray_cast(o_l, d_l)
        if not hit or face < 0 or face >= len(self.me.polygons):
            return

        # Screen-space radius -> object-space distance at the hit point
        p_world = mw @ loc
        scr = view3d_utils.location_3d_to_region_2d(region, rv3d, p_world)
        if scr is None:
            return
        edge_world = view3d_utils.region_2d_to_location_3d(
            region, rv3d, (scr.x + px_r, scr.y), p_world)
        radius = ((mwi @ edge_world) - loc).length * self.radius_scale
        if radius <= 0.0:
            return

        # Seed with the hit face's vertices (smoother than snapping to one)
        ray_me = ray_obj.data if self.use_evaluated else self.me
        seeds = []
        for vi in self.me.polygons[face].vertices:
            d0 = (Vector(ray_me.vertices[vi].co) - loc).length
            if d0 <= radius:
                seeds.append((vi, d0))
        if not seeds:
            return

        reached = self.graph.within(seeds, radius)
        if not reached:
            return

        preset, curve = bv["preset"], bv["curve"]
        mode = bv["blend"]
        target = bv["weight"]
        if self.invert:
            mode, target = flip_blend(mode, target)
        base = bv["strength"]
        if bv["pressure_strength"] and event.pressure > 0.0:
            base *= event.pressure

        auto_norm = context.tool_settings.use_auto_normalize
        allowed = self.allowed
        mirror = self.mirror

        # Blur pulls toward the average of the edge-connected neighbours.
        # Reading while writing would make the result order-dependent, so the
        # weights in range are snapshotted first.
        snap = None
        if self.smooth:
            start, dstv = self.graph.start, self.graph.dst
            snap = {}
            for vi in reached:
                if vi not in snap:
                    snap[vi] = self._get_w(vi)
                for k in range(start[vi], start[vi + 1]):
                    u = dstv[k]
                    if u not in snap:
                        snap[u] = self._get_w(u)

        changed = []
        for vi, dist in reached.items():
            if not allowed[vi]:
                continue
            f = falloff_strength(preset, curve, dist / radius)
            alpha = base * f
            if alpha <= 0.0:
                continue
            cur = self._get_w(vi)
            if snap is not None:
                start, dstv = self.graph.start, self.graph.dst
                total = snap[vi]
                cnt = 1
                for k in range(start[vi], start[vi + 1]):
                    total += snap[dstv[k]]
                    cnt += 1
                new = cur + (total / cnt - cur) * alpha
            else:
                new = blend_weight(mode, cur, target, alpha)
            if new < 0.0:
                new = 0.0
            elif new > 1.0:
                new = 1.0
            # Skip only exact no-ops. A "difference below 1e-6" cut-off would
            # keep the last speck from ever snapping to 0.
            if new == cur:
                continue
            self._set_w(vi, new)
            changed.append((vi, self.gidx))
            if mirror is not None:
                mj = int(mirror[vi])
                # The mirrored vertex gets the name-flipped group; writing the
                # same group would put Shoulder.L on the other shoulder.
                if mj >= 0 and mj != vi and allowed[mj]:
                    self._set_w(mj, new, self.mirror_gidx)
                    changed.append((mj, self.mirror_gidx))

        if not changed:
            return

        if auto_norm:
            for vi, gi in changed:
                self._normalize(vi, gi)

        self.dabs += 1
        self.touched += len(changed)
        self.me.update()
        obj.update_tag()
        self.area.tag_redraw()

    # ------------------------------------------------------------------
    # Cursor circle (resident mode only; the tool draws its own)
    # ------------------------------------------------------------------
    def _draw_cursor(self, context):
        if context.region != self.region:
            return
        color = (1.0, 0.35, 0.35, 0.9) if self.painting else (1.0, 1.0, 1.0, 0.65)
        _draw_circle_px(self.mouse[0], self.mouse[1], self.px_radius, color)


# ==========================================================================
# Surface Gradient - grow from the painted area in one go
# ==========================================================================

class PAINT_OT_geodesic_weight_spread(bpy.types.Operator):
    bl_idname = "paint.geodesic_weight_spread"
    bl_label = "Surface Gradient"
    bl_description = ("Use the vertices that already carry weight as seeds "
                      "and fall off outward along the surface")
    bl_options = {'REGISTER', 'UNDO'}

    distance: FloatProperty(
        name="Distance",
        description="Distance from the seed edge at which the weight reaches 0",
        default=0.05, min=0.0001, max=10.0, subtype='DISTANCE',
    )
    threshold: FloatProperty(
        name="Seed Threshold",
        description="Weights at or above this count as seeds",
        default=0.999, min=0.0, max=1.0,
    )
    falloff: EnumProperty(
        name="Falloff", default='SMOOTH',
        items=[('SMOOTH', "Smooth", ""), ('LIN', "Linear", ""),
               ('SHARP', "Sharp", ""), ('ROOT', "Root", ""),
               ('INVSQUARE', "Inverse Square", "")],
    )
    keep_larger: BoolProperty(
        name="Keep Larger", default=True,
        description="Keep the existing weight where it is larger",
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == 'MESH'
                and obj.vertex_groups.active is not None
                and context.mode in {'PAINT_WEIGHT', 'OBJECT'})

    def execute(self, context):
        obj = context.active_object
        me = obj.data
        vg = obj.vertex_groups.active
        gidx = vg.index
        if vg.lock_weight:
            self.report({'ERROR'}, iface_("Vertex group '%s' is locked") % vg.name)
            return {'CANCELLED'}

        seeds = []
        for v in me.vertices:
            for g in v.groups:
                if g.group == gidx and g.weight >= self.threshold:
                    seeds.append((v.index, 0.0))
                    break
        if not seeds:
            self.report({'ERROR'}, iface_(
                "No seeds found. Paint weights at or above the threshold first"))
            return {'CANCELLED'}

        graph, _ = mesh_graph_cached(me, False)
        reached = graph.within(seeds, self.distance)

        deform = _deform_group_indices(obj)
        locked = _locked_group_indices(obj)
        auto_norm = context.tool_settings.use_auto_normalize
        n = 0
        for vi, dist in reached.items():
            w = falloff_strength(self.falloff, None, dist / self.distance)
            cur = 0.0
            elem = None
            for g in me.vertices[vi].groups:
                if g.group == gidx:
                    cur = g.weight
                    elem = g
                    break
            if self.keep_larger and cur >= w:
                continue
            if elem is not None:
                elem.weight = w
            else:
                vg.add((int(vi),), w, 'REPLACE')
            n += 1

        if auto_norm:
            for vi in reached.keys():
                act = None
                others = []
                locked_total = 0.0
                for g in me.vertices[vi].groups:
                    if g.group == gidx:
                        act = g
                    elif g.group in deform:
                        if g.group in locked:
                            locked_total += g.weight
                        else:
                            others.append(g)
                if act is None:
                    continue
                if act.weight + locked_total > 1.0:
                    act.weight = max(0.0, 1.0 - locked_total)
                rest = max(0.0, 1.0 - act.weight - locked_total)
                total = sum(g.weight for g in others)
                if total > 1e-9:
                    k = rest / total
                    for g in others:
                        g.weight = g.weight * k
                elif not others and locked_total <= 0.0:
                    act.weight = 1.0

        me.update()
        obj.update_tag()
        self.report({'INFO'}, iface_("Updated %d vertices (%d reached)") % (n, len(reached)))
        return {'FINISHED'}


# ==========================================================================
# Sidebar panel
# ==========================================================================

class VIEW3D_PT_geodesic_weight(bpy.types.Panel):
    bl_label = "Surface Draw"
    bl_idname = "VIEW3D_PT_geodesic_weight"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Surface Draw"

    @classmethod
    def poll(cls, context):
        return context.mode in {'PAINT_WEIGHT', 'OBJECT'}

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        box0 = layout.box()
        box0.label(text="Toolbar tool: Surface Draw", icon='TOOL_SETTINGS')

        col = layout.column(align=True)
        col.scale_y = 1.4
        col.operator("paint.geodesic_weight_brush",
                     text="Start Resident Mode", icon='BRUSH_DATA')
        col.enabled = (context.mode == 'PAINT_WEIGHT')

        box = layout.box()
        box.label(text="Uses the standard brush settings", icon='INFO')
        bv = brush_values(context)
        row = box.row(align=True)
        row.label(text=iface_("Radius %d px") % bv["size"])
        row.label(text=iface_("Strength %.2f") % bv["strength"])
        row = box.row(align=True)
        row.label(text=iface_("Weight %.2f") % bv["weight"])
        row.label(text=bv["blend"])

        layout.separator()
        layout.label(text="Grow from the painted area")
        layout.operator("paint.geodesic_weight_spread",
                        text="Surface Gradient", icon='MOD_SMOOTH')

        if obj and obj.type == 'MESH' and obj.vertex_groups.active:
            vg = obj.vertex_groups.active
            layout.label(text=iface_("Target: %s") % vg.name,
                         icon='LOCKED' if vg.lock_weight else 'GROUP_VERTEX')
        else:
            layout.label(text="No active vertex group", icon='ERROR')


# ==========================================================================
# Toolbar tool
# ==========================================================================

_TOOL_SHADER = [None]


def _draw_circle_px(x, y, r, color):
    if _TOOL_SHADER[0] is None:
        _TOOL_SHADER[0] = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader = _TOOL_SHADER[0]
    pts = []
    seg = 48
    prev = (x + r, y)
    for i in range(1, seg + 1):
        t = i * math.tau / seg
        cur = (x + r * math.cos(t), y + r * math.sin(t))
        pts.append(prev)
        pts.append(cur)
        prev = cur
    batch = batch_for_shader(shader, 'LINES', {"pos": pts})
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set('NONE')


def _tool_supports_use_brushes():
    """Does WorkSpaceTool.setup() accept USE_BRUSHES (Blender 4.3+)?

    With it, Blender itself draws the standard brush header for the tool
    (brush selector, Weight, Radius, Strength, Brush/Stroke/Falloff/Cursor).
    """
    try:
        fn = bpy.types.WorkSpaceTool.bl_rna.functions["setup"]
        return "USE_BRUSHES" in fn.parameters["options"].enum_items.keys()
    except Exception:
        return False


_USE_BRUSHES = _tool_supports_use_brushes()


class GeodesicWeightTool(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'PAINT_WEIGHT'
    bl_idname = TOOL_IDNAME
    bl_label = "Surface Draw"
    bl_description = ("Weight brush that falls off along the surface.\n"
                      "Faces that are not connected are never painted, "
                      "however close they are")
    bl_icon = "brush.paint_weight.draw"
    bl_cursor = 'PAINT_BRUSH'
    bl_widget = None
    # Let Blender draw the standard brush header (brush selector, Weight,
    # Radius, Strength, Brush/Stroke/Falloff/Cursor) rather than imitating it.
    # 4.3+: USE_BRUSHES with no brush type (ANY) - whatever brush is selected
    #       is the one whose settings we read.
    # 4.2:  a tool with a brush data-block; 'DRAW' keeps the Draw brush active.
    #       Without it the brush popovers' poll fails and they draw greyed out.
    if _USE_BRUSHES:
        bl_options = {'USE_BRUSHES'}
    else:
        bl_data_block = 'DRAW'
    bl_keymap = (
        ("paint.geodesic_weight_brush",
         {"type": 'LEFTMOUSE', "value": 'PRESS'},
         {"properties": [("invert", False)]}),
        ("paint.geodesic_weight_brush",
         {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True},
         {"properties": [("invert", True)]}),
        ("paint.geodesic_weight_brush",
         {"type": 'LEFTMOUSE', "value": 'PRESS', "shift": True},
         {"properties": [("smooth", True)]}),
    )

    @staticmethod
    def draw_cursor(context, tool, xy):
        bv = brush_values(context)
        _draw_circle_px(xy[0], xy[1], bv["size"], (1.0, 1.0, 1.0, 0.65))

    # No draw_settings: Blender draws the brush header itself (see above).


def _pick_icon():
    """Choose an icon that exists; a missing one leaves the button blank."""
    import os
    try:
        root = bpy.utils.system_resource('DATAFILES', path="icons")
    except Exception:
        return GeodesicWeightTool.bl_icon
    for name in ("brush.paint_weight.draw",
                 "ops.paint.weight_gradient",
                 "brush.paint_texture.draw",
                 "ops.generic.select_circle"):
        if os.path.isfile(os.path.join(root, name + ".dat")):
            return name
    return "ops.generic.select_circle"


# Where to slot the tool. Names differ between versions, so try in order.
# register_tool does not raise on a missing anchor - it warns and appends -
# so only an anchor that is really present gets passed.
_TOOL_ANCHORS = ("builtin_brush.smear", "builtin_brush.average",
                 "builtin_brush.blur", "builtin.brush", "builtin.gradient")


def _existing_tool_idnames():
    try:
        from bl_ui.space_toolsystem_toolbar import VIEW3D_PT_tools_active as T
    except Exception:
        return set()
    out = set()

    def walk(node):
        if node is None or callable(node):
            return
        idn = getattr(node, "idname", None)
        if idn is not None:
            out.add(idn)
            return
        if isinstance(node, (tuple, list)):
            for x in node:
                walk(x)

    walk(T._tools.get('PAINT_WEIGHT', ()))
    return out


# ==========================================================================
# Translations (ja_JP). Source strings are English; Blender substitutes these
# when the interface language is Japanese.
# ==========================================================================

_JA = {
    # tool / operator
    "Weight brush that falls off along the surface. Faces that are not "
    "connected are never painted, however close they are":
        "辺づたいの距離で減衰するウェイトブラシ。地続きでない面は近くても塗られない",
    "Weight brush that falls off along the surface.\nFaces that are not "
    "connected are never painted, however close they are":
        "辺づたいの距離で減衰するウェイトブラシ。\n地続きでない面は近くても塗られない",
    "Radius Scale": "半径の倍率",
    "Geodesic radius as a multiple of the brush circle. 1.0 matches the circle":
        "ブラシ円に対する測地半径の倍率。1.0 で円の見た目どおり",
    "X Mirror": "X ミラー",
    "Paint symmetrically: the mirrored vertex gets the same result, written "
    "to the left/right-flipped group":
        "左右対称に塗る。鏡像の頂点に同じ結果を、名前を左右反転したグループへ書く",
    "Invert": "反転",
    "Invert the brush (same as Ctrl+drag)": "ブラシを反転する（Ctrl+ドラッグ相当）",
    "Blur": "ぼかし",
    "Blend toward the average of edge-connected neighbours (same as "
    "Shift+drag). Brush weight and blend mode are ignored":
        "辺でつながった隣接頂点の平均へ寄せる（Shift+ドラッグ相当）。"
        "ブラシのウェイトとブレンドは無視される",
    "No active vertex group": "アクティブな頂点グループがありません",
    "Vertex group '%s' is locked": "頂点グループ '%s' はロックされています",
    "Mirror group '%s' is locked, mirror skipped":
        "ミラー先のグループ '%s' がロックされているのでミラーは行いません",
    "Run this in a 3D Viewport": "3D ビューで実行してください",
    "Created vertex group %s": "頂点グループ %s を作成しました",
    "Surface Draw - LMB: paint / Ctrl+LMB: invert / Shift+LMB: blur / Esc, RMB: exit":
        "Surface Draw — 左:塗る / Ctrl+左:反転 / Shift+左:ぼかし / Esc・右クリック:終了",
    "Surface Draw stopped: %s": "Surface Draw を中断しました: %s",
    "Surface Draw finished: %d dabs, %d vertex writes":
        "Surface Draw 終了  打点 %d 回 / 延べ %d 頂点",
    # gradient
    "Use the vertices that already carry weight as seeds and fall off "
    "outward along the surface":
        "いまウェイトが乗っている範囲を種にして、辺づたいの距離で外側へ減衰させる",
    "Distance": "距離",
    "Distance from the seed edge at which the weight reaches 0":
        "種の縁から何メートル先で 0 になるか",
    "Seed Threshold": "種のしきい値",
    "Weights at or above this count as seeds": "この値以上を種とみなす",
    "Falloff": "減衰",
    "Smooth": "スムーズ",
    "Linear": "リニア",
    "Sharp": "シャープ",
    "Root": "ルート",
    "Inverse Square": "反転二乗",
    "Keep Larger": "大きい方を残す",
    "Keep the existing weight where it is larger": "既存ウェイトと比べて大きい方を採用する",
    "No seeds found. Paint weights at or above the threshold first":
        "種が見つかりません。しきい値以上のウェイトを塗ってから実行してください",
    "Updated %d vertices (%d reached)": "%d 頂点を更新（到達 %d 頂点）",
    # panel
    "Toolbar tool: Surface Draw": "本体はツールバーの「Surface Draw」",
    "Start Resident Mode": "常駐モードで開始",
    "Uses the standard brush settings": "ブラシ設定は標準のものを使います",
    "Radius %d px": "半径 %d px",
    "Strength %.2f": "強さ %.2f",
    "Weight %.2f": "ウェイト %.2f",
    "Grow from the painted area": "塗った範囲から広げる",
    "Target: %s": "対象: %s",
}


def _translation_dict():
    ja = {}
    for en, jp in _JA.items():
        # "*" is the default context; operator labels/descriptions are looked
        # up under "Operator", so register both.
        ja[("*", en)] = jp
        ja[("Operator", en)] = jp
    return {"ja_JP": ja}


# ==========================================================================
# Registration
# ==========================================================================

classes = (
    PAINT_OT_geodesic_weight_brush,
    PAINT_OT_geodesic_weight_spread,
    VIEW3D_PT_geodesic_weight,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)

    try:
        bpy.app.translations.register(__name__, _translation_dict())
    except Exception as exc:
        print("[Surface Draw] translations not registered:", exc)

    GeodesicWeightTool.bl_icon = _pick_icon()
    present = _existing_tool_idnames()
    anchor = next((a for a in _TOOL_ANCHORS if a in present), None)
    if anchor:
        bpy.utils.register_tool(GeodesicWeightTool, after={anchor}, separator=True)
    else:
        bpy.utils.register_tool(GeodesicWeightTool, separator=True)


def unregister():
    try:
        bpy.utils.unregister_tool(GeodesicWeightTool)
    except Exception:
        pass
    try:
        bpy.app.translations.unregister(__name__)
    except Exception:
        pass
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
    _CACHE.clear()
