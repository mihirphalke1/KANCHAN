"""
Region-based colouring for the generated white mesh.

The image→3D Space (Hunyuan3D-2 `/shape_generation`) returns an untextured
*white* GLB. Rather than projecting the photograph onto the mesh as a texture
(which warps badly around a bezel and never lines up), we colour the model the
way a jeweller would describe it: the metal body gets one gold material, the
set stone gets its own gem material. Two flat PBR materials, assigned to two
geometric regions — no photo mask baked onto the surface.

Pipeline:
  1. Segment the set-stone cap from the band purely from geometry (a bezel
     stone is a thick, dense blob protruding off an otherwise thin band).
  2. Sample the ACTUAL gold tone and the ACTUAL stone tone from the case
     photograph so the colours match the real item, not a generic swatch.
  3. Export a 2-material GLB (metal + stone). Falls back to an all-gold model
     when no stone was detected — which also fixes plain-metal items that would
     otherwise render as flat white.

Additive / non-decision path: colours are cosmetic and never touch fusion or
loan risk. Every failure is soft — the original white GLB is left untouched.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Gold body fallback (sRGB 0-255) when the photo can't be sampled.
DEFAULT_GOLD = (207, 165, 60)
# Per-hue gem fallbacks (sRGB 0-255) keyed by the detector's hue_class.
STONE_FALLBACK = {
    "red":        (176, 22, 40),     # ruby / garnet
    "blue":       (28, 58, 150),     # sapphire
    "green":      (16, 120, 74),     # emerald
    "other":      (140, 70, 140),    # generic coloured stone
    "colourless": (222, 226, 235),   # diamond / white sapphire
}


def _srgb01(rgb) -> list[float]:
    return [max(0.0, min(1.0, c / 255.0)) for c in rgb]


# ── Geometry: isolate the set-stone's outward FACE ───────────────────────────
def _segment_stone_faces(mesh) -> np.ndarray:
    """Boolean mask over FACES marking the visible face of the set stone.

    A set stone (bezel or prong) sits at the top of the ring and faces
    outward — its table is the flat cap at the far end of the setting, with a
    normal pointing away from the finger. Colouring the whole setting cup red
    is wrong (the cup walls and rim are metal); only the outward table is the
    gem. So we:

      1. Find the setting by its bulk — the one thick, dense blob on an
         otherwise thin band — and take its centroid to fix the stone axis
         `a` (ring centre → setting).
      2. Keep the faces at the far end along `a` whose normals also point
         along `a`: that is the outward table, not the metal side walls.
      3. Keep the single largest connected patch so stray outward faces
         elsewhere (clasp, shank curve) aren't painted.
    """
    from scipy.spatial import cKDTree

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(faces) == 0:
        return np.zeros(0, dtype=bool)

    finite = np.isfinite(vertices).all(axis=1)
    center = vertices[finite].mean(axis=0) if finite.any() else np.zeros(3)
    v = np.nan_to_num(vertices - center, nan=0.0, posinf=0.0, neginf=0.0)
    cov = np.cov(v[finite].T) if finite.any() else np.eye(3)
    _, vec = np.linalg.eigh(np.nan_to_num(cov))
    n = vec[:, 0]                                # finger-hole axis
    z = v @ n
    scale = float(np.linalg.norm(vertices.max(0) - vertices.min(0))) or 1.0

    # 1. Locate the setting blob: thick (finger axis) + locally dense.
    tree = cKDTree(vertices)
    balls = tree.query_ball_point(vertices, 0.06 * scale)
    z_range = np.array([(z[b].max() - z[b].min()) if b else 0.0 for b in balls])
    density = np.array([len(b) for b in balls], dtype=float)

    def _norm(a):
        rng = np.ptp(a)
        return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)

    blob_score = 0.7 * _norm(z_range) + 0.3 * _norm(density)
    blob = blob_score > np.percentile(blob_score, 85)
    if blob.sum() < 20:
        return np.zeros(len(faces), dtype=bool)

    # 2. Stone axis + outward table selection.
    a = v[blob].mean(axis=0)
    na = np.linalg.norm(a)
    if na < 1e-6:
        return np.zeros(len(faces), dtype=bool)
    a = a / na
    s = v @ a                                    # extent toward the stone
    fn = mesh.face_normals
    fc = vertices[faces].mean(axis=1) - center
    cap_depth = 0.11 * scale                     # how deep the table slice runs
    align_min = 0.35                             # face must look outward
    gem = (fc @ a > s.max() - cap_depth) & (fn @ a > align_min)
    if gem.sum() < 20:
        # Relax once for shallow / prong settings.
        gem = (fc @ a > s.max() - 0.16 * scale) & (fn @ a > 0.2)
    if gem.sum() < 20:
        return np.zeros(len(faces), dtype=bool)

    # 3. Largest connected outward patch = the table.
    sub = mesh.submesh([np.where(gem)[0]], only_watertight=False, append=True)
    parts = sorted(sub.split(only_watertight=False), key=lambda p: len(p.faces), reverse=True)
    if not parts:
        return np.zeros(len(faces), dtype=bool)
    keep = parts[0]
    kt = cKDTree(keep.triangles_center)
    d, _ = kt.query(vertices[faces].mean(axis=1))
    return d < 1e-6


# ── Colour sampling from the case photograph ─────────────────────────────────
def _sample_colours(image_path: Optional[Path], stones: list[dict]) -> tuple[tuple, Optional[tuple], Optional[str]]:
    """Return (gold_rgb, stone_rgb_or_None, stone_hue_class_or_None), sampling
    the real tones from the photo. Stone colour comes from the detected stone
    bbox; gold from the goldish pixels outside every stone."""
    gold = DEFAULT_GOLD
    stone_rgb = None
    hue_class = None

    # Pick the largest confirmed COLOURED stone as the representative gem.
    coloured = [
        s for s in (stones or [])
        if s.get("hue_class") and s.get("hue_class") != "colourless"
        and s.get("status", "confirmed") != "rejected"
    ]
    target = max(coloured, key=lambda s: s.get("area_pct", 0.0)) if coloured else None
    if target is None:
        # A colourless stone still deserves its own material.
        colourless = [s for s in (stones or []) if s.get("hue_class") == "colourless"]
        target = max(colourless, key=lambda s: s.get("area_pct", 0.0)) if colourless else None
    if target is not None:
        hue_class = target.get("hue_class")
        stone_rgb = STONE_FALLBACK.get(hue_class, STONE_FALLBACK["other"])

    if image_path and image_path.exists():
        try:
            import cv2
            bgr = cv2.imread(str(image_path))
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                H, W = rgb.shape[:2]
                stone_mask = np.zeros((H, W), dtype=bool)
                for s in (stones or []):
                    bb = s.get("bbox")
                    if not bb or len(bb) != 4:
                        continue
                    x, y, w, h = [int(round(t)) for t in bb]
                    x0, y0 = max(0, x), max(0, y)
                    x1, y1 = min(W, x + w), min(H, y + h)
                    if x1 > x0 and y1 > y0:
                        stone_mask[y0:y1, x0:x1] = True

                # Stone colour: median of the inner 60% of the target bbox
                # (avoids the bezel rim and facet glints at the edges).
                if target is not None and target.get("bbox"):
                    x, y, w, h = [int(round(t)) for t in target["bbox"]]
                    mx, my = int(w * 0.2), int(h * 0.2)
                    x0, y0 = max(0, x + mx), max(0, y + my)
                    x1, y1 = min(W, x + w - mx), min(H, y + h - my)
                    if x1 > x0 and y1 > y0:
                        patch = rgb[y0:y1, x0:x1].reshape(-1, 3)
                        # Drop specular highlights so a bright facet flash
                        # doesn't wash the gem toward white.
                        keep = patch[patch.max(axis=1) < 245]
                        patch = keep if len(keep) > 20 else patch
                        stone_rgb = tuple(int(c) for c in np.median(patch, axis=0))

                # Gold: goldish pixels (warm hue, decent saturation, bright)
                # anywhere outside the stone boxes.
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
                goldish = (hh >= 10) & (hh <= 40) & (ss >= 40) & (vv >= 60) & (~stone_mask)
                if goldish.sum() > 200:
                    gold = tuple(int(c) for c in np.median(rgb[goldish], axis=0))
        except Exception as e:
            logger.info("colour sampling from photo failed (%s) — using defaults", e)

    return gold, stone_rgb, hue_class


# ── Metal type: yellow / rose / white gold from the sampled tone ─────────────
def _metal_type(gold_rgb) -> tuple[str, float, float]:
    """Classify the metal from its sampled colour → (label, metalness,
    roughness). White gold is far more metallic (silvery sheen); yellow/rose
    keep a strong warm diffuse so they read gold in any lighting."""
    import colorsys
    r, g, b = [c / 255.0 for c in gold_rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hd = h * 360.0
    if s < 0.18 and v > 0.55:
        return "white", 0.85, 0.25
    if (hd < 22 or hd > 335) and s >= 0.18:
        return "rose", 0.60, 0.34
    return "yellow", 0.55, 0.38


# ── Baked shading: photographic depth as per-vertex colours ──────────────────
def _bake_vertex_colours(mesh, gem_vmask, gold_rgb, stone_rgb) -> np.ndarray:
    """Per-vertex RGBA that gives the flat regions photographic depth without
    any texture: an ambient-occlusion proxy darkens crevices (under the band,
    around the setting), and the set stone gets a bright centre table fading to
    darker edges like a real cut gem. Colours are the true tones sampled from
    the photo; this only modulates their brightness by geometry."""
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    vn = np.asarray(mesh.vertex_normals)
    scale = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0

    def _n01(a):
        rng = np.ptp(a)
        return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)

    # Ambient-occlusion proxy: a vertex whose neighbours sit outward of it along
    # its own normal is tucked into a concavity → more occluded → darker.
    tree = cKDTree(V)
    balls = tree.query_ball_point(V, 0.05 * scale)
    ao = np.zeros(len(V))
    for i, b in enumerate(balls):
        if len(b) >= 3:
            ao[i] = float(np.dot(V[b].mean(0) - V[i], vn[i]))
    ao = _n01(ao)
    shade = (1.0 - 0.5 * ao)[:, None]           # crevices up to 50% darker

    gold = np.array(_srgb01(gold_rgb))
    col = np.tile(gold, (len(V), 1)) * shade

    if gem_vmask is not None and gem_vmask.any() and stone_rgb is not None:
        gem = np.array(_srgb01(stone_rgb))
        gc = V[gem_vmask].mean(0)
        d = np.linalg.norm(V - gc, axis=1)
        dg = 1.0 - _n01(np.where(gem_vmask, d, d.max()))   # 1 at table centre
        col[gem_vmask] = gem * (0.70 + 0.55 * dg[gem_vmask, None])

    col = np.clip(col, 0, 1)
    rgba = np.empty((len(V), 4), dtype=np.uint8)
    rgba[:, :3] = (col * 255).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def _assign_pbr_materials(glb_path: Path, specs: dict) -> None:
    """trimesh exports vertex colours (COLOR_0) but leaves the material null,
    so a viewer would default it to full metalness and wash out. Attach a real
    PBR material per mesh (matched by name) with baseColorFactor white so the
    vertex colours show through, plus the right metalness/roughness/emissive."""
    from pygltflib import GLTF2, Material, PbrMetallicRoughness

    g = GLTF2().load(str(glb_path))
    if g.materials is None:
        g.materials = []
    name_to_idx = {}
    for name, s in specs.items():
        g.materials.append(Material(
            name=name,
            doubleSided=True,
            emissiveFactor=s.get("emissive", [0.0, 0.0, 0.0]),
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=s["metallic"],
                roughnessFactor=s["roughness"],
            ),
        ))
        name_to_idx[name] = len(g.materials) - 1
    default_idx = name_to_idx.get("metal", 0)
    for mesh in g.meshes:
        idx = name_to_idx.get(mesh.name, default_idx)
        for prim in mesh.primitives:
            prim.material = idx
    g.save(str(glb_path))


# ── Public entry point ───────────────────────────────────────────────────────
def colorize_glb(
    glb_path: Path,
    out_path: Path,
    stones: Optional[list[dict]] = None,
    image_path: Optional[Path] = None,
) -> dict:
    """Write a colourised copy of `glb_path` to `out_path`.

    Predictive region colouring: the true metal tone + metal type and the true
    stone colour are sampled from the photo, applied to the geometric regions
    (metal body vs the set stone's outward table), and given photographic depth
    with baked ambient-occlusion + gem-table shading. No photo is projected or
    textured onto the surface — the model is coloured as the item looks, in
    real 3D. Raises on hard failure so the caller can keep the white model.
    """
    import trimesh
    from trimesh.visual.color import ColorVisuals

    scene = trimesh.load(str(glb_path), process=False)
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values()])
    else:
        mesh = scene
    if mesh is None or len(mesh.faces) == 0:
        raise ValueError("Empty mesh — nothing to colour")

    faces = np.asarray(mesh.faces)
    gold, stone_rgb, hue_class = _sample_colours(image_path, stones or [])
    metal_label, metalness, roughness = _metal_type(gold)

    have_stone = bool(stones) and stone_rgb is not None
    stone_face_mask = np.zeros(len(faces), dtype=bool)
    if have_stone:
        stone_face_mask = _segment_stone_faces(mesh)
        if stone_face_mask.sum() < 20:
            have_stone = False

    gem_vmask = np.zeros(len(mesh.vertices), dtype=bool)
    if have_stone and stone_face_mask.any():
        gem_vmask[faces[stone_face_mask].reshape(-1)] = True

    vcolours = _bake_vertex_colours(mesh, gem_vmask if have_stone else None, gold, stone_rgb)
    mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=vcolours)

    out = trimesh.Scene()
    specs = {"metal": {"metallic": metalness, "roughness": roughness}}
    if have_stone and stone_face_mask.any():
        metal = mesh.submesh([np.where(~stone_face_mask)[0]], only_watertight=False, append=True)
        gem = mesh.submesh([np.where(stone_face_mask)[0]], only_watertight=False, append=True)
        out.add_geometry(metal, geom_name="metal")
        out.add_geometry(gem, geom_name="stone")
        specs["stone"] = {
            "metallic": 0.0, "roughness": 0.12,
            "emissive": [c * 0.12 for c in _srgb01(stone_rgb)],
        }
    else:
        out.add_geometry(mesh, geom_name="metal")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(out_path), file_type="glb")
    try:
        _assign_pbr_materials(out_path, specs)
    except Exception as e:
        logger.warning("PBR material assignment failed (%s) — vertex colours only", e)

    return {
        "gold_rgb": list(gold),
        "stone_rgb": list(stone_rgb) if (have_stone and stone_rgb) else None,
        "stone_hue_class": hue_class if have_stone else None,
        "metal_type": metal_label,
        "stone_face_pct": round(100.0 * float(stone_face_mask.mean()), 2) if have_stone else 0.0,
        "shaded": True,
        "colored": True,
    }
