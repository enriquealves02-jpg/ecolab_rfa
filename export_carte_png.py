"""
export_carte_png.py — Export PNG de la carte avec les couleurs de score
=======================================================================
Génère ecolab_carte_score.png dans le dossier ecolab/
Même palette que le Streamlit (seuils quantiles QGIS réels).
"""
import os, sys, time
import sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from shapely import wkb
import struct

sys.path.insert(0, os.path.dirname(__file__))
import config_v2 as config

GPKG      = config.CARTE_FINALE_GPKG
LAYER     = "COSIA_SCORE_finale"
OUT_PNG   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ecolab_carte_score.png")
SIMPLIFY  = 20   # tolérance simplification (mètres Lambert 93) — augmenter si lent
DPI       = 200

# Seuils calés sur les quantiles réels
BREAKS = [-0.531, -0.465, -0.396, -0.297]
COLORS = ["#1A7A2E", "#7EC850", "#F5E642", "#F5A623", "#D0021B"]
LABELS = ["Très protecteur", "Protecteur", "Neutre", "Aménageur", "Très aménageur"]

def score_to_color(s):
    if s is None: return "#AAAAAA"
    if s < BREAKS[0]: return COLORS[0]
    if s < BREAKS[1]: return COLORS[1]
    if s < BREAKS[2]: return COLORS[2]
    if s < BREAKS[3]: return COLORS[3]
    return COLORS[4]

def parse_geom(blob):
    """Parse GPKG geometry blob → shapely geometry."""
    if blob is None: return None
    try:
        # GPKG header: 2 magic + 1 version + 1 flags + 4 srs_id = 8 bytes
        flags = blob[3]
        env_code = (flags >> 1) & 7
        env_sizes = [0, 32, 48, 48, 64]
        env_size = env_sizes[env_code] if env_code < len(env_sizes) else 0
        wkb_offset = 8 + env_size
        return wkb.loads(bytes(blob[wkb_offset:]))
    except Exception:
        return None

print("=" * 60)
print("EXPORT CARTE PNG")
print("=" * 60)
print(f"GPKG : {os.path.basename(GPKG)}")
print(f"Simplification : {SIMPLIFY}m  |  DPI : {DPI}")
print()

# Lecture depuis SQLite
uri = "file:" + GPKG.replace("\\", "/") + "?mode=ro"
conn = sqlite3.connect(uri, uri=True)
conn.execute("PRAGMA cache_size = -131072")

print("Lecture des géométries (peut prendre 2-5 min)...")
t0 = time.time()
rows = conn.execute(
    f'SELECT score_polygone, geom FROM "{LAYER}" WHERE geom IS NOT NULL'
).fetchall()
conn.close()
print(f"  {len(rows):,} polygones lus ({time.time()-t0:.1f}s)")

# Parse + simplify + collect
print("Simplification et rendu...")
from shapely.geometry import mapping
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
import matplotlib.colors as mcolors

fig, ax = plt.subplots(figsize=(20, 16), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
ax.set_aspect("equal")

t1 = time.time()
skipped = 0
patches_by_color = {c: [] for c in COLORS + ["#AAAAAA"]}

for score, blob in rows:
    geom = parse_geom(blob)
    if geom is None or geom.is_empty:
        skipped += 1
        continue
    geom = geom.simplify(SIMPLIFY, preserve_topology=False)
    if geom is None or geom.is_empty:
        skipped += 1
        continue
    color = score_to_color(score)
    # Gérer polygons et multipolygons
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms) if hasattr(geom, "geoms") else []
    for poly in polys:
        if poly.is_empty: continue
        coords = list(poly.exterior.coords)
        if len(coords) < 3: continue
        xs, ys = zip(*coords)
        patches_by_color[color].append((xs, ys))

print(f"  Géométries parsées ({time.time()-t1:.1f}s)  skipped={skipped:,}")

# Dessin groupé par couleur (plus rapide)
print("Rendu matplotlib...")
t2 = time.time()
for color, polys in patches_by_color.items():
    if not polys: continue
    for xs, ys in polys:
        ax.fill(xs, ys, color=color, linewidth=0, antialiased=False)

print(f"  Rendu OK ({time.time()-t2:.1f}s)")

# Légende
legend_patches = [
    mpatches.Patch(color=COLORS[i], label=f"{LABELS[i]}")
    for i in range(len(LABELS))
]
legend_patches.append(mpatches.Patch(color="#AAAAAA", label="Non scoré"))
ax.legend(
    handles=legend_patches,
    loc="lower left",
    fontsize=11,
    framealpha=0.85,
    facecolor="#f5f5f5",
    title="Score polygone",
    title_fontsize=12,
)

ax.set_title("EcoLab — Score COSIA par parcelle\nSCoT Drôme–Ardèche–Vaucluse",
             color="white", fontsize=16, pad=12)
ax.axis("off")
plt.tight_layout(pad=0.5)

print(f"Sauvegarde → {OUT_PNG}")
plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nOK — {os.path.getsize(OUT_PNG) / 1e6:.1f} Mo")
print(f"Fichier : {OUT_PNG}")
