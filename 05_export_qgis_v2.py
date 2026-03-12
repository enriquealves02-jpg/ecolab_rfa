"""
EcoLab Pipeline V2 — Étape 5 : Export QGIS carte finale
=========================================================
Génère un seul fichier GPKG prêt à ouvrir dans QGIS :

  COSIA_BD_ecolab_carte_finale.gpkg   ← couches par quintile + score complet + tuiles
  COSIA_BD_ecolab_carte_finale.qml    ← style QGIS auto-chargé (même nom = auto-load)

Couches dans le GPKG :
  SCORE_COMPLET         ← tous les polygones avec score_polygone + quintile
  Q1_tres_protecteur    ← polygones quintile 1
  Q2_protecteur         ← polygones quintile 2
  Q3_neutre             ← polygones quintile 3
  Q4_amenageur          ← polygones quintile 4
  Q5_tres_amenageur     ← polygones quintile 5
  TUILES_SCORE          ← grille 1km×1km avec score moyen pondéré

Palette (du plus protecteur au plus aménageur) :
  Q1  #1a7a2e  vert foncé    → très protecteur (nature, biodiversité, forêt)
  Q2  #7ec850  vert clair    → protecteur
  Q3  #f5e642  beige/jaune   → neutre / mixte
  Q4  #f5a623  orange        → aménageur
  Q5  #d0021b  rouge         → très aménageur (construction, infrastructure)

Entrée :
  ecolab2_territoire_scot/COSIA_SCORE_finale_ecolab.gpkg  (étape 4)
"""

import os, sys, re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v2 as config


# =============================================================
# PALETTE
# =============================================================

QUINTILE_CONFIG = [
    (1, "Q1_tres_protecteur", "#1a7a2e"),
    (2, "Q2_protecteur",      "#7ec850"),
    (3, "Q3_neutre",          "#f5e642"),
    (4, "Q4_amenageur",       "#f5a623"),
    (5, "Q5_tres_amenageur",  "#d0021b"),
]

OUT_GPKG = config.CARTE_FINALE_GPKG
OUT_QML  = OUT_GPKG.replace(".gpkg", ".qml")


# =============================================================
# QML — style gradué auto-chargé par QGIS (même nom que GPKG)
# =============================================================

def generate_qml(path: str, scores: pd.Series) -> None:
    """Génère un fichier .qml avec 5 quintiles, palette vert→rouge."""
    q20 = float(scores.quantile(0.20))
    q40 = float(scores.quantile(0.40))
    q60 = float(scores.quantile(0.60))
    q80 = float(scores.quantile(0.80))
    vmin, vmax = float(scores.min()), float(scores.max())

    breaks = [vmin, q20, q40, q60, q80, vmax]
    ranges_xml = syms_xml = ""

    for i, (_, label, color) in enumerate(QUINTILE_CONFIG):
        lo, hi = breaks[i], breaks[i + 1]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        ranges_xml += (
            f'      <range lower="{lo}" upper="{hi}" symbol="{i}" '
            f'label="{label} ({lo:.3f} → {hi:.3f})" render="true"/>\n'
        )
        syms_xml += f"""      <symbol alpha="0.9" name="{i}" type="fill" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" value="{r},{g},{b},255" type="QString"/>
            <Option name="style" value="solid" type="QString"/>
            <Option name="outline_style" value="no" type="QString"/>
          </Option>
        </layer>
      </symbol>\n"""

    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology">
  <renderer-v2 type="graduatedSymbol" attr="score_polygone"
               enableorderby="0" forceraster="0" symbollevels="0" graduatedMethod="GraduatedColor">
    <ranges>
{ranges_xml}    </ranges>
    <symbols>
{syms_xml}    </symbols>
    <rotation/>
    <sizescale/>
  </renderer-v2>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerOpacity>1</layerOpacity>
</qgis>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(qml)
    print(f"  QML → {os.path.basename(path)}")


# =============================================================
# TUILES 1km×1km
# =============================================================

def tile_bbox(tile_name: str):
    """Extrait la bbox 1km×1km depuis le nom de tuile CoSIA (ex: D026_2023_830_6360_vecto.gpkg)."""
    m = re.search(r'_(\d+)_(\d+)_vecto', tile_name)
    if not m:
        return None
    x_km, y_km = int(m.group(1)), int(m.group(2))
    return box(x_km * 1000, y_km * 1000, (x_km + 1) * 1000, (y_km + 1) * 1000)


def build_tiles(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Construit une grille de tuiles 1km×1km avec score moyen pondéré."""
    if "tuile" not in df.columns or "surface_m2" not in df.columns:
        print("  WARN: colonnes 'tuile'/'surface_m2' absentes — tuiles ignorées")
        return gpd.GeoDataFrame()

    df_tiles = (
        df.dropna(subset=["score_polygone", "surface_m2", "tuile"])
        .groupby("tuile")
        .apply(lambda g: pd.Series({
            "score_pondere": (g["score_polygone"] * g["surface_m2"]).sum() / g["surface_m2"].sum(),
            "n_polygones":   len(g),
            "surface_m2":    g["surface_m2"].sum(),
        }))
        .reset_index()
    )

    q_cuts = np.quantile(df_tiles["score_pondere"].dropna(), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def assign_quintile(score):
        for i, (_, hi) in enumerate(zip(q_cuts[:-1], q_cuts[1:]), 1):
            if score <= hi:
                return i
        return 5

    records = []
    for _, row in df_tiles.iterrows():
        geom = tile_bbox(str(row["tuile"]))
        if geom is None:
            continue
        score = float(row["score_pondere"])
        q = assign_quintile(score)
        _, label, color = QUINTILE_CONFIG[q - 1]
        records.append({
            "tuile":        row["tuile"],
            "score_tuile":  round(score, 4),
            "quintile":     q,
            "label":        label,
            "couleur":      color,
            "n_polygones":  int(row["n_polygones"]),
            "geometry":     geom,
        })

    return gpd.GeoDataFrame(records, crs="EPSG:2154") if records else gpd.GeoDataFrame()


# =============================================================
# MAIN
# =============================================================

def run():
    print("=" * 60)
    print("PIPELINE V2 ETAPE 5 — Export QGIS carte finale unique")
    print("=" * 60)

    if not os.path.exists(config.FINALE_GPKG):
        raise FileNotFoundError(
            f"COSIA_SCORE_finale_ecolab.gpkg introuvable : {config.FINALE_GPKG}\n"
            "Lance d'abord l'étape 4."
        )

    print(f"Chargement : {os.path.basename(config.FINALE_GPKG)}")
    gdf = gpd.read_file(config.FINALE_GPKG)
    print(f"  {len(gdf):,} polygones  |  CRS : {gdf.crs}")

    col_score = "score_polygone"
    if col_score not in gdf.columns:
        candidates = [c for c in gdf.columns if "score" in c.lower()]
        col_score = candidates[0] if candidates else None
    if col_score is None:
        raise ValueError("Aucune colonne score trouvée dans le GPKG.")

    scores = gdf[col_score].fillna(0.0)

    # ── 1. Quintiles ─────────────────────────────────────────────
    q_cuts = np.quantile(scores, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print(f"\nCoupures quintiles (score_polygone) :")
    for i, (q, hi) in enumerate(zip(q_cuts[:-1], q_cuts[1:]), 1):
        _, label, color = QUINTILE_CONFIG[i - 1]
        print(f"  Q{i}  {color}  [{q:+.4f} → {hi:+.4f}]  {label}")

    print(f"\nDistribution polygones :")
    for i, (q_lo, q_hi) in enumerate(zip(q_cuts[:-1], q_cuts[1:]), 1):
        _, label, color = QUINTILE_CONFIG[i - 1]
        n = int(((scores >= q_lo) & (scores <= q_hi)).sum())
        bar = "█" * (n * 30 // max(len(gdf), 1))
        print(f"  Q{i}  {color}  {label:<25}  {n:>8,}  {n/len(gdf)*100:>5.1f}%  {bar}")

    # ── 2. Renommer FINALE_GPKG → CARTE_FINALE_GPKG (0 octet supplémentaire) ──
    # Évite de dupliquer 10 GB sur le disque.
    # CARTE_FINALE_GPKG contient déjà la couche "COSIA_SCORE_finale" (15M polygones).
    import shutil
    if os.path.exists(OUT_GPKG):
        os.remove(OUT_GPKG)
    print(f"\nRenommage vers {os.path.basename(OUT_GPKG)}...")
    shutil.move(config.FINALE_GPKG, OUT_GPKG)
    print(f"  Couche principale : COSIA_SCORE_finale ({len(gdf):,} polygones)")

    # ── 3. Tuiles 1km×1km ajoutées au même GPKG ──────────────────
    print(f"\nConstruction grille tuiles 1km×1km...")
    df_attrs = gdf.drop(columns=[gdf.geometry.name], errors="ignore")
    gdf_tuiles = build_tiles(df_attrs)
    if not gdf_tuiles.empty:
        gdf_tuiles.to_file(OUT_GPKG, layer="TUILES_SCORE", driver="GPKG", mode="a")
        print(f"  Couche 'TUILES_SCORE' : {len(gdf_tuiles)} tuiles")

    # ── 4. QML auto-style ────────────────────────────────────────
    print(f"\nGénération QML auto-style...")
    generate_qml(OUT_QML, scores)

    # ── 5. Résumé ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  GPKG carte finale → {OUT_GPKG}")
    print(f"  QML auto-style    → {OUT_QML}")
    print(f"\n  Dans QGIS :")
    print(f"    Glisse COSIA_BD_ecolab_carte_finale.gpkg")
    print(f"    → QGIS charge automatiquement les couleurs via le .qml")
    print(f"    → Couches : COSIA_SCORE_finale (15M polygones) + TUILES_SCORE")
    print(f"    → Filtrer par quintile dans QGIS : score_polygone >= X")
    print(f"{'─'*60}")
    print("ETAPE 5 — TERMINEE")


if __name__ == "__main__":
    run()
