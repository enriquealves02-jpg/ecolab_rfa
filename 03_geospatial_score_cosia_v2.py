"""
EcoLab Pipeline V2 — Étape 3 : Scoring géospatial COSIA (3 départements)
=========================================================================
Logique identique à pipeline v1/03_geospatial_scoring.py :
  - Filtre bbox SCoT (ultra-rapide, pas de clip géométrique)
  - Join attributaire pur : score_polygone attribué par classe CoSIA
  - 3 départements : D007, D026, D084
  - Glob récursif pour gérer les sous-dossiers dupliqués (7-Zip)

Sorties :
  outputs/geo/COSIA_SCORE_v2.gpkg
  outputs/geo/COSIA_SCORE_v2_attrs.csv
"""

import ast
import gc, glob, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np, pandas as pd, geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v2 as config


# =============================================================
# DIAGNOSTIC
# =============================================================

def load_topic_labels() -> dict:
    if not os.path.exists(config.TOPICS_INFO_CSV):
        return {}
    info = pd.read_csv(config.TOPICS_INFO_CSV, encoding="utf-8")
    labels = {}
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        try:
            words = ast.literal_eval(row["Representation"])
            labels[tid] = ", ".join(str(w) for w in words[:4])
        except Exception:
            labels[tid] = str(row.get("Name", tid))
    return labels


def print_mapping_diagnostic(topic_summary: pd.DataFrame, topic_labels: dict) -> None:
    scored_ids = set(topic_summary["topic"].astype(int).tolist())
    print("\n" + "─" * 80)
    print("DIAGNOSTIC — Correspondance topics BERTopic v2 → classes CoSIA")
    print(f"{'Topic':<8} {'Label TF-IDF (top 4)':<35} {'Classes CoSIA':<30} {'Score'}")
    print("─" * 80)
    missing = []
    for tid in sorted(config.TOPIC_COSIA_MAP):
        label   = topic_labels.get(tid, "???")
        classes = ", ".join(config.TOPIC_COSIA_MAP[tid])
        if tid in scored_ids:
            sc = topic_summary.loc[topic_summary["topic"] == tid, "mean_score"].values[0]
            score_str = f"{sc:+.3f}"
        else:
            score_str = "MANQUANT"
            missing.append(tid)
        print(f"  T{str(tid):<6} {label:<35} {classes:<30} {score_str}")
    print("─" * 80)
    if missing:
        print(f"  ⚠ {len(missing)} topics sans score : {missing}")
    else:
        print(f"  ✓ Tous les {len(config.TOPIC_COSIA_MAP)} topics ont un score.")
    print("─" * 80 + "\n")


# =============================================================
# SCORE PAR CLASSE
# =============================================================

def build_score_par_classe(topic_summary: pd.DataFrame) -> dict:
    score_by_topic = {int(r["topic"]): float(r["mean_score"])
                      for _, r in topic_summary.iterrows()
                      if not (isinstance(r["mean_score"], float) and np.isnan(r["mean_score"]))}
    class_scores = {}
    for tid, classes in config.TOPIC_COSIA_MAP.items():
        if tid not in score_by_topic:
            continue
        for c in classes:
            class_scores.setdefault(c, []).append(score_by_topic[tid])
    matched = sum(1 for tid in config.TOPIC_COSIA_MAP if tid in score_by_topic)
    print(f"  {matched}/{len(config.TOPIC_COSIA_MAP)} topics trouvés dans le summary")
    return {c: round(sum(v)/len(v), 4) if v else 0.0
            for c in config.CLASSES_ATTENDUES
            for v in [class_scores.get(c, [])]}


# =============================================================
# TRAITEMENT D'UNE TUILE (worker thread — lecture + bbox uniquement)
# =============================================================

def read_and_filter_tile(f, scot_bounds):
    """
    Lit une tuile COSIA et filtre par bbox SCoT.
    Pas de clip géométrique (identique à v1).
    Retourne (GeoDataFrame ou None, nom_fichier, message).
    """
    bname = os.path.basename(f)
    try:
        gdf = gpd.read_file(f)
    except Exception as e:
        return None, bname, f"ERR lecture : {e}"

    if gdf.empty:
        return None, bname, "vide"

    # Pré-filtrage bbox SCoT (ultra-rapide, pas de calcul géométrique)
    if scot_bounds is not None:
        minx, miny, maxx, maxy = scot_bounds
        gdf = gdf.cx[minx:maxx, miny:maxy]
        if gdf.empty:
            return None, bname, "hors bbox"

    return gdf, bname, f"{len(gdf)} polygones"


def process_gdf(gdf, score_par_classe, source_name):
    """Attribue score_polygone à chaque polygone selon sa classe CoSIA."""
    if gdf.empty:
        return gdf
    if gdf.geometry.name != "geom":
        gdf = gdf.rename_geometry("geom")
    if config.COL_CLASS not in gdf.columns:
        cands = [c for c in gdf.columns if "class" in c.lower()]
        if cands:
            gdf[config.COL_CLASS] = gdf[cands[0]]
    gdf["surface_m2"]     = gdf.geometry.area.round(2)
    gdf["score_polygone"] = gdf[config.COL_CLASS].map(score_par_classe).fillna(0.0).round(4)
    gdf["tuile"]          = source_name
    keep = [c for c in [config.COL_ID, config.COL_CLASS, "surface_m2", "score_polygone", "tuile", "geom"]
            if c in gdf.columns]
    return gdf[keep]


# =============================================================
# MAIN
# =============================================================

def run():
    print("=" * 60)
    print("PIPELINE V2 ETAPE 3 — Scoring géospatial COSIA (3 depts)")
    print("=" * 60)
    os.makedirs(config.GEO_DIR, exist_ok=True)

    # Scores Ollama par topic
    topic_summary = pd.read_csv(config.TOPIC_SCORE_SUMMARY_CSV)
    print(f"  {len(topic_summary)} topics scorés")

    topic_labels = load_topic_labels()
    print_mapping_diagnostic(topic_summary, topic_labels)

    score_par_classe = build_score_par_classe(topic_summary)
    print("\nScore par classe CoSIA :")
    for cls, sc in sorted(score_par_classe.items(), key=lambda x: x[1]):
        print(f"  {cls:<22} -> {sc:+.4f}")

    # Masque SCoT — bbox seulement (pas de clip, identique v1)
    scot_bounds = None
    if config.SCOT_MASK_FILE and os.path.exists(config.SCOT_MASK_FILE):
        print(f"\nChargement masque SCoT : {os.path.basename(config.SCOT_MASK_FILE)}")
        raw = gpd.read_file(config.SCOT_MASK_FILE, layer=config.SCOT_MASK_LAYER)
        scot_bounds = raw.total_bounds
        print(f"  CRS: {raw.crs}  |  {len(raw)} polygones")
        print(f"  Bbox SCoT : {scot_bounds.round(0)}")
        del raw

    # Supprimer sorties précédentes
    for f in (config.COSIA_SCORE_GPKG, config.COSIA_SCORE_CSV):
        if os.path.exists(f):
            os.remove(f)

    # Collecter toutes les tuiles des 3 depts (glob récursif = gère sous-dossiers dupliqués)
    all_files = []
    for d in config.COSIA_DIRS:
        tiles = sorted(glob.glob(os.path.join(d, "**", "*_vecto.gpkg"), recursive=True))
        if not tiles:
            tiles = sorted(glob.glob(os.path.join(d, "*_vecto.gpkg")))
        all_files.extend(tiles)
        print(f"  {os.path.basename(d)} : {len(tiles)} tuiles")

    total = len(all_files)
    print(f"\nTotal : {total} tuiles")

    # Lecture parallèle des tuiles (I/O), traitement + écriture séquentiels (comme v1)
    N_WORKERS = min(6, total)
    print(f"Parallélisation lecture : {N_WORKERS} threads (pas de clip = rapide)\n")

    first_gpkg = first_csv = True
    total_polys = done = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(read_and_filter_tile, f, scot_bounds): f
            for f in all_files
        }
        for future in as_completed(futures):
            done += 1
            bname = os.path.basename(futures[future])
            try:
                gdf, _, msg = future.result()
            except Exception as e:
                print(f"  [{done:>3}/{total}] ERR {bname} : {e}")
                continue

            if gdf is None or gdf.empty:
                print(f"  [{done:>3}/{total}] {bname} : {msg}")
                continue

            # Score (séquentiel, rapide)
            gdf = process_gdf(gdf, score_par_classe, bname)

            # Écriture GPKG (séquentielle dans le thread principal)
            if first_gpkg:
                gdf.to_file(config.COSIA_SCORE_GPKG, layer="COSIA_SCORE_v2", driver="GPKG")
                first_gpkg = False
            else:
                gdf.to_file(config.COSIA_SCORE_GPKG, layer="COSIA_SCORE_v2", driver="GPKG", mode="a")

            df_csv = gdf.drop(columns=["geom"], errors="ignore")
            df_csv.to_csv(config.COSIA_SCORE_CSV,
                          mode="w" if first_csv else "a",
                          header=first_csv, index=False, encoding="utf-8")
            first_csv = False

            total_polys += len(gdf)
            pct = done / total * 100
            print(f"  [{done:>3}/{total}] {pct:>5.1f}%  {bname} : {msg} | cumul {total_polys:,}")

            del gdf, df_csv
            gc.collect()

    print(f"\nTotal polygones scorés : {total_polys:,}")
    print(f"GPKG  → {config.COSIA_SCORE_GPKG}")
    print(f"CSV   → {config.COSIA_SCORE_CSV}")
    print("=" * 60)
    print("ETAPE 3 — TERMINÉE")
    print("=" * 60)


if __name__ == "__main__":
    run()
