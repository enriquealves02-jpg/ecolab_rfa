"""
patch_zone_imperm.py — Patch minimal ciblé
==========================================
Corrige uniquement ce qui manque :
  1. score_polygone Zone imperméable  (déjà fait si min > -0.39)
  2. art_cosia_* Surface eau          (25K lignes, batches de 5000)
  3. art_ctx_1/2 pour bdtopo_BARRAGE  (quelques centaines de lignes)
"""
import json, os, sys, sqlite3, time, pandas as pd, numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config_v2 as config

GPKG        = config.CARTE_FINALE_GPKG
LAYER       = "COSIA_SCORE_finale"
CACHE_JSON  = os.path.join(config.GEO_DIR, "articles_par_label_cache.json")
CSV_CLASSES = os.path.join(config.GEO_DATA_DIR, "COSIA_articles_par_classe.csv")
BDTOPO_COLS = [f"bdtopo_{c}" for c in config.BDTOPO_CATEGORIES]
BATCH       = 5000

print("=" * 60)
print("PATCH MINIMAL — Surface eau + Barrage + Zone imperméable")
print("=" * 60)

# ── Charger cache articles ─────────────────────────────────────
with open(CACHE_JSON, encoding="utf-8") as f:
    label_articles = json.load(f)

conn = sqlite3.connect(GPKG, timeout=120)
conn.execute("PRAGMA cache_size = -65536")

# ── 1. Zone imperméable score — vérifier si déjà fait ─────────
r = conn.execute(
    f'SELECT MIN(score_polygone), MAX(score_polygone) FROM "{LAYER}" WHERE classe = ?',
    ("Zone imperméable",)
).fetchone()
if r and r[1] and r[1] > 0.1:
    print(f"\n[1/3] Zone imperméable score : déjà OK (max={r[1]:.4f}) ✓")
else:
    print(f"\n[1/3] Zone imperméable score : correction en cours...")
    df_sc = pd.read_csv(config.TOPIC_SCORE_SUMMARY_CSV)
    score_by_topic = dict(zip(df_sc["topic"].astype(int), df_sc["mean_score"]))
    scores = [score_by_topic[t] for t, cls in config.TOPIC_COSIA_MAP.items()
              if "Zone imperméable" in cls and t in score_by_topic]
    SCORE_COSIA = round(float(np.mean(scores)), 4)
    DELTA = round(0.5 * SCORE_COSIA, 6)
    has_bdtopo = " + ".join(f'COALESCE("{c}", 0)' for c in BDTOPO_COLS)
    conn.execute(f"""
        UPDATE "{LAYER}" SET score_polygone = ROUND(
            CASE WHEN ({has_bdtopo}) > 0 THEN score_polygone + {DELTA} ELSE {SCORE_COSIA} END, 4)
        WHERE classe = ?
    """, ("Zone imperméable",))
    conn.commit()
    print(f"  OK score_cosia={SCORE_COSIA}")

# ── Helper : batch UPDATE par rowids ──────────────────────────
def batch_update(cls_or_where, sets, vals_base, where_col="classe"):
    """Met à jour par lots de BATCH rowids pour éviter journal énorme."""
    if where_col == "classe":
        rowids = [r[0] for r in conn.execute(
            f'SELECT rowid FROM "{LAYER}" WHERE classe = ?', (cls_or_where,)
        ).fetchall()]
    else:
        rowids = [r[0] for r in conn.execute(
            f'SELECT rowid FROM "{LAYER}" WHERE "{where_col}" = 1'
        ).fetchall()]
    n = len(rowids)
    for start in range(0, n, BATCH):
        batch = rowids[start:start + BATCH]
        ph = ",".join(["?"] * len(batch))
        conn.execute(
            f'UPDATE "{LAYER}" SET {", ".join(sets)} WHERE rowid IN ({ph})',
            vals_base + batch
        )
        conn.commit()
    return n

# ── 2. Surface eau — art_cosia_* ───────────────────────────────
print(f"\n[2/3] Surface eau — correction art_cosia_*...")
arts_surf = label_articles.get("Surface eau", [])[:3]
if not arts_surf:
    print("  ATTENTION : pas d'articles Surface eau dans le cache !")
else:
    sample = conn.execute(
        f'SELECT art_cosia_1_titre FROM "{LAYER}" WHERE classe = ? LIMIT 1',
        ("Surface eau",)
    ).fetchone()
    if sample and sample[0] == arts_surf[0].get("titre", ""):
        print("  Déjà OK ✓")
    else:
        t1 = time.time()
        sets, vals = [], []
        for i in range(3):
            a = arts_surf[i] if i < len(arts_surf) else {}
            sets += [f'"art_cosia_{i+1}_titre" = ?', f'"art_cosia_{i+1}_url" = ?']
            vals += [a.get("titre", ""), a.get("url", "")]
        n = batch_update("Surface eau", sets, vals)
        print(f"  OK ({time.time()-t1:.1f}s)  n={n:,}")
        print(f"  art1 = {arts_surf[0].get('titre','')[:60]}")

# ── 3. BARRAGE — art_ctx_1/2 pour bdtopo_BARRAGE = 1 ──────────
print(f"\n[3/3] BARRAGE — correction art_ctx_1/2 pour polygones barrage...")
arts_barr = label_articles.get("BARRAGE", [])[:2]
if not arts_barr:
    print("  ATTENTION : pas d'articles BARRAGE dans le cache !")
else:
    sample = conn.execute(
        f'SELECT art_ctx_1_titre FROM "{LAYER}" WHERE bdtopo_BARRAGE = 1 LIMIT 1'
    ).fetchone()
    if sample and sample[0] == arts_barr[0].get("titre", ""):
        print("  Déjà OK ✓")
    else:
        t1 = time.time()
        sets, vals = [], []
        for i in range(min(2, len(arts_barr))):
            a = arts_barr[i]
            sets += [f'"art_ctx_{i+1}_titre" = ?', f'"art_ctx_{i+1}_url" = ?']
            vals += [a.get("titre", ""), a.get("url", "")]
        n = batch_update("bdtopo_BARRAGE", sets, vals, where_col="bdtopo_BARRAGE")
        print(f"  OK ({time.time()-t1:.1f}s)  n={n:,}")
        print(f"  art1 = {arts_barr[0].get('titre','')[:60]}")

conn.close()
print("\n" + "=" * 60)
print("PATCH TERMINÉ — Relancez : streamlit run pipeline_v2/app_carte2.py")
print("=" * 60)
