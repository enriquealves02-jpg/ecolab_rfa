"""
EcoLab Pipeline V2 — Étape 6c : Injection resume_bdtopo dans le GPKG
====================================================================
Ajoute une colonne resume_bdtopo directement dans COSIA_SCORE_finale.
Plus besoin de get_feature() dans le QML — la colonne est là comme les autres.

Prérequis : BDTOPO_resume_par_combo.csv présent (étape 06b terminée).
IMPORTANT : fermer QGIS avant de lancer ce script.
"""

import os, sys, sqlite3
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v2 as config

gpkg     = config.CARTE_FINALE_GPKG
csv_path = os.path.join(os.path.dirname(gpkg), "BDTOPO_resume_par_combo.csv")

print("=" * 60)
print("PIPELINE V2 ETAPE 6c — Injection resume_bdtopo dans GPKG")
print("=" * 60)

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV introuvable : {csv_path}\nLance d'abord l'étape 06b.")

# ── 1. Charger le CSV
df = pd.read_csv(csv_path, encoding="utf-8-sig")
resume_by_key = dict(zip(df["combo_key"].fillna(""), df["resume_bdtopo"].fillna("")))
print(f"\n{len(resume_by_key)} combos dans le CSV ({sum(1 for v in resume_by_key.values() if v)} avec résumé)")

# ── 2. Ouvrir le GPKG en écriture
print(f"\nOuverture GPKG : {os.path.basename(gpkg)}")
conn = sqlite3.connect(gpkg)
conn.execute("PRAGMA journal_mode=MEMORY")   # journal en RAM → pas de fichier WAL sur disque
conn.execute("PRAGMA synchronous=OFF")       # pas de fsync → beaucoup plus rapide

# ── 3. Récupérer les colonnes bdtopo (triées alphabétiquement = même ordre que combo_key)
cols_info  = conn.execute('PRAGMA table_info("COSIA_SCORE_finale")').fetchall()
all_cols   = [c[1] for c in cols_info]
bdtopo_cols = sorted(
    [c for c in all_cols if c.startswith("bdtopo_")],
    key=lambda x: x.replace("bdtopo_", "")
)
print(f"{len(bdtopo_cols)} colonnes bdtopo : {bdtopo_cols}")

# ── 4. Ajouter la colonne si absente
if "resume_bdtopo" not in all_cols:
    conn.execute('ALTER TABLE "COSIA_SCORE_finale" ADD COLUMN resume_bdtopo TEXT DEFAULT \'\'')
    print("\nColonne resume_bdtopo ajoutée.")
else:
    print("\nColonne resume_bdtopo existe déjà — mise à jour.")

# ── 5. Récupérer toutes les combinaisons uniques du GPKG
cols_sql = ", ".join(f'"{c}"' for c in bdtopo_cols)
combos   = conn.execute(f'SELECT DISTINCT {cols_sql} FROM "COSIA_SCORE_finale"').fetchall()
print(f"{len(combos)} combinaisons uniques dans le GPKG")

# ── 6. Pour chaque combo → UPDATE ciblé
print("\nInjection des résumés...")
updated = 0
skipped = 0
for combo in combos:
    active    = [c.replace("bdtopo_", "") for c, v in zip(bdtopo_cols, combo) if v]
    combo_key = "|".join(sorted(active))
    resume    = resume_by_key.get(combo_key, "")
    if not resume:
        skipped += 1
        continue

    # WHERE exact sur chaque colonne bdtopo (COALESCE pour gérer les NULL)
    where = " AND ".join(
        f'COALESCE("{c}", 0) = {1 if v else 0}'
        for c, v in zip(bdtopo_cols, combo)
    )
    conn.execute(
        f'UPDATE "COSIA_SCORE_finale" SET resume_bdtopo = ? WHERE {where}',
        (resume,)
    )
    print(f"  [{updated+1:03d}] {combo_key or '[aucune]'} → {len(resume)} car.")
    updated += 1

conn.commit()
conn.close()

print(f"\n{'─'*60}")
print(f"  {updated} combinaisons injectées, {skipped} sans résumé")
print(f"  → Recharge la couche dans QGIS (ou redémarre QGIS)")
print(f"  → Le champ resume_bdtopo est maintenant directement dans le GPKG")
print("ETAPE 6c — TERMINEE")
