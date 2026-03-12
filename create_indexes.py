"""
Script à lancer UNE SEULE FOIS pour créer les index SQLite sur le GPKG.
Après ça, les filtres dans le Streamlit seront instantanés.
Durée estimée : 5-15 minutes.
"""
import sqlite3, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import config_v2 as config

GPKG  = config.CARTE_FINALE_GPKG
LAYER = "COSIA_SCORE_finale"

BDTOPO_COLS = [
    "bdtopo_PARC_NATURA2000", "bdtopo_PARC_PNR", "bdtopo_PARC_RESERVE",
    "bdtopo_GEOPARC", "bdtopo_FORET_PUBLIQUE", "bdtopo_ZONE_VEG_FORET",
    "bdtopo_ZONE_VEG_VIGNE", "bdtopo_ZONE_VEG_VERGER", "bdtopo_ZONE_VEG_LANDE",
    "bdtopo_COURS_EAU", "bdtopo_PLAN_EAU", "bdtopo_SURFACE_HYDRO",
    "bdtopo_EOLIENNE", "bdtopo_BARRAGE",
]

print(f"GPKG : {GPKG}")
print(f"Taille : {os.path.getsize(GPKG) / 1e9:.1f} Go")
print()

conn = sqlite3.connect(GPKG)
conn.execute("PRAGMA journal_mode = WAL")
conn.execute("PRAGMA cache_size = -262144")  # 256 MB

def create_index(name, col):
    t0 = time.time()
    print(f"  Index {name} sur {col}… ", end="", flush=True)
    conn.execute(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{LAYER}" ("{col}")')
    conn.commit()
    print(f"OK ({time.time()-t0:.0f}s)")

print("=== Création des index (une seule fois) ===")
create_index("idx_classe", "classe")
for col in BDTOPO_COLS:
    name = f"idx_{col}"
    create_index(name, col)

conn.close()

# Supprimer le cache bbox pour forcer un recalcul rapide
cache = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bbox_cache.json")
if os.path.exists(cache):
    os.remove(cache)
    print(f"\nbbox_cache.json supprimé (sera recalculé rapidement grâce aux index)")

print("\n=== Terminé ! Relancez maintenant : streamlit run app_carte2.py ===")
