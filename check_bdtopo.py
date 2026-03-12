import sqlite3, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v2 as config
from collections import Counter

gpkg = config.CARTE_FINALE_GPKG
uri = "file:" + gpkg.replace(chr(92), "/") + "?mode=ro"
conn = sqlite3.connect(uri, uri=True)
cols = [c[1] for c in conn.execute('PRAGMA table_info("COSIA_SCORE_finale")')]
bdtopo = [c for c in cols if c.startswith("bdtopo")]
print("Colonnes bdtopo:", bdtopo)

q = "SELECT DISTINCT " + ", ".join('"' + c + '"' for c in bdtopo) + ' FROM "COSIA_SCORE_finale"'
rows = conn.execute(q).fetchall()
print(f"Combinaisons uniques: {len(rows)}")
for combo in rows:
    active = [bdtopo[i].replace("bdtopo_", "") for i, v in enumerate(combo) if v]
    print(f"  {active if active else '[aucune]'}")
conn.close()
