"""
EcoLab Pipeline V2 — Étape 6b : Résumés BD TOPO par combinaison
================================================================
Pour chaque combinaison unique de catégories BD TOPO présente dans le GPKG :
  - prend top 3 articles par catégorie active (depuis le cache label_articles)
  - génère 1 résumé Ollama pour la combinaison entière
  - exporte BDTOPO_resume_par_combo.csv

Déduplique par "catégories qui ont des articles" → beaucoup moins que 788 appels Ollama.
Ne touche pas au GPKG ni aux résumés COSIA.

Prérequis : étape 6 terminée (articles_par_label_cache.json présent).
"""

import json, os, sqlite3, sys, time
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v2 as config

TOP_LLM        = 3    # articles par catégorie BD TOPO
MAX_CATS       = 4    # max catégories retenues par combo (évite les prompts trop longs)
MIN_POLYGONS   = 1    # générer un résumé pour toutes les combinaisons avec articles

OUT_DIR       = config.GEO_DIR
gpkg_in       = config.CARTE_FINALE_GPKG
OUT_CSV_COMBO = os.path.join(os.path.dirname(gpkg_in), "BDTOPO_resume_par_combo.csv")
OUT_QML       = gpkg_in.replace(".gpkg", "_articles.qml")

CACHE_LABELS  = os.path.join(OUT_DIR, "articles_par_label_cache.json")
CACHE_RESUME  = os.path.join(OUT_DIR, "resume_par_classe_cache.json")

COSIA_CLASSES = {
    "Bâtiment", "Zone imperméable", "Zone perméable", "Pelouse", "Broussaille",
    "Feuillu", "Conifère", "Culture", "Terre labourée", "Vigne", "Verger",
    "Eau", "Piscine", "Serre", "Zone de chantier"
}


# ──────────────────────────────────────────────────────────────

def ollama_resume(articles: list[dict], label: str) -> str:
    arts_txt = "\n".join(
        f"- {a['titre'][:100]} : {a.get('extrait', a.get('contenu', ''))[:200]}"
        for a in articles if a.get("titre")
    )
    prompt = (
        f"Tu es un expert en aménagement du territoire dans le SCoT Drôme-Ardèche-Vaucluse.\n"
        f"Voici des articles de presse sur : {label}\n\n"
        f"{arts_txt}\n\n"
        f"Rédige un paragraphe de 3 à 5 phrases qui résume les enjeux territoriaux "
        f"liés à ce contexte dans le SCoT. Sois factuel et synthétique.\n\nRésumé :"
    )
    try:
        r = requests.post(config.OLLAMA_URL, json={
            "model": config.OLLAMA_MODEL, "prompt": prompt,
            "stream": False, "options": {"temperature": 0.3, "num_predict": 300}
        }, timeout=config.OLLAMA_TIMEOUT)
        return r.json().get("response", "").strip()
    except Exception as e:
        print(f"  WARN Ollama: {e}")
        return ""


def generate_qml_maptips(path: str) -> None:
    P = "COSIA_articles_par_classe_"

    def art_html(prefix: str, i: int) -> str:
        t = f'"{P}{prefix}{i}_titre"'
        u = f'"{P}{prefix}{i}_url"'
        return (
            f'<b>[% {t} %]</b><br/>'
            f'<a href="[% {u} %]">[% {u} %]</a><br/>'
        )

    cosia_arts = "".join(art_html("art_cosia_", i) for i in range(1, 4))
    ctx_arts   = "".join(art_html("art_ctx_", i) for i in range(1, 7))

    # Expression QGIS qui calcule la combo_key à la volée depuis les colonnes bdtopo_*
    # puis fait get_feature sur BDTOPO_resume_par_combo
    # trim(str, char) n'existe pas en QGIS → regexp_replace pour supprimer le | final
    combo_key_expr = (
        "regexp_replace(concat("
        + ", ".join(
            f'if("{col}" = 1 OR "{col}" = \'true\' OR "{col}" = \'True\', \'{col.replace("bdtopo_", "")}|\', \'\')'
            for col in [
                "bdtopo_PARC_NATURA2000", "bdtopo_PARC_PNR", "bdtopo_PARC_RESERVE",
                "bdtopo_GEOPARC", "bdtopo_FORET_PUBLIQUE", "bdtopo_ZONE_VEG_FORET",
                "bdtopo_ZONE_VEG_VIGNE", "bdtopo_ZONE_VEG_VERGER", "bdtopo_ZONE_VEG_LANDE",
                "bdtopo_COURS_EAU", "bdtopo_PLAN_EAU", "bdtopo_SURFACE_HYDRO",
                "bdtopo_EOLIENNE", "bdtopo_BARRAGE"
            ]
        )
        + "), '\\\\|$', '')"
    )

    resume_bdtopo_expr = (
        f"get_feature('BDTOPO_resume_par_combo', 'combo_key', {combo_key_expr})['resume_bdtopo']"
    )

    html = (
        '<table width="400"><tr><td>'
        '<b>Classe :</b> [% "classe" %] &nbsp; '
        '<b>Score :</b> [% format_number("score_polygone", 3) %]<hr/>'
        '<b>Articles liés à la classe :</b><br/>' + cosia_arts +
        '<hr/><b>Articles contexte territorial :</b><br/>' + ctx_arts +
        '<hr/><b>Résumé COSIA :</b><br/>[% "cosia_resume_article" %]'
        f'<hr/><b>Résumé contexte BD TOPO :</b><br/>[% {resume_bdtopo_expr} %]'
        '</td></tr></table>'
    )

    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="MapTips|Symbology">
  <maptip><![CDATA[{html}]]></maptip>
  <renderer-v2 type="graduatedSymbol" attr="score_polygone"
               enableorderby="0" forceraster="0" symbollevels="0" graduatedMethod="GraduatedColor">
    <ranges>
      <range lower="-2.0" upper="-0.5" symbol="0" label="Très protecteur" render="true"/>
      <range lower="-0.5" upper="0.0"  symbol="1" label="Protecteur"       render="true"/>
      <range lower="0.0"  upper="0.3"  symbol="2" label="Neutre"           render="true"/>
      <range lower="0.3"  upper="0.7"  symbol="3" label="Aménageur"        render="true"/>
      <range lower="0.7"  upper="2.0"  symbol="4" label="Très aménageur"   render="true"/>
    </ranges>
    <symbols>
      <symbol alpha="0.9" name="0" type="fill"><layer class="SimpleFill" enabled="1" locked="0" pass="0"><Option type="Map"><Option name="color" value="26,122,46,255" type="QString"/><Option name="outline_style" value="no" type="QString"/></Option></layer></symbol>
      <symbol alpha="0.9" name="1" type="fill"><layer class="SimpleFill" enabled="1" locked="0" pass="0"><Option type="Map"><Option name="color" value="126,200,80,255" type="QString"/><Option name="outline_style" value="no" type="QString"/></Option></layer></symbol>
      <symbol alpha="0.9" name="2" type="fill"><layer class="SimpleFill" enabled="1" locked="0" pass="0"><Option type="Map"><Option name="color" value="245,230,66,255" type="QString"/><Option name="outline_style" value="no" type="QString"/></Option></layer></symbol>
      <symbol alpha="0.9" name="3" type="fill"><layer class="SimpleFill" enabled="1" locked="0" pass="0"><Option type="Map"><Option name="color" value="245,166,35,255" type="QString"/><Option name="outline_style" value="no" type="QString"/></Option></layer></symbol>
      <symbol alpha="0.9" name="4" type="fill"><layer class="SimpleFill" enabled="1" locked="0" pass="0"><Option type="Map"><Option name="color" value="208,2,27,255"   type="QString"/><Option name="outline_style" value="no" type="QString"/></Option></layer></symbol>
    </symbols>
  </renderer-v2>
</qgis>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(qml)
    print(f"  QML → {os.path.basename(path)}")


# ──────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("PIPELINE V2 ETAPE 6b — Résumés BD TOPO par combinaison")
    print("=" * 60)

    if not os.path.exists(CACHE_LABELS):
        raise FileNotFoundError(f"Introuvable : {CACHE_LABELS} — lance d'abord l'étape 6.")

    # ── 1. Charger cache labels
    print("\nChargement cache labels...")
    with open(CACHE_LABELS, encoding="utf-8") as f:
        label_articles: dict = json.load(f)

    resume_cache = {}
    if os.path.exists(CACHE_RESUME):
        with open(CACHE_RESUME, encoding="utf-8") as f:
            resume_cache = json.load(f)

    # Catégories BD TOPO qui ont des articles
    bdtopo_with_arts = {k for k in label_articles if k not in COSIA_CLASSES and label_articles[k]}
    print(f"  Catégories BD TOPO avec articles : {sorted(bdtopo_with_arts)}")

    # ── 2. Lire les combinaisons avec leur fréquence depuis le GPKG
    print(f"\nLecture combinaisons depuis le GPKG (avec fréquence)...")
    uri = "file:" + gpkg_in.replace(chr(92), "/") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    all_cols = [c[1] for c in conn.execute('PRAGMA table_info("COSIA_SCORE_finale")')]
    bdtopo_cols = [c for c in all_cols if c.startswith("bdtopo_")]
    cols_sql = ", ".join(f'"{c}"' for c in bdtopo_cols)
    q = f'SELECT {cols_sql}, COUNT(*) as n FROM "COSIA_SCORE_finale" GROUP BY {cols_sql} ORDER BY n DESC'
    rows_with_count = conn.execute(q).fetchall()
    conn.close()

    # Séparer les colonnes bdtopo du count
    raw_combos_freq = [(row[:-1], row[-1]) for row in rows_with_count]
    total_polys = sum(n for _, n in raw_combos_freq)
    filtered = [(combo, n) for combo, n in raw_combos_freq if n >= MIN_POLYGONS]
    covered  = sum(n for _, n in filtered)
    print(f"  {len(raw_combos_freq)} combinaisons brutes, {total_polys:,} polygones total")
    print(f"  → {len(filtered)} combos avec >= {MIN_POLYGONS} polygones ({covered/total_polys*100:.1f}% couverture)")

    def combo_to_key(row):
        """Clé pour join QGIS : toutes les cats actives."""
        active = [col.replace("bdtopo_", "") for col, val in zip(bdtopo_cols, row) if val]
        return "|".join(sorted(active))

    # Construire le mapping combo_key → fréquence (filtré)
    combo_map = {}  # combo_key → n_polygons
    for combo, n in filtered:
        key = combo_to_key(combo)
        combo_map[key] = n

    # ── 4. Générer résumés pour chaque combo filtré
    print(f"\nGénération résumés ({len(combo_map)} combos × ~30s)...")

    combo_resume = {}  # combo_key → resume
    for i, combo_key in enumerate(combo_map, 1):
        cache_key = "bdtopo_combo_" + combo_key.replace("|", "_") if combo_key else "bdtopo_combo_AUCUN"

        if cache_key in resume_cache:
            print(f"  [{i:02d}/{len(combo_map)}] {combo_key or '[aucune]'} (cache)")
            combo_resume[combo_key] = resume_cache[cache_key]
            continue

        cats = [c for c in combo_key.split("|") if c] if combo_key else []
        arts = []
        seen = set()
        for cat in cats[:MAX_CATS]:
            for a in label_articles.get(cat, [])[:TOP_LLM]:
                if a["url"] not in seen:
                    arts.append(a)
                    seen.add(a["url"])

        if not arts:
            combo_resume[combo_key] = ""
            resume_cache[cache_key] = ""
            print(f"  [{i:02d}/{len(combo_map)}] {combo_key or '[aucune]'} → pas d'articles")
            continue

        label_full = " + ".join(cats) if cats else "territoire SCoT"
        print(f"  [{i:02d}/{len(combo_map)}] {combo_key or '[aucune]'} ...", end="", flush=True)
        resume = ollama_resume(arts, label_full)
        combo_resume[combo_key] = resume
        resume_cache[cache_key] = resume
        with open(CACHE_RESUME, "w", encoding="utf-8") as f:
            json.dump(resume_cache, f, ensure_ascii=False, indent=2)
        time.sleep(config.OLLAMA_SLEEP)
        print(f" ok ({len(resume)} car.)")

    # ── 5. Export CSV : combo_key → resume_bdtopo
    print(f"\nExport {os.path.basename(OUT_CSV_COMBO)}...")
    rows = []
    for combo_key, n in combo_map.items():
        rows.append({
            "combo_key":     combo_key,
            "n_polygones":   n,
            "resume_bdtopo": combo_resume.get(combo_key, ""),
        })
    # Ligne vide pour polygones sans catégorie BD TOPO
    if "" not in combo_map:
        rows.append({"combo_key": "", "n_polygones": 0, "resume_bdtopo": ""})

    df = pd.DataFrame(rows).drop_duplicates("combo_key")
    df.to_csv(OUT_CSV_COMBO, index=False, encoding="utf-8-sig")
    print(f"  → {len(df)} lignes exportées")

    # ── 6. Régénérer le QML
    generate_qml_maptips(OUT_QML)

    print(f"\n{'─'*60}")
    print(f"  Dans QGIS :")
    print(f"    1. Ajoute {os.path.basename(OUT_CSV_COMBO)} comme couche texte délimité (sans géométrie)")
    print(f"    2. Recharge le QML (Infobulles uniquement)")
    print(f"    → Map Tips calcule la combo_key à la volée et affiche le résumé BD TOPO")
    print(f"{'─'*60}")
    print("ETAPE 6b — TERMINEE")


if __name__ == "__main__":
    run()
