"""
EcoLab Pipeline V2 — Étape 6 : Articles par parcelle
=====================================================
Pour chaque polygone COSIA, associe des articles de presse pertinents
et génère un résumé textuel via Ollama/Mistral.

Logique :
  1. Pour chaque classe COSIA (15) et chaque catégorie BD TOPO (14) :
       → embedding synthétique (texte idéal mentionnant le territoire SCoT)
       → cosine similarity sur 50K articles → top 50 candidats
       → Ollama score 0-1 pour chaque candidat → top 3 gardés
  2. Par combinaison unique (classe × bdtopo actives) :
       → articles COSIA (3) + articles BD TOPO contexte (3 par cat active)
       → Mistral génère 1 paragraphe résumé
  3. Export GPKG + QML avec Map Tips HTML (URLs cliquables dans QGIS)

Colonnes GPKG :
  art_cosia_{1,2,3}_{titre,url}   ← articles liés à la classe COSIA
  art_ctx_{1..6}_{titre,url}      ← articles liés au contexte BD TOPO (2 cats max)
  resume                          ← paragraphe Mistral

Entrée  : COSIA_BD_ecolab_carte_finale.gpkg  (étape 5)
Sorties : COSIA_BD_ecolab_articles.gpkg
          COSIA_BD_ecolab_articles.qml  (Map Tips auto-chargé dans QGIS)
"""

import ast, gc, json, os, re, sys, time
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v2 as config

# ──────────────────────────────────────────────────────────────
# PARAMÈTRES
# ──────────────────────────────────────────────────────────────
TOP_COSINE    = 15   # candidats par similarité cosinus
TOP_LLM       = 3    # articles finaux gardés après scoring LLM
MAX_BDTOPO_CATS = 2  # max catégories BD TOPO retenues par polygone

OUT_DIR  = config.GEO_DIR
OUT_GPKG = os.path.join(config.GEO_DATA_DIR, "COSIA_BD_ecolab_articles.gpkg")
OUT_QML  = OUT_GPKG.replace(".gpkg", ".qml")
EMB_CACHE_NPY = os.path.join(OUT_DIR, "articles_embeddings_cache.npy")
EMB_CACHE_IDX = os.path.join(OUT_DIR, "articles_embeddings_index.csv")

# ──────────────────────────────────────────────────────────────
# ARTICLES SYNTHÉTIQUES — mentionnent le territoire SCoT
# ──────────────────────────────────────────────────────────────
SYNTHETICS: dict[str, str] = {

    # ── CLASSES COSIA ──────────────────────────────────────────

    "Bâtiment": (
        "Dans le SCoT de la Drôme, de l'Ardèche et du Vaucluse, la construction de logements "
        "neufs s'accélère dans les communes périurbaines de Valence, Romans, Montélimar et Aubenas. "
        "Les permis de construire pour des résidences, immeubles et zones pavillonnaires se multiplient "
        "sur d'anciennes terres agricoles converties en zones à urbaniser. La rénovation énergétique "
        "des bâtiments existants est au cœur des politiques locales dans ces territoires. "
        "L'artificialisation des terres naturelles et agricoles préoccupe les élus du SCoT "
        "qui cherchent à limiter l'étalement urbain tout en répondant à la demande en logements."
    ),

    "Zone imperméable": (
        "L'imperméabilisation des sols atteint un niveau préoccupant dans la plaine de la Drôme "
        "et du Vaucluse. Routes, parkings et zones commerciales couvrent chaque année des centaines "
        "d'hectares de terres auparavant perméables dans le territoire SCoT. Le ruissellement des "
        "eaux pluviales aggrave les risques d'inondation du Rhône et de ses affluents ardéchois "
        "et drômois. La loi Zéro Artificialisation Nette impose aux communes du SCoT de réduire "
        "drastiquement les nouvelles surfaces imperméables d'ici 2030."
    ),

    "Zone perméable": (
        "Les espaces verts et prairies du territoire SCoT Drôme-Ardèche-Vaucluse constituent "
        "des zones perméables essentielles qui régulent le cycle de l'eau. Les plans locaux "
        "d'urbanisme encouragent la préservation des continuités végétales et la création "
        "de jardins de pluie dans les communes de la plaine valentinoise et des collines "
        "ardéchoises. La désimperméabilisation de parkings et la végétalisation des voiries "
        "progressent dans les communes du SCoT."
    ),

    "Pelouse": (
        "Les prairies naturelles et pelouses calcaires des Baronnies drômoises et du Vaucluse "
        "abritent une biodiversité remarquable. La gestion différenciée des espaces verts "
        "se développe dans les communes du SCoT avec des fauchages tardifs pour préserver "
        "les insectes pollinisateurs. Le retour du pastoralisme sur les pelouses sèches "
        "de l'Ardèche méridionale et des plateaux drômois contribue à restaurer ces habitats."
    ),

    "Broussaille": (
        "Les milieux broussailleux et friches agricoles du SCoT Drôme-Ardèche représentent "
        "une étape de reconquête végétale sur les terrains abandonnés. Dans les collines "
        "ardéchoises et les garrigues vauclusienne, ces formations d'épineux et d'arbustes "
        "constituent un risque incendie important, justifiant des opérations de débroussaillement "
        "préventif. La gestion pastorale est encouragée dans le territoire SCoT pour maintenir "
        "la mosaïque de milieux ouverts."
    ),

    "Feuillu": (
        "Les forêts de feuillus des massifs drômois, ardéchois et vauclusiens constituent "
        "des écosystèmes riches en biodiversité. Les chênaies et hêtraies des Baronnies "
        "et du Vercors méridional font l'objet de plans de gestion durable. Le dépérissement "
        "des forêts de feuillus, aggravé par les sécheresses successives dans la Drôme "
        "et l'Ardèche, inquiète les gestionnaires forestiers du SCoT."
    ),

    "Conifère": (
        "Les forêts de conifères des massifs ardéchois et drômois constituent une ressource "
        "économique importante pour la filière bois locale. Les sapinières et pinèdes "
        "du Vercors et des Baronnies font face aux attaques de scolytes favorisées "
        "par les sécheresses répétées dans le territoire SCoT. L'Office National des Forêts "
        "mène des programmes de reboisement en essences diversifiées dans ces massifs."
    ),

    "Culture": (
        "Les grandes cultures céréalières et maraîchères dominent les plaines agricoles "
        "de la Drôme et du Vaucluse. La Drôme est le premier département bio de France "
        "avec de nombreuses exploitations engagées dans le territoire SCoT. Les agriculteurs "
        "drômois et ardéchois adaptent leurs cultures face aux aléas climatiques et aux "
        "restrictions sur les pesticides. La préservation du foncier agricole face à "
        "l'artificialisation est un enjeu central du SCoT Drôme-Ardèche-Vaucluse."
    ),

    "Terre labourée": (
        "Les terres agricoles labourées de la plaine drômoise et du Vaucluse sont au cœur "
        "des débats sur les pratiques agricoles durables. De nombreux agriculteurs du "
        "territoire SCoT abandonnent le labour profond au profit du semis direct pour "
        "préserver la qualité des sols. La mise en place de couverts végétaux hivernaux "
        "se développe dans les exploitations céréalières de la Drôme et de l'Ardèche."
    ),

    "Vigne": (
        "Le vignoble des Côtes-du-Rhône et de Crozes-Hermitage structure l'économie agricole "
        "du nord de la Drôme et de l'Ardèche dans le SCoT. La viticulture biologique progresse "
        "rapidement dans le territoire, portée par les appellations du Vaucluse et de la Drôme. "
        "Le changement climatique modifie profondément la viticulture drômoise, avec des "
        "vendanges de plus en plus précoces. Les viticulteurs du SCoT développent l'œnotourisme "
        "pour valoriser les vins de la région."
    ),

    "Serre": (
        "La production maraîchère sous serre se développe dans les plaines du Vaucluse "
        "et de la Drôme, notamment autour de Bollène et dans la vallée du Rhône. "
        "L'agrivoltaïsme suscite des débats dans le territoire SCoT sur l'association "
        "de panneaux solaires et de production agricole. Les serres maraîchères permettent "
        "de produire des légumes locaux appréciés des circuits courts drômois et ardéchois."
    ),

    "Piscine": (
        "La prolifération des piscines privées dans les communes résidentielles du SCoT "
        "Drôme-Vaucluse soulève des questions en période de sécheresse. Des restrictions "
        "d'usage ont été imposées dans plusieurs communes de la Drôme et du Vaucluse "
        "lors des étés caniculaires. La pression sur les ressources en eau dans ces "
        "territoires impose une gestion plus sobre, avec encouragement des piscines "
        "naturelles et des équipements collectifs."
    ),

    "Surface eau": (
        "Les lacs, étangs et zones humides du SCoT Drôme-Ardèche-Vaucluse, notamment "
        "le lac de Monteynard et les étangs de la plaine drômoise, constituent des "
        "écosystèmes précieux. La qualité de l'eau des plans d'eau du territoire est "
        "menacée par les cyanobactéries favorisées par les sécheresses. La restauration "
        "des zones humides de la Drôme et de l'Ardèche fait l'objet de programmes "
        "spécifiques dans le cadre du SCoT."
    ),

    "Sol nu": (
        "Les sols nus issus de carrières, chantiers et friches industrielles du SCoT "
        "Drôme-Ardèche représentent des surfaces vulnérables à l'érosion. Les opérations "
        "de renaturation et de revégétalisation des terrains dégradés dans la plaine "
        "valentinoise et les zones industrielles ardéchoises progressent. La réhabilitation "
        "des friches industrielles de la vallée du Rhône est une priorité du SCoT."
    ),

    "Neige": (
        "L'enneigement des massifs du Vercors et des Baronnies dans la Drôme recule "
        "sous l'effet du changement climatique. Les stations de ski drômoises font face "
        "à des hivers déficitaires en neige naturelle. Le manteau neigeux printanier "
        "constitue une réserve d'eau vitale pour les rivières du SCoT Drôme-Ardèche. "
        "La diversification touristique des stations de montagne drômoises est nécessaire "
        "face au recul de l'enneigement."
    ),

    # ── CATÉGORIES BD TOPO ─────────────────────────────────────

    "PARC_NATURA2000": (
        "Le réseau Natura 2000 protège de nombreux sites dans le SCoT Drôme-Ardèche-Vaucluse, "
        "notamment dans les Baronnies, le Vercors méridional et les gorges ardéchoises. "
        "Ces sites drômois et ardéchois hébergent des espèces protégées comme l'aigle de Bonelli "
        "et de nombreuses orchidées sauvages. Les documents d'objectifs Natura 2000 du SCoT "
        "définissent les mesures de conservation compatibles avec l'agriculture et le tourisme locaux. "
        "Des conflits surgissent parfois entre protection Natura 2000 et projets d'aménagement "
        "dans la Drôme et le Vaucluse."
    ),

    "PARC_PNR": (
        "Le Parc Naturel Régional du Vercors et le PNR des Baronnies Provençales couvrent "
        "une large partie du SCoT Drôme-Ardèche. Ces parcs protègent des paysages emblématiques "
        "comme les hauts plateaux karstiques drômois et les alpilles ardéchoises. La charte "
        "des parcs naturels régionaux du territoire SCoT soutient l'agriculture locale, "
        "les produits du terroir drômois et le tourisme doux. Ces PNR sont des acteurs clés "
        "de la biodiversité et du développement durable dans la Drôme et le Vaucluse."
    ),

    "PARC_RESERVE": (
        "Les réserves naturelles et arrêtés de protection de biotope du SCoT Drôme-Ardèche "
        "protègent les habitats les plus vulnérables du territoire. Les réserves biologiques "
        "des forêts ardéchoises et drômoises permettent l'évolution naturelle des écosystèmes. "
        "Le Conservatoire des Espaces Naturels de la Drôme et de l'Ardèche gère un réseau "
        "de sites préservés dans le SCoT. Ces espaces constituent des refuges de biodiversité "
        "essentiels dans la trame verte et bleue du territoire."
    ),

    "GEOPARC": (
        "Le Géoparc mondial UNESCO du Massif du Vercors valorise le patrimoine géologique "
        "exceptionnel de la Drôme et de l'Isère. Les formations karstiques, grottes et canyons "
        "du Vercors drômois témoignent de millions d'années d'histoire géologique. Ce géoparc "
        "attire des visiteurs et géologues dans le SCoT Drôme pour découvrir les fossiles "
        "marins jurassiques des plateaux calcaires. La préservation du patrimoine géologique "
        "et la sensibilisation des jeunes générations drômoises sont au cœur de sa mission."
    ),

    "FORET_PUBLIQUE": (
        "Les forêts domaniales gérées par l'ONF couvrent de larges surfaces dans le SCoT "
        "Drôme-Ardèche-Vaucluse, notamment dans le Vercors, les Baronnies et les Cévennes. "
        "L'Office National des Forêts mène des coupes de bois programmées et des programmes "
        "de reboisement dans ces massifs forestiers publics du territoire. Face au dépérissement "
        "forestier lié aux sécheresses drômoises et ardéchoises, l'ONF diversifie les essences "
        "reboisées. Ces forêts publiques jouent un rôle de protection contre les risques naturels "
        "dans le SCoT."
    ),

    "ZONE_VEG_FORET": (
        "Les massifs forestiers du SCoT Drôme-Ardèche-Vaucluse, dont le Vercors, les Baronnies "
        "et les Cévennes ardéchoises, constituent des réservoirs de biodiversité essentiels. "
        "Les incendies de forêt constituent une menace croissante dans ces territoires secs, "
        "aggravée par les canicules et sécheresses. La gestion forestière durable et la "
        "sylviculture proche de la nature se développent dans les forêts drômoises et ardéchoises. "
        "Le reboisement post-incendie et la diversification des essences sont prioritaires "
        "pour la résilience des forêts du SCoT."
    ),

    "ZONE_VEG_VIGNE": (
        "Les vignobles des Côtes-du-Rhône, Crozes-Hermitage, Cornas et Saint-Joseph "
        "structurent le paysage agricole du SCoT Drôme-Ardèche nord et du Vaucluse. "
        "La viticulture biologique et biodynamique progresse dans ces appellations du "
        "territoire SCoT. Le changement climatique avance les vendanges dans la Drôme "
        "et l'Ardèche, modifiant les profils des vins. Les vignerons indépendants et "
        "caves coopératives du SCoT développent l'œnotourisme local."
    ),

    "ZONE_VEG_VERGER": (
        "L'arboriculture fruitière occupe les vallées de la Drôme et du Vaucluse avec "
        "des vergers d'abricotiers, pommiers et poiriers caractéristiques du SCoT. "
        "Les abricots de la Drôme et les cerises de la région de Buis-les-Baronnies "
        "sont des productions emblématiques du territoire. Les producteurs bio du SCoT "
        "développent des techniques alternatives aux pesticides. Les marchés locaux "
        "et circuits courts valorisent ces fruits du terroir drômois et vauclusien."
    ),

    "ZONE_VEG_LANDE": (
        "Les landes à genêts et bruyères des garrigues ardéchoises et des plateaux calcaires "
        "du SCoT Drôme-Vaucluse constituent des habitats semi-naturels précieux. Les haies "
        "bocagères du territoire ardéchois abritent une faune abondante d'insectes pollinisateurs "
        "et de rapaces. La restauration des haies dans le cadre des plans de biodiversité "
        "du SCoT Drôme progresse. Les landes méditerranéennes du Vaucluse et de l'Ardèche "
        "méridionale régressent sous l'effet de la reforestation spontanée."
    ),

    "COURS_EAU": (
        "La Drôme, l'Ardèche, l'Eygues et leurs affluents constituent le réseau hydrographique "
        "vital du SCoT. La qualité des rivières drômoises et ardéchoises est menacée par "
        "les nitrates agricoles et les étiages sévères lors des sécheresses estivales. "
        "Les programmes de restauration de la continuité écologique de la Drôme et de "
        "l'Ardèche visent la libre circulation des poissons migrateurs. Les zones humides "
        "riveraines du SCoT jouent un rôle épurateur naturel essentiel."
    ),

    "PLAN_EAU": (
        "Le lac de Monteynard-Avignonet et les retenues artificielles du SCoT Drôme-Ardèche "
        "constituent des réservoirs d'eau douce essentiels pour l'irrigation et l'eau potable. "
        "Les sécheresses estivales provoquent une baisse préoccupante du niveau de ces "
        "retenues dans la Drôme et le Vaucluse. La qualité de l'eau des plans d'eau "
        "drômois est menacée par les cyanobactéries lors des étés caniculaires. "
        "La pêche de loisir et les activités nautiques constituent des usages économiques "
        "importants de ces lacs dans le SCoT."
    ),

    "SURFACE_HYDRO": (
        "Les zones humides, mares et prairies inondables du SCoT Drôme-Ardèche-Vaucluse "
        "abritent des amphibiens, libellules et plantes aquatiques protégées. La destruction "
        "des zones humides par le drainage et l'urbanisation dans la plaine drômoise "
        "a conduit à leur disparition partielle. Des programmes de restauration de mares "
        "et zones humides se développent dans le territoire SCoT. Ces milieux stockent "
        "du carbone et régulent les inondations dans les vallées ardéchoises et drômoises."
    ),

    "EOLIENNE": (
        "Les parcs éoliens se développent sur les crêtes et plateaux du SCoT Drôme-Ardèche, "
        "notamment dans les Baronnies et le plateau ardéchois. L'implantation d'éoliennes "
        "dans ces territoires suscite des oppositions locales liées à l'impact paysager "
        "et sur la faune, notamment les rapaces nicheurs drômois. Les procédures "
        "d'autorisation environnementale sont complexes dans le SCoT. L'éolien représente "
        "une source d'énergie renouvelable dans la transition énergétique du territoire."
    ),

    "BARRAGE": (
        "Les barrages hydroélectriques de Montelimar, Saint-Vallier et Beauchastel "
        "sur le Rhône bordent le territoire SCoT Drôme-Ardèche. Ces ouvrages produisent "
        "de l'électricité renouvelable mais perturbent la continuité écologique du Rhône. "
        "Les passes à poissons et la restauration des frayères sont des enjeux majeurs "
        "pour la faune piscicole du SCoT. La gestion des débits pour l'irrigation agricole "
        "dans la Drôme et le Vaucluse est au cœur des conflits d'usage de l'eau."
    ),
}


# ──────────────────────────────────────────────────────────────
# EMBEDDINGS
# ──────────────────────────────────────────────────────────────

def get_or_build_embeddings(model, docs: list[str], urls: list[str]) -> np.ndarray:
    """Charge le cache embeddings ou recalcule si articles changés."""
    if os.path.exists(EMB_CACHE_NPY) and os.path.exists(EMB_CACHE_IDX):
        cached = pd.read_csv(EMB_CACHE_IDX)["url"].tolist()
        if cached == urls:
            print("  Cache embeddings chargé.")
            return np.load(EMB_CACHE_NPY)
        print("  Cache obsolète, recalcul...")
    print(f"  Encoding {len(docs)} articles...")
    embs = model.encode(docs, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
    np.save(EMB_CACHE_NPY, embs)
    pd.DataFrame({"url": urls}).to_csv(EMB_CACHE_IDX, index=False)
    return embs


# ──────────────────────────────────────────────────────────────
# OLLAMA
# ──────────────────────────────────────────────────────────────

def ollama_score(titre: str, extrait: str, label: str) -> float:
    """Score 0.0-1.0 de pertinence d'un article pour un label COSIA/BD TOPO."""
    prompt = (
        f"Tu évalues la pertinence d'un article de presse pour une parcelle de territoire.\n"
        f"La parcelle est de type : {label}\n\n"
        f"Article :\n"
        f"Titre : {titre[:150]}\n"
        f"Extrait : {extrait[:300]}\n\n"
        f"Question : Dans quelle mesure cet article est-il directement pertinent pour "
        f"comprendre les enjeux d'une telle parcelle dans le SCoT Drôme-Ardèche-Vaucluse ?\n\n"
        f"Réponds avec UN seul nombre entre 0.0 et 1.0 :\n"
        f"- 0.9 à 1.0 = article très directement lié à ce type de parcelle et au territoire\n"
        f"- 0.6 à 0.8 = article lié au sujet ou au territoire\n"
        f"- 0.3 à 0.5 = article partiellement pertinent\n"
        f"- 0.0 à 0.2 = article hors sujet\n\n"
        f"Score (un seul nombre, ex: 0.75) ="
    )
    try:
        r = requests.post(config.OLLAMA_URL, json={
            "model": config.OLLAMA_MODEL, "prompt": prompt,
            "stream": False, "options": {"temperature": 0, "num_predict": 50}
        }, timeout=config.OLLAMA_TIMEOUT)
        txt = r.json().get("response", "0").strip()
        m = re.search(r"([01]?\.\d+|\d(?!\d))", txt)
        val = float(m.group(1)) if m else 0.0
        return min(1.0, max(0.0, val))
    except Exception:
        return 0.0


def ollama_resume(articles: list[dict], label: str) -> str:
    """Génère un paragraphe résumé de 3-5 phrases à partir des articles."""
    arts_txt = "\n".join(
        f"- {a['titre'][:100]} : {a['extrait'][:200]}"
        for a in articles[:6] if a.get("titre")
    )
    prompt = (
        f"Tu es expert en aménagement du territoire du SCoT Drôme-Ardèche-Vaucluse.\n"
        f"En 3 à 5 phrases, explique le contexte territorial pour une parcelle de type '{label}' "
        f"en t'inspirant de ces articles de presse locaux :\n{arts_txt}\n\n"
        f"Résumé :"
    )
    try:
        r = requests.post(config.OLLAMA_URL, json={
            "model": config.OLLAMA_MODEL, "prompt": prompt,
            "stream": False, "options": {"temperature": 0.3, "num_predict": 300}
        }, timeout=config.OLLAMA_TIMEOUT)
        return r.json().get("response", "").strip()
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────
# SÉLECTION ARTICLES PAR LABEL
# ──────────────────────────────────────────────────────────────

def select_top_articles_for_label(
    label: str, synth_emb: np.ndarray,
    df_arts: pd.DataFrame, art_embs: np.ndarray,
    top_n: int = TOP_LLM,
) -> list[dict]:
    """
    Cosine similarity → top TOP_COSINE → Ollama score → top_n.
    Retourne liste de dicts {titre, url, extrait, score_llm}.
    """
    sims = cosine_similarity(synth_emb.reshape(1, -1), art_embs)[0]
    top_idx = np.argsort(sims)[::-1][:TOP_COSINE]

    scored = []
    for idx in top_idx:
        row = df_arts.iloc[idx]
        titre   = str(row.get("titre", "") or "")
        extrait = str(row.get("contenu", "") or "")[:300]
        url     = str(row.get("url", "") or "")
        score   = ollama_score(titre, extrait, label)
        scored.append({"titre": titre, "url": url, "extrait": extrait,
                        "score_llm": score, "score_cos": float(sims[idx])})
        time.sleep(config.OLLAMA_SLEEP)

    # Fallback cosinus si LLM n'a pas scoré (tous à 0.0)
    if all(s["score_llm"] == 0.0 for s in scored):
        scored.sort(key=lambda x: x["score_cos"], reverse=True)
    else:
        scored.sort(key=lambda x: x["score_llm"], reverse=True)
    return scored[:top_n]


# ──────────────────────────────────────────────────────────────
# QML MAP TIPS
# ──────────────────────────────────────────────────────────────

def generate_qml_maptips(path: str) -> None:
    """Génère un QML avec Map Tips HTML pour affichage dans QGIS.

    Les articles viennent du CSV COSIA_articles_par_classe.csv joint par 'classe'.
    QGIS préfixe les champs joints : "COSIA_articles_par_classe_art_cosia_1_titre"
    → Dans QGIS : joindre le CSV sur le champ 'classe' avant d'appliquer ce QML.
    """
    # Préfixe QGIS pour la jointure (nom du CSV sans extension)
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

    html = (
        '<table width="400"><tr><td>'
        '<b>Classe :</b> [% "classe" %] &nbsp; '
        '<b>Score :</b> [% format_number("score_polygone", 3) %]<hr/>'
        '<b>Articles liés à la classe :</b><br/>' + cosia_arts +
        '<hr/><b>Articles contexte territorial :</b><br/>' + ctx_arts +
        '<hr/><b>Résumé :</b><br/>[% "cosia_resume_article" %]'
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
</qgis>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(qml)
    print(f"  QML Map Tips → {os.path.basename(path)}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("PIPELINE V2 ETAPE 6 — Articles par parcelle")
    print("=" * 60)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Lire SEULEMENT les valeurs distinctes (pas les 15M lignes !)
    gpkg_in = config.CARTE_FINALE_GPKG
    if not os.path.exists(gpkg_in):
        raise FileNotFoundError(f"Introuvable : {gpkg_in} — lance d'abord l'étape 5.")
    print(f"\nLecture DISTINCT depuis {os.path.basename(gpkg_in)} (sqlite3 read-only)...")
    import sqlite3
    _uri = "file:" + gpkg_in.replace("\\", "/") + "?mode=ro"
    _conn = sqlite3.connect(_uri, uri=True)
    # Classes COSIA distinctes
    classes = sorted([
        r[0] for r in _conn.execute(f'SELECT DISTINCT "{config.COL_CLASS}" FROM "COSIA_SCORE_finale"')
        if r[0]
    ])
    # Catégories BD TOPO présentes dans les colonnes
    _col_info = _conn.execute('PRAGMA table_info("COSIA_SCORE_finale")').fetchall()
    bdtopo_cats = [c[1].replace("bdtopo_", "") for c in _col_info if c[1].startswith("bdtopo_")]
    _conn.close()
    print(f"  {len(classes)} classes COSIA : {classes}")
    print(f"  {len(bdtopo_cats)} catégories BD TOPO")

    # ── 2. Charger les articles (toutes les sources) ─────────────
    print("\nChargement articles (toutes sources)...")
    NEWS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "donnéesNews")
    REAL_SOURCES = [
        os.path.join(NEWS_DIR, "ledauphine_articles.csv"),
        os.path.join(NEWS_DIR, "ledauphine_articles_202026.csv"),
        os.path.join(NEWS_DIR, "ledauphine_articles_2022.csv"),
        os.path.join(NEWS_DIR, "cerema_actualites.csv"),
        os.path.join(NEWS_DIR, "actu_environnement_actualites.csv"),
        os.path.join(NEWS_DIR, "lemonde_articles.csv"),
        os.path.join(NEWS_DIR, "vert_eco_articles.csv"),
    ]
    frames = []
    for src in REAL_SOURCES:
        if os.path.exists(src):
            df_src = pd.read_csv(src, encoding="utf-8")
            frames.append(df_src)
            print(f"  {os.path.basename(src):<45} {len(df_src):>7,} articles")
        else:
            print(f"  WARN: introuvable → {src}")
    df_arts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for col in ("titre", "contenu", "url"):
        if col not in df_arts.columns:
            df_arts[col] = ""
    df_arts = (
        df_arts.dropna(subset=["url"])
               .drop_duplicates("url")
               .reset_index(drop=True)
    )
    df_arts["_doc"] = (df_arts["titre"].fillna("") + ". " + df_arts["contenu"].fillna("")).str[:500]
    print(f"  → {len(df_arts):,} articles uniques (toutes sources)")

    # ── 3. Embeddings articles (cache) ────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("pip install sentence-transformers")

    print("\nChargement modèle sentence-transformers...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    art_embs = get_or_build_embeddings(model, df_arts["_doc"].tolist(), df_arts["url"].tolist())
    print(f"  Embeddings : {art_embs.shape}")

    # ── 4. Embeddings synthétiques ────────────────────────────────
    print("\nEncoding articles synthétiques...")
    synth_labels = list(SYNTHETICS.keys())
    synth_embs   = model.encode(list(SYNTHETICS.values()), convert_to_numpy=True, show_progress_bar=False)
    synth_map    = {label: emb for label, emb in zip(synth_labels, synth_embs)}
    print(f"  {len(synth_labels)} labels encodés")

    # ── 5. Top articles par label (avec cache JSON) ───────────────
    cache_json = os.path.join(OUT_DIR, "articles_par_label_cache.json")
    if os.path.exists(cache_json):
        print(f"\nCache articles par label trouvé → chargement...")
        with open(cache_json, encoding="utf-8") as f:
            label_articles: dict = json.load(f)
    else:
        label_articles = {}

    print(f"\nScoring LLM par label ({len(synth_labels)} labels × {TOP_COSINE} candidats)...")
    for i, label in enumerate(synth_labels, 1):
        if label in label_articles:
            print(f"  [{i:02d}/{len(synth_labels)}] {label:<30} (cache)")
            continue
        print(f"  [{i:02d}/{len(synth_labels)}] {label:<30} ...", end="", flush=True)
        arts = select_top_articles_for_label(label, synth_map[label], df_arts, art_embs)
        label_articles[label] = arts
        with open(cache_json, "w", encoding="utf-8") as f:
            json.dump(label_articles, f, ensure_ascii=False, indent=2)
        top_score = arts[0]["score_llm"] if arts else 0.0
        print(f" top_score={top_score:.2f}  '{arts[0]['titre'][:40] if arts else '—'}'")

    print(f"\n  {len(label_articles)} labels scorés, cache sauvegardé")

    # ── 6. Résumé Mistral par classe COSIA (15 labels, pas 15M lignes)
    resume_cache_json = os.path.join(OUT_DIR, "resume_par_classe_cache.json")
    if os.path.exists(resume_cache_json):
        with open(resume_cache_json, encoding="utf-8") as f:
            resume_cache: dict = json.load(f)
    else:
        resume_cache = {}

    # Meilleurs bdtopo globaux (top 2 par score LLM moyen)
    bdtopo_scores = {
        cat: max((a["score_llm"] for a in label_articles.get(cat, [])), default=0.0)
        for cat in bdtopo_cats
    }
    top_bdtopo = sorted(bdtopo_scores, key=bdtopo_scores.get, reverse=True)[:MAX_BDTOPO_CATS]

    print(f"\nGénération résumés Mistral ({len(classes)} classes)...")
    classe_data: dict[str, dict] = {}

    for classe in classes:
        arts_cosia = label_articles.get(classe, [])[:TOP_LLM]
        arts_ctx   = []
        seen_urls  = {a["url"] for a in arts_cosia}
        for cat in top_bdtopo:
            for a in label_articles.get(cat, [])[:TOP_LLM]:
                if a["url"] not in seen_urls:
                    arts_ctx.append(a)
                    seen_urls.add(a["url"])
        arts_ctx = arts_ctx[:6]

        if classe not in resume_cache:
            all_arts = arts_cosia + arts_ctx
            resume_cache[classe] = ollama_resume(all_arts, classe)
            with open(resume_cache_json, "w", encoding="utf-8") as f:
                json.dump(resume_cache, f, ensure_ascii=False, indent=2)
            time.sleep(config.OLLAMA_SLEEP)
            print(f"  {classe:<25} → résumé généré")
        else:
            print(f"  {classe:<25} (cache)")

        classe_data[classe] = {
            "arts_cosia": arts_cosia,
            "arts_ctx":   arts_ctx,
            "resume":     resume_cache.get(classe, ""),
        }

    # ── 7. Export CSV articles par classe (15 lignes, ~quelques Ko)
    # Aucune modification du GPKG → zéro problème OneDrive / disque
    OUT_CSV = os.path.join(os.path.dirname(gpkg_in), "COSIA_articles_par_classe.csv")
    rows_csv = []
    for classe, d in classe_data.items():
        row = {"classe": classe}
        for i, a in enumerate(d["arts_cosia"][:3], 1):
            row[f"art_cosia_{i}_titre"] = a.get("titre", "")
            row[f"art_cosia_{i}_url"]   = a.get("url",   "")
        for j in range(len(d["arts_cosia"]), 3):
            row[f"art_cosia_{j+1}_titre"] = ""
            row[f"art_cosia_{j+1}_url"]   = ""
        for i, a in enumerate(d["arts_ctx"][:6], 1):
            row[f"art_ctx_{i}_titre"] = a.get("titre", "")
            row[f"art_ctx_{i}_url"]   = a.get("url",   "")
        for j in range(len(d["arts_ctx"]), 6):
            row[f"art_ctx_{j+1}_titre"] = ""
            row[f"art_ctx_{j+1}_url"]   = ""
        row["resume"] = d["resume"]
        rows_csv.append(row)

    df_csv = pd.DataFrame(rows_csv)
    df_csv.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  → {os.path.basename(OUT_CSV)} ({len(df_csv)} lignes, {os.path.getsize(OUT_CSV)//1024} Ko)")

    # ── 8. QML Map Tips ────────────────────────────────────────────
    OUT_QML_FINAL = gpkg_in.replace(".gpkg", "_articles.qml")
    generate_qml_maptips(OUT_QML_FINAL)

    # ── Résumé ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Classes COSIA         : {len(classe_data)}")
    print(f"  Labels scorés         : {len(label_articles)}")
    print(f"  Résumés générés       : {len(resume_cache)}")
    print(f"\n  Dans QGIS :")
    gpkg_name = os.path.basename(gpkg_in)
    csv_name  = os.path.basename(OUT_CSV)
    qml_name  = os.path.basename(OUT_QML_FINAL)
    print(f"    1. Glisse {gpkg_name}  (couche COSIA_SCORE_finale)")
    print(f"    2. Ajoute couche texte délimité : {csv_name}  (sans géométrie)")
    print(f"    3. Couche → Propriétés → Jointures → joindre par 'classe'")
    print(f"    4. Applique le QML : {qml_name}")
    print(f"    → Map Tips : survol/clic → articles + résumé")
    print(f"{'─'*60}")
    print("ETAPE 6 — TERMINEE")


if __name__ == "__main__":
    run()
