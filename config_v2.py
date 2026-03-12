import os

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(os.path.dirname(BASE_DIR), "pipeline")
GEO_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "ecolab2_territoire_scot")
OUT_DIR      = os.path.join(BASE_DIR, "outputs")
GEO_DIR      = os.path.join(OUT_DIR, "geo")

ARTICLES_WITH_TOPICS_CSV = os.path.join(PIPELINE_DIR, "outputs", "bertopic_v2", "articles_with_topic_labels.csv")
TOPICS_INFO_CSV          = os.path.join(PIPELINE_DIR, "outputs", "bertopic_v2", "topics_info_v2.csv")
DOCS_SCORED_CSV         = os.path.join(OUT_DIR, "docs_scored_v2.csv")
TOPIC_SCORE_SUMMARY_CSV = os.path.join(OUT_DIR, "topic_score_summary_v2.csv")

COSIA_DIRS = [os.path.join(GEO_DATA_DIR, d) for d in [
    "COSIA_1-0__GPKG_LAMB93_D007_2023-01-01",
    "COSIA_1-0__GPKG_LAMB93_D026_2023-01-01",
    "COSIA_1-0__GPKG_LAMB93_D084_2024-01-01",
]]

COL_ID    = "numero"
COL_CLASS = "classe"
SCOT_MASK_FILE  = os.path.join(GEO_DATA_DIR, "Parcelles_SCOT.gpkg")
SCOT_MASK_LAYER = None

def _bdtopo_root(dept):
    d = f"BDTOPO_3-5_TOUSTHEMES_SHP_LAMB93_{dept}_2025-12-15"
    ed = d.replace("BDTOPO_3-5_TOUSTHEMES_SHP_LAMB93_", "BDT_3-5_SHP_LAMB93_").replace("_2025-12-15", "_ED2025-12-15")
    return os.path.join(GEO_DATA_DIR, d, d, "BDTOPO", "1_DONNEES_LIVRAISON_2025-12-00073", ed)

BDTOPO_ROOTS = [_bdtopo_root(d) for d in ("D007", "D026", "D084")]
BDTOPO_ROOT  = BDTOPO_ROOTS[1]  # D026 (rétrocompatibilité)


CLASSES_ATTENDUES = [
    "Bâtiment", "Zone imperméable", "Zone perméable", "Pelouse", "Broussaille",
    "Feuillu", "Conifère", "Culture", "Terre labourée", "Vigne", "Serre",
    "Piscine", "Surface eau", "Sol nu", "Neige",
]

COSIA_SCORE_GPKG = os.path.join(GEO_DIR, "COSIA_SCORE_v2.gpkg")
COSIA_SCORE_CSV  = os.path.join(GEO_DIR, "COSIA_SCORE_v2_attrs.csv")
FINALE_GPKG      = os.path.join(GEO_DATA_DIR, "COSIA_SCORE_finale_ecolab.gpkg")
FINALE_CSV       = os.path.join(GEO_DIR, "COSIA_SCORE_finale_attrs.csv")
CARTE_FINALE_GPKG    = os.path.join(GEO_DATA_DIR, "COSIA_BD_ecolab_carte_finale.gpkg")
ARTICLES_GPKG        = os.path.join(GEO_DATA_DIR, "COSIA_BD_ecolab_articles.gpkg")

OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "mistral"
OLLAMA_TIMEOUT = 120
OLLAMA_SLEEP   = 0.3
MIN_DOC_LEN    = 5

TOPIC_COSIA_MAP = {
     0: ["Zone imperméable"],
    11: ["Zone imperméable", "Zone perméable"],
   127: ["Zone imperméable"],
   135: ["Zone imperméable", "Sol nu"],
     8: ["Culture", "Terre labourée", "Vigne"],
    27: ["Culture", "Vigne", "Serre"],
    57: ["Culture", "Terre labourée", "Vigne"],
    91: ["Culture", "Terre labourée"],
   198: ["Culture", "Terre labourée"],
   205: ["Culture", "Terre labourée", "Vigne"],
   413: ["Zone perméable", "Culture"],
   513: ["Culture", "Terre labourée"],
    15: ["Surface eau", "Piscine"],
    25: ["Surface eau", "Zone perméable"],
    59: ["Surface eau", "Zone perméable"],
    62: ["Surface eau"],
    74: ["Surface eau", "Piscine"],
   113: ["Surface eau"],
   184: ["Surface eau"],
   337: ["Surface eau", "Zone perméable"],
    29: ["Feuillu", "Conifère"],
    35: ["Feuillu", "Conifère", "Zone perméable"],
    53: ["Feuillu", "Conifère"],
   141: ["Feuillu", "Conifère"],
   291: ["Feuillu", "Conifère"],
   499: ["Feuillu", "Conifère"],
    22: ["Zone perméable", "Pelouse", "Broussaille"],
    26: ["Broussaille", "Feuillu", "Zone perméable"],
    60: ["Zone perméable", "Pelouse", "Broussaille"],
    65: ["Zone perméable", "Pelouse", "Broussaille"],
    66: ["Zone perméable", "Pelouse"],
    68: ["Zone perméable", "Culture"],
   108: ["Zone perméable", "Feuillu", "Broussaille"],
   143: ["Broussaille", "Feuillu"],
   155: ["Zone perméable", "Pelouse", "Broussaille"],
   297: ["Zone perméable", "Surface eau"],
   303: ["Zone perméable", "Feuillu"],
   325: ["Zone perméable", "Feuillu", "Broussaille"],
   385: ["Zone perméable", "Pelouse"],
   153: ["Sol nu", "Zone perméable"],
   175: ["Zone perméable", "Sol nu", "Culture"],
   251: ["Zone imperméable", "Sol nu"],
   321: ["Zone imperméable", "Sol nu"],
   361: ["Zone perméable"],
     9: ["Bâtiment", "Zone imperméable"],
   244: ["Bâtiment", "Zone imperméable"],
   323: ["Zone perméable", "Pelouse"],
   392: ["Bâtiment"],
   468: ["Zone imperméable", "Bâtiment"],
   527: ["Zone perméable", "Pelouse"],
    18: ["Bâtiment"],
    33: ["Zone imperméable", "Bâtiment"],
    79: ["Bâtiment"],
   138: ["Bâtiment"],
   160: ["Bâtiment"],
   209: ["Bâtiment", "Zone imperméable"],
   260: ["Bâtiment"],
   262: ["Bâtiment"],
    12: ["Serre", "Culture", "Bâtiment"],
    41: ["Zone perméable"],
   186: ["Zone perméable"],
    19: ["Neige"],
    37: ["Zone perméable", "Broussaille", "Feuillu"],
    44: ["Neige"],
    48: ["Neige", "Zone perméable"],
    34: ["Zone perméable", "Neige"],
    84: ["Zone imperméable"],
    90: ["Zone imperméable"],
   181: ["Zone perméable"],
   518: ["Zone perméable"],
   445: ["Zone perméable"],
}

_P = os.path.join  # alias pour lisibilité
BDTOPO_CATEGORIES = {
    # Chemins relatifs à chaque BDTOPO_ROOT (le script 04 itère sur les 3 depts)
    "PARC_NATURA2000": {"shp": _P("ZONES_REGLEMENTEES", "PARC_OU_RESERVE.shp"), "field": "NATURE", "values": ["Site Natura 2000"]},
    "PARC_PNR":        {"shp": _P("ZONES_REGLEMENTEES", "PARC_OU_RESERVE.shp"), "field": "NATURE", "values": ["Parc naturel régional"]},
    "PARC_RESERVE":    {"shp": _P("ZONES_REGLEMENTEES", "PARC_OU_RESERVE.shp"), "field": "NATURE", "values": ["Réserve naturelle", "Réserve biologique", "Réserve nationale de chasse et de faune sauvage", "Réserve de biosphère", "Arrêté de protection", "Site acquis ou assimilé des conservatoires d'es", "Zone naturelle"]},
    "GEOPARC":         {"shp": _P("ZONES_REGLEMENTEES", "PARC_OU_RESERVE.shp"), "field": "NATURE", "values": ["Géoparc mondial UNESCO"]},
    "FORET_PUBLIQUE":  {"shp": _P("ZONES_REGLEMENTEES", "FORET_PUBLIQUE.shp"),  "field": None, "values": None},
    "ZONE_VEG_FORET":  {"shp": _P("OCCUPATION_DU_SOL", "ZONE_DE_VEGETATION.shp"), "field": "NATURE", "values": ["Forêt fermée de conifères", "Forêt fermée de feuillus", "Forêt fermée mixte", "Forêt ouverte", "Bois"]},
    "ZONE_VEG_VIGNE":  {"shp": _P("OCCUPATION_DU_SOL", "ZONE_DE_VEGETATION.shp"), "field": "NATURE", "values": ["Vigne"]},
    "ZONE_VEG_VERGER": {"shp": _P("OCCUPATION_DU_SOL", "ZONE_DE_VEGETATION.shp"), "field": "NATURE", "values": ["Verger", "Peupleraie"]},
    "ZONE_VEG_LANDE":  {"shp": _P("OCCUPATION_DU_SOL", "ZONE_DE_VEGETATION.shp"), "field": "NATURE", "values": ["Lande ligneuse", "Haie"]},
    "COURS_EAU":       {"shp": _P("HYDROGRAPHIE", "COURS_D_EAU.shp"), "field": None, "values": None},
    "PLAN_EAU":        {"shp": _P("HYDROGRAPHIE", "PLAN_D_EAU.shp"),  "field": None, "values": None},
    "SURFACE_HYDRO":   {"shp": _P("HYDROGRAPHIE", "SURFACE_HYDROGRAPHIQUE.shp"), "field": None, "values": None},
    "EOLIENNE":        {"shp": _P("BATI", "CONSTRUCTION_PONCTUELLE.shp"), "field": "NATURE", "values": ["Eolienne"]},
    "BARRAGE":         {"shp": _P("BATI", "CONSTRUCTION_SURFACIQUE.shp"), "field": "NATURE", "values": ["Barrage"]},
}

TOPIC_BDTOPO_MAP = {
    22:  ["PARC_NATURA2000", "PARC_PNR", "PARC_RESERVE", "GEOPARC"],
    26:  ["PARC_RESERVE", "ZONE_VEG_FORET"],
    60:  ["PARC_RESERVE", "PARC_NATURA2000"],
    65:  ["PARC_RESERVE", "ZONE_VEG_FORET", "ZONE_VEG_LANDE"],
    66:  ["PARC_RESERVE", "ZONE_VEG_VERGER", "ZONE_VEG_LANDE"],
   108:  ["ZONE_VEG_FORET", "PARC_RESERVE"],
   143:  ["ZONE_VEG_FORET", "PARC_RESERVE"],
   155:  ["PARC_RESERVE", "PARC_NATURA2000"],
   297:  ["COURS_EAU", "PLAN_EAU", "SURFACE_HYDRO"],
   303:  ["ZONE_VEG_FORET", "PARC_RESERVE"],
   325:  ["PARC_NATURA2000", "PARC_PNR", "PARC_RESERVE"],
   385:  ["PARC_RESERVE", "ZONE_VEG_LANDE"],
    29:  ["ZONE_VEG_FORET", "FORET_PUBLIQUE"],
    35:  ["ZONE_VEG_FORET", "FORET_PUBLIQUE"],
    53:  ["FORET_PUBLIQUE", "ZONE_VEG_FORET"],
   141:  ["FORET_PUBLIQUE", "ZONE_VEG_FORET"],
   291:  ["ZONE_VEG_FORET", "ZONE_VEG_VERGER"],
   499:  ["ZONE_VEG_FORET", "FORET_PUBLIQUE"],
    15:  ["COURS_EAU", "PLAN_EAU", "SURFACE_HYDRO", "BARRAGE"],
    25:  ["COURS_EAU", "PLAN_EAU", "SURFACE_HYDRO"],
    59:  ["COURS_EAU", "PLAN_EAU", "SURFACE_HYDRO", "BARRAGE"],
    62:  ["PLAN_EAU", "SURFACE_HYDRO"],
    74:  ["COURS_EAU", "PLAN_EAU", "SURFACE_HYDRO"],
   113:  ["BARRAGE", "PLAN_EAU", "SURFACE_HYDRO"],
   184:  ["PLAN_EAU", "SURFACE_HYDRO"],
   337:  ["COURS_EAU", "PLAN_EAU"],
     8:  ["ZONE_VEG_VIGNE", "ZONE_VEG_VERGER"],
    27:  ["ZONE_VEG_VERGER", "ZONE_VEG_VIGNE"],
    57:  ["ZONE_VEG_VIGNE", "ZONE_VEG_VERGER"],
    91:  ["ZONE_VEG_VIGNE", "ZONE_VEG_VERGER"],
   198:  ["ZONE_VEG_VIGNE", "ZONE_VEG_VERGER"],
   205:  ["ZONE_VEG_VIGNE", "ZONE_VEG_VERGER"],
   153:  ["PARC_NATURA2000", "PARC_PNR", "PARC_RESERVE"],
   175:  ["PARC_NATURA2000", "PARC_PNR", "PARC_RESERVE", "ZONE_VEG_FORET"],
   251:  ["PARC_NATURA2000", "PARC_PNR"],
   361:  ["PARC_RESERVE", "PARC_NATURA2000"],
    37:  ["PARC_PNR", "PARC_RESERVE", "GEOPARC"],
    41:  ["EOLIENNE", "ZONE_VEG_LANDE"],
   186:  ["EOLIENNE"],
}
