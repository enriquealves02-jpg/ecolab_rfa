"""
EcoLab — Carte interactive v2 (Flask + Leaflet)
================================================
Architecture :
  - Flask (port 5050) : API bbox → GeoJSON, détail par fid
  - Leaflet.js (navigateur) : rendu carte, pan/zoom fluide sans Python
  - Streamlit : panneau détail à droite + filtres dans la sidebar

Lancement : streamlit run app_carte2.py
"""

import json, os, sys, sqlite3, struct, threading
from functools import lru_cache
import pandas as pd
import shapely.wkb, shapely.geometry, shapely.ops
import streamlit as st
import streamlit.components.v1 as components
from flask import Flask, jsonify, request, Response
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # pipeline_v2/
ROOT_DIR = os.path.dirname(BASE_DIR)                    # ecolab/

IMG_BARPLOT    = os.path.join(ROOT_DIR, "donnéesNews", "bertopic_out", "figures_named", "barplot_topic_scores_named.png")
IMG_TIMESERIES = os.path.join(ROOT_DIR, "donnéesNews", "bertopic_out", "figures", "timeseries_sentiment.png")
sys.path.insert(0, BASE_DIR)
import config_v2 as config

GPKG       = config.CARTE_FINALE_GPKG
GEO_DIR    = os.path.dirname(GPKG)
CSV_BDTOPO = os.path.join(GEO_DIR, "BDTOPO_resume_par_combo.csv")
CSV_COSIA  = os.path.join(GEO_DIR, "COSIA_articles_par_classe.csv")
LAYER      = "COSIA_SCORE_finale"
RTREE      = f"rtree_{LAYER}_geom"
FLASK_PORT      = 5050
MAX_FEAT        = 600
MAX_FEAT_ALL    = 2000   # limite pour le mode "charger tout" (filtres avec peu de résultats)
TILES_DIR       = os.path.join(ROOT_DIR, "tiles")
VTILES_DIR      = os.path.join(ROOT_DIR, "tiles_vector")
BBOX_CACHE_FILE = os.path.join(ROOT_DIR, "bbox_cache.json")

COSIA_CLASSES = config.CLASSES_ATTENDUES  # 15 classes

# ── Connexion SQLite persistante par thread Flask ──────────────
_thread_local = threading.local()

def _get_conn():
    if not getattr(_thread_local, "conn", None):
        uri = "file:" + GPKG.replace("\\", "/") + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.execute("PRAGMA cache_size = -131072")
        conn.execute("PRAGMA mmap_size = 268435456")   # 256 MB mmap
        _thread_local.conn = conn
    return _thread_local.conn

# ── Cache géométries WGS84 (LRU, 8000 entrées ≈ ~200 MB) ──────
@lru_cache(maxsize=8000)
def _get_geom_wgs84(fid: int, blob: bytes, zoom: int):
    """Parse, simplifie (selon zoom) et projette en WGS84. Mis en cache par fid+zoom."""
    geom = parse_gpkg_geom(blob)
    if geom is None or geom.is_empty:
        return None
    tol = 1 if zoom >= 16 else 3 if zoom >= 14 else 6
    geom = geom.simplify(tol, preserve_topology=True)
    if geom is None or geom.is_empty:
        return None
    return _to_wgs84(geom)

BDTOPO_COLS = [
    "bdtopo_PARC_NATURA2000", "bdtopo_PARC_PNR", "bdtopo_PARC_RESERVE",
    "bdtopo_GEOPARC", "bdtopo_FORET_PUBLIQUE", "bdtopo_ZONE_VEG_FORET",
    "bdtopo_ZONE_VEG_VIGNE", "bdtopo_ZONE_VEG_VERGER", "bdtopo_ZONE_VEG_LANDE",
    "bdtopo_COURS_EAU", "bdtopo_PLAN_EAU", "bdtopo_SURFACE_HYDRO",
    "bdtopo_EOLIENNE", "bdtopo_BARRAGE",
]
_VALID_BDTOPO = {col.replace("bdtopo_", "") for col in BDTOPO_COLS}

BDTOPO_LABELS = {
    "PARC_NATURA2000": "Natura 2000",
    "PARC_PNR":        "Parc naturel régional",
    "PARC_RESERVE":    "Réserve naturelle",
    "GEOPARC":         "Géoparc UNESCO",
    "FORET_PUBLIQUE":  "Forêt publique",
    "ZONE_VEG_FORET":  "Zone végétation — Forêt",
    "ZONE_VEG_VIGNE":  "Zone végétation — Vigne",
    "ZONE_VEG_VERGER": "Zone végétation — Verger",
    "ZONE_VEG_LANDE":  "Zone végétation — Lande",
    "COURS_EAU":       "Cours d'eau",
    "PLAN_EAU":        "Plan d'eau",
    "SURFACE_HYDRO":   "Surface hydrographique",
    "EOLIENNE":        "Éolienne",
    "BARRAGE":         "Barrage",
}


# ── Parsing géométrie GPKG ─────────────────────────────────────
_ENV_SIZES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}

def parse_gpkg_geom(blob):
    if not blob or blob[:2] != b"GP":
        return None
    flags    = blob[3]
    env_code = (flags >> 1) & 0x07
    header   = 8 + _ENV_SIZES.get(env_code, 0)
    try:
        return shapely.wkb.loads(blob[header:])
    except Exception:
        return None


# ── Transformer (initialisation unique) ───────────────────────
def _make_transformer():
    uri = "file:" + GPKG.replace("\\", "/") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    row = conn.execute(
        f"SELECT srs_id FROM gpkg_geometry_columns WHERE table_name = '{LAYER}'"
    ).fetchone()
    srs_id = row[0] if row else 4326
    if srs_id == 4326:
        conn.close()
        return None, 4326
    srs_row = conn.execute(
        "SELECT organization_coordsys_id FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
        (srs_id,)
    ).fetchone()
    epsg = srs_row[0] if srs_row else srs_id
    conn.close()
    return Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True), epsg

_transformer, _epsg = _make_transformer()


def _to_wgs84(geom):
    if _transformer is None:
        return geom
    def _transform_coords(coords):
        arr = list(coords)
        if not arr:
            return arr
        xs = [c[0] for c in arr]
        ys = [c[1] for c in arr]
        lons, lats = _transformer.transform(xs, ys)
        return list(zip(lons, lats))
    if geom.geom_type == "Polygon":
        ext  = _transform_coords(geom.exterior.coords)
        ints = [_transform_coords(r.coords) for r in geom.interiors]
        return shapely.geometry.Polygon(ext, ints)
    elif geom.geom_type == "MultiPolygon":
        return shapely.geometry.MultiPolygon([_to_wgs84(p) for p in geom.geoms])
    return geom


def _score_to_color(score):
    if score is None:
        return "#AAAAAA"
    try:
        s = float(score)
    except Exception:
        return "#AAAAAA"
    # Seuils calés sur les quantiles réels QGIS (-0.824 → +0.725)
    if s < -0.531: return "#1A7A2E"   # Q1 — Très protecteur
    if s < -0.465: return "#7EC850"   # Q2 — Protecteur
    if s < -0.396: return "#F5E642"   # Q3 — Neutre
    if s < -0.297: return "#F5A623"   # Q4 — Aménageur
    return "#D0021B"                   # Q5 — Très aménageur


# ── Résumés ────────────────────────────────────────────────────
def _load_resumes():
    if not os.path.exists(CSV_BDTOPO):
        return {}
    df = pd.read_csv(CSV_BDTOPO, encoding="utf-8-sig")
    return dict(zip(df["combo_key"].fillna(""), df["resume_bdtopo"].fillna("")))

_bdtopo_resumes = _load_resumes()


def _load_cosia_resumes():
    if not os.path.exists(CSV_COSIA):
        return {}
    df = pd.read_csv(CSV_COSIA, encoding="utf-8-sig")
    return dict(zip(df["classe"].fillna(""), df["resume"].fillna("")))

_cosia_resumes = _load_cosia_resumes()


def _load_cosia_articles():
    """Charge les articles COSIA par classe depuis le CSV (fallback si GPKG vide)."""
    if not os.path.exists(CSV_COSIA):
        return {}
    df = pd.read_csv(CSV_COSIA, encoding="utf-8-sig")
    art_cols = [c for c in df.columns if c.startswith("art_")]
    result = {}
    for _, row in df.iterrows():
        cls = row.get("classe", "")
        if not cls:
            continue
        result[cls] = {c: (row[c] if pd.notna(row[c]) else "") for c in art_cols}
    return result

_cosia_articles = _load_cosia_articles()


def _combo_key(row_dict):
    cats = sorted(
        col.replace("bdtopo_", "")
        for col in BDTOPO_COLS
        if row_dict.get(col)
    )
    return "|".join(cats)


# ── Précomputation des bboxes (background au démarrage) ────────
# _bbox_cache stocke une bbox WGS84 par filtre simple :
#   key "classe:Feuillu"       → {"bbox": [[lat1,lon1],[lat2,lon2]], "count": N}
#   key "bdtopo:EOLIENNE"      → idem
# _bbox_ready indique si le calcul est terminé
_bbox_cache = {}
_bbox_ready = threading.Event()


def _bbox_to_wgs84(minx, miny, maxx, maxy):
    if _transformer:
        lon1, lat1 = _transformer.transform(minx, miny)
        lon2, lat2 = _transformer.transform(maxx, maxy)
    else:
        lon1, lat1, lon2, lat2 = minx, miny, maxx, maxy
    return [[lat1, lon1], [lat2, lon2]]


def _precompute_bboxes():
    """Calcule toutes les bboxes par classe et bdtopo en un seul scan, avec cache fichier."""
    global _bbox_cache

    # ── Tentative de chargement depuis le cache fichier ────────────────
    try:
        if os.path.exists(BBOX_CACHE_FILE):
            cache_mtime = os.path.getmtime(BBOX_CACHE_FILE)
            gpkg_mtime  = os.path.getmtime(GPKG)
            if cache_mtime >= gpkg_mtime:
                with open(BBOX_CACHE_FILE, "r", encoding="utf-8") as f:
                    _bbox_cache = json.load(f)
                _bbox_ready.set()
                print(f"[bbox] Cache chargé depuis fichier — {len(_bbox_cache)} entrées (instantané)")
                return
    except Exception as e:
        print(f"[bbox] Impossible de lire le cache fichier : {e}")

    # ── Calcul depuis le GPKG (un seul scan GROUP BY) ─────────────────
    print("[bbox] Calcul en cours (première fois, ~10 min)…")
    try:
        uri = "file:" + GPKG.replace("\\", "/") + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("PRAGMA cache_size = -131072")

        # Agrégats bdtopo dans le même GROUP BY que classe → UN SEUL SCAN
        bdtopo_aggs = []
        for col in BDTOPO_COLS:
            cat = col.replace("bdtopo_", "")
            bdtopo_aggs += [
                f'MIN(CASE WHEN f."{col}"=1 THEN r.minx END) AS bmnx_{cat}',
                f'MIN(CASE WHEN f."{col}"=1 THEN r.miny END) AS bmny_{cat}',
                f'MAX(CASE WHEN f."{col}"=1 THEN r.maxx END) AS bmxx_{cat}',
                f'MAX(CASE WHEN f."{col}"=1 THEN r.maxy END) AS bmxy_{cat}',
                f'SUM(CASE WHEN f."{col}"=1 THEN 1 ELSE 0 END) AS cnt_{cat}',
            ]
        agg_sql = ",\n".join(bdtopo_aggs)

        rows = conn.execute(f"""
            SELECT f.classe,
                   MIN(r.minx), MIN(r.miny), MAX(r.maxx), MAX(r.maxy), COUNT(*),
                   {agg_sql}
            FROM "{LAYER}" f
            JOIN "{RTREE}" r ON f.fid = r.id
            GROUP BY f.classe
        """).fetchall()
        conn.close()

        # Accumulateurs pour les bboxes bdtopo (agrégées sur toutes les classes)
        bdtopo_acc = {}  # cat → [mnx, mny, mxx, mxy, count]

        for row in rows:
            cls, mnx, mny, mxx, mxy, cnt = row[:6]
            if cls and mnx is not None:
                _bbox_cache[f"classe:{cls}"] = {
                    "bbox": _bbox_to_wgs84(mnx, mny, mxx, mxy), "count": cnt
                }
            # Récupérer les valeurs bdtopo pour cette classe
            idx = 6
            for col in BDTOPO_COLS:
                cat = col.replace("bdtopo_", "")
                bmnx, bmny, bmxx, bmxy, bcnt = row[idx:idx+5]
                idx += 5
                if bmnx is not None and bcnt:
                    if cat not in bdtopo_acc:
                        bdtopo_acc[cat] = [bmnx, bmny, bmxx, bmxy, 0]
                    else:
                        bdtopo_acc[cat][0] = min(bdtopo_acc[cat][0], bmnx)
                        bdtopo_acc[cat][1] = min(bdtopo_acc[cat][1], bmny)
                        bdtopo_acc[cat][2] = max(bdtopo_acc[cat][2], bmxx)
                        bdtopo_acc[cat][3] = max(bdtopo_acc[cat][3], bmxy)
                    bdtopo_acc[cat][4] += int(bcnt)

        for cat, (bmnx, bmny, bmxx, bmxy, bcnt) in bdtopo_acc.items():
            _bbox_cache[f"bdtopo:{cat}"] = {
                "bbox": _bbox_to_wgs84(bmnx, bmny, bmxx, bmxy), "count": bcnt
            }

        # ── Sauvegarde du cache ────────────────────────────────────────
        with open(BBOX_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_bbox_cache, f, ensure_ascii=False)
        print(f"[bbox] Cache sauvegardé → {BBOX_CACHE_FILE}")

    except Exception as e:
        print(f"[bbox] Erreur : {e}")
    finally:
        _bbox_ready.set()
        print(f"[bbox] Terminé — {len(_bbox_cache)} entrées en cache")


def _intersect_bboxes(bboxes):
    """Retourne l'union (not intersection) des bboxes WGS84 [[lat1,lon1],[lat2,lon2]]."""
    if not bboxes:
        return None
    lat1 = min(b[0][0] for b in bboxes)
    lon1 = min(b[0][1] for b in bboxes)
    lat2 = max(b[1][0] for b in bboxes)
    lon2 = max(b[1][1] for b in bboxes)
    return [[lat1, lon1], [lat2, lon2]]


# ── Flask API ──────────────────────────────────────────────────
flask_app = Flask(__name__)
_selected  = {"fid": None}
_view      = {"lat": 44.7, "lon": 4.9, "zoom": 10}


def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@flask_app.route("/api/set_view")
def api_set_view():
    try:
        _view["lat"]  = float(request.args.get("lat",  _view["lat"]))
        _view["lon"]  = float(request.args.get("lon",  _view["lon"]))
        _view["zoom"] = int(float(request.args.get("zoom", _view["zoom"])))
    except Exception:
        pass
    return _cors(jsonify({"ok": True}))


@flask_app.route("/tiles/<int:z>/<int:x>/<int:y>.png")
def serve_tile(z, x, y):
    path = os.path.join(TILES_DIR, str(z), str(x), f"{y}.png")
    if not os.path.exists(path):
        return Response(status=404)
    with open(path, "rb") as f:
        return _cors(Response(f.read(), mimetype="image/png",
                              headers={"Cache-Control": "public, max-age=86400"}))


@flask_app.route("/api/tiles_ready")
def api_tiles_ready():
    z8 = os.path.join(TILES_DIR, "8")
    ready = os.path.isdir(z8) and any(True for _ in os.scandir(z8))
    return _cors(jsonify({"ready": ready}))


@flask_app.route("/vtiles/<int:z>/<int:x>/<int:y>.pbf")
def serve_vtile(z, x, y):
    path = os.path.join(VTILES_DIR, str(z), str(x), f"{y}.pbf")
    if not os.path.exists(path):
        return Response(status=404)
    with open(path, "rb") as f:
        data = f.read()
    headers = {"Cache-Control": "public, max-age=86400"}
    if data[:2] == b'\x1f\x8b':
        headers["Content-Encoding"] = "gzip"
    return _cors(Response(data, mimetype="application/x-protobuf", headers=headers))


@flask_app.route("/api/filter_bbox")
def api_filter_bbox():
    if not _bbox_ready.is_set():
        return _cors(jsonify({"computing": True}))

    classes_param  = request.args.get("classes", "")
    bdtopo_param   = request.args.get("bdtopo",  "")
    filter_classes = [c for c in classes_param.split(",") if c] if classes_param else []
    filter_bdtopo  = [b for b in bdtopo_param.split(",") if b in _VALID_BDTOPO] if bdtopo_param else []

    if not filter_classes and not filter_bdtopo:
        return _cors(jsonify({"bbox": None, "count": 0}))

    bboxes = []
    total_count = 0
    for cls in filter_classes:
        entry = _bbox_cache.get(f"classe:{cls}")
        if entry:
            bboxes.append(entry["bbox"])
            total_count += entry["count"]
    for cat in filter_bdtopo:
        entry = _bbox_cache.get(f"bdtopo:{cat}")
        if entry:
            bboxes.append(entry["bbox"])
            total_count += entry["count"]

    if not bboxes:
        return _cors(jsonify({"bbox": None, "count": 0}))

    return _cors(jsonify({"bbox": _intersect_bboxes(bboxes), "count": total_count}))


@flask_app.route("/api/features")
def api_features():
    # Filtres optionnels (communs aux deux modes)
    classes_param  = request.args.get("classes", "")
    bdtopo_param   = request.args.get("bdtopo",  "")
    filter_classes = [c for c in classes_param.split(",") if c] if classes_param else []
    filter_bdtopo  = [b for b in bdtopo_param.split(",")  if b in _VALID_BDTOPO] if bdtopo_param else []

    where_extra = []
    params = []
    if filter_classes:
        placeholders = ",".join("?" * len(filter_classes))
        where_extra.append(f"f.classe IN ({placeholders})")
        params.extend(filter_classes)
    for cat in filter_bdtopo:
        where_extra.append(f'f."bdtopo_{cat}" = 1')

    load_all = request.args.get("all", "0") == "1"

    conn = _get_conn()
    zoom = int(request.args.get("zoom", 14))

    if load_all:
        # ── Mode "tout charger" : pas de bbox, retourne toutes les parcelles filtrées ──
        if not filter_classes and not filter_bdtopo:
            return _cors(jsonify({"type": "FeatureCollection", "features": []}))
        where_str = ("WHERE " + " AND ".join(where_extra)) if where_extra else ""
        q = f"""
            SELECT f.fid, f.classe, f.score_polygone, f.geom
            FROM "{LAYER}" f
            {where_str}
            LIMIT {MAX_FEAT_ALL}
        """
        rows = conn.execute(q, params).fetchall()

    else:
        # ── Mode viewport : chargement par bbox (comportement par défaut) ──
        try:
            mnlat = float(request.args["minlat"])
            mnlon = float(request.args["minlon"])
            mxlat = float(request.args["maxlat"])
            mxlon = float(request.args["maxlon"])
            zoom  = int(request.args.get("zoom", 10))
        except Exception:
            return _cors(jsonify({"type": "FeatureCollection", "features": []}))

        if zoom < 13:
            return _cors(jsonify({"type": "FeatureCollection", "features": []}))

        if _transformer:
            x1, y1 = _transformer.transform(float(mnlon), float(mnlat), direction="INVERSE")
            x2, y2 = _transformer.transform(float(mxlon), float(mxlat), direction="INVERSE")
            bminx, bmaxx = min(x1, x2), max(x1, x2)
            bminy, bmaxy = min(y1, y2), max(y1, y2)
        else:
            bminx, bmaxx = mnlon, mxlon
            bminy, bmaxy = mnlat, mxlat

        where_str = ("AND " + " AND ".join(where_extra)) if where_extra else ""
        q = f"""
            SELECT f.fid, f.classe, f.score_polygone, f.geom
            FROM "{LAYER}" f
            JOIN "{RTREE}" r ON f.fid = r.id
            WHERE r.minx <= {bmaxx} AND r.maxx >= {bminx}
              AND r.miny <= {bmaxy} AND r.maxy >= {bminy}
              {where_str}
            LIMIT {MAX_FEAT}
        """
        rows = conn.execute(q, params).fetchall()

    features = []
    for fid, classe, score, blob in rows:
        geom = _get_geom_wgs84(fid, bytes(blob) if blob else b"", zoom)
        if geom is None or geom.is_empty:
            continue
        features.append({
            "type": "Feature",
            "geometry": shapely.geometry.mapping(geom),
            "properties": {
                "fid":    fid,
                "classe": classe or "—",
                "score":  round(float(score), 3) if score is not None else None,
                "color":  _score_to_color(score),
            }
        })

    return _cors(jsonify({"type": "FeatureCollection", "features": features}))


@flask_app.route("/api/select/<int:fid>")
def api_select(fid):
    _selected["fid"] = fid
    return _cors(jsonify({"ok": True}))


@flask_app.route("/api/selected")
def api_selected():
    return _cors(jsonify({"fid": _selected["fid"]}))


@flask_app.route("/api/detail/<int:fid>")
def api_detail(fid):
    uri = "file:" + GPKG.replace("\\", "/") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    cols_info = conn.execute(f'PRAGMA table_info("{LAYER}")').fetchall()
    col_names = [c[1] for c in cols_info if c[1] != "geom"]
    cols_sql  = ", ".join(f'"{c}"' for c in col_names)
    row = conn.execute(f'SELECT {cols_sql} FROM "{LAYER}" WHERE fid = ?', (fid,)).fetchone()
    conn.close()
    if row is None:
        return _cors(jsonify({}))
    d = dict(zip(col_names, row))
    # Priorité : colonne GPKG (écrite par 06b) → CSV lookup → vide
    gpkg_resume = d.get("resume_bdtopo") or ""
    ck = _combo_key(d)
    found = _bdtopo_resumes.get(ck, "")
    if not gpkg_resume and not found and ck:
        print(f"[bdtopo] combo_key manquant dans CSV : '{ck}'")
    d["resume_bdtopo"] = gpkg_resume or found
    d["resume_cosia"]  = _cosia_resumes.get(d.get("classe", ""), "")

    # Fallback articles depuis CSV si GPKG vide
    cls = d.get("classe", "")
    if cls and not (d.get("art_cosia_1_titre") or "").strip():
        csv_arts = _cosia_articles.get(cls, {})
        for col, val in csv_arts.items():
            if col.startswith("art_cosia_") and not (d.get(col) or "").strip():
                d[col] = val

    return _cors(jsonify(d))


@st.cache_resource
def start_flask_server():
    threading.Thread(
        target=lambda: flask_app.run(port=FLASK_PORT, debug=False, use_reloader=False),
        daemon=True
    ).start()
    threading.Thread(target=_precompute_bboxes, daemon=True).start()
    return True


# ── HTML Leaflet (stable) ───────────────────────────────────────
def _make_leaflet_html(filter_classes: list, filter_bdtopo: list) -> str:
    lat    = _view["lat"]
    lon    = _view["lon"]
    zoom   = _view["zoom"]
    cls_js = json.dumps(filter_classes)
    bd_js  = json.dumps(filter_bdtopo)
    has_filters   = bool(filter_classes or filter_bdtopo)
    fit_bounds_js = "map.fitBounds(SCOT_BOUNDS, {padding: [30, 30]});" if not has_filters else ""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin:0; padding:0; }}
    #map {{ height:100vh; width:100%; }}
    #info {{
      position:absolute; bottom:10px; left:10px; z-index:1000;
      background:rgba(255,255,255,0.93); padding:6px 10px; border-radius:4px;
      font-size:12px; box-shadow:0 1px 4px rgba(0,0,0,.3); max-width:340px;
    }}
    #filter-badge {{
      position:absolute; top:10px; left:50px; z-index:1000;
      background:#1976D2; color:white; padding:4px 10px;
      border-radius:12px; font-size:11px; font-weight:bold;
      display: {'block' if has_filters else 'none'};
    }}
  </style>
</head>
<body>
<div id="map"></div>
<div id="info">{'Filtre actif — recherche en cours\u2026' if has_filters else 'Zoomez jusqu\u0027au niveau 13 pour voir les parcelles'}</div>
<div id="filter-badge">Filtre actif</div>
<script>
var API            = 'http://localhost:{FLASK_PORT}';
var FILTER_CLASSES = {cls_js};
var FILTER_BDTOPO  = {bd_js};
var MAX_FEAT       = {MAX_FEAT};
var MAX_FEAT_ALL   = {MAX_FEAT_ALL};
var hasFilters     = FILTER_CLASSES.length > 0 || FILTER_BDTOPO.length > 0;

var map = L.map('map').setView([{lat}, {lon}], {zoom});

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 20
}}).addTo(map);

// ── Contour territoire SCoT ───────────────────────────────────
var SCOT_BOUNDS = [[44.154, 4.460], [44.730, 5.709]];
L.rectangle(SCOT_BOUNDS, {{
    color: '#1976D2', weight: 2.5, dashArray: '8 5',
    fill: false, opacity: 0.85, interactive: false
}}).addTo(map).bindTooltip('Territoire SCoT Drôme-Ardèche-Vaucluse', {{sticky: true}});
{fit_bounds_js}

// ── Tuiles raster COSIA (fond, si générées) ───────────────────
fetch(API + '/api/tiles_ready')
  .then(function(r) {{ return r.json(); }})
  .then(function(d) {{
    if (d.ready) L.tileLayer(API + '/tiles/{{z}}/{{x}}/{{y}}.png',
      {{ opacity: 0.85, maxNativeZoom: 13, maxZoom: 20 }}).addTo(map);
  }}).catch(function() {{}});

var layer = null, loading = false, allFeaturesLoaded = false;

function buildFeaturesUrl() {{
    var b = map.getBounds();
    var url = API + '/api/features?minlat=' + b.getSouth() + '&minlon=' + b.getWest()
        + '&maxlat=' + b.getNorth() + '&maxlon=' + b.getEast() + '&zoom=' + map.getZoom();
    if (FILTER_CLASSES.length > 0) url += '&classes=' + encodeURIComponent(FILTER_CLASSES.join(','));
    if (FILTER_BDTOPO.length  > 0) url += '&bdtopo='  + encodeURIComponent(FILTER_BDTOPO.join(','));
    return url;
}}

function _applyGeoJSON(data) {{
    if (layer) map.removeLayer(layer);
    if (!data.features.length) {{
        layer = null;
        document.getElementById('info').textContent = hasFilters
            ? '\u26a0\ufe0f Aucune parcelle ne correspond aux filtres'
            : 'Aucune parcelle dans cette zone';
        loading = false; return;
    }}
    layer = L.geoJSON(data, {{
        style: function(f) {{ return {{ fillColor: f.properties.color, fillOpacity: 0.78, color: 'transparent', weight: 0 }}; }},
        onEachFeature: function(f, l) {{
            l.bindTooltip('<b>' + f.properties.classe + '</b><br/>Score: ' + (f.properties.score !== null ? f.properties.score : '-'), {{sticky:true}});
            l.on('mouseover', function() {{ l.setStyle({{color:'#222',weight:1.5,fillOpacity:0.92}}); }});
            l.on('mouseout',  function() {{ l.setStyle({{color:'transparent',weight:0,fillOpacity:0.78}}); }});
            l.on('click', function(e) {{
                L.DomEvent.stopPropagation(e);
                fetch(API + '/api/select/' + f.properties.fid);
                window.parent.postMessage({{type:'ecolab_select', fid: f.properties.fid}}, '*');
            }});
        }}
    }}).addTo(map);
    var n = data.features.length;
    var msg = n + ' parcelle' + (n>1?'s':'') + ' affich\u00e9e' + (n>1?'s':'');
    if (n >= MAX_FEAT_ALL) msg += ' (limite \u2014 affinez)';
    else if (n >= MAX_FEAT) msg += ' (limite \u2014 zoomez)';
    document.getElementById('info').textContent = msg;
    loading = false;
}}

function loadAllFiltered() {{
    if (loading) return; loading = true;
    document.getElementById('info').textContent = 'Chargement\u2026';
    var url = API + '/api/features?all=1';
    if (FILTER_CLASSES.length > 0) url += '&classes=' + encodeURIComponent(FILTER_CLASSES.join(','));
    if (FILTER_BDTOPO.length  > 0) url += '&bdtopo='  + encodeURIComponent(FILTER_BDTOPO.join(','));
    fetch(url).then(function(r) {{ return r.json(); }})
    .then(function(data) {{
        _applyGeoJSON(data); allFeaturesLoaded = true;
        if (layer && layer.getBounds().isValid())
            map.fitBounds(layer.getBounds(), {{padding:[40,40],maxZoom:17}});
    }}).catch(function() {{ loading = false; }});
}}

function loadFeatures() {{
    if (allFeaturesLoaded) return;
    var zoom = map.getZoom();
    if (!hasFilters && zoom < 13) {{
        document.getElementById('info').textContent = 'Zoomez jusqu\u2019au niveau 13 (zoom: ' + Math.round(zoom) + ')';
        if (layer) {{ map.removeLayer(layer); layer = null; }}
        return;
    }}
    if (loading) return; loading = true;
    document.getElementById('info').textContent = 'Chargement\u2026';
    fetch(buildFeaturesUrl()).then(function(r) {{ return r.json(); }})
    .then(function(data) {{ _applyGeoJSON(data); }})
    .catch(function() {{ loading = false; }});
}}

var _moveTimer = null;
map.on('moveend', function() {{
    var c = map.getCenter();
    fetch(API + '/api/set_view?lat=' + c.lat + '&lon=' + c.lng + '&zoom=' + map.getZoom());
    clearTimeout(_moveTimer);
    _moveTimer = setTimeout(loadFeatures, 300);
}});

function buildBboxUrl() {{
    var url = API + '/api/filter_bbox', sep = '?';
    if (FILTER_CLASSES.length > 0) {{ url += sep + 'classes=' + encodeURIComponent(FILTER_CLASSES.join(',')); sep = '&'; }}
    if (FILTER_BDTOPO.length  > 0) {{ url += sep + 'bdtopo='  + encodeURIComponent(FILTER_BDTOPO.join(',')); }}
    return url;
}}

function fetchBboxAndZoom() {{
    fetch(buildBboxUrl()).then(function(r) {{ return r.json(); }})
    .then(function(data) {{
        if (data.computing) {{
            document.getElementById('info').textContent = '\u23f3 Calcul index\u2026';
            setTimeout(fetchBboxAndZoom, 4000); return;
        }}
        if (!data.bbox || data.count === 0) {{
            document.getElementById('info').textContent = '\u26a0\ufe0f Aucune parcelle ne correspond';
            return;
        }}
        if (data.count <= MAX_FEAT_ALL) {{ loadAllFiltered(); }}
        else {{ map.fitBounds(data.bbox, {{padding:[30,30],maxZoom:16}}); }}
    }}).catch(function() {{ loadFeatures(); }});
}}

if (hasFilters) {{ fetchBboxAndZoom(); }} else {{ loadFeatures(); }}
</script>
</body>
</html>"""


# ── Rendu détail ───────────────────────────────────────────────
def render_detail(d: dict):
    score = d.get("score_polygone")

    def _s(v):
        try: return f"{float(v):.3f}"
        except: return "—"

    color = _score_to_color(score)
    txt_color = "white" if color in ("#F5A623", "#D0021B") else "#222"
    st.markdown(
        f'<div style="background:{color};padding:10px;border-radius:6px;margin-bottom:12px">'
        f'<span style="font-size:1.3em;font-weight:bold;color:{txt_color}">'
        f'{d.get("classe","—")}</span></div>', unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Score global",  _s(score))
    c2.metric("Score COSIA",   _s(d.get("score_cosia")))
    c3.metric("Score BD TOPO", _s(d.get("score_bdtopo")))

    surf = d.get("surface_m2")
    if surf:
        st.caption(f"Surface : {surf:,.0f} m²  |  FID : {d.get('fid')}")

    active_bdtopo = [col.replace("bdtopo_","") for col in BDTOPO_COLS if d.get(col)]
    if active_bdtopo:
        st.markdown("**Contexte BD TOPO :**")
        st.markdown(" ".join(
            f"`{BDTOPO_LABELS.get(c, c)}`" for c in active_bdtopo
        ))

    st.divider()

    st.markdown("**Articles liés à la classe**")
    found_articles = False
    for i in range(1, 4):
        t = d.get(f"art_cosia_{i}_titre") or ""
        u = d.get(f"art_cosia_{i}_url")   or ""
        if t:
            st.markdown(f"- [{t}]({u})" if u else f"- {t}")
            found_articles = True
    if not found_articles:
        st.caption("_(aucun article associé à cette classe)_")

    ctx = [(d.get(f"art_ctx_{i}_titre",""), d.get(f"art_ctx_{i}_url",""))
           for i in range(1, 7) if d.get(f"art_ctx_{i}_titre")]
    if ctx:
        st.markdown("**Articles contexte territorial**")
        for t, u in ctx:
            st.markdown(f"- [{t}]({u})" if u else f"- {t}")

    st.divider()

    resume = d.get("resume_cosia") or d.get("resume") or ""
    if resume:
        st.markdown("**Résumé COSIA**")
        st.markdown(resume)

    resume_bd = d.get("resume_bdtopo") or ""
    if resume_bd:
        st.markdown("**Résumé contexte BD TOPO**")
        st.markdown(resume_bd)
    elif active_bdtopo:
        st.caption("_(résumé BD TOPO non généré — lance 06b)_")


# ── APP PRINCIPALE ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="EcoLab — Carte parcelles",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar : filtres ──────────────────────────────────────
    with st.sidebar:
        st.header("Filtres")

        st.subheader("Classes COSIA")
        selected_classes = st.multiselect(
            label="Sélectionner des classes",
            options=COSIA_CLASSES,
            default=[],
            placeholder="Toutes les classes (aucun filtre)",
            label_visibility="collapsed",
        )

        st.subheader("Catégories BD TOPO")
        bdtopo_options = sorted(BDTOPO_LABELS.keys())
        selected_bdtopo_labels = st.multiselect(
            label="Sélectionner des catégories BD TOPO",
            options=bdtopo_options,
            format_func=lambda k: BDTOPO_LABELS[k],
            default=[],
            placeholder="Toutes / aucun filtre",
            label_visibility="collapsed",
        )

        if selected_classes or selected_bdtopo_labels:
            st.divider()
            st.info(
                "Seules les parcelles vérifiant **toutes** les conditions cochées "
                "sont affichées.\n\n"
                "Si aucune parcelle ne correspond dans la zone, un message s'affiche sur la carte."
            )
            if st.button("Réinitialiser les filtres", use_container_width=True):
                st.rerun()

        st.divider()
        st.markdown("**Légende — Score**")
        legend_items = [
            ("< -0.531",          "#1A7A2E", "white", "Très protecteur"),
            ("-0.531 → -0.465",   "#7EC850", "#222",  "Protecteur"),
            ("-0.465 → -0.396",   "#F5E642", "#222",  "Neutre"),
            ("-0.396 → -0.297",   "#F5A623", "#222",  "Aménageur"),
            ("> -0.297",          "#D0021B", "white",  "Très aménageur"),
        ]
        for rng, bg, fg, lbl in legend_items:
            st.markdown(
                f'<div style="background:{bg};color:{fg};padding:3px 8px;'
                f'border-radius:4px;margin:2px 0;font-size:0.82em">'
                f'<b>{rng}</b> — {lbl}</div>',
                unsafe_allow_html=True
            )

    # ── Corps principal ────────────────────────────────────────
    st.title("EcoLab — Carte des parcelles COSIA")

    start_flask_server()

    tab_carte, tab_analyse = st.tabs(["🗺️ Carte", "📊 Analyses BERTopic"])

    with tab_analyse:
        st.subheader("Score moyen d'artificialisation par topic")
        if os.path.exists(IMG_BARPLOT):
            st.image(IMG_BARPLOT, width='stretch')
        else:
            st.warning(f"Image introuvable : {IMG_BARPLOT}")

        st.subheader("Série temporelle des sentiments")
        if os.path.exists(IMG_TIMESERIES):
            st.image(IMG_TIMESERIES, width='stretch')
        else:
            st.warning(f"Image introuvable : {IMG_TIMESERIES}")

    with tab_carte:
        map_col, detail_col = st.columns([7, 3])

        with map_col:
            components.html(
                _make_leaflet_html(selected_classes, selected_bdtopo_labels),
                height=700,
                scrolling=False,
            )

        with detail_col:
            if "selected_fid" not in st.session_state:
                st.session_state.selected_fid = None

            col_a, col_b = st.columns([3, 1])
            with col_b:
                refresh = st.button("🔄 Actualiser", help="Cliquez après avoir sélectionné une parcelle")
            with col_a:
                if st.session_state.selected_fid:
                    st.caption(f"FID sélectionné : {st.session_state.selected_fid}")

            if refresh or st.session_state.selected_fid is None:
                try:
                    import urllib.request
                    with urllib.request.urlopen(
                        f"http://localhost:{FLASK_PORT}/api/selected", timeout=1
                    ) as resp:
                        fid = json.load(resp).get("fid")
                    if fid is not None:
                        st.session_state.selected_fid = fid
                except Exception:
                    pass

            if st.session_state.selected_fid:
                try:
                    import urllib.request
                    with urllib.request.urlopen(
                        f"http://localhost:{FLASK_PORT}/api/detail/{st.session_state.selected_fid}",
                        timeout=2
                    ) as resp:
                        details = json.load(resp)
                    if details:
                        render_detail(details)
                    else:
                        st.info("Parcelle introuvable.")
                except Exception as e:
                    st.error(f"Erreur chargement détail : {e}")
            else:
                st.markdown("### Cliquez sur une parcelle")
                st.markdown("Puis **🔄 Actualiser** pour voir les détails.")
                st.markdown("- Classe COSIA + scores")
                st.markdown("- Catégories BD TOPO actives")
                st.markdown("- Articles liés")
                st.markdown("- Résumés COSIA et BD TOPO")


if __name__ == "__main__":
    main()
