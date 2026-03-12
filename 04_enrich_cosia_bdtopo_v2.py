import ast
import gc, os, sys
import numpy as np, pandas as pd, geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v2 as config


def load_topic_labels() -> dict:
    """Charge topics_info_v2.csv → {topic_id: label TF-IDF top-4}."""
    if not os.path.exists(config.TOPICS_INFO_CSV):
        print(f"  WARN: topics_info_v2.csv introuvable ({config.TOPICS_INFO_CSV})")
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


def print_bdtopo_mapping_diagnostic(score_by_topic: dict, topic_labels: dict) -> None:
    """
    Affiche le tableau de correspondance :
      T{id} | label TF-IDF réel | catégories BD TOPO associées | score Ollama
    Signale les topics de TOPIC_BDTOPO_MAP absents du summary.
    """
    print("\n" + "─" * 85)
    print("DIAGNOSTIC — Correspondance topics BERTopic v2 → catégories BD TOPO")
    print(f"{'Topic':<8} {'Label TF-IDF (top 4)':<35} {'Catégories BD TOPO':<28} {'Score'}")
    print("─" * 85)
    missing = []
    for tid in sorted(config.TOPIC_BDTOPO_MAP):
        label = topic_labels.get(tid, "???")
        cats  = ", ".join(config.TOPIC_BDTOPO_MAP[tid])
        if tid in score_by_topic:
            score_str = f"{score_by_topic[tid]:+.3f}"
        else:
            score_str = "MANQUANT"
            missing.append(tid)
        print(f"  T{str(tid):<6} {label:<35} {cats:<28} {score_str}")
    print("─" * 85)
    if missing:
        print(f"  ⚠ {len(missing)} topics dans TOPIC_BDTOPO_MAP sans score : {missing}")
        print("    → Ces topics ne contribueront pas au score BD TOPO.")
    else:
        print(f"  ✓ Tous les {len(config.TOPIC_BDTOPO_MAP)} topics de la map ont un score.")
    print("─" * 85 + "\n")


def load_bdtopo_layer(cat_info):
    """Charge une couche BD TOPO depuis les 3 départements et les concatène."""
    gdfs = []
    for root in config.BDTOPO_ROOTS:
        shp = os.path.join(root, cat_info["shp"])
        if not os.path.exists(shp):
            continue
        gdf = gpd.read_file(shp)
        if cat_info["field"] and cat_info["values"]:
            gdf = gdf[gdf[cat_info["field"]].isin(cat_info["values"])].copy()
        if not gdf.empty:
            gdfs.append(gdf[[gdf.geometry.name]].copy())
    if not gdfs:
        return None
    if len(gdfs) == 1:
        return gdfs[0]
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)


def process_category(cat_name, cat_info, cosia_sub, cosia_bbox, roots_ok, zone_veg_cache, zone_veg_rel):
    """
    Worker thread : charge la couche BD TOPO pour une catégorie,
    pré-filtre par bbox COSIA, fait le sjoin → retourne les indices matchants.
    """
    minx, miny, maxx, maxy = cosia_bbox

    if cat_info["shp"] == zone_veg_rel:
        parts = []
        for root in roots_ok:
            veg = zone_veg_cache.get(root)
            if veg is None:
                continue
            sub = (veg[veg["NATURE"].isin(cat_info["values"])][["geometry"]].copy()
                   if cat_info["values"] else veg[["geometry"]].copy())
            if not sub.empty:
                parts.append(sub)
        layer = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs) if parts else None
    else:
        layer = load_bdtopo_layer(cat_info)

    if layer is None or layer.empty:
        return cat_name, set()

    # Reprojection si nécessaire
    if layer.crs != cosia_sub.crs:
        layer = layer.to_crs(cosia_sub.crs)

    # Pré-filtrage BD TOPO par bbox COSIA globale
    layer = layer.cx[minx:maxx, miny:maxy]
    if layer.empty:
        return cat_name, set()

    # layer à GAUCHE (petit), cosia_sub à DROITE (sindex 15M déjà construit)
    # → geopandas interroge le sindex pré-construit de cosia_sub, N_layer fois seulement
    layer_geom = layer[[layer.geometry.name]].copy()
    layer_geom.index = range(len(layer_geom))
    joined = gpd.sjoin(layer_geom, cosia_sub, how="inner", predicate="intersects")
    indices = set(joined["_i"])
    return cat_name, indices


def compute_score_bdtopo(cosia, score_by_topic):
    cat_mean = {}
    for tid, cats in config.TOPIC_BDTOPO_MAP.items():
        s = score_by_topic.get(int(tid))
        if s is None: continue
        for cat in cats:
            cat_mean.setdefault(cat, []).append(float(s))
    cat_mean = {c: float(np.mean(v)) for c, v in cat_mean.items()}

    score_sum = np.zeros(len(cosia), dtype=float)
    count     = np.zeros(len(cosia), dtype=int)
    for cat, ms in cat_mean.items():
        col = f"bdtopo_{cat}"
        if col not in cosia.columns: continue
        mask = cosia[col].values.astype(bool)
        score_sum += mask * ms
        count     += mask.astype(int)
    return np.where(count > 0, score_sum / count, np.nan)


def run():
    print("=" * 60)
    print("PIPELINE V2 ETAPE 4 - Enrichissement COSIA x BDTOPO")
    print("=" * 60)
    os.makedirs(config.GEO_DIR, exist_ok=True)

    if not os.path.exists(config.COSIA_SCORE_GPKG):
        raise FileNotFoundError(f"COSIA_SCORE_v2.gpkg introuvable : {config.COSIA_SCORE_GPKG} - Lance etape 2 d abord.")

    print(f"Chargement COSIA_SCORE_v2.gpkg...")
    cosia = gpd.read_file(config.COSIA_SCORE_GPKG)
    if "score_polygone" in cosia.columns:
        cosia = cosia.rename(columns={"score_polygone": "score_cosia"})
    elif "score_cosia" not in cosia.columns:
        cosia["score_cosia"] = 0.0
    print(f"  {len(cosia):,} polygones (CRS: {cosia.crs})")

    df_sc = pd.read_csv(config.TOPIC_SCORE_SUMMARY_CSV)
    score_by_topic = dict(zip(df_sc["topic"].astype(int), df_sc["mean_score"]))
    print(f"  {len(score_by_topic)} topics avec score Ollama")

    # Diagnostic : lien topic_id → label TF-IDF réel → catégories BD TOPO
    topic_labels = load_topic_labels()
    print(f"  {len(topic_labels)} labels TF-IDF chargés depuis topics_info_v2.csv")
    print_bdtopo_mapping_diagnostic(score_by_topic, topic_labels)

    bdtopo_ok = any(os.path.isdir(r) for r in config.BDTOPO_ROOTS)
    if not bdtopo_ok:
        print("WARN: aucun dossier BDTOPO trouvé - score BDTOPO = NaN")
        for cat in config.BDTOPO_CATEGORIES: cosia[f"bdtopo_{cat}"] = False
        cosia["score_bdtopo"] = np.nan
        cosia["score_polygone"] = cosia["score_cosia"]
    else:
        roots_ok = [r for r in config.BDTOPO_ROOTS if os.path.isdir(r)]
        print(f"  BDTOPO disponible : {len(roots_ok)}/3 département(s)")

        # Diagnostic : vérifier que les SHP clés existent
        print("\n  Diagnostic chemins SHP :")
        test_shps = [
            os.path.join("ZONES_REGLEMENTEES", "PARC_OU_RESERVE.shp"),
            os.path.join("OCCUPATION_DU_SOL",  "ZONE_DE_VEGETATION.shp"),
            os.path.join("HYDROGRAPHIE",        "COURS_D_EAU.shp"),
        ]
        for root in roots_ok:
            dept = os.path.basename(root)[-4:]
            for rel in test_shps:
                full = os.path.join(root, rel)
                status = "OK" if os.path.exists(full) else "MANQUANT"
                print(f"    [{dept}] {status}  {rel}")
        print()

        cosia_sub  = gpd.GeoDataFrame({"_i": cosia.index.tolist()}, geometry=cosia.geometry.values, crs=cosia.crs)
        cosia_bbox = cosia.total_bounds
        minx, miny, maxx, maxy = cosia_bbox

        # Pré-chargement ZONE_DE_VEGETATION — pré-filtrée par bbox COSIA (évite 594K features en mémoire)
        ZONE_VEG_REL = os.path.join("OCCUPATION_DU_SOL", "ZONE_DE_VEGETATION.shp")
        zone_veg_cache = {}
        for root in roots_ok:
            shp = os.path.join(root, ZONE_VEG_REL)
            if os.path.exists(shp):
                veg = gpd.read_file(shp)
                veg_bbox = veg.cx[minx:maxx, miny:maxy]
                if not veg_bbox.empty:
                    zone_veg_cache[root] = veg_bbox
                    print(f"  ZONE_DE_VEGETATION ({os.path.basename(root)[-4:]}) : "
                          f"{len(veg_bbox):,} zones dans bbox SCoT")
                del veg; gc.collect()

        # Pré-construction de l'index spatial une seule fois (évite de le reconstruire 14x)
        print(f"\n  Construction index spatial COSIA ({len(cosia_sub):,} polygones)...")
        _ = cosia_sub.sindex
        print(f"  Index prêt.")

        print(f"\n  Traitement {len(config.BDTOPO_CATEGORIES)} catégories BD TOPO (séquentiel)...")
        for i, (cat_name, cat_info) in enumerate(config.BDTOPO_CATEGORIES.items(), 1):
            prefix = f"  [{i:02d}/{len(config.BDTOPO_CATEGORIES)}] {cat_name}"
            try:
                _, indices = process_category(cat_name, cat_info, cosia_sub, cosia_bbox,
                                              roots_ok, zone_veg_cache, ZONE_VEG_REL)
                cosia[f"bdtopo_{cat_name}"] = cosia.index.isin(indices)
                print(f"{prefix} → {len(indices):,} polygones")
            except Exception as e:
                print(f"{prefix} ERREUR : {e}")
                cosia[f"bdtopo_{cat_name}"] = False

        del zone_veg_cache, cosia_sub; gc.collect()

        # Toponymie cours d eau (3 depts)
        print("Toponymie cours d eau (3 departements)...")
        ce_parts = []
        for root in roots_ok:
            shp_ce = os.path.join(root, "HYDROGRAPHIE", "COURS_D_EAU.shp")
            if os.path.exists(shp_ce):
                df_ce = gpd.read_file(shp_ce)[["TOPONYME", "IMPORTANCE", "geometry"]]
                ce_parts.append(df_ce)
        if ce_parts:
            ce = gpd.GeoDataFrame(pd.concat(ce_parts, ignore_index=True), crs=ce_parts[0].crs)
            ce = ce[ce["TOPONYME"].notna() & (ce["TOPONYME"].str.strip() != "")]
            ce["_imp"] = pd.to_numeric(ce["IMPORTANCE"], errors="coerce").fillna(99)
            if ce.crs != cosia.crs: ce = ce.to_crs(cosia.crs)
            cosia_idx = gpd.GeoDataFrame({"_i": cosia.index.tolist()}, geometry=cosia.geometry.values, crs=cosia.crs)
            # rivières à gauche (petit), cosia à droite (index déjà construit) → N_rivières requêtes seulement
            jce = gpd.sjoin(ce[["_imp","TOPONYME","geometry"]], cosia_idx, how="inner", predicate="intersects")
            best = jce.sort_values("_imp").groupby("_i")["TOPONYME"].first()
            cosia["nom_cours_eau"] = cosia.index.map(best).fillna("")
            print(f"  {int((cosia['nom_cours_eau'] != '').sum()):,} polygones avec cours d eau nomme")
            del ce, cosia_idx, jce; gc.collect()
        else:
            cosia["nom_cours_eau"] = ""

        # Score BDTOPO
        print("Calcul score BDTOPO...")
        score_bdtopo = compute_score_bdtopo(cosia, score_by_topic)
        cosia["score_bdtopo"] = np.round(score_bdtopo, 4)
        print(f"  {int((~np.isnan(score_bdtopo)).sum()):,} polygones avec contexte BDTOPO")

        # Score final : 50% COSIA + 50% BDTOPO
        has_bd = ~np.isnan(score_bdtopo)
        score_final = np.where(
            has_bd,
            0.50 * cosia["score_cosia"].values + 0.50 * np.nan_to_num(score_bdtopo),
            cosia["score_cosia"].values,
        )
        cosia["score_polygone"] = np.round(score_final, 4)

    # Export FINALE_GPKG dans ecolab2_territoire_scot
    for path in (config.FINALE_GPKG, config.FINALE_CSV):
        if os.path.exists(path): os.remove(path)

    cosia.to_file(config.FINALE_GPKG, layer="COSIA_SCORE_finale", driver="GPKG")
    print(f"FINALE_GPKG -> {config.FINALE_GPKG}")

    geom_col = cosia.geometry.name
    cosia.drop(columns=[geom_col]).to_csv(config.FINALE_CSV, index=False, encoding="utf-8")
    print(f"FINALE_CSV  -> {config.FINALE_CSV}")

    print(f"score_cosia    : min={cosia['score_cosia'].min():.4f} max={cosia['score_cosia'].max():.4f}")
    print(f"score_polygone : min={cosia['score_polygone'].min():.4f} max={cosia['score_polygone'].max():.4f}")
    print(f"Polygones total : {len(cosia):,}")
    print("ETAPE 4 TERMINEE")


if __name__ == "__main__":
    run()
