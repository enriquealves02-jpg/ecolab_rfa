"""
EcoLab Pipeline — Étape 1 v2 : BERTopic Dauphiné Libéré
=========================================================
Entraîne BERTopic sur les articles du Dauphiné Libéré (3 fichiers CSV).
Plus de topics que le v1, adapté à la presse locale.

Sorties dans outputs/bertopic_v2/ :
  docs_with_topics_v2.csv
  topics_info_v2.csv
  model_v2/
  topics_overview_v2.png   ← liste des topics + nb docs + top-3 mots-clés
"""

import ast
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION V2
# ═══════════════════════════════════════════════════════════════
DATA_DIR = config.DATA_DIR
OUT_DIR  = config.OUT_DIR

BERTOPIC_V2_DIR = os.path.join(OUT_DIR, "bertopic_v2")
DOCS_CSV_V2     = os.path.join(BERTOPIC_V2_DIR, "docs_with_topics_v2.csv")
TOPICS_CSV_V2   = os.path.join(BERTOPIC_V2_DIR, "topics_info_v2.csv")
MODEL_DIR_V2    = os.path.join(BERTOPIC_V2_DIR, "model_v2")
PNG_V2          = os.path.join(BERTOPIC_V2_DIR, "topics_overview_v2.png")

# Toutes les sources : nationales (config) + Dauphiné (les fichiers manquants sont ignorés)
SOURCES = config.NEWS_SOURCES + [
    {"path": os.path.join(DATA_DIR, "ledauphine_articles.csv"),        "source": "dauphine"},
    {"path": os.path.join(DATA_DIR, "ledauphine_articles_2022.csv"),   "source": "dauphine"},
    {"path": os.path.join(DATA_DIR, "ledauphine_articles_202026.csv"), "source": "dauphine"},
]

NR_TOPICS      = None   # None = garder tous les topics bruts HDBSCAN
MIN_TOPIC_SIZE = 8      # petits clusters → plus de topics
MIN_DOC_LEN    = 30     # chars minimum pour qu'un doc soit conservé
# ═══════════════════════════════════════════════════════════════


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def clean_text(s) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s.lower())
    return s.strip()


def load_csv(path: str, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    for col in ("titre", "contenu", "url"):
        if col not in df.columns:
            df[col] = ""
    df["source"] = source
    df["doc"] = (
        df["titre"].fillna("") + ". " + df["contenu"].fillna("")
    ).map(clean_text)
    df["doc_len"] = df["doc"].str.len()
    return df[df["doc_len"] >= MIN_DOC_LEN]


# ───────────────────────────────────────────────────────────────
# VISUALISATION PNG
# ───────────────────────────────────────────────────────────────

def save_topics_png(topics_info: pd.DataFrame, png_path: str, n_total_docs: int) -> None:
    """
    Génère un graphique horizontal listant chaque topic avec :
      - son nombre de documents (barre)
      - ses 3 mots-clés TF-IDF principaux (annotation)
    """
    # Exclure le topic -1 (outliers) et trier par count desc
    df = (
        topics_info[topics_info["Topic"] != -1]
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    if df.empty:
        print("  Aucun topic à visualiser.")
        return

    n = len(df)
    row_h = 0.38          # hauteur par ligne (inches)
    fig_h = max(8, n * row_h + 2.5)
    fig, ax = plt.subplots(figsize=(16, fig_h))

    counts = df["Count"].values
    max_count = counts.max()

    # Couleurs : dégradé selon count
    norm_counts = counts / max_count
    cmap = plt.cm.Blues
    colors = [cmap(0.35 + 0.6 * v) for v in norm_counts]

    y_pos = np.arange(n)

    bars = ax.barh(y_pos, counts, color=colors, edgecolor="white", height=0.75)

    # Annotations : "mot1, mot2, mot3   (N docs)"
    for i, (_, row) in enumerate(df.iterrows()):
        # Extraire les mots-clés
        rep = row.get("Representation", "[]")
        if isinstance(rep, str):
            try:
                words = ast.literal_eval(rep)
            except Exception:
                words = []
        else:
            words = list(rep) if rep else []
        top3 = ", ".join(str(w) for w in words[:3])

        count = counts[i]
        bar_end = count
        # Texte à droite de la barre
        ax.text(
            bar_end + max_count * 0.01,
            i,
            f"{top3}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#333333",
        )
        # Count dans la barre (si assez large) ou à gauche
        if count > max_count * 0.08:
            ax.text(
                bar_end * 0.5,
                i,
                str(count),
                va="center",
                ha="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    # Étiquettes Y : "Topic N — mot_principal"
    y_labels = []
    for _, row in df.iterrows():
        rep = row.get("Representation", "[]")
        if isinstance(rep, str):
            try:
                words = ast.literal_eval(rep)
            except Exception:
                words = []
        else:
            words = list(rep) if rep else []
        label = f"T{int(row['Topic'])}  {words[0] if words else ''}"
        y_labels.append(label)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.invert_yaxis()

    ax.set_xlabel("Nombre de documents", fontsize=10)
    ax.set_xlim(0, max_count * 1.55)   # marge à droite pour les annotations
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    n_topics = len(df)
    n_outliers = int(topics_info.loc[topics_info["Topic"] == -1, "Count"].sum()) if -1 in topics_info["Topic"].values else 0
    ax.set_title(
        f"BERTopic v2 — toutes sources\n"
        f"{n_topics} topics  ·  {n_total_docs} documents  ·  {n_outliers} outliers (-1)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PNG sauvegardé → {png_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run() -> None:
    print("\n" + "=" * 60)
    print("ÉTAPE 1 v2 — BERTopic Dauphiné Libéré")
    print("=" * 60)

    os.makedirs(BERTOPIC_V2_DIR, exist_ok=True)

    # ── 1. CHARGEMENT ─────────────────────────────────────────
    print(f"\n[{_ts()}] Chargement des sources...")
    frames = []
    for src in SOURCES:
        path = src["path"]
        if not os.path.exists(path):
            print(f"  IGNORÉ (introuvable) : {os.path.basename(path)}")
            continue
        df_src = load_csv(path, src["source"])
        print(f"  {src['source']:<12} {os.path.basename(path):<42} → {len(df_src):>6} docs")
        frames.append(df_src)

    if not frames:
        print("Aucun fichier chargé. Abandon.")
        return

    df = pd.concat(frames, ignore_index=True)
    df = (
        df.dropna(subset=["url"])
          .drop_duplicates(subset=["url"])
          .reset_index(drop=True)
    )
    docs = df["doc"].tolist()
    sources_counts = df["source"].value_counts().to_dict()
    print(f"\n  Total après déduplication : {len(docs)} documents")
    print(f"  Longueur moyenne          : {df['doc_len'].mean():.0f} chars")
    print(f"  Par source                : {sources_counts}")

    # ── 2. EMBEDDINGS ─────────────────────────────────────────
    print(f"\n[{_ts()}] Embeddings (paraphrase-multilingual-MiniLM-L12-v2)...")
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    t0 = time.time()
    embeddings = embedding_model.encode(
        docs,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )
    print(f"  OK — shape {embeddings.shape} — {time.time()-t0:.1f}s")

    # ── 3. MODÈLE ─────────────────────────────────────────────
    print(f"\n[{_ts()}] Configuration BERTopic (MIN_TOPIC_SIZE={MIN_TOPIC_SIZE}, NR_TOPICS={NR_TOPICS})...")

    vectorizer_model = CountVectorizer(
        stop_words=config.FRENCH_STOPWORDS,
        ngram_range=(1, 2),
        min_df=2,
    )

    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=NR_TOPICS,
        calculate_probabilities=True,
        verbose=True,
        language="multilingual",
    )

    # ── 4. UMAP ───────────────────────────────────────────────
    print(f"\n[{_ts()}] Réduction UMAP...")
    t0 = time.time()
    umap_embeddings = umap_model.fit_transform(embeddings)
    print(f"  OK — {time.time()-t0:.1f}s")

    # ── 5. HDBSCAN ────────────────────────────────────────────
    print(f"\n[{_ts()}] Clustering HDBSCAN...")
    t0 = time.time()
    hdbscan_model.fit(umap_embeddings)
    n_clusters = len(set(hdbscan_model.labels_)) - (1 if -1 in hdbscan_model.labels_ else 0)
    n_noise    = (hdbscan_model.labels_ == -1).sum()
    print(f"  OK — {n_clusters} clusters, {n_noise} outliers — {time.time()-t0:.1f}s")

    # ── 6. FIT_TRANSFORM ──────────────────────────────────────
    print(f"\n[{_ts()}] fit_transform BERTopic...")
    t0 = time.time()
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    print(f"  OK — {time.time()-t0:.1f}s")

    df["topic"]      = topics
    df["topic_prob"] = probs.max(axis=1) if probs is not None else np.nan

    # ── 7. SAUVEGARDE ─────────────────────────────────────────
    print(f"\n[{_ts()}] Sauvegarde...")

    df.to_csv(DOCS_CSV_V2, index=False, encoding="utf-8")
    print(f"  docs_with_topics_v2.csv  → {DOCS_CSV_V2}")

    topics_info = topic_model.get_topic_info()
    topics_info.to_csv(TOPICS_CSV_V2, index=False, encoding="utf-8")
    print(f"  topics_info_v2.csv       → {TOPICS_CSV_V2}")

    topic_model.save(MODEL_DIR_V2)
    print(f"  modèle BERTopic          → {MODEL_DIR_V2}")

    # ── 8. PNG ────────────────────────────────────────────────
    print(f"\n[{_ts()}] Génération du PNG...")
    save_topics_png(topics_info, PNG_V2, len(docs))

    # ── 9. RÉSUMÉ ─────────────────────────────────────────────
    n_topics   = (topics_info["Topic"] != -1).sum()
    n_outliers = (df["topic"] == -1).sum()
    print(f"\n{'─'*60}")
    print(f"  Topics                 : {n_topics}")
    print(f"  Outliers (-1)          : {n_outliers} ({100*n_outliers/len(docs):.1f}%)")
    print(f"  Docs par topic (moy.)  : {len(docs)/max(n_topics,1):.1f}")
    print(f"{'─'*60}")
    print("ÉTAPE 1 v2 — TERMINÉE")
    print("=" * 60)


if __name__ == "__main__":
    run()
