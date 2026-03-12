"""
EcoLab Pipeline — Post-traitement BERTopic v2
==============================================
Filtre les articles des topics pertinents pour le projet EcoLab et produit :

  donnéesNews/Best_articles_for_ecolab.csv
      -> titre, date, contenu, url

  outputs/bertopic_v2/topic_best_articles_bertopicv2.csv
      -> topic, topic_prob, source, titre, date, contenu, url
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

DOCS_CSV   = os.path.join(config.OUT_DIR, "bertopic_v2", "docs_with_topics_v2.csv")
OUT_DIR_BT = os.path.join(config.OUT_DIR, "bertopic_v2")
NEWS_DIR   = config.DATA_DIR

BEST_CSV  = os.path.join(NEWS_DIR,   "Best_articles_for_ecolab.csv")
TOPIC_CSV = os.path.join(OUT_DIR_BT, "topic_best_articles_bertopicv2.csv")

RELEVANT_TOPICS = {
    0, 11, 127, 135,           # mobilite/transport
    8, 27, 57, 91, 198, 205, 413, 513,   # agriculture
    15, 25, 59, 62, 74, 113, 184, 337,   # eau
    29, 35, 53, 141, 291, 499,            # foret
    22, 26, 60, 65, 66, 68, 108, 143, 155, 297, 303, 325, 385,  # biodiversite
    153, 175, 251, 321, 361,              # sols/artificialisation
    9, 244, 323, 392, 468, 527,           # urbanisme/amenagement
    18, 33, 79, 138, 160, 209, 260, 262, # bati/energie
    12, 41, 186,                          # ENR
    19, 37, 44, 48,                       # montagne
    34, 84, 90, 181, 518,                 # climat/pollution
    445,                                  # dechets
}

def run():
    print("=" * 60)
    print("POST-TRAITEMENT — BERTopic v2 : meilleurs articles EcoLab")
    print("=" * 60)

    print(f"Chargement : {DOCS_CSV}")
    df = pd.read_csv(DOCS_CSV, encoding="utf-8")
    print(f"  {len(df):,} documents charges")

    df_rel = df[df["topic"].isin(RELEVANT_TOPICS)].copy()
    print(f"  {len(df_rel):,} documents dans les {len(RELEVANT_TOPICS)} topics retenus")

    df_rel = df_rel.sort_values(["topic", "topic_prob"], ascending=[True, False])

    # CSV 1
    os.makedirs(NEWS_DIR, exist_ok=True)
    df_best = df_rel[["titre", "date", "contenu", "url"]].drop_duplicates(subset=["url"])
    df_best.to_csv(BEST_CSV, index=False, encoding="utf-8")
    print(f"  Best_articles_for_ecolab.csv -> {BEST_CSV}")
    print(f"    {len(df_best):,} articles uniques")

    # CSV 2
    os.makedirs(OUT_DIR_BT, exist_ok=True)
    cols2 = [c for c in ["topic","topic_prob","source","titre","date","contenu","url"] if c in df_rel.columns]
    df_topic = df_rel[cols2].drop_duplicates(subset=["url"])
    df_topic.to_csv(TOPIC_CSV, index=False, encoding="utf-8")
    print(f"  topic_best_articles_bertopicv2.csv -> {TOPIC_CSV}")
    print(f"    {len(df_topic):,} articles")

    n_content = int(df_best["contenu"].notna().sum())
    print(f"Topics retenus : {len(RELEVANT_TOPICS)} | Articles : {len(df_best):,} | Avec contenu : {n_content:,}")
    print("TERMINE")

if __name__ == "__main__":
    run()
