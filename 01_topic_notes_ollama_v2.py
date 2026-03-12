"""
EcoLab Pipeline — Étape 1 : Scoring des articles via Ollama
=============================================================
Pour chaque article (issu de articles_with_topic_labels.csv), envoie le texte
à Ollama qui renvoie un score de -2 à +2 selon l'orientation territoriale.

Ensuite agrège les scores par topic → moyenne pondérée.

Sorties :
  outputs/docs_scored_v2.csv          ← tous les articles + score Ollama
  outputs/topic_score_summary_v2.csv  ← score moyen par topic (→ étape 3)

Reprise automatique si docs_scored_v2.csv existe déjà (articles déjà scorés
sont ignorés, on reprend où on s'est arrêté).
"""

import json
import os
import re
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v2 as config

# =============================================================
# PROMPT
# =============================================================

# --- Pré-prompt système (contexte et règles fixes) -----------
SYSTEM_PROMPT = """\
Tu es un expert en analyse foncière et territoriale. Ta mission : évaluer chaque article \
selon son orientation vis-à-vis de l'artificialisation des sols en France.

L'artificialisation = transformation de terres naturelles, agricoles ou forestières \
en surfaces bâties, imperméabilisées ou aménagées par l'homme.

=== GRILLE DE NOTATION ===

score = -2 → L'article résiste à l'artificialisation ou documente ses impacts négatifs :
- Changement climatique, réchauffement, émissions de GES, effondrement du vivant
- Risques naturels : inondations, sécheresse, feux de forêt, érosion, submersion
- Biodiversité : espèces menacées, faune, flore, corridors écologiques, forêts, zones humides
- Agriculture : terres agricoles, paysans, souveraineté alimentaire, pesticides, sols vivants
- Eau : qualité des rivières, nappes phréatiques, assèchement, milieux aquatiques
- Protection foncière : ZAN défendu, sobriété foncière, renaturation, trame verte et bleue
- Espaces protégés : Natura 2000, parcs naturels, réserves, zones naturelles, Géoparc
- Contentieux : recours contre un projet béton, annulation d'une ZAC, victoire juridique écologique

score = +2 → L'article favorise l'artificialisation ou traite d'aménagement humain du territoire :
- Urbanisme et construction : logements, bâtiments, permis de construire, PLU, ZAC, ZAE
- Infrastructure : routes, autoroutes, ponts, tunnels, aéroports, parkings, signalisation
- Mobilité motorisée : réseau routier, voiture, camion, véhicule électrique
- Immobilier et foncier : promotion immobilière, prix du foncier, vente de terrains
- Aménagement : zones d'activité, lotissements, centres commerciaux, projets d'aménagement
- Politique foncière : ZAN critiqué ou assoupli, étalement urbain défendu, densification refusée
- Énergie des bâtiments : rénovation thermique, isolation, chauffage, performance énergétique

score = -1 → Dominante environnementale légère ou sujet mixte penchant vers la nature :
- Mobilité douce en ville (vélo, marche) sans construction lourde associée
- Agriculture durable avec volet économique fort
- Énergie renouvelable défendant des espaces naturels existants

score = +1 → Dominante aménagement légère ou sujet mixte penchant vers le bâti :
- Transport ferroviaire ou fluvial (moins artificialisant que la route, mais infrastructure)
- Réhabilitation ou reconversion d'une friche industrielle existante
- Éolien ou solaire sur terrain déjà artificialisé ou dégradé

score = 0 → Aucun lien avec le territoire, la nature ou l'aménagement :
- Politique étrangère, conflit armé, sport, santé sans lien foncier, finance pure, culture

=== RÈGLES DE DÉCISION ===
1. Juge le SUJET PRINCIPAL de l'article, pas les mentions secondaires ni le ton.
2. -2 ou +2 : le sujet est clairement et entièrement centré sur l'environnement ou sur l'aménagement.
3. -1 ou +1 : le sujet est mixte, ou l'angle est indirect (ex : mobilité douce, réhabilitation de friche).
4. 0 : aucun lien avec le territoire, la nature ou l'aménagement.
5. ZAN favorable aux constructeurs = +2. ZAN défendu par les écologistes = -2.
6. Éoliennes en mer = -1 (offshore, n'artificialise pas les terres).
7. Rénovation énergétique seule, sans nouveau bâti = +1 (amélioration de l'existant).
8. Pesticides, sols agricoles, qualité de l'eau = -2 même si le ton est neutre ou technique.

=== FORMAT DE RÉPONSE OBLIGATOIRE ===
Réponds UNIQUEMENT avec cette ligne exacte, sans explication ni ponctuation supplémentaire :
score=X
(X est un entier parmi : -2, -1, 0, 1, 2)
"""

# --- Template par article (variables dynamiques) -------------
PROMPT_TEMPLATE = """\
{SYSTEM}
=== ARTICLE À NOTER ===
Titre : {TITRE}
Texte : {TEXTE}

score="""

# =============================================================
# OLLAMA
# =============================================================

def ollama_call(prompt: str) -> str:
    payload = {
        "model":  config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    r = requests.post(config.OLLAMA_URL, json=payload, timeout=config.OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    if "response" in data:
        return data["response"]
    if "message" in data and isinstance(data["message"], dict):
        return data["message"].get("content", "")
    raise RuntimeError(f"Réponse inattendue : {json.dumps(data)[:200]}")


def parse_score(raw: str):
    """
    Extrait le score depuis la réponse du modèle.
    Le prompt se termine par 'score=' donc le modèle complète avec le chiffre.
    Robuste aux espaces, retours à la ligne, texte parasite.
    """
    # Cas principal : le modèle a complété "score=X" ou renvoyé "score=X"
    m = re.search(r"score\s*=\s*([+-]?\s*[012])", raw.strip(), re.IGNORECASE)
    if m:
        s = int(m.group(1).replace(" ", ""))
        return s if s in (-2, -1, 0, 1, 2) else None
    # Fallback : premier entier valide dans la réponse
    for token in raw.strip().split():
        token = token.strip(".,;:")
        if token in ("-2", "-1", "0", "1", "2", "+1", "+2"):
            return int(token)
    return None


# =============================================================
# AGRÉGATION PAR TOPIC
# =============================================================

def compute_topic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le score moyen par topic à partir des articles scorés.
    Retourne un DataFrame avec : topic, mean_score, median_score, n_docs
    """
    scored = df[df["llm_score"].notna() & (df["topic"] != -1)].copy()
    summary = (
        scored.groupby("topic")["llm_score"]
        .agg(mean_score="mean", median_score="median", n_docs="count")
        .reset_index()
    )
    summary["mean_score"]   = summary["mean_score"].round(4)
    summary["median_score"] = summary["median_score"].round(4)
    return summary


# =============================================================
# MAIN
# =============================================================

def run() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("ÉTAPE 1 — Scoring des articles via Ollama")
    print("=" * 60)

    os.makedirs(config.OUT_DIR, exist_ok=True)

    if not os.path.exists(config.ARTICLES_WITH_TOPICS_CSV):
        raise FileNotFoundError(
            f"docs_with_topics.csv introuvable : {config.ARTICLES_WITH_TOPICS_CSV}\n"
            "Lance d'abord l'étape 1 (01_bertopic.py)."
        )

    df = pd.read_csv(config.ARTICLES_WITH_TOPICS_CSV)
    print(f"Articles chargés : {len(df)}")

    # Filtrer les articles trop courts
    df = df[df["contenu"].notna() & (df["contenu"].str.len() >= config.MIN_DOC_LEN)].reset_index(drop=True)
    print(f"Articles à scorer (longueur ≥ {config.MIN_DOC_LEN} chars) : {len(df)}")

    # ── Reprise ───────────────────────────────────────────────
    if os.path.exists(config.DOCS_SCORED_CSV):
        df_prev = pd.read_csv(config.DOCS_SCORED_CSV)
        # Identifier les URLs déjà scorées (score non nul)
        done_urls = set(
            df_prev.loc[df_prev["llm_score"].notna(), "url"].tolist()
        )
        print(f"Reprise : {len(done_urls)} articles déjà scorés")
        # Mettre à jour df avec les scores existants
        df = df.merge(
            df_prev[["url", "llm_score"]],
            on="url", how="left"
        )
    else:
        df["llm_score"] = None
        done_urls = set()

    to_score = df[df["llm_score"].isna()].index.tolist()
    total    = len(to_score)
    print(f"Articles restants à scorer : {total}")
    print(f"Modèle Ollama : {config.OLLAMA_MODEL}\n")

    # ── Boucle de scoring ─────────────────────────────────────
    ok = skip = err = 0
    SAVE_EVERY   = 50
    BAR_WIDTH    = 30
    SCORE_LABELS = {-2: "██ -2", -1: "█  -1", 0: "·   0", 1: "   +1 █", 2: "   +2 ██"}

    for i, idx in enumerate(to_score):
        row   = df.loc[idx]
        titre = str(row.get("titre", ""))[:60].ljust(60)
        texte = str(row.get("contenu", "") or "")[:1500]

        prompt = PROMPT_TEMPLATE.format(
            SYSTEM=SYSTEM_PROMPT,
            TITRE=str(row.get("titre", ""))[:200],
            TEXTE=texte,
        )

        try:
            raw   = ollama_call(prompt)
            score = parse_score(raw)

            if score is None:
                skip += 1
                df.at[idx, "llm_score"] = None
                score_disp = "  ?"
            else:
                ok += 1
                df.at[idx, "llm_score"] = score
                score_disp = SCORE_LABELS.get(score, str(score))

        except Exception as e:
            err += 1
            df.at[idx, "llm_score"] = None
            score_disp = " ERR"
            print(f"\n  ERR idx={idx} : {e}")

        # Barre de progression
        pct      = (i + 1) / total
        filled   = int(BAR_WIDTH * pct)
        bar      = "█" * filled + "░" * (BAR_WIDTH - filled)
        elapsed  = f"{ok+skip+err}/{total}"
        print(
            f"\r  [{bar}] {100*pct:5.1f}%  {elapsed:>12}  "
            f"score={score_disp}  titre={titre}  "
            f"(OK={ok} skip={skip} err={err})",
            end="", flush=True,
        )

        # Sauvegarde intermédiaire
        if (i + 1) % SAVE_EVERY == 0:
            print()   # saut de ligne avant le message de sauvegarde
            df.to_csv(config.DOCS_SCORED_CSV, index=False, encoding="utf-8")
            print(f"  → sauvegarde intermédiaire ({i+1} articles)")

        time.sleep(config.OLLAMA_SLEEP)

    print()   # saut de ligne final après la barre

    # ── Sauvegarde finale des articles ───────────────────────
    df.to_csv(config.DOCS_SCORED_CSV, index=False, encoding="utf-8")
    print(f"\nOK={ok} | SKIP={skip} | ERR={err}")
    print(f"docs_scored_v2.csv → {config.DOCS_SCORED_CSV}")

    # ── Agrégation par topic ──────────────────────────────────
    print("\nAgrégation par topic...")
    summary = compute_topic_summary(df)
    summary.to_csv(config.TOPIC_SCORE_SUMMARY_CSV, index=False, encoding="utf-8")
    print(f"topic_score_summary_v2.csv → {config.TOPIC_SCORE_SUMMARY_CSV}")

    print(f"\nTopics avec score : {len(summary)}")
    print(summary.sort_values("mean_score").to_string(index=False))

    print("=" * 60)
    print("ÉTAPE 1 — TERMINÉE")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run()
