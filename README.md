# EcoLab Pipeline V2

Pipeline de scoring d'artificialisation des sols par croisement d'articles de presse (NLP/BERTopic) et de donnees geospatiales (COSIA + BD TOPO) sur le territoire SCoT Drome-Ardeche-Vaucluse.

## Architecture

```
Scraping articles  -->  BERTopic (topic modeling)  -->  Notation Ollama/Mistral
       |                        |                              |
       v                        v                              v
  data/news/            data/bertopic_v2/              outputs/docs_scored_v2.csv
                                                               |
                                                               v
                        Score geospatial COSIA  <--  topic_score_summary_v2.csv
                                |
                                v
                        Enrichissement BD TOPO (14 categories)
                                |
                                v
                        Export QGIS (quintiles + grille)
                                |
                                v
                        Articles par parcelle (similarite cosinus)
                                |
                                v
                        Resumes BD TOPO (Ollama)  -->  Injection dans GPKG
                                |
                                v
                        Index SQLite  -->  Streamlit interactif
```

## Prerequis

### Python

```bash
pip install -r requirements.txt
```

### Ollama (LLM local)

Installer Ollama : https://ollama.com/download

```bash
ollama pull mistral
```

Pour paralleliser la notation (etape 1) :

```bash
# Windows - avant de lancer le script
set OLLAMA_NUM_PARALLEL=6
set OLLAMA_MAX_LOADED_MODELS=1
```

### Donnees geospatiales (hors Git, trop volumineuses)

Ces fichiers doivent etre telecharges et places dans `../ecolab2_territoire_scot/` (repertoire frere de `pipeline_v2/`) :

| Donnee | Source | Taille |
|--------|--------|--------|
| COSIA D007, D026, D084 | [geoservices.ign.fr](https://geoservices.ign.fr) | ~5 GB chaque |
| BD TOPO D007, D026, D084 | [geoservices.ign.fr](https://geoservices.ign.fr) | ~2 GB chaque |
| Parcelles_SCOT.gpkg | Decoupage SCoT en amont | 257 MB |

Structure attendue :

```
parent/
├── pipeline_v2/                  <-- ce repo
├── ecolab2_territoire_scot/      <-- donnees geo (a telecharger)
│   ├── COSIA_1-0__GPKG_LAMB93_D007_2023-01-01/
│   ├── COSIA_1-0__GPKG_LAMB93_D026_2023-01-01/
│   ├── COSIA_1-0__GPKG_LAMB93_D084_2024-01-01/
│   ├── BDTOPO_3-5_TOUSTHEMES_SHP_LAMB93_D007_2025-12-15/
│   ├── BDTOPO_3-5_TOUSTHEMES_SHP_LAMB93_D026_2025-12-15/
│   ├── BDTOPO_3-5_TOUSTHEMES_SHP_LAMB93_D084_2025-12-15/
│   └── Parcelles_SCOT.gpkg
└── donnéesNews/                  <-- copie dans data/news/ (dans Git)
```

## Lancement

### Pipeline complete (Windows)

```bash
cd pipeline_v2
run_pipeline_v2.bat
```

### Etape par etape

| Etape | Script | Description | Duree estimee |
|-------|--------|-------------|---------------|
| 0a | `01_bertopic_v2.py` | Topic modeling BERTopic sur les articles | ~30 min |
| 0b | `trouver_bon_articles.py` | Selection meilleurs articles par topic | ~10 min |
| 0c | `post_processing_bertopic_v2_best_article.py` | Post-processing des resultats BERTopic | ~5 min |
| 1 | `01_topic_notes_ollama_v2.py` | Notation des articles via Ollama/Mistral | ~3h (6 workers) |
| 2 | `02_topic_viz_v2.py` | Visualisations (barplot, distribution) | ~1 min |
| 3 | `03_geospatial_score_cosia_v2.py` | Score geospatial sur polygones COSIA | ~2h |
| 4 | `04_enrich_cosia_bdtopo_v2.py` | Enrichissement BD TOPO (14 categories) | ~3h |
| 5 | `05_export_qgis_v2.py` | Export GPKG pour QGIS | ~4h |
| 6 | `06_articles_par_parcelle_v2.py` | Liaison articles-parcelles | ~2h |
| 7a | `06b_resume_bdtopo.py` | Resumes BD TOPO par combo (Ollama) | ~1h |
| 7b | `06c_inject_resume_bdtopo.py` | Injection resumes dans le GPKG | ~10 min |
| 8 | `create_indexes.py` | Index SQLite pour filtrage rapide | ~40 min |

### Streamlit (carte interactive)

```bash
streamlit run app_carte2.py
```

Ouvre un navigateur avec :
- Carte Leaflet zoomable avec les parcelles colorees par score
- Sidebar : filtres par classe COSIA et categories BD TOPO
- Panel detail : score, articles lies, resumes COSIA et BD TOPO
- Onglet Analyses : barplot des topics + serie temporelle des sentiments

## Donnees incluses (dans Git)

```
data/
├── bertopic_v2/          CSVs issus du topic modeling
│   ├── articles_with_topic_labels.csv
│   ├── topics_info_v2.csv
│   └── topic_best_articles_bertopicv2.csv
├── news/                 Articles de presse scrapes (8 sources)
├── figures/              Visualisations BERTopic (PNG)
└── geo/                  Resumes et articles par classe (CSV)
```

## Scripts utilitaires

| Script | Description |
|--------|-------------|
| `check_bdtopo.py` | Diagnostic des colonnes BD TOPO dans le GPKG |
| `patch_zone_imperm.py` | Correction ciblee sans re-run complet |
| `export_carte_png.py` | Export carte haute resolution en PNG |
| `config_v2.py` | Configuration centralisee (chemins, mappings topic-COSIA) |

## GPU

Teste sur RTX 5060 Laptop (8 GB VRAM). Mistral 7B Q4 utilise ~4.1 GB, permettant 6 workers paralleles pour la notation.
