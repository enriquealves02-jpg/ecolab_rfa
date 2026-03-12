# Donnees externes requises pour la pipeline v2

Ces fichiers sont trop volumineux pour Git. Ils doivent etre places dans les
repertoires indiques AVANT de lancer `run_pipeline_v2.bat`.

## Structure attendue (repertoire parent de pipeline_v2/)

```
ecolab/
├── pipeline_v2/           <-- ce dossier (dans Git)
│   ├── data/
│   │   ├── bertopic_v2/   articles_with_topic_labels.csv, topics_info_v2.csv (dans Git)
│   │   ├── news/          CSVs articles de presse (dans Git)
│   │   ├── figures/       barplot + timeseries PNG (dans Git)
│   │   └── geo/           BDTOPO_resume_par_combo.csv, COSIA_articles_par_classe.csv (dans Git)
│   ├── outputs/           genere par la pipeline
│   └── *.py               scripts
│
├── ecolab2_territoire_scot/   <-- A TELECHARGER (hors Git)
│   ├── COSIA_1-0__GPKG_LAMB93_D007_2023-01-01/
│   ├── COSIA_1-0__GPKG_LAMB93_D026_2023-01-01/
│   ├── COSIA_1-0__GPKG_LAMB93_D084_2024-01-01/
│   ├── BDTOPO_3-5_TOUSTHEMES_SHP_LAMB93_D007_2025-12-15/
│   ├── BDTOPO_3-5_TOUSTHEMES_SHP_LAMB93_D026_2025-12-15/
│   ├── BDTOPO_3-5_TOUSTHEMES_SHP_LAMB93_D084_2025-12-15/
│   └── Parcelles_SCOT.gpkg
│
├── donnéesNews/               <-- Copie dans data/news/ (dans Git)
│   └── bertopic_out/          <-- Copie dans data/figures/ (dans Git)
│
└── pipeline/outputs/bertopic_v2/  <-- Copie dans data/bertopic_v2/ (dans Git)
```

## Sources des donnees geospatiales (hors Git)

| Donnee | Source | Taille approx |
|--------|--------|---------------|
| COSIA (3 departements) | geoservices.ign.fr | ~5 GB chaque |
| BD TOPO (3 departements) | geoservices.ign.fr | ~2 GB chaque |
| Parcelles_SCOT.gpkg | Genere en amont (decoupage SCoT) | 257 MB |

## Prerequis logiciels

- Python 3.10+
- Ollama avec modele `mistral` charge (`ollama pull mistral`)
- Packages: geopandas, shapely, pyproj, pandas, streamlit, flask, sentence-transformers
