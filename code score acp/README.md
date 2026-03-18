# Score de potentiel écologique par IRIS
## CarHAB + BD Forêt + COSIA

Ce script calcule un score de potentiel écologique par IRIS à partir de trois sources de données : CarHAB, BD Forêt et COSIA. Le score final (0-100) est produit par ACP pondérée et cartographié par IRIS.

---

## Librairies R à installer

```r
install.packages(c("sf", "dplyr", "tidyverse", "FactoMineR", "factoextra", "tmap"))
```

| Librairie | Utilisation |
|-----------|-------------|
| `sf` | Lecture et manipulation des données spatiales |
| `dplyr` | Manipulation des tableaux |
| `tidyverse` | Manipulation des données (pivot, map, etc.) |
| `FactoMineR` | Analyse en Composantes Principales (ACP) |
| `factoextra` | Visualisation des résultats de l'ACP |
| `tmap` | Cartographie des scores par IRIS |

---

## Données nécessaires

### 1. CarHAB — Habitats naturels
Cartographie des habitats naturels et semi-naturels.

| Fichier | Département | Format |
|--------|-------------|--------|
| `CarHab_26_Drome_Habitats_CarHab.gpkg` | Drôme (26) | GeoPackage |
| `CarHab_84_Vaucluse_Habitats_CarHab.gpkg` | Vaucluse (84) | GeoPackage |

Colonnes utilisées : `milieu`, `occupation`, `humidite_edaphique`, `cd_hab`, `surface`

Source : [Carmen — Ministère de l'Écologie](https://carmen.carmencarto.fr/)

---

### 2. Contours IRIS — IGN
Découpage géographique de référence.

| Fichier | Format |
|--------|--------|
| `CONTOURS-IRIS_SRPB_2024-01-01.shp` | Shapefile |

Colonne utilisée : `CODE_IRIS`

Source : [IGN — Géoservices](https://geoservices.ign.fr/contoursiris)

---

### 3. BD Forêt v2 — IGN
Base de données des formations végétales forestières.

| Fichier | Département | Format |
|--------|-------------|--------|
| `FORMATION_VEGETALE.shp` (D007) | Ardèche (07) | Shapefile |
| `FORMATION_VEGETALE.shp` (D026) | Drôme (26) | Shapefile |
| `FORMATION_VEGETALE.shp` (D084) | Vaucluse (84) | Shapefile |

Colonnes utilisées : `TFV` (type de formation végétale), `ESSENCE`

Source : [IGN — Géoservices](https://geoservices.ign.fr/bdforet)

---

### 4. COSIA — Occupation du sol
Données d'occupation du sol par classe (bâti, végétation, eau, etc.).

| Fichier | Format |
|--------|--------|
| `inter_cosia_iris_2.csv` | CSV |

Colonnes utilisées : `CODE_IRIS`, `classe`, `surface_inter_m2`

> Ce fichier est le résultat d'une intersection préalable réalisée sous **QGIS** entre les données COSIA et les contours IRIS, à l'aide de l'outil **Intersection** (Vecteur > Géotraitements > Intersection). Le fichier résultant a ensuite été exporté au format CSV. Cette étape doit être effectuée en amont de l'exécution du script R.

---

## Structure des chemins attendus

```
E:/claude/donnee/
├── CarHab_26_Drome/
│   └── CarHab_26_Drome/
│       └── CarHab_26_Drome_Habitats_CarHab.gpkg
├── CarHab_84_Vaucluse/
│   └── CarHab_84_Vaucluse/
│       └── CarHab_84_Vaucluse_Habitats_CarHab.gpkg
├── CONTOURS IRIS_IGN/
│   └── CONTOURS-IRIS_SRPB_2024-01-01.shp
├── BDFORET_2-0__SHP_LAMB93_D007_2014-04-01/
│   └── .../FORMATION_VEGETALE.shp
├── BDFORET_2-0__SHP_LAMB93_D026_2014-04-01/
│   └── .../FORMATION_VEGETALE.shp
├── BDFORET_2-0__SHP_LAMB93_D084_2022-04-01/
│   └── .../FORMATION_VEGETALE.shp
└── inter_cosia_iris_2.csv
```

> Les chemins sont définis en dur dans le script. Les adapter si nécessaire.

---

## Résultat produit

Le script génère un fichier CSV :

**`score_potentiel_ecologique_final.csv`**

| Colonne | Description |
|--------|-------------|
| `CODE_IRIS` | Identifiant IRIS |
| `score_biodiv_final` | Score brut normalisé 0-100 |
| `score_final_clean` | Score final (NA si données insuffisantes) |
| `donnees_manquantes` | TRUE si l'IRIS n'a aucune donnée exploitable |
