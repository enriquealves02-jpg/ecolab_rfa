# =============================================================================
# Score de potentiel écologique par IRIS
# Sources : CarHAB, BD Forêt, COSIA
# =============================================================================

library(sf)
library(dplyr)
library(tidyverse)
library(FactoMineR)
library(factoextra)
library(tmap)

options(sf_use_s2 = FALSE)


# =============================================================================
# 1. CHARGEMENT DES DONNEES
# =============================================================================

# CarHAB
carhab <- bind_rows(
  st_read("C:/Users/anna/Documents/dev durable/claude/donnee/CarHab_26_Drome/CarHab_26_Drome/CarHab_26_Drome_Habitats_CarHab.gpkg", quiet = TRUE),
  st_read("C:/Users/anna/Documents/dev durable/claude/donnee/CarHab_84_Vaucluse/CarHab_84_Vaucluse/CarHab_84_Vaucluse_Habitats_CarHab.gpkg", quiet = TRUE)
) %>%
  st_make_valid() %>%
  select(milieu, occupation, humidite_edaphique, cd_hab, surface, geom)

# IRIS
iris_sf <- st_read("C:/Users/anna/Documents/dev durable/claude/donnee/CONTOURS IRIS_IGN/CONTOURS-IRIS_SRPB_2024-01-01.shp") %>%
  st_make_valid()

surface_iris <- iris_sf %>%
  mutate(surface_iris_m2 = st_area(.) %>% as.numeric()) %>%
  st_drop_geometry() %>%
  select(CODE_IRIS, surface_iris_m2)

# BD Forêt
foret <- bind_rows(
  st_read("C:/Users/anna/Documents/dev durable/claude/donnee/BDFORET_2-0__SHP_LAMB93_D007_2014-04-01/BDFORET_2-0__SHP_LAMB93_D007_2014-04-01/BDFORET/1_DONNEES_LIVRAISON/BDF_2-0_SHP_LAMB93_D007/FORMATION_VEGETALE.shp"),
  st_read("C:/Users/anna/Documents/dev durable/claude/donnee/BDFORET_2-0__SHP_LAMB93_D026_2014-04-01/BDFORET_2-0__SHP_LAMB93_D026_2014-04-01/BDFORET/1_DONNEES_LIVRAISON/BDF_2-0_SHP_LAMB93_D026/FORMATION_VEGETALE.shp"),
  st_read("C:/Users/anna/Documents/dev durable/claude/donnee/BDFORET_2-0__SHP_LAMB93_D084_2022-04-01/BDFORET_2-0__SHP_LAMB93_D084_2022-04-01/BDFORET/1_DONNEES_LIVRAISON/BDF_2-0_SHP_LAMB93_D084/FORMATION_VEGETALE.shp")
) %>% st_make_valid()

# COSIA
base_cosia <- read.csv("C:/Users/anna/Documents/dev durable/claude/donnee/inter_cosia_iris_2.csv") %>%
  select(CODE_IRIS, classe, surface_inter_m2) %>%
  mutate(CODE_IRIS = as.character(CODE_IRIS)) %>%
  left_join(surface_iris, by = "CODE_IRIS")


# =============================================================================
# 2. INDICATEURS CARHAB
# =============================================================================

inter_carhab <- st_intersection(
  iris_sf %>% select(CODE_IRIS) %>% st_make_valid(),
  carhab
) %>%
  mutate(surface_inter_m2 = st_area(.) %>% as.numeric())

base_inter <- inter_carhab %>%
  st_drop_geometry() %>%
  left_join(surface_iris, by = "CODE_IRIS")

milieux_enjeu  <- c("prairies_temp", "surface_en_eau", "autres_perennes", "m_ouvert")
humide_vals    <- c("humide à nappe circulante", "aquatique et amphibie à nappe stagnante",
                    "aquatique et amphibie à nappe circulante", "humide à nappe stagnante",
                    "légèrement humide", "détrempé à nappe stagnante")
xero_vals      <- c("très sec", "perxérique", "assez sec")

ind1 <- base_inter %>%
  mutate(est_enjeu = milieu %in% milieux_enjeu) %>%
  group_by(CODE_IRIS) %>%
  summarise(pct_milieu_enjeu = sum(surface_inter_m2[est_enjeu]) / first(surface_iris_m2) * 100, .groups = "drop")

ind2 <- base_inter %>%
  mutate(est_humide = humidite_edaphique %in% humide_vals,
         est_xero   = humidite_edaphique %in% xero_vals) %>%
  group_by(CODE_IRIS) %>%
  summarise(
    pct_zone_humide = sum(surface_inter_m2[est_humide]) / first(surface_iris_m2) * 100,
    pct_zone_xero   = sum(surface_inter_m2[est_xero])   / first(surface_iris_m2) * 100,
    .groups = "drop"
  )

ind3 <- base_inter %>%
  group_by(CODE_IRIS, milieu) %>%
  summarise(surf_milieu = sum(surface_inter_m2), .groups = "drop") %>%
  group_by(CODE_IRIS) %>%
  mutate(p = surf_milieu / sum(surf_milieu)) %>%
  summarise(
    shannon_milieu  = -sum(p * log(p + 1e-9)),
    nb_types_milieu = n_distinct(milieu),
    .groups = "drop"
  )

ind4 <- base_inter %>%
  mutate(est_agri_occup = occupation == "agri") %>%
  group_by(CODE_IRIS) %>%
  summarise(pct_occup_agri = sum(surface_inter_m2[est_agri_occup]) / first(surface_iris_m2) * 100, .groups = "drop")

df_carhab_iris <- surface_iris %>%
  left_join(ind1, by = "CODE_IRIS") %>%
  left_join(ind2, by = "CODE_IRIS") %>%
  left_join(ind3, by = "CODE_IRIS") %>%
  left_join(ind4, by = "CODE_IRIS") %>%
  replace(is.na(.), 0)


# =============================================================================
# 3. INDICATEURS BD FORET
# =============================================================================

feuillus_enjeu <- c(
  "Forêt fermée de hêtre pur", "Forêt fermée de chênes décidus purs",
  "Forêt fermée de chênes sempervirents purs", "Forêt fermée de châtaignier pur",
  "Forêt fermée à mélange de feuillus", "Forêt fermée de feuillus purs en îlots",
  "Forêt ouverte de feuillus purs",
  "Forêt fermée à mélange de feuillus prépondérants et conifères"
)

coniferes <- c(
  "Forêt fermée de pin sylvestre pur", "Forêt fermée de pin maritime pur",
  "Forêt fermée de pin laricio ou pin noir pur", "Forêt fermée de pin d\u0092Alep pur",
  "Forêt fermée de douglas pur", "Forêt fermée de sapin ou épicéa",
  "Forêt fermée de mélèze pur", "Forêt fermée de conifères purs en îlots",
  "Forêt ouverte de conifères purs", "Forêt fermée à mélange de conifères",
  "Forêt fermée à mélange de pins purs", "Forêt fermée à mélange d\u0092autres conifères",
  "Forêt fermée d\u0092un autre conifère pur autre que pin", "Forêt fermée d\u0092un autre pin pur",
  "Forêt fermée à mélange de conifères prépondérants et feuillus"
)

foret <- foret %>% st_transform(st_crs(iris_sf))

inter_foret <- st_intersection(
  iris_sf %>% select(CODE_IRIS) %>% st_make_valid(),
  foret %>% select(TFV, ESSENCE)
) %>%
  mutate(surface_inter_m2 = st_area(.) %>% as.numeric())

base_foret <- inter_foret %>%
  st_drop_geometry() %>%
  left_join(surface_iris, by = "CODE_IRIS")

ind_foret1 <- base_foret %>%
  mutate(est_feuillu_enjeu = TFV %in% feuillus_enjeu) %>%
  group_by(CODE_IRIS) %>%
  summarise(pct_feuillu_enjeu = sum(surface_inter_m2[est_feuillu_enjeu]) / first(surface_iris_m2) * 100, .groups = "drop")

ind_foret2 <- base_foret %>%
  mutate(est_conifere = TFV %in% coniferes) %>%
  group_by(CODE_IRIS) %>%
  summarise(pct_conifere = sum(surface_inter_m2[est_conifere]) / first(surface_iris_m2) * 100, .groups = "drop")

ind_foret3 <- base_foret %>%
  filter(!ESSENCE %in% c("NC", "NR")) %>%
  group_by(CODE_IRIS, ESSENCE) %>%
  summarise(surf_essence = sum(surface_inter_m2), .groups = "drop") %>%
  group_by(CODE_IRIS) %>%
  mutate(p = surf_essence / sum(surf_essence)) %>%
  summarise(nb_essences = n_distinct(ESSENCE), .groups = "drop")

df_foret_iris <- surface_iris %>%
  left_join(ind_foret1, by = "CODE_IRIS") %>%
  left_join(ind_foret2, by = "CODE_IRIS") %>%
  left_join(ind_foret3, by = "CODE_IRIS") %>%
  replace(is.na(.), 0)


# =============================================================================
# 4. INDICATEURS COSIA
# =============================================================================

cosia_pivot <- base_cosia %>%
  group_by(CODE_IRIS, classe) %>%
  summarise(surface_inter_m2 = sum(surface_inter_m2), .groups = "drop") %>%
  left_join(surface_iris, by = "CODE_IRIS") %>%
  mutate(pct = surface_inter_m2 / surface_iris_m2 * 100) %>%
  select(CODE_IRIS, classe, pct) %>%
  pivot_wider(names_from = classe, values_from = pct, values_fill = 0)

names(cosia_pivot) <- names(cosia_pivot) %>%
  tolower() %>% gsub(" ", "_", .) %>%
  gsub("é|è|ê", "e", .) %>% gsub("â", "a", .) %>%
  iconv(to = "ASCII//TRANSLIT")

df_cosia_iris <- cosia_pivot %>%
  mutate(
    pct_veg_naturelle = feuillu + pelouse + broussaille + surface_eau,
    pct_artificialise = batiment + zone_impermeable + serre + piscine,
    pct_agri_cosia    = culture + terre_labouree + vigne,
    shannon_cosia = apply(
      cbind(broussaille, batiment, conifere, culture, feuillu,
            pelouse, piscine, serre, sol_nu, surface_eau,
            terre_labouree, vigne, zone_impermeable, zone_permeable),
      1, function(x) { p <- x[x > 0] / 100; -sum(p * log(p + 1e-9)) }
    )
  ) %>%
  select(code_iris, pct_veg_naturelle, pct_artificialise, pct_agri_cosia, shannon_cosia) %>%
  rename(CODE_IRIS = code_iris)


# =============================================================================
# 5. ACP ET SCORE FINAL (CarHAB + BD Forêt + COSIA)
# =============================================================================

df_final <- df_carhab_iris %>%
  select(CODE_IRIS, pct_milieu_enjeu, pct_zone_humide,
         shannon_milieu, nb_types_milieu, pct_occup_agri) %>%
  left_join(
    df_foret_iris %>% select(CODE_IRIS, pct_feuillu_enjeu, pct_conifere, nb_essences),
    by = "CODE_IRIS"
  ) %>%
  left_join(df_cosia_iris, by = "CODE_IRIS") %>%
  replace(is.na(.), 0)

mat_final <- df_final %>% column_to_rownames("CODE_IRIS")

res_pca_final <- PCA(mat_final, scale.unit = TRUE, graph = FALSE, ncp = 5)

fviz_eig(res_pca_final, addlabels = TRUE)

fviz_pca_var(res_pca_final,
             col.var = "contrib",
             gradient.cols = c("grey", "orange", "red"),
             repel = TRUE) +
  ggtitle("Cercle des corrélations — Dim.1 vs Dim.2")

fviz_pca_var(res_pca_final,
             axes = c(1, 3),
             col.var = "contrib",
             gradient.cols = c("grey", "orange", "red"),
             repel = TRUE) +
  ggtitle("Cercle des corrélations — Dim.1 vs Dim.3")

res_pca_final$var$contrib %>% round(2)
res_pca_final$var$coord[, 1:3] %>% round(2)

# Score pondéré par la variance expliquée (3 premières dimensions)
variances_final <- res_pca_final$eig[1:3, "percentage of variance"]
coords_final    <- as.data.frame(res_pca_final$ind$coord)

score_matrix <- cbind(
  coords_final$Dim.1,   # naturalité / diversité habitat
  -coords_final$Dim.2,  # artificialisation pénalisée (axe inversé)
  coords_final$Dim.3    # zones humides vs pression agricole
)

df_final$score_brut <- as.vector(score_matrix %*% variances_final / sum(variances_final))

df_final$score_biodiv_final <- with(df_final,
  (score_brut - min(score_brut)) / (max(score_brut) - min(score_brut)) * 100
)

df_final <- df_final %>%
  mutate(
    donnees_manquantes = (pct_milieu_enjeu == 0 & pct_veg_naturelle == 0 &
                            shannon_milieu == 0 & nb_essences == 0),
    score_final_clean  = ifelse(donnees_manquantes, NA, score_biodiv_final)
  )


# =============================================================================
# 6. CARTOGRAPHIE ET EXPORT
# =============================================================================

iris_score_final <- iris_sf %>%
  left_join(df_final %>% select(CODE_IRIS, score_final_clean, donnees_manquantes),
            by = "CODE_IRIS")

tm_shape(iris_score_final) +
  tm_fill("score_final_clean",
          palette  = "YlGn", n = 7,
          colorNA  = "lightgrey", textNA = "Données insuffisantes",
          title    = "Score potentiel\nécologique (0-100)") +
  tm_borders(alpha = 0.3) +
  tm_layout(title = "Score potentiel écologique final — CarHAB + BD Forêt + COSIA",
            legend.outside = TRUE)

write.csv(
  df_final %>% select(CODE_IRIS, score_final_clean, score_biodiv_final, donnees_manquantes),
  "score_potentiel_ecologique_final.csv",
  row.names = FALSE
)


# =============================================================================
# 7. TEST DE SENSIBILITE DES PONDERATIONS
# =============================================================================

scenarios <- list(
  pca  = c(39.4, 17.7, 10.0),  # poids ACP (actuel)
  egal = c(1, 1, 1),            # poids égaux
  dom1 = c(60, 25, 15)          # Dim.1 dominante
)

scores_scenarios <- map_dfc(scenarios, function(w) {
  score <- as.vector(score_matrix %*% w / sum(w))
  (score - min(score)) / (max(score) - min(score)) * 100
})

names(scores_scenarios) <- names(scenarios)
scores_scenarios$CODE_IRIS <- df_final$CODE_IRIS

cor(scores_scenarios %>% select(-CODE_IRIS))
