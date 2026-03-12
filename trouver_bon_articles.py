"""
EcoLab — Trouver les meilleurs articles par label COSIA / BD TOPO
==================================================================
Principe :
  1. Encoder les articles synthétiques (un par label COSIA + catégorie BD TOPO)
  2. Encoder tous les articles réels (6 sources CSV)  [cache .npy]
  3. Similarité cosinus → top-5 articles réels les plus proches de chaque label
  4. Sortie CSV : outputs/trouver_bon_articles/top5_par_label.csv

Modifie la section ARTICLES_SYNTHETIQUES pour affiner les textes.
"""

import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ═══════════════════════════════════════════════════════════════
# SORTIES
# ═══════════════════════════════════════════════════════════════
OUT_DIR   = os.path.join(config.OUT_DIR, "trouver_bon_articles")
OUT_CSV   = os.path.join(OUT_DIR, "top5_par_label.csv")
CACHE_EMB = os.path.join(OUT_DIR, "embeddings_cache.npy")
CACHE_IDX = os.path.join(OUT_DIR, "embeddings_cache_index.csv")

TOP_N = 5   # nombre d'articles à retenir par label

# ═══════════════════════════════════════════════════════════════
# SOURCES D'ARTICLES RÉELS
# (config.NEWS_SOURCES + 3 CSV Dauphiné)
# ═══════════════════════════════════════════════════════════════
REAL_SOURCES = config.NEWS_SOURCES + [
    {"path": os.path.join(config.DATA_DIR, "ledauphine_articles.csv"),        "source": "dauphine"},
    {"path": os.path.join(config.DATA_DIR, "ledauphine_articles_2022.csv"),   "source": "dauphine"},
    {"path": os.path.join(config.DATA_DIR, "ledauphine_articles_202026.csv"), "source": "dauphine"},
]

# ═══════════════════════════════════════════════════════════════
# ARTICLES SYNTHÉTIQUES — 1 par label COSIA + 1 par catégorie BD TOPO
#
# Clé   = identifiant du label (libre, utilisé dans le CSV de sortie)
# Valeur = texte de l'article synthétique "idéal" pour ce label
#
# ⚠ COMPLÈTE CES TEXTES selon tes besoins avant de lancer le script.
# ═══════════════════════════════════════════════════════════════
ARTICLES_SYNTHETIQUES: dict[str, str] = {

    # ── LABELS COSIA ────────────────────────────────────────────

    "Bâtiment": (
        "La construction de logements neufs s'accélère dans les communes périurbaines. "
        "Face à la demande croissante de logements, plusieurs municipalités ont accordé "
        "de nouveaux permis de construire pour des résidences, des immeubles collectifs "
        "et des zones pavillonnaires. Les promoteurs immobiliers s'installent sur d'anciennes "
        "terres agricoles converties en zones à urbaniser. La rénovation énergétique des "
        "bâtiments existants est aussi au cœur des politiques locales, avec des aides "
        "pour l'isolation thermique, le remplacement des chaudières et l'installation "
        "de pompes à chaleur. Le secteur du bâtiment représente un enjeu majeur dans la "
        "réduction des émissions de gaz à effet de serre, tant pour les constructions neuves "
        "que pour le parc immobilier ancien. L'étalement urbain inquiète les associations "
        "qui dénoncent la disparition progressive des terres naturelles et agricoles au profit "
        "de zones résidentielles et commerciales."
    ),

    "Zone imperméable": (
        "L'imperméabilisation des sols atteint un niveau préoccupant dans les zones périurbaines. "
        "Routes, parkings, zones commerciales et voiries recouvrent chaque année des milliers "
        "d'hectares de terres auparavant perméables. Le ruissellement des eaux pluviales "
        "augmente, aggravant les risques d'inondation en aval des bassins versants. "
        "Les collectivités cherchent à limiter l'artificialisation en imposant des quotas "
        "de surfaces imperméables dans les nouveaux projets d'aménagement. Des solutions "
        "comme les revêtements drainants, les noues et les jardins de pluie sont désormais "
        "intégrées aux plans d'urbanisme. Les autoroutes, les zones industrielles et "
        "les grandes surfaces commerciales constituent les principales sources d'imperméabilisation. "
        "La loi Zéro Artificialisation Nette impose aux communes de réduire drastiquement "
        "la création de nouvelles surfaces imperméables d'ici 2030."
    ),

    "Zone perméable": (
        "Les espaces verts urbains jouent un rôle crucial dans la régulation thermique "
        "et hydrologique des villes. Parcs, jardins publics, prairies urbaines et friches "
        "végétalisées constituent des zones perméables essentielles qui permettent à l'eau "
        "de s'infiltrer dans le sol et d'alimenter les nappes phréatiques. Les villes "
        "intègrent de plus en plus la nature en milieu urbain grâce aux toitures végétalisées, "
        "aux coulées vertes et aux jardins partagés. La désimperméabilisation de certaines "
        "voiries, le retrait du bitume sur d'anciens parkings et la création de jardins de "
        "pluie participent à la reconquête de sols perméables. Ces surfaces jouent également "
        "un rôle de fraîcheur en période de canicule, réduisant l'effet d'îlot de chaleur "
        "urbain. Les plans locaux d'urbanisme encouragent désormais la préservation et "
        "la restauration des continuités végétales en milieu bâti."
    ),

    "Pelouse": (
        "Les prairies naturelles et les pelouses sèches abritent une biodiversité remarquable "
        "souvent méconnue. Ces milieux ouverts accueillent des dizaines d'espèces d'insectes "
        "pollinisateurs, d'orchidées sauvages et d'oiseaux nicheurs. La fauche tardive, "
        "réalisée après la floraison des plantes, permet de préserver les cycles de reproduction "
        "de la faune et de la flore. De nombreuses communes adoptent désormais une gestion "
        "différenciée de leurs espaces verts, laissant pousser l'herbe librement dans certaines "
        "zones pour favoriser la biodiversité. Les pâturages extensifs avec des troupeaux "
        "de moutons ou de bovins entretiennent ces pelouses tout en maintenant des paysages "
        "ouverts caractéristiques. Le retour du pâturage sur des pelouses calcaires abandonnées "
        "contribue à restaurer des habitats d'intérêt communautaire menacés par l'embroussaillement "
        "progressif."
    ),

    "Broussaille": (
        "Les milieux broussailleux représentent une étape dynamique dans la reconquête "
        "végétale des terrains abandonnés. Friches agricoles, talus ferroviaires et anciennes "
        "carrières se transforment progressivement en fourrés d'épineux, de ronces et "
        "d'arbustes pionniers. Ces habitats semi-ouverts offrent gîte et couvert à de nombreuses "
        "espèces animales, notamment les passereaux et les mammifères comme le chevreuil et "
        "le lièvre. Cependant, la fermeture des milieux par l'avancée des broussailles menace "
        "les pelouses sèches et les landes ouvertes. La gestion par débroussaillage mécanique "
        "ou pastoralisme est nécessaire pour maintenir la mosaïque de milieux. En zone "
        "méditerranéenne, les broussailles constituent également un risque incendie important, "
        "justifiant des opérations régulières de débroussaillement préventif autour des habitations "
        "et des forêts."
    ),

    "Feuillu": (
        "Les forêts de feuillus représentent le couvert forestier dominant des plaines "
        "et des coteaux tempérés. Chênes, hêtres, charmes, frênes et érables forment "
        "des écosystèmes riches en biodiversité, abritant champignons, insectes saproxyliques "
        "et oiseaux forestiers. La sylviculture à couvert continu tend à remplacer les coupes "
        "rases pour préserver les fonctions écologiques de ces forêts. Chaque année, plusieurs "
        "milliers d'hectares de feuillus sont récoltés pour alimenter les filières bois-énergie "
        "et bois d'œuvre. Les forêts de feuillus jouent un rôle majeur dans le stockage du "
        "carbone et la régulation du cycle de l'eau. La dépérissement des chênaies et des "
        "frênaies, aggravé par les sécheresses successives et les pathogènes émergents, "
        "inquiète forestiers et écologues qui observent un affaiblissement généralisé "
        "de ces arbres emblématiques."
    ),

    "Conifère": (
        "Les forêts de conifères couvrent les massifs montagnards et constituent une ressource "
        "économique majeure pour la filière bois. Sapins, épicéas, pins sylvestres et mélèzes "
        "sont exploités pour la construction, la papeterie et le bois de chauffage. "
        "Les sylviculteurs font face à des défis sans précédent avec les attaques de scolytes, "
        "des insectes ravageurs favorisés par les sécheresses répétées qui affaiblissent "
        "les résineux. Des milliers d'hectares de pessières et de sapinières dépérissent "
        "chaque année, contraignant les propriétaires forestiers à des coupes sanitaires d'urgence. "
        "Le reboisement en essences diversifiées et résistantes au changement climatique "
        "devient une priorité pour l'Office National des Forêts. La certification "
        "PEFC garantit une gestion durable des forêts de conifères, en préservant "
        "leur rôle de régulation hydrologique et de protection contre l'érosion "
        "dans les zones de montagne."
    ),

    "Culture": (
        "Les grandes cultures céréalières dominent les plaines agricoles françaises. "
        "Blé tendre, maïs, colza, tournesol et orge constituent l'essentiel des surfaces "
        "cultivées. Les agriculteurs adaptent leurs assolements face aux aléas climatiques "
        "et aux nouvelles contraintes réglementaires sur l'usage des pesticides. "
        "L'agriculture de précision, avec ses drones et ses capteurs connectés, permet "
        "de réduire les intrants tout en maintenant les rendements. La conversion à "
        "l'agriculture biologique progresse dans certaines exploitations, répondant à "
        "une demande croissante des consommateurs. Les terres cultivées constituent "
        "un enjeu central dans la politique de Zéro Artificialisation Nette, qui vise "
        "à stopper la conversion des terres agricoles en zones bâties. La préservation "
        "du foncier agricole face à la pression urbaine est au cœur des débats sur "
        "l'aménagement du territoire et la souveraineté alimentaire."
    ),

    "Terre labourée": (
        "Le labour est une pratique agricole fondamentale qui consiste à retourner "
        "et ameublir la couche superficielle du sol avant les semis. Les terres labourées, "
        "reconnaissables à leur couleur sombre et leur surface travaillée, sont caractéristiques "
        "des exploitations en agriculture conventionnelle. Cependant, de nombreux agriculteurs "
        "abandonnent le labour profond au profit du semis direct ou du travail superficiel "
        "du sol, afin de préserver la vie microbienne, limiter l'érosion et réduire "
        "la consommation de carburant. La déchaumaison après les récoltes laisse le sol "
        "nu et vulnérable à l'érosion hydrique et éolienne. Les couverts végétaux "
        "intercalaires permettent de protéger ces terres labourées durant l'hiver "
        "et d'enrichir le sol en matière organique. La qualité des sols agricoles, "
        "mesurée par leur teneur en carbone organique, est aujourd'hui un indicateur "
        "clé de la durabilité des pratiques culturales."
    ),

    "Vigne": (
        "La viticulture constitue un pilier de l'économie agricole dans de nombreuses "
        "régions françaises. Les vignobles en appellation d'origine contrôlée produisent "
        "des vins reconnus mondialement, tandis que les vins de pays et les vins de table "
        "répondent à une demande plus quotidienne. Les viticulteurs font face au défi "
        "du changement climatique qui avance les dates de vendanges et modifie les "
        "profils aromatiques des vins. La conversion en viticulture biologique ou "
        "biodynamique progresse, réduisant l'usage des pesticides et du soufre. "
        "La taille de la vigne, réalisée en hiver, la protection contre le gel de "
        "printemps et la lutte contre le mildiou constituent les principaux travaux "
        "viticoles. Les caves coopératives regroupent des centaines de viticulteurs "
        "qui mutualisent la vinification et la commercialisation. Le tourisme viticole, "
        "avec les visites de domaines et les œnotouristes, génère des revenus "
        "complémentaires pour les exploitations."
    ),

    "Serre": (
        "La production maraîchère sous serre connaît un essor important pour répondre "
        "à la demande en légumes frais toute l'année. Tomates, concombres, poivrons "
        "et salades sont cultivés en serres chauffées ou froides selon les saisons. "
        "Les serres modernes intègrent des systèmes d'irrigation au goutte-à-goutte, "
        "de récupération des eaux de pluie et de contrôle climatique automatisé. "
        "L'agrivoltaïsme, qui consiste à associer production agricole et panneaux "
        "solaires, se développe sur les serres agricoles, permettant de produire "
        "simultanément de l'énergie et des aliments. Les serres photovoltaïques "
        "suscitent cependant des débats sur leur impact sur la biodiversité et "
        "les paysages agricoles. La filière horticulture florale produit également "
        "sous abri des fleurs coupées et des plantes en pot pour les marchés locaux "
        "et nationaux."
    ),

    "Piscine": (
        "La prolifération des piscines privées soulève des questions environnementales "
        "croissantes dans un contexte de sécheresse récurrente. Chaque piscine nécessite "
        "plusieurs dizaines de mètres cubes d'eau pour son remplissage initial et "
        "son entretien annuel, ce qui pèse sur les ressources en eau potable. "
        "Certaines communes ont instauré des restrictions d'usage lors des périodes "
        "de sécheresse, interdisant le remplissage des piscines. Les piscines occupent "
        "également des surfaces imperméabilisées qui imperméabilisent le sol autour "
        "des habitations. Le traitement chimique des eaux avec le chlore soulève "
        "des inquiétudes sur la pollution des nappes phréatiques en cas de fuites. "
        "Les piscines naturelles, filtrées par des plantes aquatiques et sans produits "
        "chimiques, se développent comme alternative écologique. Les piscines publiques "
        "municipales, mutualisées, sont davantage encouragées que les bassins privés "
        "dans les plans de sobriété hydrique."
    ),

    "Surface eau": (
        "Les lacs, étangs et zones humides constituent des écosystèmes précieux "
        "pour la biodiversité et la régulation du cycle de l'eau. Ces milieux aquatiques "
        "abritent des espèces végétales et animales rares, notamment des amphibiens, "
        "des libellules et des oiseaux d'eau migrateurs. Les zones humides jouent "
        "un rôle de tampon contre les inondations en stockant temporairement les crues. "
        "Leur destruction progressive par le drainage, le remblayage et l'urbanisation "
        "a conduit à la disparition de la moitié des zones humides françaises en "
        "cinquante ans. Les plans de restauration des milieux humides se multiplient "
        "pour reconstituer ces habitats menacés. La qualité de l'eau des lacs et "
        "étangs est menacée par les proliférations de cyanobactéries, favorisées "
        "par l'eutrophisation due aux apports d'azote et de phosphore agricoles. "
        "Les retenues collinaires créées pour l'irrigation agricole suscitent des "
        "controverses sur leur impact sur les milieux aquatiques naturels."
    ),

    "Sol nu": (
        "Les sols nus représentent une surface vulnérable exposée aux phénomènes "
        "d'érosion hydrique et éolienne. Carrières, chantiers de construction, "
        "zones de remblai et terrains vagues constituent autant de surfaces dénudées "
        "qui perdent rapidement leurs horizons organiques sous l'action de la pluie "
        "et du vent. L'érosion des sols agricoles laissés nus après les récoltes "
        "emporte des milliers de tonnes de terre chaque année. Les friches industrielles "
        "polluées nécessitent des opérations de dépollution coûteuses avant toute "
        "réhabilitation. Les opérations de terrassement et de génie civil mettent à nu "
        "de larges surfaces pendant la durée des travaux, aggravant les risques "
        "d'inondation et de pollution des cours d'eau voisins. La revegetation rapide "
        "des sols nus après les travaux est une exigence réglementaire pour limiter "
        "ces impacts environnementaux."
    ),

    "Neige": (
        "L'enneigement des massifs montagnards recule inexorablement sous l'effet "
        "du changement climatique. Les stations de ski font face à des hivers de plus "
        "en plus déficitaires en neige naturelle, contraignant les exploitants à "
        "investir massivement dans les canons à neige artificielle. Le manteau neigeux "
        "printanier, qui constitue une réserve d'eau vitale pour les vallées alpines, "
        "fond de plus en plus tôt dans la saison. Les glaciers régressent à un rythme "
        "alarmant, menaçant à terme les ressources en eau des rivières qui en dépendent "
        "en été. Le risque avalanche reste une préoccupation majeure dans les zones "
        "habitées et les voies de communication de montagne. Les bulletins d'enneigement "
        "et les prévisions météo-neige mobilisent des services spécialisés pour anticiper "
        "et prévenir les accidents. La diversification des activités touristiques "
        "hivernales, au-delà du ski alpin, devient une nécessité économique pour "
        "les territoires de montagne."
    ),

    # ── CATÉGORIES BD TOPO ──────────────────────────────────────

    "PARC_NATURA2000": (
        "Le réseau Natura 2000 protège les habitats naturels et les espèces sauvages "
        "les plus menacés d'Europe à travers un maillage de sites d'importance communautaire "
        "et de zones de protection spéciale pour les oiseaux. En France, ce réseau couvre "
        "près de 13 % du territoire terrestre. Les sites Natura 2000 sont gérés de manière "
        "contractuelle avec les acteurs locaux, agriculteurs, forestiers et collectivités, "
        "sans interdiction systématique des activités humaines. Les documents d'objectifs "
        "fixent les mesures de conservation compatibles avec les usages traditionnels. "
        "Les habitats prioritaires comme les tourbières, les pelouses calcaires et les "
        "forêts alluviales bénéficient d'une protection renforcée. Des conflits surgissent "
        "parfois entre les exigences de conservation et les projets d'aménagement, notamment "
        "l'implantation d'éoliennes ou l'extension de carrières en zone Natura 2000."
    ),

    "PARC_PNR": (
        "Les parcs naturels régionaux constituent des territoires ruraux habités, "
        "reconnus pour la qualité de leurs paysages et de leur patrimoine naturel et culturel. "
        "Leur charte, approuvée par l'État pour douze ans, fixe les orientations de "
        "développement durable du territoire. Les parcs naturels régionaux soutiennent "
        "l'agriculture locale et les produits du terroir, valorisent l'artisanat traditionnel "
        "et développent un tourisme doux respectueux de l'environnement. Ils jouent un rôle "
        "de laboratoire pour des pratiques innovantes en matière de transition écologique. "
        "Le Parc naturel régional du Vercors, par exemple, protège des milieux emblématiques "
        "comme les hauts plateaux karstiques et les forêts de résineux tout en maintenant "
        "une vie économique locale dynamique. Les parcs régionaux sont des acteurs clés "
        "dans la mise en œuvre des politiques de biodiversité à l'échelon local."
    ),

    "PARC_RESERVE": (
        "Les réserves naturelles offrent une protection stricte aux espèces et habitats "
        "les plus vulnérables du patrimoine naturel français. Créées par décret, elles "
        "réglementent strictement les activités humaines autorisées sur leur périmètre. "
        "Les réserves biologiques intégrales, gérées par l'ONF, permettent à la forêt "
        "d'évoluer librement sans intervention humaine, pour étudier la dynamique naturelle "
        "des écosystèmes forestiers. Les arrêtés de protection de biotope protègent "
        "quant à eux les habitats d'espèces sauvages particulièrement sensibles comme "
        "le desman des Pyrénées ou certaines orchidées. Le conservatoire des espaces "
        "naturels anime un réseau de sites préservés, souvent acquis grâce à des fonds "
        "publics et gérés par des bénévoles. Ces espaces protégés jouent un rôle de "
        "refuges de biodiversité et de réservoirs écologiques au sein de la trame "
        "verte et bleue."
    ),

    "GEOPARC": (
        "Les géoparcs mondiaux UNESCO valorisent les territoires où l'histoire géologique "
        "est exceptionnellement lisible dans le paysage. Le Géoparc du Massif du Vercors "
        "illustre les processus karstiques à travers ses grottes, avens et canyons "
        "sculptés dans les calcaires jurassiques. Les visiteurs peuvent y observer "
        "des fossiles marins datant de plusieurs dizaines de millions d'années, témoins "
        "d'anciens fonds océaniques. La géodiversité de ces territoires, avec leurs "
        "formations rocheuses spectaculaires, attire géologues, randonneurs et touristes. "
        "Le géoparc mène des programmes d'éducation à la géologie dans les écoles locales "
        "pour sensibiliser les jeunes générations au patrimoine de leur territoire. "
        "La gestion durable du géoparc concilie préservation du patrimoine géologique, "
        "développement touristique et vie économique locale."
    ),

    "ZONE_SILENCE": (
        "Les zones de calme sont des espaces où la pollution sonore est volontairement "
        "limitée pour préserver la tranquillité naturelle et le bien-être des habitants "
        "et de la faune. La prise de conscience de l'impact du bruit sur la santé "
        "humaine et la biodiversité a conduit certains territoires à créer des réserves "
        "de silence. L'absence de bruit de fond artificiel permet d'entendre les sons "
        "naturels, chants d'oiseaux, bruissement des feuilles, ruissellement de l'eau, "
        "constituant un patrimoine paysager à préserver. Les nuisances sonores des "
        "infrastructures routières, ferroviaires et aériennes perturbent le comportement "
        "des espèces animales, notamment leur communication et leur reproduction. "
        "Plusieurs parcs naturels intègrent désormais des cartographies du silence "
        "dans leurs plans de gestion pour identifier et protéger les zones encore "
        "épargnées par la pollution acoustique."
    ),

    "FORET_PUBLIQUE": (
        "Les forêts domaniales et publiques, gérées par l'Office National des Forêts, "
        "couvrent près de 30 % de la surface forestière française. L'ONF assure leur "
        "gestion durable selon les principes de la sylviculture proche de la nature, "
        "en privilégiant la régénération naturelle et la diversité des essences. "
        "Ces forêts publiques sont accessibles à tous pour la promenade, la cueillette "
        "et les loisirs de pleine nature, dans le respect du règlement forestier. "
        "Les coupes de bois programmées permettent de financer l'entretien des forêts "
        "et de produire du bois d'œuvre et de chauffage pour les filières locales. "
        "Face au dépérissement forestier lié aux sécheresses et aux pathogènes, "
        "l'ONF mène des programmes de reboisement en essences diversifiées, mieux "
        "adaptées aux conditions climatiques futures. La forêt publique joue également "
        "un rôle de régulation des risques naturels, notamment la protection contre "
        "les avalanches et les glissements de terrain en montagne."
    ),

    "ZONE_VEG_FORET": (
        "Les massifs forestiers constituent des écosystèmes complexes qui abritent "
        "une biodiversité exceptionnelle et rendent de nombreux services environnementaux. "
        "La forêt stocke du carbone, régule le cycle de l'eau, prévient l'érosion des sols "
        "et offre des conditions climatiques favorables aux espèces qui y vivent. "
        "Les incendies de forêt constituent l'une des principales menaces pesant sur "
        "ces écosystèmes, aggravés par les vagues de chaleur et les épisodes de sécheresse. "
        "La déforestation mondiale, notamment en Amazonie et en Asie du Sud-Est, "
        "provoque chaque année la disparition de millions d'hectares de forêts tropicales. "
        "En France, la surface forestière progresse depuis un siècle grâce à la déprise "
        "agricole, mais la qualité et la biodiversité des forêts se dégradent. "
        "Le reboisement après incendie et les plans de gestion forestière durable "
        "cherchent à restaurer des forêts résilientes face au changement climatique."
    ),

    "ZONE_VEG_VIGNE": (
        "Les vignobles façonnent des paysages agricoles caractéristiques des régions "
        "viticoles, notamment en Bourgogne, en Bordeaux, en Champagne et dans la vallée "
        "du Rhône. La culture de la vigne en appellation d'origine protégée répond à "
        "des cahiers des charges stricts sur les cépages autorisés, les rendements "
        "et les pratiques culturales. La viticulture biologique et biodynamique connaît "
        "une progression rapide, portée par les attentes des consommateurs et des "
        "contraintes réglementaires sur les produits phytosanitaires. Le changement "
        "climatique modifie profondément la viticulture, avançant les dates de vendanges "
        "et favorisant l'implantation de nouveaux cépages résistants à la chaleur. "
        "Les vignerons indépendants et les caves coopératives valorisent leur terroir "
        "à travers l'œnotourisme, attirant des visiteurs dans les chais et les routes "
        "des vins. La protection des paysages viticoles, parfois classés au patrimoine "
        "mondial de l'UNESCO, est un enjeu culturel et économique majeur."
    ),

    "ZONE_VEG_VERGER": (
        "L'arboriculture fruitière occupe des vallées entières dans les régions de "
        "polyculture. Pommes, poires, cerises, pêches et noix sont cultivés dans "
        "des vergers qui structurent le paysage agricole. Les vergers traditionnels "
        "à haute tige, moins productifs que les vergers intensifs modernes, abritent "
        "une biodiversité importante et constituent des habitats de prédilection pour "
        "les rapaces et les chauves-souris. La peupleraie, autre culture ligneuse "
        "de plaine, fournit du bois de trituration pour l'industrie papetière mais "
        "transforme profondément les milieux rivulaires. Les producteurs bio développent "
        "des techniques alternatives aux pesticides comme la confusion sexuelle "
        "pour lutter contre les carpocapses. Les marchés de producteurs locaux "
        "et les circuits courts valorisent les fruits de qualité issus de variétés "
        "anciennes, alternatives aux variétés industrielles standardisées."
    ),

    "ZONE_VEG_LANDE": (
        "Les haies bocagères constituent un élément fondamental des paysages ruraux "
        "traditionnels et un maillon essentiel de la trame verte et bleue. Ces alignements "
        "d'arbres et d'arbustes délimitent les parcelles agricoles, brise-vent naturels "
        "qui protègent les cultures et le bétail. Elles abritent une faune abondante, "
        "notamment les insectes pollinisateurs, les rapaces nicheurs et les reptiles "
        "qui régulent les populations de rongeurs. La suppression massive des haies "
        "lors du remembrement agricole des années 1960-1980 a provoqué l'érosion des "
        "sols, la disparition d'espèces et la détérioration de la qualité de l'eau. "
        "Les programmes de replantation de haies se multiplient désormais dans le cadre "
        "des plans de reconquête de la biodiversité. Les landes à genêts et à bruyères, "
        "habitats semi-naturels issus de pratiques agropastorales ancestrales, "
        "régressent face à la reforestation spontanée et à l'intensification agricole."
    ),

    "COURS_EAU": (
        "Les rivières et les cours d'eau constituent des artères vitales pour les "
        "écosystèmes terrestres et aquatiques. La qualité physico-chimique et biologique "
        "des cours d'eau s'est fortement dégradée depuis l'après-guerre sous l'effet "
        "des pollutions agricoles, industrielles et domestiques. Les nitrates et les "
        "pesticides agricoles contaminent les nappes phréatiques et les cours d'eau, "
        "menaçant la faune piscicole et la santé humaine. Les programmes de restauration "
        "de la continuité écologique visent à supprimer les obstacles à la migration "
        "des poissons comme les seuils et les barrages obsolètes. Les étiages sévères "
        "durant les étés de sécheresse assèchent certains tronçons de rivières, menaçant "
        "la survie des truites et autres espèces exigeantes en eau fraîche et oxygénée. "
        "Les zones humides riveraines, ripisylves et prairies inondables, jouent un rôle "
        "épurateur naturel en filtrant les polluants avant leur arrivée dans la rivière."
    ),

    "PLAN_EAU": (
        "Les lacs naturels et les retenues artificielles constituent des réservoirs "
        "d'eau douce essentiels pour l'alimentation en eau potable, l'irrigation agricole "
        "et la production hydroélectrique. Le niveau des lacs alpins suit de près "
        "la courbe de l'enneigement hivernal et de la fonte printanière des glaciers. "
        "Les étés de sécheresse provoquent une baisse préoccupante du niveau des retenues, "
        "entraînant des restrictions d'irrigation et des conflits d'usage entre agriculteurs, "
        "collectivités et environnementalistes. La qualité de l'eau des plans d'eau "
        "est menacée par les cyanobactéries toxiques, dont les proliférations sont "
        "favorisées par le réchauffement des eaux et les apports nutritifs excessifs. "
        "La pêche de loisir et les activités nautiques constituent des usages économiques "
        "importants des lacs et retenues, soumis à des réglementations strictes pour "
        "préserver les peuplements piscicoles et la tranquillité des riverains."
    ),

    "SURFACE_HYDRO": (
        "Les zones humides, mares, marais et prairies inondables constituent des "
        "écosystèmes parmi les plus riches de la planète en termes de biodiversité. "
        "Elles abritent des espèces végétales et animales rares, notamment des amphibiens "
        "comme la grenouille agile, des libellules et des plantes aquatiques protégées. "
        "La destruction des zones humides par le drainage, le remblayage et l'urbanisation "
        "a conduit à leur disparition massive, avec des conséquences dramatiques sur "
        "la régulation des inondations et la purification naturelle de l'eau. "
        "Les mares rurales, qui ponctuaient autrefois le paysage bocager, ont été "
        "comblées par milliers avec la mécanisation agricole. Des programmes de "
        "restauration et de création de mares se développent pour reconstituer ces "
        "habitats précieux. Les zones humides stockent également du carbone dans "
        "leurs tourbières et contribuent à l'atténuation du changement climatique."
    ),

    "ZONE_PECHE": (
        "La pêche de loisir est la principale activité sportive de plein air en France, "
        "pratiquée par plusieurs millions de pêcheurs à la ligne. Les rivières, étangs "
        "et lacs sont gérés par les associations agréées de pêche et de protection "
        "du milieu aquatique qui veillent à la qualité des eaux et à l'abondance "
        "des peuplements piscicoles. Les alevinage réguliers compensent les prélèvements "
        "et maintiennent des populations de truites, carpes et brochets suffisantes "
        "pour la pratique de la pêche. La pisciculture en eau douce produit des poissons "
        "de consommation dans des bassins et étangs spécialement aménagés. Les conflits "
        "entre pêcheurs et autres usagers de l'eau, notamment les kayakistes et les "
        "baigneurs, sont fréquents en période estivale. La pollution des eaux et les "
        "sécheresses menacent les frayères et les habitats piscicoles, nécessitant "
        "une surveillance constante de la qualité des milieux aquatiques."
    ),

    "EOLIENNE": (
        "L'énergie éolienne constitue l'une des principales sources d'énergie renouvelable "
        "en plein développement en France. Les parcs éoliens terrestres se multiplient "
        "dans les zones venteuses, notamment les plateaux et les crêtes des massifs "
        "montagnards. Chaque éolienne peut produire plusieurs mégawatts d'électricité "
        "selon sa taille et les conditions de vent. L'implantation de nouveaux parcs "
        "éoliens suscite des oppositions locales liées à l'impact paysager, au bruit "
        "et aux effets sur la faune, notamment les oiseaux et les chauves-souris. "
        "Les procédures d'autorisation environnementale sont longues et complexes, "
        "avec de nombreux recours juridiques qui retardent la mise en service "
        "des projets. L'éolien offshore, plus puissant et moins contesté que l'éolien "
        "terrestre, se développe sur les façades maritimes avec des fermes flottantes "
        "en haute mer. L'objectif de la France est d'atteindre une part importante "
        "d'électricité éolienne dans son mix énergétique d'ici 2030."
    ),

    "BARRAGE": (
        "Les barrages hydroélectriques constituent l'épine dorsale de la production "
        "d'électricité renouvelable en France, qui possède le deuxième plus grand parc "
        "hydraulique d'Europe. Ces ouvrages de génie civil monumentaux retiennent "
        "des millions de mètres cubes d'eau dans des réservoirs de montagne, "
        "libérés au fil des besoins pour actionner les turbines et produire de l'électricité. "
        "Les barrages jouent également un rôle dans la régulation des crues et "
        "le soutien des débits d'étiage en été, permettant l'irrigation agricole. "
        "La continuité écologique des cours d'eau est fortement perturbée par ces "
        "ouvrages qui bloquent la migration des poissons comme le saumon et la truite "
        "de mer. Des passes à poissons sont installées pour permettre la remontée "
        "des migrateurs. La révision des concessions hydrauliques est un enjeu "
        "stratégique pour les producteurs d'électricité et les territoires concernés."
    ),

    # ── ARTICLES SPÉCIFIQUES DRÔME ──────────────────────────────

    "Agriculture Drôme": (
        "La Drôme est un département agricole diversifié où se côtoient viticulture, "
        "arboriculture fruitière, grandes cultures et élevage. La Drôme est le premier "
        "département bio de France, avec près d'un quart de la surface agricole utile "
        "cultivée en agriculture biologique. La lavande et les plantes aromatiques "
        "constituent des productions emblématiques des plateaux du Diois et du Tricastin. "
        "Les abricots de la Drôme, produits dans les vallées du Rhône et de l'Isère, "
        "sont reconnus pour leur qualité. La viticulture des Côtes-du-Rhône et de "
        "Crozes-Hermitage structure l'économie agricole du nord du département. "
        "Les agriculteurs drômois font face aux conséquences du changement climatique, "
        "avec des épisodes de gel tardif, de grêle et de sécheresse qui menacent "
        "les récoltes. La préservation du foncier agricole est un enjeu crucial "
        "dans une Drôme soumise à une forte pression urbaine, notamment autour "
        "de Valence et dans la plaine de Romans. Les circuits courts et la vente "
        "directe se développent pour valoriser les productions locales."
    ),

    "Artificialisation Drôme": (
        "La Drôme connaît une artificialisation croissante de ses terres naturelles "
        "et agricoles sous la pression de l'urbanisation. Le schéma de cohérence "
        "territoriale du Pays de Romans-Bourg-de-Péage intègre les objectifs de "
        "Zéro Artificialisation Nette, imposant aux communes de limiter la consommation "
        "de foncier naturel. Les zones d'activités économiques se multiplient le long "
        "des axes routiers, notamment la RN 532 et l'autoroute A7, imperméabilisant "
        "des terrains agricoles et naturels. Les zones pavillonnaires s'étendent dans "
        "les communes périurbaines de Valence, Romans-sur-Isère et Montélimar, "
        "absorbant d'anciennes terres cultivées. Le SCOT de la Drôme des Collines "
        "et le SCoT du Pays de Bièvre-Valloire tentent de canaliser cette urbanisation "
        "diffuse. Les associations de protection de la nature alertent sur la disparition "
        "progressive des corridors écologiques entre les massifs du Vercors et des "
        "Baronnies. La renaturation des zones imperméabilisées et la densification "
        "du tissu urbain existant sont présentées comme les solutions pour concilier "
        "développement et préservation des terres drômoises."
    ),
}

# ═══════════════════════════════════════════════════════════════
# FONCTIONS
# ═══════════════════════════════════════════════════════════════

def load_real_articles() -> pd.DataFrame:
    """Charge et déduplique tous les articles réels."""
    import re

    def clean(s):
        if not isinstance(s, str):
            return ""
        return re.sub(r"\s+", " ", s).strip()

    frames = []
    for src in REAL_SOURCES:
        path = src["path"]
        if not os.path.exists(path):
            print(f"  IGNORÉ (introuvable) : {os.path.basename(path)}")
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
            for col in ("titre", "contenu", "url"):
                if col not in df.columns:
                    df[col] = ""
            df["source"] = src["source"]
            df["doc"] = (df["titre"].fillna("") + ". " + df["contenu"].fillna("")).map(clean)
            frames.append(df[["titre", "contenu", "url", "source", "doc"]])
            print(f"  {src['source']:<12} {os.path.basename(path):<45} → {len(df):>6} docs")
        except Exception as e:
            print(f"  ERREUR {os.path.basename(path)} : {e}")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = (
        df_all.dropna(subset=["url"])
              .drop_duplicates(subset=["url"])
              .reset_index(drop=True)
    )
    df_all = df_all[df_all["doc"].str.len() >= 20].reset_index(drop=True)
    return df_all


def get_embeddings(model: SentenceTransformer, texts: list[str],
                   cache_npy: str | None = None,
                   cache_idx_csv: str | None = None,
                   index_col: pd.Series | None = None) -> np.ndarray:
    """
    Encode une liste de textes.
    Si cache_npy existe et que l'index correspond, charge depuis le cache.
    """
    if cache_npy and cache_idx_csv and os.path.exists(cache_npy) and os.path.exists(cache_idx_csv):
        cached_idx = pd.read_csv(cache_idx_csv)["url"].tolist()
        if index_col is not None and list(index_col) == cached_idx:
            print("  Cache embeddings chargé.")
            return np.load(cache_npy)
        else:
            print("  Cache obsolète, recalcul...")

    print(f"  Encoding {len(texts)} textes...")
    embs = model.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True)

    if cache_npy and cache_idx_csv and index_col is not None:
        np.save(cache_npy, embs)
        pd.DataFrame({"url": list(index_col)}).to_csv(cache_idx_csv, index=False)
        print(f"  Cache sauvegardé → {cache_npy}")

    return embs


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run() -> None:
    print("\n" + "=" * 60)
    print("Trouver les meilleurs articles par label COSIA / BD TOPO")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Charger les articles réels ─────────────────────────
    print("\n[1/4] Chargement des articles réels...")
    df_real = load_real_articles()
    print(f"  → {len(df_real)} articles uniques au total")

    # ── 2. Embeddings réels (avec cache) ──────────────────────
    print("\n[2/4] Embeddings des articles réels...")
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    real_embs = get_embeddings(
        model,
        df_real["doc"].tolist(),
        cache_npy=CACHE_EMB,
        cache_idx_csv=CACHE_IDX,
        index_col=df_real["url"],
    )
    print(f"  Embeddings réels : shape {real_embs.shape}")

    # ── 3. Embeddings synthétiques ────────────────────────────
    print("\n[3/4] Embeddings des articles synthétiques...")
    labels    = list(ARTICLES_SYNTHETIQUES.keys())
    synth_texts = list(ARTICLES_SYNTHETIQUES.values())
    synth_embs  = model.encode(synth_texts, show_progress_bar=False, convert_to_numpy=True)
    print(f"  {len(labels)} articles synthétiques encodés")

    # ── 4. Similarité cosinus + top-5 par label ───────────────
    print("\n[4/4] Calcul des similarités et sélection top-5...")
    sim_matrix = cosine_similarity(synth_embs, real_embs)  # (n_labels, n_docs)

    rows = []
    for i, label in enumerate(labels):
        sims = sim_matrix[i]
        top_idx = np.argsort(sims)[::-1][:TOP_N]
        for rank, idx in enumerate(top_idx, 1):
            row = df_real.iloc[idx]
            rows.append({
                "label":   label,
                "rank":    rank,
                "score":   round(float(sims[idx]), 4),
                "source":  row["source"],
                "titre":   row["titre"],
                "url":     row["url"],
                "contenu": str(row["contenu"] or "")[:300] if pd.notna(row["contenu"]) else "",
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n  Résultat sauvegardé → {OUT_CSV}")

    # ── Résumé console ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"{'Label':<25} {'Meilleur article (score)'}")
    print(f"{'─'*60}")
    for label in labels:
        top = df_out[(df_out["label"] == label) & (df_out["rank"] == 1)].iloc[0]
        titre_court = str(top["titre"])[:45]
        print(f"  {label:<23} {top['score']:.3f}  {titre_court}")
    print(f"{'─'*60}")
    print("TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    run()
