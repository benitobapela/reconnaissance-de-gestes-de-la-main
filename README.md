# Reconnaissance de gestes de la main par traitement d'images et apprentissage automatique

## 1. Présentation du projet

Ce projet s'inscrit dans le cadre du cours de Reconnaissance des formes. Il vise à concevoir un système capable de reconnaître cinq gestes de la main à partir d'images, en utilisant des techniques de traitement d'images et d'apprentissage automatique.

L'approche étudiée repose sur trois éléments principaux :

- **Segmentation par HSV** : isoler les régions susceptibles de correspondre à la peau.
- **Moments de Hu** : représenter la forme du contour principal par sept caractéristiques numériques.
- **k plus proches voisins (k-NN)** : classifier les gestes à partir des caractéristiques extraites.

Le travail est organisé en plusieurs étapes afin de rendre le traitement progressif, vérifiable et reproductible.

## 2. Objectifs

Les objectifs du projet sont les suivants :

- explorer et comprendre le dataset utilisé ;
- vérifier l'intégrité des images ;
- prétraiter les images pour faciliter leur traitement ;
- segmenter les régions de peau avec l'espace colorimétrique HSV ;
- détecter le contour principal de la main ;
- calculer les sept moments de Hu et appliquer une transformation logarithmique signée ;
- construire et valider un dataset de caractéristiques numériques ;
- préparer ces caractéristiques pour la classification k-NN ;
- évaluer les performances du système et analyser ses erreurs dans les étapes suivantes.

## 3. Dataset utilisé

Le projet utilise le dataset **HandGesture-5G**, constitué de 2 653 fichiers image répartis en cinq classes.

| Classe | Nombre d'images |
|---|---:|
| Fist | 542 |
| L-Sign | 537 |
| Open Palm_Open Hand | 661 |
| Pointing Up | 405 |
| V-Sign | 508 |
| **Total** | **2 653** |

Les fichiers comprennent des images JPG, JPEG et HEIC. Les images sont de grande résolution et présentent des variations de position, d'orientation, de taille de la main, d'éclairage et de couleur de peau.

L'analyse exploratoire a également révélé la présence de doublons exacts. Ces doublons doivent être pris en compte lors de la séparation des données d'entraînement et de test.

> Remarque : la détection de doublons réalisée à ce stade porte sur les images strictement identiques après décodage. Elle ne permet pas d'identifier tous les quasi-doublons.

## 4. Organisation du projet

L'arborescence du projet est organisée comme suit :

```text
reconnaissance-de-gestes-de-la-main/
│
├── data/
│   ├── raw/                 # Dataset brut
│   ├── processed/
│   │   ├── color/           # Images redimensionnées
│   │   ├── hsv/             # Images prétraitées dans l'espace HSV
│   │   ├── masks_hsv_initials/
│   │   └── masks_hsv_morphologie/
│   └── final/
│       └── features.csv     # Dataset final des caractéristiques
│
├── docs/                    # Documentation des étapes
├── figures/                 # Figures et visualisations
├── notebooks/               # Notebooks éventuels
├── results/                 # Rapports et tableaux de résultats
├── src/                     # Scripts Python
├── tests/                   # Tests éventuels
│
├── .gitignore
├── README.md
└── requirements.txt
```

Les dossiers de masques et certains fichiers intermédiaires peuvent être créés par les scripts lorsqu'ils sont exécutés. Leur présence dépend donc des étapes déjà lancées.

## 5. Environnement technique

### Langage et bibliothèques

- Python 3.13.16
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow
- pillow-heif

### Installation

Il est recommandé de travailler à la racine du projet et d'utiliser un environnement virtuel Python.

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Le fichier `requirements.txt` contient les dépendances suivantes :

```text
opencv-python
numpy
pandas
matplotlib
scikit-learn
pillow
pillow-heif
```

Si PowerShell empêche l'activation de l'environnement virtuel, il est possible d'utiliser l'interpréteur Python de l'environnement directement, ou de lancer les commandes depuis un terminal adapté. Il n'est pas nécessaire de modifier les règles de sécurité du système sans raison.

## 6. Les étapes réalisées

### Étape 1 — Choix du dataset et préparation du projet

Le dataset HandGesture-5G a été retenu pour travailler sur la reconnaissance de cinq gestes de la main.

Le projet Python a été initialisé avec une organisation séparant les données brutes, les données intermédiaires, les résultats, les figures, la documentation et le code source.

**Résultat :** une base de travail structurée et un dataset de référence défini.

### Étape 2 — Préparation de l'environnement et des outils

Les bibliothèques nécessaires au chargement, au traitement, à l'analyse et à la visualisation des images ont été installées.

Une attention particulière a été portée aux fichiers HEIC, qui nécessitent une prise en charge adaptée pour être décodés.

**Résultat :** un environnement Python permettant de traiter les différents formats d'image du dataset.

### Étape 3 — Analyse exploratoire et contrôle de l'intégrité

Cette étape a permis d'examiner la répartition des classes, les formats, les dimensions et l'état des fichiers.

Les contrôles ont porté notamment sur :

- les fichiers vides et illisibles ;
- les dimensions suspectes ;
- les images uniformes ;
- les doublons exacts ;
- la visualisation d'échantillons de chaque classe.

Les 2 653 fichiers ont pu être lus. L'analyse des doublons a identifié 846 groupes de copies exactes, représentant 1 692 fichiers appartenant à ces groupes. Aucun doublon exact détecté n'a été identifié entre des classes différentes.

**Résultat :** une meilleure connaissance du dataset et des risques de fuite de données lors de l'évaluation du modèle.

### Étape 4 — Prétraitement des images

Les images ont été redimensionnées en conservant leurs proportions, avec une longueur maximale de 256 pixels. Les images n'ont pas été agrandies lorsque leur taille initiale était inférieure à cette limite.

Le redimensionnement a été réalisé avec l'interpolation `INTER_AREA`, adaptée à la réduction d'images.

Les images brutes ont été conservées et les versions traitées enregistrées séparément. Les images couleur et leurs représentations HSV ont été générées pour les étapes suivantes.

Le traitement des 2 653 images s'est terminé sans échec.

**Résultat :** des images de taille réduite, plus pratiques à manipuler pendant la segmentation.

### Étape 5 — Segmentation de la peau par HSV et sélection du contour principal

L'espace colorimétrique HSV a été utilisé pour construire un masque des régions susceptibles de correspondre à la peau.

Les seuils retenus sont :

- borne inférieure : `[0, 20, 40]` ;
- borne supérieure : `[25, 255, 255]`.

Une fermeture morphologique avec un noyau elliptique de 5 × 5 a été appliquée, suivie d'une ouverture avec un noyau elliptique de 3 × 3.

Les contours externes ont ensuite été détectés avec OpenCV. Le contour de plus grande aire a été sélectionné comme contour principal.

Des essais sur un échantillon de cinq images ont permis de comparer plusieurs configurations morphologiques. Ils ont aussi montré que des fragments de peau ou des parties de la main peuvent parfois être mal segmentés.

**Résultat :** une méthode de segmentation et de sélection du contour principal définie pour l'extraction des caractéristiques.

### Étape 6 — Extraction et analyse des moments de Hu

Les sept moments de Hu ont été calculés à partir du contour principal sélectionné.

Afin de réduire les différences d'échelle entre les valeurs, une transformation logarithmique signée a été appliquée :

\[
H_i'=-\operatorname{sign}(H_i)
\log_{10}\left(\max(|H_i|,\varepsilon)\right)
\]

avec \(\varepsilon=10^{-30}\).

L'extraction a d'abord été testée sur un échantillon d'images, puis la distribution des caractéristiques a été étudiée sur 150 images, soit 30 images par classe.

Cette analyse a montré que les distributions de certains moments diffèrent entre les classes, mais présentent aussi des chevauchements. Les moments de Hu seuls ne garantissent donc pas une séparation parfaite des gestes.

**Résultat :** une méthode d'extraction des sept caractéristiques, accompagnée d'une analyse préliminaire de leur distribution.

### Étape 7 — Construction et validation du dataset final

Le script de construction a été exécuté sur toutes les images HSV disponibles.

Pour chaque image, il a appliqué la segmentation, la morphologie, la sélection du contour, le calcul des sept moments de Hu et leur transformation logarithmique signée.

Le livrable généré est :

`data/final/features.csv`

Ce fichier contient neuf colonnes :

- `image_id` ;
- `class` ;
- `Hu1`, `Hu2`, `Hu3`, `Hu4`, `Hu5`, `Hu6`, `Hu7`.

La construction a produit 2 653 lignes sans échec d'extraction. Le script de validation a confirmé la structure attendue, la répartition des cinq classes, l'absence de valeurs manquantes ou non finies et l'unicité des identifiants.

**Résultat :** un dataset de caractéristiques validé sur le plan structurel et numérique, prêt à être utilisé pour la classification.

## 7. Pipeline général

Le pipeline de traitement réalisé jusqu'à la construction du dataset final est le suivant :

```text
Dataset HandGesture-5G
          |
          v
Analyse exploratoire et contrôle
          |
          v
Prétraitement et redimensionnement
          |
          v
Conversion et stockage HSV
          |
          v
Segmentation de la peau
          |
          v
Morphologie et contour principal
          |
          v
Calcul des sept moments de Hu
          |
          v
Transformation logarithmique signée
          |
          v
Construction de features.csv
          |
          v
Validation du dataset final
          |
          v
Classification k-NN (étape suivante)
```

## 8. Scripts principaux

Les scripts ci-dessous sont utilisés pour les contrôles, les tests et la construction du dataset. Les scripts de test et d'analyse portent sur des échantillons, tandis que le script de construction traite l'ensemble des images.

| Script | Rôle |
|---|---|
| `src/dataset_check.py` | Contrôle initial du dataset |
| `src/check_heic.py` | Vérification de la lecture des fichiers HEIC |
| `src/explore_dataset.py` | Analyse exploratoire des dimensions et des classes |
| `src/visualize_samples.py` | Visualisation d'échantillons |
| `src/check_image_channels.py` | Contrôle des représentations couleur |
| `src/check_dataset_integrity.py` | Contrôles d'intégrité |
| `src/check_duplicates.py` | Détection des doublons exacts |
| `src/analyze_duplicates.py` | Analyse des doublons détectés |
| `src/test_hsv_feasibility.py` | Test préliminaire de faisabilité de la segmentation HSV |
| `src/compare_working_sizes.py` | Comparaison de tailles de traitement |
| `src/preprocess_dataset.py` | Prétraitement de toutes les images |
| `src/test_segmentation_hsv_initiale.py` | Test initial de segmentation |
| `src/test_morphologie_hsv.py` | Comparaison des opérations morphologiques |
| `src/test_contour_principal.py` | Test de sélection du contour principal |
| `src/test_moments_hu.py` | Test des moments de Hu sur un échantillon |
| `src/analyser_distribution_hu.py` | Analyse de la distribution des moments de Hu |
| `src/valider_distribution_hu.py` | Validation du tableau d'analyse des moments |
| `src/construire_dataset_final.py` | Construction de `features.csv` |
| `src/valider_dataset_final.py` | Validation du dataset final |

Les noms indiqués correspondent aux scripts utilisés dans le projet à ce stade. Il convient de vérifier leur présence effective dans le dépôt avant toute livraison.

## 9. Reproduire la construction du dataset final

Après installation des dépendances et préparation des images HSV dans `data/processed/hsv/`, exécuter :

```powershell
python src/construire_dataset_final.py
python src/valider_dataset_final.py
```

Le premier script génère `data/final/features.csv`. Le second contrôle sa structure, les effectifs des classes, les identifiants et les valeurs des caractéristiques.

Les images HSV prétraitées doivent être disponibles pour cette opération. Le dataset brut et les fichiers intermédiaires ne sont pas recréés automatiquement par ces deux commandes seules.

## 10. Limites connues

Plusieurs limites doivent être considérées lors de l'interprétation des résultats :

- la segmentation HSV dépend des conditions d'éclairage et de la couleur de peau ;
- des fragments de peau peuvent être détectés séparément ou certaines parties de la main peuvent être omises ;
- le contour de plus grande aire n'est pas nécessairement la main entière d# Reconnaissance de gestes de la main par traitement d'images et apprentissage automatique

## 1. Présentation du projet

Ce projet est réalisé dans le cadre du cours de Reconnaissance des formes. Il vise à concevoir un système capable de reconnaître cinq gestes de la main à partir d'images, en combinant des techniques de traitement d'images et d'apprentissage automatique.

L'approche retenue, appelée variante B, repose sur trois éléments principaux :

- **Segmentation HSV** : isoler les régions susceptibles de correspondre à la peau.
- **Moments de Hu** : représenter la forme du contour principal par sept caractéristiques numériques.
- **k plus proches voisins (k-NN)** : classifier les gestes à partir des caractéristiques extraites.

Le projet suit une démarche progressive et reproductible, allant de l'analyse du dataset à l'évaluation et à la sauvegarde du modèle entraîné.

## 2. Objectifs

Les objectifs du projet sont les suivants :

- explorer et vérifier l'intégrité du dataset ;
- prétraiter les images pour faciliter leur traitement ;
- segmenter les régions susceptibles de correspondre à la peau dans l'espace HSV ;
- détecter le contour principal de la main ;
- extraire les sept moments de Hu et appliquer une transformation logarithmique signée ;
- construire et valider un dataset de caractéristiques numériques ;
- préparer les données pour l'apprentissage automatique ;
- rechercher une configuration appropriée du classificateur k-NN ;
- entraîner et évaluer le modèle ;
- analyser les erreurs de classification et les limites de l'approche ;
- sauvegarder le modèle et vérifier qu'il peut être rechargé sans modifier ses prédictions.

## 3. Dataset utilisé

Le projet utilise le dataset **HandGesture-5G**, composé de 2 653 images réparties en cinq classes.

| Classe | Nombre d'images |
|---|---:|
| Fist | 542 |
| L-Sign | 537 |
| Open Palm_Open Hand | 661 |
| Pointing Up | 405 |
| V-Sign | 508 |
| **Total** | **2 653** |

Les fichiers comprennent des images JPG, JPEG et HEIC. La diversité des images implique des variations de position, d'orientation, de taille de la main, d'éclairage et d'arrière-plan.

L'analyse des doublons exacts a révélé :

- 2 653 images examinées ;
- 1 807 empreintes distinctes ;
- 846 groupes de doublons exacts de deux fichiers ;
- 1 692 fichiers appartenant à ces groupes ;
- 961 images sans doublon exact détecté ;
- aucun groupe de doublons exacts détecté entre plusieurs classes.

La détection porte sur les images strictement identiques après décodage. Elle ne garantit pas l'identification de toutes les images similaires ou presque identiques.

## 4. Organisation du projet

L'organisation générale du dépôt est la suivante :

```text
reconnaissance-de-gestes-de-la-main/
│
├── data/
│   ├── raw/                         # Images originales
│   ├── processed/
│   │   ├── color/                   # Images redimensionnées
│   │   ├── hsv/                     # Images HSV
│   │   ├── masks_hsv_initials/      # Exemples de masques initiaux
│   │   └── masks_hsv_morphologie/   # Exemples de masques après morphologie
│   └── final/
│       └── features.csv             # Caractéristiques finales
│
├── docs/                            # Documentation des étapes
├── figures/                         # Figures et visualisations
├── models/
│   ├── knn_pipeline.pkl              # Pipeline entraîné et sauvegardé
│   └── model_config.json             # Configuration du modèle
├── notebooks/                       # Notebooks éventuels
├── results/                         # Résultats, métriques et rapports
├── src/                             # Scripts Python
├── tests/                           # Tests éventuels
│
├── .gitignore
├── README.md
├── README_model.md
└── requirements.txt
```

Les fichiers intermédiaires et les figures sont générés par les différents scripts. Leur présence dépend des opérations déjà exécutées.

Le dossier `data/raw/` contient les images originales. Il est exclu du suivi Git afin d'éviter de versionner inutilement le dataset complet.

## 5. Environnement technique

### 5.1. Technologies utilisées

- Python 3.13.16
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow
- pillow-heif
- Joblib

### 5.2. Installation

Depuis la racine du projet, sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Le fichier `requirements.txt` doit contenir les dépendances nécessaires au projet, notamment :

```text
opencv-python
numpy
pandas
matplotlib
scikit-learn
pillow
pillow-heif
joblib
```

Les versions réellement utilisées peuvent être vérifiées dans l'environnement Python. Pour faciliter la reproduction exacte des expériences, il est recommandé de conserver également les versions des bibliothèques.

Si l'activation de l'environnement virtuel est bloquée par PowerShell, il est possible d'utiliser directement l'interpréteur Python de l'environnement virtuel ou un terminal adapté.

# 6. Les dix étapes réalisées

## Étape 1 — Choix du dataset et cadrage du projet

Le dataset HandGesture-5G a été retenu pour la reconnaissance de cinq gestes de la main.

Le problème a été formulé comme une tâche de classification supervisée : à partir d'une image de main, le système doit prédire l'une des cinq classes définies dans le dataset.

L'approche étudiée repose sur la segmentation HSV, l'extraction des moments de Hu et la classification par k-NN.

**Résultat :** le problème, les classes et l'approche de traitement ont été définis.

## Étape 2 — Préparation de l'environnement et des outils

L'environnement Python a été préparé pour permettre la lecture, le traitement et l'analyse des images.

Une attention particulière a été portée aux fichiers HEIC, qui nécessitent une bibliothèque compatible pour leur décodage.

Les dossiers du projet ont été organisés afin de séparer les images brutes, les données intermédiaires, les caractéristiques finales, les résultats et les scripts.

**Résultat :** un environnement de travail permettant de traiter les différents formats d'images du dataset.

## Étape 3 — Analyse exploratoire et contrôle de l'intégrité

L'ensemble des 2 653 images a été examiné afin de vérifier leur exploitabilité.

Les contrôles ont porté sur :

- les effectifs des classes ;
- les formats et les dimensions ;
- la lecture et le décodage des fichiers ;
- les images vides, illisibles ou uniformes ;
- les doublons exacts ;
- la visualisation d'exemples de chaque classe.

Les 2 653 fichiers ont pu être décodés. Aucun fichier vide, illisible ou entièrement uniforme n'a été détecté lors des contrôles réalisés.

L'analyse des doublons a identifié 846 groupes de deux images strictement identiques. Aucun groupe détecté ne réunissait des images appartenant à des classes différentes.

**Résultat :** un dataset analysé, avec une connaissance des doublons à prendre en compte lors de la séparation des données.

## Étape 4 — Prétraitement des images

Les images ont été redimensionnées en conservant leurs proportions, avec une longueur maximale de 256 pixels. Aucun agrandissement n'a été appliqué aux images plus petites.

L'interpolation `INTER_AREA` a été utilisée pour la réduction des images.

Les images originales ont été conservées. Les versions prétraitées et les représentations HSV ont été enregistrées dans les dossiers intermédiaires.

Le traitement des 2 653 images s'est terminé sans échec.

**Résultat :** des images plus légères et adaptées aux opérations de segmentation et d'extraction des caractéristiques.

## Étape 5 — Segmentation HSV et extraction du contour principal

L'espace colorimétrique HSV a été utilisé pour construire un masque des régions susceptibles de correspondre à la peau.

Les seuils retenus sont :

- borne inférieure : `[0, 20, 40]` ;
- borne supérieure : `[25, 255, 255]`.

Les opérations morphologiques appliquées sont :

1. une fermeture avec un noyau elliptique de 5 × 5, pendant une itération ;
2. une ouverture avec un noyau elliptique de 3 × 3, pendant une itération.

Les contours externes sont ensuite détectés à l'aide d'OpenCV. Le contour ayant la plus grande aire est sélectionné comme contour principal.

Des essais sur des échantillons ont permis de comparer différentes configurations de segmentation et de morphologie. Ils ont également montré que certaines images peuvent présenter des fragments de peau détectés séparément, des parties de main omises ou des contours affectés par le cadrage.

**Résultat :** une méthode de segmentation et de sélection du contour principal définie pour l'extraction des caractéristiques.

## Étape 6 — Extraction et analyse des moments de Hu

Les sept moments de Hu sont calculés à partir du contour principal sélectionné.

Pour limiter les différences d'échelle entre les valeurs, une transformation logarithmique signée est appliquée :

\[
H_i'=-\operatorname{sign}(H_i)
\log_{10}\left(\max(|H_i|,10^{-30})\right)
\]

Cette transformation est appliquée aux sept moments.

L'extraction a d'abord été testée sur un échantillon d'images. La distribution des caractéristiques a ensuite été étudiée sur 150 images, soit 30 images par classe.

Cette analyse a montré que certains moments présentent des distributions différentes selon les classes, mais aussi des chevauchements. Les sept moments de Hu ne garantissent donc pas, à eux seuls, une séparation parfaite des gestes.

**Résultat :** une représentation numérique de chaque image par sept caractéristiques de forme, accompagnée d'une analyse préliminaire de leur distribution.

## Étape 7 — Construction et validation du dataset final

Le script `src/construire_dataset_final.py` a été exécuté sur l'ensemble des images du dataset.

Pour chaque image, il réalise la segmentation, les opérations morphologiques, la sélection du contour principal, le calcul des sept moments de Hu et leur transformation logarithmique signée.

Le fichier produit est :

`data/final/features.csv`

Il contient neuf colonnes :

- `image_id` ;
- `class` ;
- `Hu1`, `Hu2`, `Hu3`, `Hu4`, `Hu5`, `Hu6`, `Hu7`.

Le fichier comporte 2 653 lignes. Les contrôles réalisés ont confirmé l'absence de valeurs manquantes ou non finies, la présence des cinq classes et l'unicité des identifiants.

**Résultat :** un dataset de caractéristiques validé sur le plan structurel et numérique, prêt pour la classification.

## Étape 8 — Préparation des données et sélection des paramètres du k-NN

Cette étape a consisté à préparer les caractéristiques pour l'apprentissage automatique et à rechercher une configuration du classificateur k-NN.

### 8.1. Vérification du dataset

Le fichier `data/final/features.csv` a été contrôlé afin de vérifier :

- la présence des colonnes attendues ;
- la validité des sept caractéristiques de Hu ;
- l'absence de valeurs manquantes et non finies ;
- la cohérence des classes et des identifiants ;
- la correspondance entre les groupes de doublons et les caractéristiques.

Les 1 692 lignes associées aux doublons détectés ont pu être rapprochées des caractéristiques correspondantes.

### 8.2. Séparation des données

La séparation a été réalisée avec `StratifiedGroupKFold`, en cinq plis, avec mélange et `random_state=42`.

Les images appartenant à un même groupe de doublons exacts ont été maintenues ensemble afin d'éviter qu'un même contenu soit présent simultanément dans les ensembles d'entraînement et de test.

La répartition obtenue est la suivante :

| Ensemble | Nombre d'images | Proportion |
|---|---:|---:|
| Entraînement | 2 122 | 79,98 % |
| Test | 531 | 20,02 % |
| **Total** | **2 653** | **100 %** |

Aucun groupe de doublons n'est partagé entre les deux ensembles.

### 8.3. Standardisation

Les caractéristiques ont été standardisées avec `StandardScaler`.

Le scaler est ajusté uniquement sur les données d'entraînement. Les données de test sont transformées avec les paramètres appris sur l'entraînement, sans réajustement.

Cette opération est importante pour le k-NN, car le calcul des distances peut être influencé par les échelles des caractéristiques.

### 8.4. Recherche des paramètres

Plusieurs valeurs de \(k\) ont été évaluées, avec les pondérations `uniform` et `distance`. La distance euclidienne a été utilisée.

La recherche a été conduite sur l'ensemble d'entraînement avec une validation croisée groupée et stratifiée en cinq plis. Le critère de sélection était le F1-score macro.

La meilleure moyenne observée dans cette recherche était obtenue avec \(k=1\), avec environ 85,72 % de F1 macro en validation croisée.

Le groupe a néanmoins retenu la configuration suivante pour le modèle final :

- nombre de voisins : \(k=3\) ;
- pondération : `distance` ;
- métrique : `euclidean`.

Ce choix constitue une décision méthodologique du projet. Les résultats de validation croisée ne démontrent pas que \(k=3\) est supérieur à \(k=1\).

**Résultat :** un jeu d'entraînement et un jeu de test séparés sans chevauchement de groupes de doublons, ainsi qu'une configuration k-NN définie.

## Étape 9 — Entraînement et évaluation du modèle

### 9.1. Entraînement

Le modèle final a été construit sous la forme d'un pipeline scikit-learn composé de deux étapes :

1. `StandardScaler` ;
2. `KNeighborsClassifier`.

Le classificateur utilise trois voisins, une pondération par distance et une distance euclidienne.

L'entraînement a été effectué sur les 2 122 images de l'ensemble d'entraînement. Les 531 images de test n'ont pas été utilisées pour ajuster le modèle.

### 9.2. Prédictions sur le jeu de test

Le modèle a produit une prédiction pour chacune des 531 images du jeu de test.

Les résultats ont été enregistrés dans :

`results/predictions_knn.csv`

Ce fichier contient notamment l'identifiant de l'image, sa classe réelle et sa classe prédite.

### 9.3. Métriques globales

Les résultats obtenus sur le jeu de test sont les suivants :

| Métrique | Résultat |
|---|---:|
| Accuracy | 87,01 % |
| Précision macro | 87,25 % |
| Rappel macro | 86,89 % |
| F1-score macro | 86,99 % |
| F1-score pondéré | 86,99 % |
| Prédictions correctes | 462 / 531 |
| Prédictions incorrectes | 69 / 531 |

L'accuracy correspond à la proportion d'images correctement classées. Le F1-score macro calcule le F1-score de chaque classe puis en fait la moyenne sans pondération par les effectifs.

### 9.4. Résultats par classe

| Classe | Précision | Rappel | F1-score | Effectif test |
|---|---:|---:|---:|---:|
| Fist | 88,99 % | 89,81 % | 89,40 % | 108 |
| L-Sign | 80,70 % | 85,19 % | 82,88 % | 108 |
| Open Palm_Open Hand | 88,19 % | 84,85 % | 86,49 % | 132 |
| Pointing Up | 90,41 % | 81,48 % | 85,71 % | 81 |
| V-Sign | 87,96 % | 93,14 % | 90,48 % | 102 |

Le V-Sign présente le F1-score le plus élevé, tandis que le L-Sign présente le plus faible.

### 9.5. Matrice de confusion

La matrice de confusion permet de comparer les classes réelles aux classes prédites.

```text
Classe réelle            Fist  L-Sign  Open Palm  Pointing Up  V-Sign
Fist                       97       5          4            0       2
L-Sign                      3      92          9            2       2
Open Palm_Open Hand         7      13        112            0       0
Pointing Up                 2       2          2           66       9
V-Sign                      0       2          0            5      95
```

Les principales confusions observées concernent :

- L-Sign classé comme Open Palm : 9 images ;
- Open Palm classé comme L-Sign : 13 images ;
- Pointing Up classé comme V-Sign : 9 images ;
- V-Sign classé comme Pointing Up : 5 images.

Ces erreurs indiquent que certaines classes présentent des difficultés de distinction avec la représentation choisie.

Les résultats sont enregistrés dans :

- `results/metriques_knn.csv` ;
- `results/matrice_confusion_knn.csv` ;
- `results/predictions_knn.csv` ;
- `figures/matrice_confusion_knn.png`.

**Résultat :** un modèle entraîné et évalué sur un jeu de test distinct, avec des métriques globales, des résultats par classe et une matrice de confusion.

## Étape 10 — Analyse des résultats et sauvegarde du modèle

Cette étape a permis d'interpréter les performances, d'examiner certaines erreurs et de sauvegarder le pipeline final.

### 10.1. Analyse des erreurs

Les résultats montrent que le modèle reconnaît correctement une majorité des images du jeu de test, mais que certaines classes sont plus difficiles à distinguer.

Les principales confusions sont observées entre L-Sign et Open Palm, ainsi qu'entre Pointing Up et V-Sign.

Une inspection visuelle de six exemples d'erreurs de classification a été réalisée. Elle a permis d'observer les images originales, les masques segmentés et les contours sélectionnés.

Dans les exemples examinés, la silhouette de la main était généralement conservée. Certaines images présentaient toutefois un cadrage proche des bords ou quelques fragments de masque parasites.

Ces observations ne permettent pas d'affirmer que la segmentation est la cause unique des erreurs. D'autres facteurs peuvent intervenir, notamment les informations de forme conservées par les moments de Hu et les distances utilisées par le k-NN.

### 10.2. Limites identifiées

Les principales limites sont les suivantes :

- la segmentation HSV peut être sensible aux conditions d'éclairage et aux couleurs de l'arrière-plan ;
- la sélection du contour de plus grande aire peut échouer si la main n'est pas la région dominante ;
- les sept moments de Hu représentent la forme de manière compacte et peuvent ne pas préserver tous les détails locaux des doigts ;
- certaines classes ont des silhouettes qui peuvent se ressembler ;
- le choix de \(k=3\) est un choix méthodologique et n'a pas obtenu la meilleure moyenne lors de la recherche par validation croisée ;
- les performances mesurées sur le jeu de test ne garantissent pas les mêmes résultats sur des images prises dans des conditions entièrement nouvelles.

### 10.3. Sauvegarde du pipeline

Le pipeline final a été sauvegardé dans :

`models/knn_pipeline.pkl`

Cette sauvegarde contient le `StandardScaler` ajusté sur les données d'entraînement et le classificateur K-NN entraîné.

Les paramètres et les informations nécessaires à la réutilisation du modèle sont documentés dans :

`models/model_config.json`

Le fichier `README_model.md` décrit les conditions de chargement et d'utilisation du pipeline.

### 10.4. Vérification de la sauvegarde

Le pipeline sauvegardé a été rechargé avec Joblib, puis utilisé pour prédire les classes des 531 images de test.

Les nouvelles prédictions ont été comparées aux prédictions initiales :

- prédictions comparées : 531 ;
- prédictions identiques : 531 ;
- prédictions différentes : 0.

Cette vérification confirme que le pipeline sauvegardé conserve le comportement du modèle évalué.

Elle ne constitue pas une nouvelle évaluation de la généralisation du modèle.

### 10.5. Rapport récapitulatif

Un rapport de synthèse des résultats a été généré dans :

`results/evaluation_report.txt`

Il complète les fichiers CSV de métriques, de prédictions et de matrice de confusion.

**Résultat :** le modèle est sauvegardé, sa configuration est documentée et son rechargement a été vérifié.

# 7. Pipeline général du projet

Le pipeline global réalisé au cours des dix étapes est le suivant :

```text
Dataset HandGesture-5G
          |
          v
Analyse exploratoire et contrôle
          |
          v
Prétraitement et redimensionnement
          |
          v
Conversion en HSV
          |
          v
Segmentation de la peau
          |
          v
Morphologie et contour principal
          |
          v
Calcul des sept moments de Hu
          |
          v
Transformation logarithmique signée
          |
          v
Construction et validation de features.csv
          |
          v
Séparation groupée entraînement / test
          |
          v
Standardisation des caractéristiques
          |
          v
Recherche des paramètres du k-NN
          |
          v
Entraînement du pipeline final
          |
          v
Prédiction sur le jeu de test
          |
          v
Évaluation et analyse des erreurs
          |
          v
Sauvegarde et vérification du pipeline
```

# 8. Scripts principaux

Les scripts ci-dessous correspondent aux principales opérations du projet. Certains scripts servent à tester une méthode sur un échantillon, tandis que d'autres traitent l'ensemble des données.

| Script | Rôle |
|---|---|
| `src/construire_dataset_final.py` | Prétraitement, segmentation et construction des caractéristiques finales |
| `src/valider_dataset_final.py` | Validation du dataset final |
| `src/inspecter_dataset_knn.py` | Inspection des caractéristiques destinées au k-NN |
| `src/verifier_correspondance_doublons.py` | Vérification de la correspondance entre doublons et caractéristiques |
| `src/preparer_split_knn.py` | Préparation de la séparation entraînement/test |
| `src/standardiser_features_knn.py` | Préparation des caractéristiques standardisées |
| `src/rechercher_meilleur_knn.py` | Recherche des paramètres du classificateur |
| `src/preparer_evaluation_knn.py` | Vérifications préalables à l'évaluation |
| `src/entrainer_knn.py` | Entraînement et sauvegarde du pipeline |
| `src/predire_test_knn.py` | Prédictions sur le jeu de test |
| `src/evaluer_metriques_knn.py` | Calcul des métriques de classification |
| `src/matrice_confusion_knn.py` | Génération de la matrice de confusion |
| `src/inspecter_segmentation_erreurs.py` | Inspection visuelle d'exemples mal classés |
| `src/verifier_sauvegarde_modele.py` | Vérification des prédictions après rechargement |
| `src/generer_rapport_evaluation.py` | Génération du rapport récapitulatif |

La liste n'est pas exhaustive : d'autres scripts ont servi à l'exploration, aux essais de segmentation, aux tests morphologiques et à l'analyse des moments de Hu.

# 9. Reproduire les principales opérations

Les commandes suivantes doivent être exécutées depuis la racine du projet, dans un environnement Python où les dépendances sont installées.

## 9.1. Construire et valider les caractéristiques

```powershell
python .\src\construire_dataset_final.py
python .\src\valider_dataset_final.py
```

Les images HSV nécessaires doivent être disponibles avant la construction du dataset final.

## 9.2. Préparer les données et rechercher les paramètres

```powershell
python .\src\inspecter_dataset_knn.py
python .\src\verifier_correspondance_doublons.py
python .\src\preparer_split_knn.py
python .\src\standardiser_features_knn.py
python .\src\rechercher_meilleur_knn.py
```

Ces opérations permettent de vérifier les caractéristiques, de préparer la séparation des données et d'effectuer la recherche des paramètres.

## 9.3. Entraîner et évaluer le modèle

```powershell
python .\src\preparer_evaluation_knn.py
python .\src\entrainer_knn.py
python .\src\predire_test_knn.py
python .\src\evaluer_metriques_knn.py
python .\src\matrice_confusion_knn.py
```

Ces scripts génèrent les prédictions, les métriques et la matrice de confusion.

## 9.4. Analyser les erreurs et vérifier la sauvegarde

```powershell
python .\src\inspecter_segmentation_erreurs.py
python .\src\verifier_sauvegarde_modele.py
python .\src\generer_rapport_evaluation.py
```

Ces commandes permettent d'inspecter des exemples d'erreurs, de vérifier le pipeline sauvegardé et de générer le rapport récapitulatif.

**Important :** pour reproduire exactement les résultats, il faut conserver la cohérence entre les fichiers de caractéristiques, le split, les prédictions et les métriques. Une réexécution dans un ordre différent peut régénérer certains fichiers intermédiaires.

# 10. Résultats principaux

Le modèle final retenu est un pipeline `StandardScaler + KNeighborsClassifier`, avec \(k=3\), une pondération par distance et une distance euclidienne.

Les résultats obtenus sur le jeu de test sont :

- 2 122 images pour l'entraînement ;
- 531 images pour le test ;
- 87,01 % d'accuracy ;
- 86,99 % de F1-score macro ;
- 462 prédictions correctes ;
- 69 prédictions incorrectes.

La classe V-Sign présente le meilleur F1-score, à 90,48 %. La classe L-Sign présente le plus faible, à 82,88 %.

Les principales confusions concernent L-Sign et Open Palm, ainsi que Pointing Up et V-Sign.

Ces résultats sont ceux du protocole expérimental réalisé sur le dataset HandGesture-5G. Ils ne garantissent pas les mêmes performances sur toutes les images nouvelles.

# 11. Limites et améliorations possibles

Plusieurs pistes pourraient améliorer le système :

1. améliorer la segmentation pour réduire les effets des variations d'éclairage et d'arrière-plan ;
2. examiner davantage d'images mal classées ;
3. comparer les moments de Hu avec des caractéristiques géométriques complémentaires ;
4. comparer plusieurs classificateurs en conservant un protocole d'évaluation identique ;
5. tester le système sur des images réellement nouvelles, provenant de conditions différentes ;
6. développer une interface permettant de charger une image et d'afficher la prédiction ;
7. documenter les erreurs et les limites observées pendant la démonstration.

La prochaine étape du travail de groupe consiste à réutiliser le pipeline de traitement et le modèle sauvegardé pour construire une démonstration capable de prédire le geste à partir d'une nouvelle image.

# 12. Livrables produits

Les principaux livrables des dix étapes sont :

| Livrable | Fichier ou dossier |
|---|---|
| Dataset de caractéristiques | `data/final/features.csv` |
| Séparation entraînement/test | `results/split_train_test.csv` |
| Prédictions sur le jeu de test | `results/predictions_knn.csv` |
| Métriques de classification | `results/metriques_knn.csv` |
| Matrice de confusion | `results/matrice_confusion_knn.csv` |
| Visualisation de la matrice de confusion | `figures/matrice_confusion_knn.png` |
| Inspection d'exemples d'erreurs | `figures/inspection_segmentation_erreurs.png` |
| Pipeline entraîné | `models/knn_pipeline.pkl` |
| Configuration du modèle | `models/model_config.json` |
| Documentation du modèle | `README_model.md` |
| Rapport récapitulatif | `results/evaluation_report.txt` |
| Vérification de la sauvegarde | `src/verifier_sauvegarde_modele.py` |

Les fichiers de données brutes et les données intermédiaires volumineuses peuvent rester exclus du dépôt Git. Leur disponibilité est toutefois nécessaire pour reproduire certaines étapes depuis les images originales.

# 13. Conclusion

Au terme des dix étapes, un système expérimental de reconnaissance de cinq gestes de la main a été construit à partir du dataset HandGesture-5G.

La démarche comprend l'analyse et la validation des images, leur prétraitement, la segmentation HSV, l'extraction des moments de Hu, la construction du dataset de caractéristiques, la préparation des données, l'entraînement du k-NN, l'évaluation et l'analyse des erreurs.

Le pipeline final atteint une accuracy de 87,01 % et un F1-score macro de 86,99 % sur les 531 images du jeu de test. Sa sauvegarde et son rechargement ont également été vérifiés, avec des prédictions identiques sur les 531 images.

Ces résultats montrent que la méthode retenue est capable de reconnaître une majorité des gestes dans les conditions de l'expérience. Des erreurs subsistent néanmoins, notamment entre certaines classes dont les formes se ressemblent.

Le modèle constitue donc une base pour la prochaine phase du projet : la création d'une démonstration fonctionnelle permettant de charger une nouvelle image, d'appliquer le traitement prévu et d'afficher la classe prédite.
ans toutes les images ;
- les moments de Hu ne garantissent pas une séparation complète des cinq classes ;
- les doublons exacts peuvent fausser l'évaluation s'ils sont répartis entre l'entraînement et le test ;
- la validation structurelle de `features.csv` ne constitue pas une mesure de la performance de reconnaissance.

## 11. Étapes suivantes

La suite du projet consiste à :

1. préparer une séparation rigoureuse des données d'entraînement et de test ;
2. éviter les fuites de données dues aux doublons exacts ;
3. standardiser les caractéristiques en ajustant le scaler uniquement sur l'ensemble d'entraînement ;
4. entraîner un classificateur k-NN ;
5. comparer plusieurs valeurs de \(k\), par exemple 1, 3, 5, 7 et 9 ;
6. mesurer l'accuracy, la précision, le rappel et le F1-score ;
7. analyser la matrice de confusion et les erreurs de classification ;
8. documenter les résultats et les limites du système.

Les performances finales ne sont pas encore établies par les étapes décrites dans ce README.

## 12. Livrable principal

Le livrable produit par les sept premières étapes est :

`data/final/features.csv`

Il contient une ligne par image traitée avec succès, sa classe et ses sept caractéristiques numériques. Il sert d'entrée au travail de classification réalisé dans la suite du projet.

