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
- le contour de plus grande aire n'est pas nécessairement la main entière dans toutes les images ;
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

