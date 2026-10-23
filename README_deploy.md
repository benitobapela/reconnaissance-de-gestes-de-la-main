# Reconnaissance de gestes de la main par vision par ordinateur et apprentissage automatique

## 1. Présentation du projet

Ce projet porte sur la reconnaissance automatique de gestes de la main à partir d'une image.

L'objectif est de développer une chaîne complète de traitement permettant de prendre une image d'une main, d'en extraire les caractéristiques pertinentes et de déterminer automatiquement le geste réalisé.

Le système utilise des techniques de :

- traitement d'image ;
- segmentation de la main ;
- extraction de contour ;
- calcul des moments de Hu ;
- normalisation des caractéristiques ;
- apprentissage automatique ;
- classification avec l'algorithme K-NN ;
- évaluation du modèle ;
- déploiement sous forme d'une application Streamlit.

Le projet est réalisé dans le cadre des travaux pratiques en intelligence artificielle et en science des données.

---

## 2. Objectif

L'objectif principal est de construire un système capable de reconnaître cinq types de gestes de la main :

1. `Fist`
2. `L-Sign`
3. `Open Palm_Open Hand`
4. `Pointing Up`
5. `V-Sign`

À partir d'une nouvelle image, le système doit :

1. charger l'image ;
2. la prétraiter ;
3. détecter la région correspondant à la main ;
4. segmenter la main ;
5. extraire son contour principal ;
6. calculer ses caractéristiques géométriques ;
7. transformer les moments de Hu ;
8. standardiser les caractéristiques ;
9. appliquer le modèle K-NN ;
10. produire la classe prédite.

---

## 3. Architecture générale

Le fonctionnement global du système peut être résumé comme suit :

```text
Image de la main
       │
       ▼
Prétraitement de l'image
       │
       ▼
Redimensionnement
       │
       ▼
Conversion BGR → HSV
       │
       ▼
Segmentation de la peau
       │
       ▼
Extraction du contour principal
       │
       ▼
Calcul des moments de Hu
       │
       ▼
Transformation logarithmique
       │
       ▼
Standardisation des caractéristiques
       │
       ▼
Classification K-NN
       │
       ▼
Geste reconnu
```

---

## 4. Classes du dataset

Le système travaille avec cinq catégories de gestes :

| Classe | Description |
|---|---|
| Fist | Poing fermé |
| L-Sign | Geste formant la lettre L |
| Open Palm_Open Hand | Main ouverte |
| Pointing Up | Index dirigé vers le haut |
| V-Sign | Geste en V |

---

## 5. Prétraitement des images

Avant l'extraction des caractéristiques, les images sont préparées afin d'obtenir des données plus homogènes.

Le prétraitement comprend notamment :

- lecture des images ;
- prise en charge de différents formats ;
- gestion des informations EXIF ;
- redimensionnement des images ;
- limitation de la taille maximale des images ;
- préparation des images pour OpenCV.

Le redimensionnement permet de réduire le coût de calcul tout en conservant les informations importantes nécessaires à la reconnaissance du geste.

Le traitement est principalement implémenté dans :

```text
src/preprocess_dataset.py
```

---

## 6. Segmentation de la main

La segmentation permet de séparer la main du reste de l'image.

Pour cela, l'image est convertie de l'espace colorimétrique BGR vers HSV.

L'espace HSV permet de travailler séparément sur :

- la teinte ;
- la saturation ;
- la luminosité.

Une plage de valeurs HSV est ensuite utilisée afin de construire un masque correspondant à la région de peau.

Le masque obtenu permet de distinguer la zone de la main de l'arrière-plan.

Des opérations morphologiques peuvent également être utilisées afin de réduire le bruit et améliorer la qualité du masque.

---

## 7. Extraction du contour principal

Après segmentation, plusieurs régions peuvent éventuellement être présentes dans l'image.

Le système recherche alors le contour principal correspondant à la région la plus pertinente.

L'objectif est de conserver le contour de la main afin de pouvoir analyser sa forme indépendamment de l'image originale.

Cette étape est réalisée dans :

```text
src/construire_dataset_final.py
```

---

## 8. Extraction des moments de Hu

Les moments de Hu sont utilisés pour représenter la forme de la main sous forme de caractéristiques numériques.

Le système calcule les sept moments de Hu :

```text
Hu1
Hu2
Hu3
Hu4
Hu5
Hu6
Hu7
```

Ces caractéristiques permettent de décrire la forme du contour.

Afin de rendre les valeurs plus adaptées à l'apprentissage automatique, une transformation logarithmique est appliquée aux moments de Hu.

Les sept caractéristiques obtenues constituent donc le vecteur utilisé pour représenter chaque image :

```text
[Hu1, Hu2, Hu3, Hu4, Hu5, Hu6, Hu7]
```

---

## 9. Standardisation des caractéristiques

Les caractéristiques extraites peuvent avoir des échelles différentes.

Une standardisation est donc appliquée avant la classification.

Le principe consiste à centrer les caractéristiques autour de leur moyenne et à les ramener à une échelle comparable.

Cette étape permet au modèle K-NN de calculer les distances entre les observations de manière plus cohérente.

---

## 10. Classification avec K-NN

Le modèle utilisé pour la reconnaissance des gestes est l'algorithme :

**K-Nearest Neighbors (K-NN)**.

Le principe du K-NN consiste à rechercher les observations d'apprentissage les plus proches d'une nouvelle observation.

Dans ce projet, le modèle utilise :

```text
K = 3
```

Pour une nouvelle image :

1. les caractéristiques de l'image sont extraites ;
2. elles sont standardisées ;
3. le modèle recherche les trois observations les plus proches ;
4. les classes de ces voisins sont comparées ;
5. la classe majoritaire devient la prédiction.

---

## 11. Pipeline du modèle

Le modèle final est sauvegardé sous forme d'un pipeline :

```text
models/knn_pipeline.pkl
```

Le pipeline permet de conserver les étapes nécessaires à l'utilisation du modèle.

La configuration du modèle est également conservée dans :

```text
models/model_config.json
```

---

## 12. Entraînement du modèle

L'entraînement du modèle est réalisé à partir des caractéristiques extraites des images.

Le script principal d'entraînement est :

```text
src/entrainer_knn.py
```

D'autres scripts permettent d'étudier différentes configurations du modèle et de rechercher les paramètres les plus adaptés.

Parmi les scripts utilisés :

```text
src/rechercher_meilleur_knn.py
src/standardiser_features_knn.py
src/preparer_split_knn.py
src/preparer_evaluation_knn.py
```

---

## 13. Évaluation

Plusieurs scripts ont été développés afin d'analyser les performances du modèle.

Ils permettent notamment de produire :

- les métriques de classification ;
- la matrice de confusion ;
- l'analyse des erreurs ;
- l'analyse de la distribution des caractéristiques ;
- la vérification de la qualité du dataset.

Scripts concernés :

```text
src/evaluer_metriques_knn.py
src/matrice_confusion_knn.py
src/generer_rapport_evaluation.py
src/inspecter_segmentation_erreurs.py
src/analyser_distribution_hu.py
src/valider_distribution_hu.py
```

---

## 14. Vérification et contrôle du dataset

Plusieurs scripts ont été développés pour contrôler la qualité des données avant l'entraînement.

Ils permettent notamment de vérifier :

- l'intégrité des images ;
- les doublons ;
- les formats d'images ;
- les canaux des images ;
- la correspondance entre certaines données ;
- la distribution du dataset.

Scripts utilisés :

```text
src/check_dataset_integrity.py
src/check_duplicates.py
src/analyze_duplicates.py
src/verifier_correspondance_doublons.py
src/check_image_channels.py
src/check_heic.py
src/dataset_check.py
src/valider_dataset_final.py
```

---

## 15. Tests de segmentation et de traitement d'image

Différents tests ont été réalisés afin de vérifier le fonctionnement des différentes étapes du traitement d'image.

Par exemple :

```text
src/test_hsv_feasibility.py
src/test_segmentation_hsv_initiale.py
src/test_morphologie_hsv.py
src/test_contour_principal.py
src/test_moments_hu.py
```

Ces tests ont permis de vérifier progressivement :

1. la faisabilité de la segmentation HSV ;
2. la qualité du masque ;
3. l'amélioration du masque par morphologie ;
4. l'extraction du contour ;
5. le calcul des moments de Hu.

---

## 16. Inférence sur une nouvelle image

Une fois le modèle entraîné, le script :

```text
src/inference.py
```

permet d'effectuer une prédiction sur une nouvelle image.

Le fonctionnement est :

```text
Nouvelle image
      ↓
Prétraitement
      ↓
Segmentation
      ↓
Contour
      ↓
7 moments de Hu
      ↓
Standardisation
      ↓
K-NN
      ↓
Classe prédite
```

Le script produit également un masque de segmentation dans :

```text
results/masque_test.png
```

---

## 17. Application Streamlit

Une interface utilisateur a ensuite été développée avec Streamlit.

Le fichier principal de l'application est :

```text
src/app.py
```

L'utilisateur peut importer une image de main et obtenir automatiquement la prédiction du geste.

L'application permet de visualiser le processus de reconnaissance et le résultat final.

L'application peut être lancée localement avec :

```bash
python -m streamlit run src/app.py
```

Elle est ensuite accessible localement à l'adresse :

```text
http://localhost:8501
```

---

## 18. Dépendances principales

Les principales bibliothèques utilisées dans le projet sont :

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Pillow
- Pillow-HEIF
- Streamlit

Les dépendances sont regroupées dans :

```text
requirements.txt
```

---

## 19. Organisation du projet

```text
reconnaissance-de-gestes-de-la-main/
│
├── data/
│   ├── demo/
│   └── ...
│
├── models/
│   ├── knn_pipeline.pkl
│   └── model_config.json
│
├── results/
│   └── ...
│
├── src/
│   ├── analyser_distribution_hu.py
│   ├── analyze_duplicates.py
│   ├── check_dataset_integrity.py
│   ├── check_duplicates.py
│   ├── check_heic.py
│   ├── check_image_channels.py
│   ├── compare_working_sizes.py
│   ├── construire_dataset_final.py
│   ├── dataset_check.py
│   ├── entrainer_knn.py
│   ├── evaluer_metriques_knn.py
│   ├── explore_dataset.py
│   ├── generer_rapport_evaluation.py
│   ├── inference.py
│   ├── inspecter_dataset_knn.py
│   ├── inspecter_segmentation_erreurs.py
│   ├── matrice_confusion_knn.py
│   ├── predire_test_knn.py
│   ├── preprocess_dataset.py
│   ├── rechercher_meilleur_knn.py
│   ├── standardiser_features_knn.py
│   ├── test_contour_principal.py
│   ├── test_hsv_feasibility.py
│   ├── test_moments_hu.py
│   ├── test_morphologie_hsv.py
│   ├── test_segmentation_hsv_initiale.py
│   ├── valider_dataset_final.py
│   ├── valider_distribution_hu.py
│   ├── verifier_correspondance_doublons.py
│   ├── verifier_sauvegarde_modele.py
│   ├── visualize_samples.py
│   └── app.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 20. Résumé de la démarche expérimentale

Le développement du projet a été réalisé progressivement.

### Étape 1 — Exploration des données

Analyse du dataset et vérification des différentes classes de gestes.

### Étape 2 — Nettoyage et contrôle

Recherche des images problématiques, doublons, formats particuliers et incohérences.

### Étape 3 — Prétraitement

Lecture, gestion des images et redimensionnement.

### Étape 4 — Segmentation

Utilisation de l'espace HSV pour isoler la région de peau.

### Étape 5 — Extraction du contour

Identification du contour principal de la main.

### Étape 6 — Extraction des caractéristiques

Calcul des sept moments de Hu et transformation de leurs valeurs.

### Étape 7 — Standardisation

Mise à l'échelle des caractéristiques avant l'apprentissage.

### Étape 8 — Apprentissage

Entraînement d'un classifieur K-NN.

### Étape 9 — Évaluation

Analyse des performances à l'aide de métriques et d'une matrice de confusion.

### Étape 10 — Inférence

Test du modèle sur de nouvelles images.

### Étape 11 — Application

Création d'une interface Streamlit permettant d'utiliser le modèle de manière interactive.

---

## 21. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Langage principal |
| OpenCV | Traitement d'image |
| NumPy | Calcul numérique |
| Pandas | Manipulation des données |
| Scikit-learn | Machine Learning |
| Joblib | Sauvegarde du modèle |
| Pillow | Lecture et traitement des images |
| Pillow-HEIF | Gestion des images HEIC |
| Streamlit | Interface utilisateur |
| Git | Gestion de versions |
| GitLab / GitHub | Hébergement du code |

---

## 22. Conclusion

Ce projet met en œuvre une chaîne complète de vision par ordinateur et d'apprentissage automatique pour reconnaître des gestes de la main.

L'approche repose sur l'extraction de caractéristiques géométriques à partir du contour de la main, notamment les sept moments de Hu, puis sur leur classification à l'aide de K-NN.

Le projet ne se limite donc pas à l'entraînement d'un modèle : il comprend également la préparation des données, le traitement des images, la segmentation, l'extraction des caractéristiques, l'évaluation, l'inférence et la création d'une interface permettant d'utiliser le modèle.

Cette organisation permet également de disposer d'une base technique pour la rédaction du rapport et la présentation orale du projet.