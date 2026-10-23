
# Étape 2 — Acquisition et organisation du dataset

## 1. Objectif

Cette étape consiste à sélectionner, récupérer et organiser le jeu de données qui sera utilisé pour développer et évaluer notre système de reconnaissance des gestes de la main.

Le dataset doit contenir des images correspondant aux cinq classes de gestes retenues pour le projet. Cette étape permet également de vérifier la structure des dossiers, le nombre d'images par classe et les formats des fichiers disponibles.

## 2. Dataset sélectionné

**Nom :** HandGesture-5G

**Source :** Mendeley Data

**Nombre total d'images attendu :** 2 653

**Nombre de classes :** 5

Le dataset a été retenu comme base de travail pour notre système de reconnaissance des gestes de la main.

Il contient des images réparties entre les cinq catégories suivantes :

| Classe | Nombre d'images |
|---|---:|
| Open Palm_Open Hand | 661 |
| Fist | 542 |
| Pointing Up | 405 |
| L-Sign | 537 |
| V-Sign | 508 |
| **Total** | **2 653** |

Les effectifs ci-dessus ont été comparés aux fichiers présents dans les dossiers du dataset. Ils devront être vérifiés de nouveau si le contenu des données est modifié.

## 3. Organisation des données

Les images originales sont conservées dans le répertoire `data/raw/`.

Chaque classe possède son propre sous-dossier afin de faciliter l'identification des étiquettes associées aux images.

Structure utilisée :

```text
data/
├── raw/
│   ├── Open Palm_Open Hand/
│   ├── Fist/
│   ├── Pointing Up/
│   ├── L-Sign/
│   └── V-Sign/
├── processed/
└── final/
```

### Rôle des répertoires

- `data/raw/` : contient les images originales du dataset.
- `data/processed/` : accueillera les données après les opérations de prétraitement.
- `data/final/` : accueillera les données préparées pour les étapes ultérieures du projet, selon les besoins du pipeline.

Les images originales doivent être conservées afin de pouvoir reprendre le traitement sans perdre les données de départ.

## 4. Formats des fichiers

L'inspection des extensions a révélé la répartition suivante :

| Format | Nombre de fichiers |
|---|---:|
| JPG | 1 758 |
| JPEG | 885 |
| HEIC | 10 |
| **Total** | **2 653** |

Les fichiers JPG et JPEG sont des formats courants d'images. Les dix fichiers HEIC nécessitent une prise en charge spécifique, car ils ne sont pas directement lus par défaut par la fonction `cv2.imread()` d'OpenCV dans notre environnement.

## 5. Vérifications réalisées

Un script Python de vérification a été utilisé pour contrôler le contenu des dossiers et la lisibilité des images.

Les vérifications réalisées sont les suivantes :

- comptage des images dans chaque classe ;
- vérification de la présence des cinq dossiers ;
- identification des extensions de fichiers ;
- contrôle de la lecture des images JPG et JPEG avec OpenCV ;
- vérification spécifique des dix fichiers HEIC à l'aide de Pillow et de pillow-heif.

### Résultats des contrôles

| Vérification | Résultat |
|---|---|
| Nombre total de fichiers dans les classes | 2 653 |
| Nombre d'images JPG/JPEG reconnues par OpenCV | 2 643 |
| Images JPG/JPEG illisibles détectées par OpenCV | 0 |
| Fichiers HEIC à traiter avec un lecteur adapté | 10 |
| Fichiers HEIC vérifiés avec Pillow et pillow-heif | 10 valides |

Les dix fichiers HEIC ont été ouverts avec succès à l'aide du lecteur adapté. Ils ne doivent donc pas être considérés comme des fichiers corrompus simplement parce qu'OpenCV ne les lit pas directement dans cette configuration.

## 6. Environnement et dépendances

Le projet utilise Python pour le traitement des images.

Les principales bibliothèques mobilisées pour la gestion et la vérification des images sont :

- OpenCV (`opencv-python`) : lecture et traitement des images ;
- Pillow (`pillow`) : ouverture des images avec un lecteur adapté ;
- pillow-heif : prise en charge des fichiers HEIC.

D'autres bibliothèques seront utilisées lors des prochaines étapes, notamment pour la manipulation des données, la visualisation et l'apprentissage automatique.

## 7. Précautions de gestion des données

Les précautions suivantes sont retenues pour la suite du projet :

1. Conserver les images originales dans `data/raw/`.
2. Ne pas modifier les fichiers originaux directement.
3. Préserver les noms des classes afin de maintenir la correspondance entre les images et leurs étiquettes.
4. Prendre en charge explicitement les formats d'images rencontrés.
5. Ne pas considérer qu'une image est corrompue uniquement parce qu'une bibliothèque ne prend pas en charge son format.
6. Vérifier les éventuels doublons et la qualité visuelle du dataset avant l'entraînement.
7. Éviter qu'une même image, ou une copie de celle-ci, soit présente à la fois dans les données d'entraînement et de test.

## 8. Résultat de l'étape

À la fin de cette étape, le dataset est téléchargé, organisé par classe et soumis à des contrôles initiaux de comptage et de lisibilité.

Les cinq classes attendues sont présentes et les effectifs correspondent aux nombres de référence. Les fichiers JPG/JPEG sont lisibles par OpenCV et les fichiers HEIC ont été validés avec un lecteur adapté.

Le dataset est ainsi prêt pour l'analyse exploratoire, qui constituera l'étape suivante du projet.

## 9. Étape suivante

L'étape 3 portera sur l'analyse exploratoire du dataset. Elle consistera notamment à :

- étudier les dimensions des images ;
- examiner la répartition des classes ;
- visualiser des exemples représentatifs de chaque classe ;
- observer les variations de couleur, de luminosité, d'arrière-plan et de position des mains ;
- déterminer les difficultés potentielles pour la segmentation de la peau dans l'espace HSV.

Les observations de cette étape permettront de préparer les décisions de prétraitement et de segmentation.
