# Documentation du modèle — HandGesture-5G

## 1. Présentation

Ce projet utilise un modèle de classification K-NN (*K-Nearest Neighbors*) pour reconnaître cinq catégories de gestes de la main à partir d'images.

Le modèle exploite les sept moments invariants de Hu calculés à partir du plus grand contour externe détecté après segmentation de la peau dans l'espace colorimétrique HSV.

## 2. Classes reconnues

Le modèle peut prédire l'une des cinq classes suivantes :

- `Fist`
- `L-Sign`
- `Open Palm_Open Hand`
- `Pointing Up`
- `V-Sign`

## 3. Fichiers du modèle

- `models/knn_pipeline.pkl` : pipeline entraîné comprenant la standardisation et le classifieur K-NN.
- `models/model_config.json` : paramètres du modèle, caractéristiques attendues et configuration du prétraitement.
- `results/metriques_knn.csv` : métriques d'évaluation.
- `results/matrice_confusion_knn.csv` : matrice de confusion au format CSV.
- `figures/matrice_confusion_knn.png` : visualisation de la matrice de confusion.
- `results/predictions_knn.csv` : prédictions produites sur le jeu de test.

## 4. Architecture du pipeline

Le traitement des caractéristiques suit cet ordre :

1. Redimensionnement de l'image, avec un côté maximal de 256 pixels sans agrandissement.
2. Conversion de l'image dans l'espace HSV.
3. Segmentation à l'aide des bornes HSV définies dans `model_config.json`.
4. Fermeture morphologique avec un noyau elliptique de 5 × 5.
5. Ouverture morphologique avec un noyau elliptique de 3 × 3.
6. Sélection du plus grand contour externe.
7. Calcul des sept moments de Hu.
8. Application de la transformation logarithmique signée.
9. Standardisation par le `StandardScaler` appris sur les données d'entraînement.
10. Classification par K-NN avec trois voisins, pondération par distance et distance euclidienne.

L'ordre des caractéristiques attendu par le modèle est :

`Hu1, Hu2, Hu3, Hu4, Hu5, Hu6, Hu7`

## 5. Charger le modèle

L'environnement Python doit disposer des bibliothèques nécessaires, notamment `joblib`, `numpy`, `pandas`, `opencv-python` et `scikit-learn`, avec des versions compatibles avec l'environnement d'entraînement.

Exemple de chargement :

```python
from pathlib import Path
import joblib

RACINE_PROJET = Path(__file__).resolve().parent

FICHIER_MODELE = RACINE_PROJET / "models" / "knn_pipeline.pkl"

modele = joblib.load(FICHIER_MODELE)

print("Modèle chargé avec succès.")
print("Classes :", modele.named_steps["knn"].classes_)
```

Cet exemple suppose que le script se trouve à la racine du projet. Si le code est placé dans un autre dossier, le chemin devra être adapté.

**Sécurité :** ne charger un fichier `.pkl` qu'en provenance d'une source fiable.

## 6. Effectuer une prédiction

Le modèle attend sept caractéristiques numériques déjà calculées et présentées dans le même ordre que pendant l'entraînement.

Exemple avec une ligne de caractéristiques existante :

```python
import pandas as pd

colonnes_hu = [
    "Hu1", "Hu2", "Hu3", "Hu4",
    "Hu5", "Hu6", "Hu7"
]

exemple = pd.DataFrame(
    [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
    columns=colonnes_hu
)

prediction = modele.predict(exemple)

print("Classe prédite :", prediction[0])
```

Les valeurs de cet exemple sont fictives : elles servent uniquement à illustrer l'appel à `predict()` et ne représentent pas les caractéristiques d'une véritable image.

Pour reconnaître un nouveau geste à partir d'une image, il faut d'abord appliquer exactement le prétraitement du projet et calculer ses sept moments de Hu transformés. Le modèle sauvegardé ne réalise pas automatiquement ces opérations sur une image brute.

## 7. Résultats de l'évaluation

Le modèle a été évalué sur 531 images de test, séparées des données d'entraînement en tenant compte des groupes de doublons.

Résultats obtenus :

- Exactitude (*accuracy*) : 87,01 %.
- Précision macro : 87,25 %.
- Rappel macro : 86,89 %.
- F1 macro : 86,99 %.
- Prédictions correctes : 462 sur 531.
- Prédictions incorrectes : 69 sur 531.

Ces résultats concernent le jeu de test utilisé pendant le projet. Ils ne garantissent pas les mêmes performances sur toutes les nouvelles images ou dans toutes les conditions de prise de vue.

## 8. Reproductibilité

Pour réutiliser correctement le modèle :

- conserver l'ordre des sept caractéristiques ;
- appliquer la même transformation logarithmique signée aux moments de Hu ;
- respecter la configuration de segmentation et les opérations morphologiques ;
- utiliser le `StandardScaler` déjà sauvegardé dans le pipeline ;
- éviter de réentraîner ou de réajuster le scaler sur les données de test.

La configuration détaillée est disponible dans `models/model_config.json`.

## 9. Limites

Les performances dépendent notamment de la qualité de la segmentation, de la visibilité de la main, du cadrage et de la capacité des moments de Hu à distinguer les gestes proches.

Une évaluation sur des images externes et des conditions de prise de vue variées sera nécessaire avant de considérer le système comme généralisable à tous les utilisateurs.
