
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# --------------------------------------------------
# 1. Définir les chemins
# --------------------------------------------------

RACINE_PROJET = Path(__file__).resolve().parent.parent

FICHIER_FEATURES = RACINE_PROJET / "data" / "final" / "features.csv"
FICHIER_SPLIT = RACINE_PROJET / "results" / "split_train_test.csv"

DOSSIER_MODELES = RACINE_PROJET / "models"
FICHIER_MODELE = DOSSIER_MODELES / "knn_pipeline.pkl"

# --------------------------------------------------
# 2. Charger les données
# --------------------------------------------------

features = pd.read_csv(FICHIER_FEATURES)
split = pd.read_csv(FICHIER_SPLIT)

# Récupérer uniquement les identifiants des images d'entraînement
ids_train = split.loc[
    split["split"] == "train", "image_id"
]

# Associer les caractéristiques aux identifiants
donnees_train = features[
    features["image_id"].isin(ids_train)
].copy()

# Définir les caractéristiques et les étiquettes
colonnes_hu = [f"Hu{i}" for i in range(1, 8)]

X_train = donnees_train[colonnes_hu]
y_train = donnees_train["class"]


# --------------------------------------------------
# 3. Construire le modèle K-NN
# --------------------------------------------------

modele_knn = Pipeline([
    ("standardisation", StandardScaler()),
    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=3,
            weights="distance",
            metric="euclidean",
        ),
    ),
])


# --------------------------------------------------
# 4. Entraîner le modèle
# --------------------------------------------------

print("=" * 55)
print("ÉTAPE 9.2 — ENTRAÎNEMENT DU MODÈLE K-NN")
print("=" * 55)

print(f"\nNombre d'images d'entraînement : {len(X_train)}")
print(f"Nombre de caractéristiques     : {X_train.shape[1]}")
print(f"Nombre de classes              : {y_train.nunique()}")

print("\nParamètres retenus :")
print("Algorithme       : K-NN")
print("Nombre de voisins: 3")
print("Pondération      : distance")
print("Distance         : euclidienne")
print("Standardisation  : StandardScaler")

# Le scaler et le modèle sont ajustés sur les données train seulement
modele_knn.fit(X_train, y_train)

# Créer le dossier de sauvegarde s'il n'existe pas
DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)

# Sauvegarder le pipeline complet : StandardScaler + K-NN
joblib.dump(modele_knn, FICHIER_MODELE)

print(f"\nPipeline sauvegardé dans : {FICHIER_MODELE}")


# --------------------------------------------------
# 5. Vérifier l'entraînement
# --------------------------------------------------

knn_entraine = modele_knn.named_steps["knn"]

print("\n" + "-" * 55)
print("BILAN DE L'ENTRAÎNEMENT")
print("-" * 55)

print("Entraînement terminé avec succès.")
print(f"Nombre d'images apprises : {knn_entraine.n_samples_fit_}")
print(f"Classes apprises         : {list(knn_entraine.classes_)}")

print("\nAucune prédiction sur les données de test n'a été effectuée.")
print("L'étape 9.2 est terminée.")
