
from pathlib import Path

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
FICHIER_PREDICTIONS = RACINE_PROJET / "results" / "predictions_knn.csv"


# --------------------------------------------------
# 2. Charger les données
# --------------------------------------------------

features = pd.read_csv(FICHIER_FEATURES)
split = pd.read_csv(FICHIER_SPLIT)

# Séparer les identifiants selon la répartition déjà établie
ids_train = split.loc[
    split["split"] == "train", "image_id"
]

ids_test = split.loc[
    split["split"] == "test", "image_id"
]

# Récupérer les lignes des deux ensembles
donnees_train = features[
    features["image_id"].isin(ids_train)
].copy()

donnees_test = features[
    features["image_id"].isin(ids_test)
].copy()

colonnes_hu = [f"Hu{i}" for i in range(1, 8)]

X_train = donnees_train[colonnes_hu]
y_train = donnees_train["class"]

X_test = donnees_test[colonnes_hu]


# --------------------------------------------------
# 3. Construire le pipeline K-NN
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
# 4. Entraîner sur train uniquement
# --------------------------------------------------

modele_knn.fit(X_train, y_train)


# --------------------------------------------------
# 5. Prédire les classes des images de test
# --------------------------------------------------

predictions = modele_knn.predict(X_test)


# --------------------------------------------------
# 6. Enregistrer les prédictions
# --------------------------------------------------

resultats = donnees_test[
    ["image_id", "class"]
].copy()

resultats = resultats.rename(
    columns={"class": "classe_reelle"}
)

resultats["classe_predite"] = predictions

FICHIER_PREDICTIONS.parent.mkdir(
    parents=True,
    exist_ok=True,
)

resultats.to_csv(
    FICHIER_PREDICTIONS,
    index=False,
    encoding="utf-8",
)


# --------------------------------------------------
# 7. Afficher le bilan
# --------------------------------------------------

print("=" * 55)
print("ÉTAPE 9.3 — PRÉDICTIONS SUR LES DONNÉES DE TEST")
print("=" * 55)

print(f"\nImages d'entraînement : {len(X_train)}")
print(f"Images de test        : {len(X_test)}")
print(f"Prédictions produites : {len(predictions)}")

print("\nAperçu des premières prédictions :")
print(resultats.head(10).to_string(index=False))

print("\nNombre de prédictions par classe prédite :")
print(
    resultats["classe_predite"]
    .value_counts()
    .sort_index()
    .to_string()
)

print(f"\nFichier enregistré : {FICHIER_PREDICTIONS}")
print("Les prédictions sont terminées.")
print("Aucune métrique d'évaluation n'a encore été calculée.")
