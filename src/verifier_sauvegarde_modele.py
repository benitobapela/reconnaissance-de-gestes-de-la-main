
from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# 1. Définir les chemins
# --------------------------------------------------

RACINE_PROJET = Path(__file__).resolve().parent.parent

FICHIER_MODELE = RACINE_PROJET / "models" / "knn_pipeline.pkl"
FICHIER_FEATURES = RACINE_PROJET / "data" / "final" / "features.csv"
FICHIER_SPLIT = RACINE_PROJET / "results" / "split_train_test.csv"
FICHIER_PREDICTIONS = RACINE_PROJET / "results" / "predictions_knn.csv"


# --------------------------------------------------
# 2. Vérifier l'existence des fichiers
# --------------------------------------------------

fichiers = [
    FICHIER_MODELE,
    FICHIER_FEATURES,
    FICHIER_SPLIT,
    FICHIER_PREDICTIONS,
]

for fichier in fichiers:
    if not fichier.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {fichier}"
        )


# --------------------------------------------------
# 3. Charger le pipeline sauvegardé
# --------------------------------------------------

modele = joblib.load(FICHIER_MODELE)

print("=" * 60)
print("VÉRIFICATION DU MODÈLE SAUVEGARDÉ")
print("=" * 60)

print("\nPipeline chargé avec succès.")
print("Étapes :", list(modele.named_steps.keys()))
print("Classes :", list(modele.named_steps["knn"].classes_))


# --------------------------------------------------
# 4. Charger les données de test
# --------------------------------------------------

features = pd.read_csv(FICHIER_FEATURES)
split = pd.read_csv(FICHIER_SPLIT)
predictions_initiales = pd.read_csv(FICHIER_PREDICTIONS)

ids_test = split.loc[
    split["split"] == "test", "image_id"
]

donnees_test = features[
    features["image_id"].isin(ids_test)
].copy()

colonnes_hu = [f"Hu{i}" for i in range(1, 8)]

X_test = donnees_test[colonnes_hu]


# --------------------------------------------------
# 5. Produire de nouvelles prédictions
# --------------------------------------------------

predictions_rechargees = modele.predict(X_test)

resultats = pd.DataFrame({
    "image_id": donnees_test["image_id"].to_numpy(),
    "classe_predite_rechargee": predictions_rechargees,
})


# --------------------------------------------------
# 6. Comparer avec les prédictions initiales
# --------------------------------------------------

comparaison = resultats.merge(
    predictions_initiales[
        ["image_id", "classe_predite"]
    ],
    on="image_id",
    how="outer",
    validate="one_to_one",
    indicator=True,
)

if not (comparaison["_merge"] == "both").all():
    raise ValueError(
        "Les identifiants ne correspondent pas entre "
        "les prédictions initiales et celles du modèle rechargé."
    )

identiques = (
    comparaison["classe_predite_rechargee"]
    == comparaison["classe_predite"]
)

nombre_total = len(comparaison)
nombre_identique = int(identiques.sum())
nombre_different = nombre_total - nombre_identique


# --------------------------------------------------
# 7. Afficher le bilan
# --------------------------------------------------

print("\n" + "-" * 60)
print("RÉSULTAT DE LA COMPARAISON")
print("-" * 60)

print(f"Nombre de prédictions comparées : {nombre_total}")
print(f"Prédictions identiques           : {nombre_identique}")
print(f"Prédictions différentes          : {nombre_different}")

if nombre_total == 531 and nombre_different == 0:
    print("\nSUCCÈS : les 531 prédictions sont identiques.")
    print("La sauvegarde et le rechargement sont validés.")
else:
    print("\nATTENTION : la vérification n'est pas entièrement validée.")
    print("Vérifie les fichiers et les résultats ci-dessus.")
