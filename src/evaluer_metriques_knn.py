
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# --------------------------------------------------
# 1. Définir les chemins
# --------------------------------------------------

RACINE_PROJET = Path(__file__).resolve().parent.parent

FICHIER_PREDICTIONS = (
    RACINE_PROJET / "results" / "predictions_knn.csv"
)

FICHIER_METRIQUES = (
    RACINE_PROJET / "results" / "metriques_knn.csv"
)

CLASSES = [
    "Fist",
    "L-Sign",
    "Open Palm_Open Hand",
    "Pointing Up",
    "V-Sign",
]


# --------------------------------------------------
# 2. Charger et vérifier les prédictions
# --------------------------------------------------

if not FICHIER_PREDICTIONS.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER_PREDICTIONS}"
    )

df = pd.read_csv(FICHIER_PREDICTIONS)

colonnes_requises = {
    "image_id",
    "classe_reelle",
    "classe_predite",
}

if not colonnes_requises.issubset(df.columns):
    raise ValueError(
        "Colonnes manquantes dans le fichier des prédictions."
    )

if df[list(colonnes_requises)].isna().any().any():
    raise ValueError(
        "Le fichier contient des valeurs manquantes."
    )

if df["image_id"].duplicated().any():
    raise ValueError(
        "Des identifiants d'images sont dupliqués."
    )

if len(df) != 531:
    raise ValueError(
        f"531 prédictions attendues, mais {len(df)} trouvées."
    )

y_reel = df["classe_reelle"]
y_predit = df["classe_predite"]


# --------------------------------------------------
# 3. Calculer les métriques globales
# --------------------------------------------------

accuracy = accuracy_score(y_reel, y_predit)

precision_macro = precision_score(
    y_reel,
    y_predit,
    labels=CLASSES,
    average="macro",
    zero_division=0,
)

rappel_macro = recall_score(
    y_reel,
    y_predit,
    labels=CLASSES,
    average="macro",
    zero_division=0,
)

f1_macro = f1_score(
    y_reel,
    y_predit,
    labels=CLASSES,
    average="macro",
    zero_division=0,
)

f1_weighted = f1_score(
    y_reel,
    y_predit,
    labels=CLASSES,
    average="weighted",
    zero_division=0,
)


# --------------------------------------------------
# 4. Afficher le rapport par classe
# --------------------------------------------------

print("=" * 65)
print("ÉTAPE 9.4 — MÉTRIQUES D'ÉVALUATION DU K-NN")
print("=" * 65)

print(f"\nNombre d'images évaluées : {len(df)}")

print("\n--- Métriques globales ---")
print(f"Accuracy             : {accuracy:.4f} ({accuracy * 100:.2f} %)")
print(f"Précision macro      : {precision_macro:.4f}")
print(f"Rappel macro         : {rappel_macro:.4f}")
print(f"F1-score macro       : {f1_macro:.4f}")
print(f"F1-score pondéré     : {f1_weighted:.4f}")

print("\n--- Rapport détaillé par classe ---")

rapport = classification_report(
    y_reel,
    y_predit,
    labels=CLASSES,
    target_names=CLASSES,
    digits=4,
    zero_division=0,
)

print(rapport)


# --------------------------------------------------
# 5. Enregistrer les métriques globales
# --------------------------------------------------

metriques = pd.DataFrame([
    {"metrique": "accuracy", "valeur": accuracy},
    {"metrique": "precision_macro", "valeur": precision_macro},
    {"metrique": "rappel_macro", "valeur": rappel_macro},
    {"metrique": "f1_macro", "valeur": f1_macro},
    {"metrique": "f1_weighted", "valeur": f1_weighted},
])

metriques.to_csv(
    FICHIER_METRIQUES,
    index=False,
    encoding="utf-8",
)

print(f"\nMétriques enregistrées dans : {FICHIER_METRIQUES}")
print("Étape 9.4 terminée.")
