
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Définir les chemins et les caractéristiques
# --------------------------------------------------

RACINE = Path(__file__).resolve().parent.parent

FICHIER_FEATURES = RACINE / "data" / "final" / "features.csv"
FICHIER_SPLIT = RACINE / "results" / "split_train_test.csv"

DOSSIER_SORTIE = RACINE / "results"
FICHIER_TRAIN = DOSSIER_SORTIE / "features_train_standardisees.csv"
FICHIER_TEST = DOSSIER_SORTIE / "features_test_standardisees.csv"

COLONNES_HU = [f"Hu{i}" for i in range(1, 8)]


def main():
    print("=" * 65)
    print("ÉTAPE 8.3 — STANDARDISATION DES CARACTÉRISTIQUES HU")
    print("=" * 65)

    # --------------------------------------------------
    # 2. Charger les données et la séparation validée
    # --------------------------------------------------

    features = pd.read_csv(FICHIER_FEATURES)
    split = pd.read_csv(FICHIER_SPLIT)

    # Récupérer l'affectation train/test enregistrée.
    donnees = features.merge(
        split[["image_id", "split"]],
        on="image_id",
        how="left",
        validate="one_to_one"
    )

    if donnees["split"].isna().any():
        raise ValueError("Certaines images n'ont pas de séparation.")

    if len(donnees) != len(features):
        raise ValueError("Le nombre d'images a changé après la fusion.")

    # Vérifier les valeurs numériques.
    valeurs = donnees[COLONNES_HU].to_numpy(dtype=float)

    if not np.isfinite(valeurs).all():
        raise ValueError("Les caractéristiques Hu contiennent des valeurs invalides.")

    train = donnees[donnees["split"] == "train"].copy()
    test = donnees[donnees["split"] == "test"].copy()

    print(f"\nImages d'entraînement : {len(train)}")
    print(f"Images de test : {len(test)}")

    # --------------------------------------------------
    # 3. Apprendre la standardisation sur train uniquement
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(train[COLONNES_HU])
    X_test = scaler.transform(test[COLONNES_HU])

    # --------------------------------------------------
    # 4. Vérifier les résultats
    # --------------------------------------------------

    print("\nMoyennes des caractéristiques standardisées (train) :")
    print(np.mean(X_train, axis=0).round(6))

    print("\nÉcarts-types des caractéristiques standardisées (train) :")
    print(np.std(X_train, axis=0).round(6))

    print("\nMoyennes des caractéristiques standardisées (test) :")
    print(np.mean(X_test, axis=0).round(6))

    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("La standardisation a produit des valeurs invalides.")

    # --------------------------------------------------
    # 5. Enregistrer les données standardisées
    # --------------------------------------------------

    train_sortie = train[["image_id", "class"]].copy()
    test_sortie = test[["image_id", "class"]].copy()

    train_sortie[COLONNES_HU] = X_train
    test_sortie[COLONNES_HU] = X_test

    train_sortie.to_csv(FICHIER_TRAIN, index=False, encoding="utf-8")
    test_sortie.to_csv(FICHIER_TEST, index=False, encoding="utf-8")

    print("\nFichiers enregistrés :")
    print(FICHIER_TRAIN.relative_to(RACINE))
    print(FICHIER_TEST.relative_to(RACINE))

    print("\nRÉSULTAT : STANDARDISATION VALIDÉE")


if __name__ == "__main__":
    main()
