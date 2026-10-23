
from pathlib import Path

import numpy as np
import pandas as pd

# Chemins du projet
RACINE = Path(__file__).resolve().parent.parent
FICHIER_FEATURES = RACINE / "data" / "final" / "features.csv"

COLONNES_HU = [f"Hu{i}" for i in range(1, 8)]
COLONNES_ATTENDUES = ["image_id", "class", *COLONNES_HU]


def main():
    print("=" * 60)
    print("ÉTAPE 8.1 — INSPECTION DU DATASET POUR LE K-NN")
    print("=" * 60)

    # 1. Vérifier l'existence du fichier
    if not FICHIER_FEATURES.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {FICHIER_FEATURES}"
        )

    # 2. Charger le dataset
    df = pd.read_csv(FICHIER_FEATURES)

    print(f"\nNombre de lignes : {len(df)}")
    print(f"Nombre de colonnes : {len(df.columns)}")
    print(f"Colonnes trouvées : {list(df.columns)}")

    # 3. Vérifier les colonnes
    if list(df.columns) != COLONNES_ATTENDUES:
        raise ValueError(
            "Les colonnes ne correspondent pas à la structure attendue."
        )

    print("\nStructure des colonnes : OK")

    # 4. Vérifier les valeurs manquantes
    print("\nValeurs manquantes par colonne :")
    print(df.isna().sum())

    # 5. Vérifier que les caractéristiques sont numériques
    for colonne in COLONNES_HU:
        df[colonne] = pd.to_numeric(df[colonne], errors="coerce")

    X = df[COLONNES_HU]
    y = df["class"]

    print("\nValeurs non numériques ou manquantes dans Hu :",
          int(X.isna().sum().sum()))

    # 6. Vérifier les valeurs infinies
    valeurs = X.to_numpy(dtype=float)
    print("Valeurs infinies dans Hu :",
          int(np.isinf(valeurs).sum()))

    # 7. Afficher la répartition des classes
    print("\nEffectifs par classe :")
    print(y.value_counts().sort_index())

    print("\nProportions par classe (%) :")
    print((y.value_counts(normalize=True).sort_index() * 100).round(2))

    # 8. Rechercher les caractéristiques Hu identiques
    nombre_groupes_hu = int(
        X.duplicated(keep=False).sum()
    )

    print(
        "\nNombre de lignes appartenant à un groupe "
        f"de caractéristiques Hu identiques : {nombre_groupes_hu}"
    )

    # 9. Vérifier les identifiants
    print("\nIdentifiants manquants :", int(df["image_id"].isna().sum()))
    print("Identifiants dupliqués :",
          int(df["image_id"].duplicated().sum()))

    # 10. Aperçu
    print("\nAperçu des cinq premières lignes :")
    print(df.head().to_string(index=False))

    # 11. Résultat global
    valide = (
        len(df) == 2653
        and not df[COLONNES_ATTENDUES].isna().any().any()
        and np.isfinite(X.to_numpy(dtype=float)).all()
        and df["image_id"].is_unique
    )

    print("\n" + "=" * 60)
    print("RÉSULTAT :", "VÉRIFICATIONS RÉUSSIES" if valide
          else "ANOMALIES À EXAMINER")
    print("=" * 60)


if __name__ == "__main__":
    main()
