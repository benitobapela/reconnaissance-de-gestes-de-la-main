
from pathlib import Path

import numpy as np
import pandas as pd


RACINE = Path(__file__).resolve().parent.parent
FICHIER_CSV = RACINE / "data" / "final" / "features.csv"

CLASSES_ATTENDUES = {
    "Fist": 542,
    "L-Sign": 537,
    "Open Palm_Open Hand": 661,
    "Pointing Up": 405,
    "V-Sign": 508,
}

COLONNES_ATTENDUES = [
    "image_id", "class",
    "Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7",
]


def main():
    print("=" * 60)
    print("VALIDATION DU DATASET FINAL")
    print("=" * 60)

    if not FICHIER_CSV.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {FICHIER_CSV}"
        )

    df = pd.read_csv(FICHIER_CSV)

    # 1. Dimensions et structure
    print(f"\nNombre de lignes : {len(df)}")
    print(f"Nombre de colonnes : {len(df.columns)}")

    structure_ok = df.columns.tolist() == COLONNES_ATTENDUES
    print(f"Structure des colonnes : {'OK' if structure_ok else 'ERREUR'}")

    if not structure_ok:
        print("Colonnes trouvées :", df.columns.tolist())
        raise ValueError("Les colonnes ne correspondent pas au format attendu.")

    # 2. Nombre total de lignes
    print(f"Nombre total attendu : {sum(CLASSES_ATTENDUES.values())}")
    print(f"Nombre total obtenu : {'OK' if len(df) == 2653 else 'ERREUR'}")

    # 3. Répartition des classes
    effectifs = df["class"].value_counts().to_dict()

    print("\nRépartition par classe :")
    print(df["class"].value_counts().sort_index().to_string())

    classes_ok = effectifs == CLASSES_ATTENDUES
    print(f"Effectifs des classes : {'OK' if classes_ok else 'ERREUR'}")

    # 4. Identifiants manquants ou dupliqués
    ids_manquants = int(df["image_id"].isna().sum())
    ids_dupliques = int(df["image_id"].duplicated().sum())

    print(f"\nIdentifiants manquants : {ids_manquants}")
    print(f"Identifiants dupliqués : {ids_dupliques}")

    # 5. Valeurs manquantes
    valeurs_manquantes = int(df.isna().sum().sum())
    print(f"Valeurs manquantes dans le CSV : {valeurs_manquantes}")

    # 6. Vérification des sept caractéristiques numériques
    colonnes_hu = [f"Hu{i}" for i in range(1, 8)]

    hu = df[colonnes_hu].apply(pd.to_numeric, errors="coerce")
    valeurs_non_numeriques = int(hu.isna().sum().sum())

    if hu.size:
        valeurs_finies = np.isfinite(hu.to_numpy(dtype=float))
        valeurs_non_finies = int((~valeurs_finies).sum())
    else:
        valeurs_non_finies = 0

    print(f"Valeurs Hu non numériques ou manquantes : {valeurs_non_numeriques}")
    print(f"Valeurs Hu non finies (NaN/infini) : {valeurs_non_finies}")

    # 7. Aperçu du fichier
    print("\nAperçu des cinq premières lignes :")
    print(df.head().to_string(index=False))

    # 8. Conclusion
    valide = (
        len(df) == 2653
        and structure_ok
        and classes_ok
        and ids_manquants == 0
        and ids_dupliques == 0
        and valeurs_manquantes == 0
        and valeurs_non_numeriques == 0
        and valeurs_non_finies == 0
    )

    print("\n" + "=" * 60)
    print("RÉSULTAT :", "VALIDATION RÉUSSIE" if valide else "ANOMALIES À EXAMINER")
    print("=" * 60)

    if not valide:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
