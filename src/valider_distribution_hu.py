
from pathlib import Path

import numpy as np
import pandas as pd


RACINE = Path(__file__).resolve().parent.parent
FICHIER_CSV = RACINE / "results" / "distribution_hu_echantillon.csv"

CLASSES_ATTENDUES = {
    "Fist",
    "L-Sign",
    "Open Palm_Open Hand",
    "Pointing Up",
    "V-Sign",
}

COLONNES_HU = [f"Hu{i}" for i in range(1, 8)]
COLONNES_ATTENDUES = ["image_id", "class"] + COLONNES_HU


def main():
    print("=" * 55)
    print("VALIDATION DES MOMENTS DE HU")
    print("=" * 55)

    if not FICHIER_CSV.exists():
        print(f"ERREUR : fichier introuvable : {FICHIER_CSV}")
        return

    df = pd.read_csv(FICHIER_CSV)

    # 1. Vérifier la structure du fichier
    colonnes_manquantes = [
        c for c in COLONNES_ATTENDUES if c not in df.columns
    ]

    if colonnes_manquantes:
        print(f"ERREUR : colonnes manquantes : {colonnes_manquantes}")
        return

    print(f"\nNombre de lignes : {len(df)}")
    print(f"Nombre de colonnes : {len(df.columns)}")
    print("Structure du CSV : OK")

    # 2. Vérifier les classes
    classes_observees = set(df["class"].dropna().unique())

    print("\nRépartition par classe :")
    print(df.groupby("class").size().to_string())

    if classes_observees != CLASSES_ATTENDUES:
        print("ERREUR : les classes observées ne correspondent pas.")
    elif not (df.groupby("class").size() == 30).all():
        print("ATTENTION : chaque classe ne contient pas 30 lignes.")
    else:
        print("Répartition des cinq classes : OK")

    # 3. Vérifier les identifiants
    if df["image_id"].isna().any() or df["class"].isna().any():
        print("ERREUR : identifiant ou classe manquant.")
    elif df.duplicated(["class", "image_id"]).any():
        print("ATTENTION : des identifiants sont répétés dans une classe.")
    else:
        print("Identifiants des images : OK")

    # 4. Vérifier les sept moments
    valeurs = df[COLONNES_HU].apply(pd.to_numeric, errors="coerce")
    tableau = valeurs.to_numpy(dtype=float)

    nb_manquantes = int(valeurs.isna().sum().sum())
    nb_non_finies = int((~np.isfinite(tableau)).sum())

    print(f"\nValeurs manquantes ou non numériques : {nb_manquantes}")
    print(f"Valeurs non finies (NaN/infini) : {nb_non_finies}")

    if nb_manquantes == 0 and nb_non_finies == 0:
        print("Valeurs des sept moments : OK")
    else:
        print("ERREUR : certaines valeurs doivent être examinées.")

    # 5. Résumé numérique pour repérer des valeurs inhabituelles
    if nb_manquantes == 0 and nb_non_finies == 0:
        print("\nRésumé des moments transformés :")
        print(valeurs.describe().round(4).to_string())

    print("\nValidation terminée.")
    print("Consulter les messages ci-dessus pour le bilan de chaque test.")


if __name__ == "__main__":
    main()
