
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent

FICHIER_FEATURES = RACINE / "data" / "final" / "features.csv"
FICHIER_DOUBLONS = RACINE / "results" / "duplicates_exact.csv"
FICHIER_SORTIE = RACINE / "results" / "correspondance_doublons_features.csv"


def construire_image_id(nom, classe):
    """Reconstruit l'identifiant utilisé dans features.csv."""
    fichier = Path(str(nom))
    extension = fichier.suffix.lower().lstrip(".")
    nom_sans_extension = fichier.stem

    return f"{classe}/{nom_sans_extension}__{extension}.npy"


def main():
    print("=" * 65)
    print("VÉRIFICATION DES DOUBLONS ET DU DATASET FINAL")
    print("=" * 65)

    features = pd.read_csv(FICHIER_FEATURES)
    doublons = pd.read_csv(FICHIER_DOUBLONS)

    colonnes_requises = {"empreinte", "nom", "classe", "chemin"}
    if not colonnes_requises.issubset(doublons.columns):
        raise ValueError(
            f"Colonnes attendues dans le rapport : {colonnes_requises}"
        )

    # Reconstruire l'identifiant de chaque image du rapport
    doublons["image_id"] = doublons.apply(
        lambda ligne: construire_image_id(
            ligne["nom"], ligne["classe"]
        ),
        axis=1,
    )

    # Vérifier la correspondance avec le dataset final
    ids_features = set(features["image_id"])
    doublons["present_dans_features"] = doublons["image_id"].isin(
        ids_features
    )

    non_correspondants = doublons[
        ~doublons["present_dans_features"]
    ]

    print(f"\nLignes du rapport de doublons : {len(doublons)}")
    print(
        "Lignes correspondant à features.csv :",
        int(doublons["present_dans_features"].sum()),
    )
    print("Lignes non correspondantes :", len(non_correspondants))

    # Vérifier les groupes d'images identiques
    groupes = doublons.groupby("empreinte").agg(
        nombre_fichiers=("image_id", "size"),
        nombre_identifiants=("image_id", "nunique"),
        nombre_classes=("classe", "nunique"),
    )

    print("\nNombre de groupes d'images identiques :", len(groupes))
    print(
        "Groupes contenant plusieurs fichiers :",
        int((groupes["nombre_fichiers"] > 1).sum()),
    )
    print(
        "Groupes présents dans plusieurs classes :",
        int((groupes["nombre_classes"] > 1).sum()),
    )

    # Enregistrer le rapport de correspondance
    doublons.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")

    print(f"\nRapport enregistré : {FICHIER_SORTIE.relative_to(RACINE)}")

    if not non_correspondants.empty:
        print("\nExemples de lignes non correspondantes :")
        print(
            non_correspondants[
                ["nom", "classe", "image_id"]
            ].head(10).to_string(index=False)
        )

    print("\nVérification terminée.")


if __name__ == "__main__":
    main()
