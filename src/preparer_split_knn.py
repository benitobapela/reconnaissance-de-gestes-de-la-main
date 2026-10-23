
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# --------------------------------------------------
# 1. Définir les chemins
# --------------------------------------------------

RACINE = Path(__file__).resolve().parent.parent

FICHIER_FEATURES = RACINE / "data" / "final" / "features.csv"
FICHIER_DOUBLONS = RACINE / "results" / "correspondance_doublons_features.csv"
FICHIER_SORTIE = RACINE / "results" / "split_train_test.csv"


def construire_image_id(nom, classe):
    """Reconstruire l'identifiant utilisé dans features.csv."""
    fichier = Path(str(nom))
    extension = fichier.suffix.lower().lstrip(".")
    nom_sans_extension = fichier.stem

    return f"{classe}/{nom_sans_extension}__{extension}.npy"


def main():
    print("=" * 65)
    print("ÉTAPE 8.2 — SÉPARATION ENTRAÎNEMENT / TEST")
    print("=" * 65)

    # --------------------------------------------------
    # 2. Charger les données
    # --------------------------------------------------

    features = pd.read_csv(FICHIER_FEATURES)
    doublons = pd.read_csv(FICHIER_DOUBLONS)

    if features["image_id"].duplicated().any():
        raise ValueError("Des image_id sont dupliqués dans features.csv.")

    # --------------------------------------------------
    # 3. Construire les groupes de doublons
    # --------------------------------------------------

    # Chaque empreinte représente un groupe d'images
    # strictement identiques.
    image_vers_groupe = {}

    for _, ligne in doublons.iterrows():
        image_id = construire_image_id(
            ligne["nom"], ligne["classe"]
        )
        groupe_id = f"duplicate:{ligne['empreinte']}"

        if image_id in image_vers_groupe:
            if image_vers_groupe[image_id] != groupe_id:
                raise ValueError(
                    f"Plusieurs groupes pour {image_id}"
                )

        image_vers_groupe[image_id] = groupe_id

    # Les images sans doublon reçoivent chacune un groupe
    # individuel afin de ne pas les mélanger entre les ensembles.
    features["group_id"] = features["image_id"].map(
        image_vers_groupe
    )

    masque_sans_groupe = features["group_id"].isna()

    features.loc[masque_sans_groupe, "group_id"] = (
        "unique:" + features.loc[masque_sans_groupe, "image_id"]
    )

    # Vérifier que chaque image du rapport est présente.
    ids_features = set(features["image_id"])
    ids_doublons = set(image_vers_groupe)

    absents = ids_doublons - ids_features

    if absents:
        raise ValueError(
            f"{len(absents)} images du rapport sont absentes "
            "de features.csv."
        )

    # Vérifier qu'un groupe ne contient pas plusieurs classes.
    classes_par_groupe = features.groupby("group_id")["class"].nunique()

    if (classes_par_groupe > 1).any():
        raise ValueError(
            "Un groupe contient des images de classes différentes."
        )

    print(f"\nNombre total d'images : {len(features)}")
    print(f"Groupes de doublons : {len(doublons['empreinte'].unique())}")
    print(f"Images sans doublon exact : {masque_sans_groupe.sum()}")
    print(f"Nombre total de groupes : {features['group_id'].nunique()}")

    # --------------------------------------------------
    # 4. Effectuer une séparation groupée et stratifiée
    # --------------------------------------------------

    X = features[
        ["Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7"]
    ]
    y = features["class"]
    groupes = features["group_id"]

    separateur = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Le premier pli fournit environ 20 % des données pour le test.
    indices_train, indices_test = next(
        separateur.split(X, y, groups=groupes)
    )

    resultat = features[["image_id", "class", "group_id"]].copy()
    resultat["split"] = "train"
    resultat.loc[indices_test, "split"] = "test"

    # --------------------------------------------------
    # 5. Vérifier qu'aucun groupe n'est partagé
    # --------------------------------------------------

    groupes_train = set(
        resultat.loc[resultat["split"] == "train", "group_id"]
    )
    groupes_test = set(
        resultat.loc[resultat["split"] == "test", "group_id"]
    )

    groupes_communs = groupes_train.intersection(groupes_test)

    if groupes_communs:
        raise ValueError(
            f"Fuite de données : {len(groupes_communs)} groupes "
            "sont présents dans train et test."
        )

    # Vérifier que toutes les classes sont représentées.
    classes_train = set(
        resultat.loc[resultat["split"] == "train", "class"]
    )
    classes_test = set(
        resultat.loc[resultat["split"] == "test", "class"]
    )

    if classes_train != classes_test:
        raise ValueError(
            "Toutes les classes ne sont pas présentes "
            "dans les deux ensembles."
        )

    # --------------------------------------------------
    # 6. Enregistrer la séparation
    # --------------------------------------------------

    resultat.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8")

    print("\nRépartition obtenue :")

    tableau = pd.crosstab(
        resultat["class"],
        resultat["split"],
        margins=True
    )
    print(tableau.to_string())

    print("\nProportions des ensembles :")
    print(
        (resultat["split"].value_counts(normalize=True) * 100)
        .round(2)
        .astype(str)
        .add("%")
        .to_string()
    )

    print("\nVérification de la fuite de données :")
    print("Groupes communs entre train et test :", len(groupes_communs))

    print(f"\nFichier enregistré : {FICHIER_SORTIE.relative_to(RACINE)}")
    print("\nRÉSULTAT : SÉPARATION VALIDÉE")


if __name__ == "__main__":
    main()
