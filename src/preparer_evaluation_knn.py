
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Définition des chemins
# --------------------------------------------------

RACINE_PROJET = Path(__file__).resolve().parent.parent

FICHIER_FEATURES = RACINE_PROJET / "data" / "final" / "features.csv"
FICHIER_SPLIT = RACINE_PROJET / "results" / "split_train_test.csv"

CLASSES_ATTENDUES = {
    "Fist",
    "L-Sign",
    "Open Palm_Open Hand",
    "Pointing Up",
    "V-Sign",
}

COLONNES_HU = [f"Hu{i}" for i in range(1, 8)]


# --------------------------------------------------
# 2. Vérifier l'existence des fichiers
# --------------------------------------------------

print("=" * 60)
print("ÉTAPE 9.1 — PRÉPARATION DES DONNÉES D'ÉVALUATION")
print("=" * 60)

if not FICHIER_FEATURES.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER_FEATURES}"
    )

if not FICHIER_SPLIT.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER_SPLIT}"
    )

print("\n[OK] Les deux fichiers nécessaires existent.")


# --------------------------------------------------
# 3. Charger les fichiers
# --------------------------------------------------

features = pd.read_csv(FICHIER_FEATURES)
split = pd.read_csv(FICHIER_SPLIT)

print("\n--- Dimensions des fichiers ---")
print(f"Caractéristiques : {features.shape}")
print(f"Répartition      : {split.shape}")


# --------------------------------------------------
# 4. Vérifier les colonnes nécessaires
# --------------------------------------------------

colonnes_features = {"image_id", "class", *COLONNES_HU}
colonnes_split = {"image_id", "class", "group_id", "split"}

if not colonnes_features.issubset(features.columns):
    raise ValueError(
        "Il manque des colonnes dans features.csv : "
        f"{colonnes_features - set(features.columns)}"
    )

if not colonnes_split.issubset(split.columns):
    raise ValueError(
        "Il manque des colonnes dans split_train_test.csv : "
        f"{colonnes_split - set(split.columns)}"
    )

print("[OK] Toutes les colonnes nécessaires sont présentes.")


# --------------------------------------------------
# 5. Vérifier les identifiants et les correspondances
# --------------------------------------------------

if features["image_id"].isna().any():
    raise ValueError("Des identifiants d'images sont manquants.")

if split["image_id"].isna().any():
    raise ValueError("Des identifiants sont manquants dans le split.")

if features["image_id"].duplicated().any():
    raise ValueError("Des image_id sont dupliqués dans features.csv.")

if split["image_id"].duplicated().any():
    raise ValueError("Des image_id sont dupliqués dans le fichier split.")

ids_features = set(features["image_id"])
ids_split = set(split["image_id"])

if ids_features != ids_split:
    absents_du_split = ids_features - ids_split
    absents_des_features = ids_split - ids_features

    raise ValueError(
        "Les identifiants des deux fichiers ne correspondent pas.\n"
        f"Absents du split : {len(absents_du_split)}\n"
        f"Absents des caractéristiques : {len(absents_des_features)}"
    )

print("[OK] Chaque image possède une seule ligne dans chaque fichier.")


# --------------------------------------------------
# 6. Vérifier les classes
# --------------------------------------------------

classes_features = set(features["class"].unique())
classes_split = set(split["class"].unique())

if classes_features != CLASSES_ATTENDUES:
    raise ValueError(
        f"Classes inattendues dans features.csv : {classes_features}"
    )

if classes_split != CLASSES_ATTENDUES:
    raise ValueError(
        f"Classes inattendues dans le fichier split : {classes_split}"
    )

correspondance_classes = features[
    ["image_id", "class"]
].merge(
    split[["image_id", "class"]],
    on="image_id",
    suffixes=("_features", "_split"),
)

if not (
    correspondance_classes["class_features"]
    == correspondance_classes["class_split"]
).all():
    raise ValueError(
        "Certaines images n'ont pas la même classe dans les deux fichiers."
    )

print("[OK] Les cinq classes sont correctes et cohérentes.")


# --------------------------------------------------
# 7. Vérifier les sept moments de Hu
# --------------------------------------------------

valeurs_hu = features[COLONNES_HU].to_numpy(dtype=float)

if not np.isfinite(valeurs_hu).all():
    raise ValueError(
        "Les caractéristiques Hu contiennent des valeurs "
        "manquantes ou infinies."
    )

print("[OK] Les sept moments de Hu sont tous valides.")


# --------------------------------------------------
# 8. Vérifier la répartition entraînement/test
# --------------------------------------------------

repartition_attendue = {"train": 2122, "test": 531}

if set(split["split"].unique()) != {"train", "test"}:
    raise ValueError(
        "Le fichier doit contenir uniquement les ensembles "
        "'train' et 'test'."
    )

effectifs = split["split"].value_counts().to_dict()

print("\n--- Répartition des données ---")
for nom in ["train", "test"]:
    nombre = effectifs.get(nom, 0)
    pourcentage = nombre / len(split) * 100
    print(f"{nom:>5} : {nombre:4d} images ({pourcentage:.2f} %)")

if effectifs != repartition_attendue:
    raise ValueError(
        f"Répartition inattendue : {effectifs}"
    )

print("[OK] Les effectifs attendus sont respectés.")


# --------------------------------------------------
# 9. Vérifier les classes dans chaque ensemble
# --------------------------------------------------

print("\n--- Répartition par classe ---")

tableau_classes = pd.crosstab(
    split["class"],
    split["split"],
).reindex(sorted(CLASSES_ATTENDUES), fill_value=0)

print(tableau_classes)

if (tableau_classes == 0).any().any():
    raise ValueError(
        "Au moins une classe est absente d'un des ensembles."
    )

print("[OK] Les cinq classes sont présentes dans train et test.")


# --------------------------------------------------
# 10. Vérifier l'absence de fuite entre les groupes
# --------------------------------------------------

if split["group_id"].isna().any():
    raise ValueError("Certains group_id sont manquants.")

groupes_train = set(
    split.loc[split["split"] == "train", "group_id"]
)

groupes_test = set(
    split.loc[split["split"] == "test", "group_id"]
)

groupes_communs = groupes_train.intersection(groupes_test)

print("\n--- Vérification des doublons ---")
print(f"Groupes dans train : {len(groupes_train)}")
print(f"Groupes dans test  : {len(groupes_test)}")
print(f"Groupes communs    : {len(groupes_communs)}")

if groupes_communs:
    raise ValueError(
        "Fuite de données détectée : certains groupes "
        "sont présents dans train et test."
    )

print("[OK] Aucun groupe commun entre train et test.")


# --------------------------------------------------
# 11. Bilan final
# --------------------------------------------------

print("\n" + "=" * 60)
print("BILAN DE L'ÉTAPE 9.1")
print("=" * 60)
print(f"Nombre total d'images : {len(features)}")
print(f"Images d'entraînement : {effectifs['train']}")
print(f"Images de test        : {effectifs['test']}")
print(f"Nombre de classes     : {len(CLASSES_ATTENDUES)}")
print("Toutes les vérifications sont réussies.")
print("Les données sont prêtes pour l'étape 9.2.")
