
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ==================================================
# 1. CHEMINS DU PROJET
# ==================================================

RACINE = Path(__file__).resolve().parent.parent

DOSSIER_HSV = RACINE / "data" / "processed" / "hsv"
DOSSIER_FINAL = RACINE / "data" / "final"
DOSSIER_RESULTS = RACINE / "results"

FICHIER_CSV = DOSSIER_FINAL / "features.csv"
FICHIER_ERREURS = DOSSIER_RESULTS / "erreurs_extraction_hu.csv"


# ==================================================
# 2. PARAMETRES DE SEGMENTATION HSV
# ==================================================

HSV_BAS = np.array([0, 20, 40], dtype=np.uint8)
HSV_HAUT = np.array([25, 255, 255], dtype=np.uint8)

EPSILON = 1e-30

COLONNES = [
    "image_id",
    "class",
    "Hu1",
    "Hu2",
    "Hu3",
    "Hu4",
    "Hu5",
    "Hu6",
    "Hu7",
]


# ==================================================
# 3. SEGMENTATION ET MORPHOLOGIE
# ==================================================

def construire_masque(image_hsv):
    """Segmente la peau et nettoie le masque."""

    masque = cv2.inRange(
        image_hsv,
        HSV_BAS,
        HSV_HAUT
    )

    noyau_fermeture = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    noyau_ouverture = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    masque = cv2.morphologyEx(
        masque,
        cv2.MORPH_CLOSE,
        noyau_fermeture
    )

    masque = cv2.morphologyEx(
        masque,
        cv2.MORPH_OPEN,
        noyau_ouverture
    )

    return masque


# ==================================================
# 4. EXTRACTION DU CONTOUR PRINCIPAL
# ==================================================

def extraire_contour_principal(masque):
    """Retourne le plus grand contour externe valide."""

    contours, _ = cv2.findContours(
        masque.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("Aucun contour détecté.")

    contour = max(contours, key=cv2.contourArea)

    if len(contour) < 3:
        raise ValueError("Contour insuffisant pour calculer Hu.")

    aire = cv2.contourArea(contour)

    if aire <= 0:
        raise ValueError("Aire du contour nulle ou invalide.")

    return contour


# ==================================================
# 5. CALCUL DES SEPT MOMENTS DE HU
# ==================================================

def extraire_hu_transformes(image_hsv):
    """Retourne les sept moments de Hu transformés."""

    if (
        image_hsv.ndim != 3
        or image_hsv.shape[2] != 3
        or image_hsv.dtype != np.uint8
    ):
        raise ValueError("Image HSV invalide.")

    masque = construire_masque(image_hsv)
    contour = extraire_contour_principal(masque)

    moments = cv2.moments(contour)

    if abs(moments["m00"]) < 1e-12:
        raise ValueError("Moment m00 nul.")

    hu = cv2.HuMoments(moments).flatten()

    if hu.shape != (7,):
        raise ValueError("Le calcul n'a pas produit 7 moments.")

    if not np.all(np.isfinite(hu)):
        raise ValueError("Moments de Hu bruts non finis.")

    # Transformation logarithmique signée
    hu_transformes = (
        -np.sign(hu)
        * np.log10(np.maximum(np.abs(hu), EPSILON))
    )

    if not np.all(np.isfinite(hu_transformes)):
        raise ValueError("Moments transformés non finis.")

    return hu_transformes


# ==================================================
# 6. CONSTRUCTION DU DATASET COMPLET
# ==================================================

def main():

    DOSSIER_FINAL.mkdir(parents=True, exist_ok=True)
    DOSSIER_RESULTS.mkdir(parents=True, exist_ok=True)

    if not DOSSIER_HSV.exists():
        raise FileNotFoundError(
            f"Dossier HSV introuvable : {DOSSIER_HSV}"
        )

    dossiers_classes = sorted(
        dossier
        for dossier in DOSSIER_HSV.iterdir()
        if dossier.is_dir()
    )

    if not dossiers_classes:
        raise RuntimeError(
            "Aucun dossier de classe trouvé dans le dossier HSV."
        )

    lignes = []
    erreurs = []
    total_images = 0

    print("=" * 65)
    print("CONSTRUCTION DU DATASET FINAL")
    print("=" * 65)

    for dossier_classe in dossiers_classes:

        classe = dossier_classe.name

        fichiers = sorted(dossier_classe.glob("*.npy"))

        print(f"\nClasse : {classe}")
        print(f"Images à traiter : {len(fichiers)}")

        reussites_classe = 0
        echecs_classe = 0

        for fichier in fichiers:

            total_images += 1

            try:
                image_hsv = np.load(
                    fichier,
                    allow_pickle=False
                )

                hu = extraire_hu_transformes(image_hsv)

                ligne = {
                    # Le chemin relatif distingue les fichiers
                    # même si deux classes ont le même nom de fichier.
                    "image_id": fichier.relative_to(
                        DOSSIER_HSV
                    ).as_posix(),

                    "class": classe,
                }

                for i, valeur in enumerate(hu, start=1):
                    ligne[f"Hu{i}"] = float(valeur)

                lignes.append(ligne)
                reussites_classe += 1

            except Exception as erreur:

                echecs_classe += 1

                erreurs.append({
                    "image_id": fichier.relative_to(
                        DOSSIER_HSV
                    ).as_posix(),
                    "class": classe,
                    "erreur": str(erreur),
                })

        print(f"Réussites : {reussites_classe}")
        print(f"Échecs : {echecs_classe}")

    # Construire le tableau avec les colonnes attendues.
    df = pd.DataFrame(lignes, columns=COLONNES)

    # Vérifier la structure et les valeurs avant l'enregistrement.
    if not df.empty:

        colonnes_hu = [f"Hu{i}" for i in range(1, 8)]

        if df["image_id"].duplicated().any():
            raise ValueError(
                "Des identifiants d'images sont dupliqués."
            )

        valeurs_hu = df[colonnes_hu].to_numpy(dtype=float)

        if not np.all(np.isfinite(valeurs_hu)):
            raise ValueError(
                "Le dataset contient des valeurs NaN ou infinies."
            )

        if df["class"].isna().any():
            raise ValueError(
                "Certaines lignes n'ont pas de classe."
            )

    # Enregistrer le dataset, même si aucune image n'a réussi :
    # les en-têtes seront alors conservés.
    df.to_csv(
        FICHIER_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # Enregistrer le détail des échecs, s'il y en a.
    if erreurs:

        pd.DataFrame(erreurs).to_csv(
            FICHIER_ERREURS,
            index=False,
            encoding="utf-8-sig"
        )

    elif FICHIER_ERREURS.exists():

        # Éviter de conserver un ancien rapport trompeur.
        FICHIER_ERREURS.unlink()

    # Bilan final
    print("\n" + "=" * 65)
    print("BILAN DE LA CONSTRUCTION")
    print("=" * 65)

    print(f"Images examinées : {total_images}")
    print(f"Images réussies : {len(df)}")
    print(f"Échecs : {len(erreurs)}")
    print(f"Colonnes du CSV : {len(df.columns)}")
    print(f"\nDataset final : {FICHIER_CSV}")

    if erreurs:
        print(f"Rapport des erreurs : {FICHIER_ERREURS}")

    print("\nNombre de lignes par classe :")

    if not df.empty:
        print(df.groupby("class").size().to_string())
    else:
        print("Aucune ligne valide.")

    print("\nConstruction terminée.")


if __name__ == "__main__":
    main()
