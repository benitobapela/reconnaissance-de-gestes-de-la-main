
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. Chemins et paramètres
# --------------------------------------------------

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_HSV = RACINE / "data" / "processed" / "hsv"

SEED = 42

HSV_BAS = np.array([0, 20, 40], dtype=np.uint8)
HSV_HAUT = np.array([25, 255, 255], dtype=np.uint8)


def construire_masque(image_hsv):
    """Segmente la main et applique la morphologie retenue."""

    masque = cv2.inRange(image_hsv, HSV_BAS, HSV_HAUT)

    noyau_fermeture = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )
    noyau_ouverture = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    masque = cv2.morphologyEx(
        masque, cv2.MORPH_CLOSE, noyau_fermeture
    )
    masque = cv2.morphologyEx(
        masque, cv2.MORPH_OPEN, noyau_ouverture
    )

    return masque


def extraire_contour_principal(masque):
    """Retourne le plus grand contour externe, ou None."""

    contours, _ = cv2.findContours(
        masque.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def calculer_hu(contour):
    """Calcule les sept moments de Hu bruts."""

    if contour is None or len(contour) < 3:
        return None

    moments = cv2.moments(contour)

    if abs(moments["m00"]) < 1e-12:
        return None

    hu = cv2.HuMoments(moments).flatten()

    if not np.all(np.isfinite(hu)):
        return None

    return hu

def transformer_hu(hu):
    """Applique la transformation logarithmique signée."""

    epsilon = 1e-30

    hu_transformes = (
        -np.sign(hu)
        * np.log10(np.maximum(np.abs(hu), epsilon))
    )

    return hu_transformes


# --------------------------------------------------
# 2. Sélectionner une image par classe
# --------------------------------------------------

def main():
    if not DOSSIER_HSV.exists():
        print(f"Dossier introuvable : {DOSSIER_HSV}")
        return

    generateur = np.random.default_rng(SEED)
    dossiers_classes = sorted(
        dossier for dossier in DOSSIER_HSV.iterdir()
        if dossier.is_dir()
    )

    if not dossiers_classes:
        print("Aucun dossier de classe trouvé.")
        return

    print("=" * 65)
    print("TEST DES MOMENTS DE HU — 1 IMAGE PAR CLASSE")
    print("=" * 65)

    for dossier_classe in dossiers_classes:
        fichiers = sorted(dossier_classe.glob("*.npy"))

        if not fichiers:
            print(f"\n{dossier_classe.name} : aucune image HSV.")
            continue

        index = int(generateur.integers(0, len(fichiers)))
        fichier = fichiers[index]

        image_hsv = np.load(fichier)

        if (
            image_hsv.ndim != 3
            or image_hsv.shape[2] != 3
            or image_hsv.dtype != np.uint8
        ):
            print(f"\nImage HSV invalide : {fichier.name}")
            continue

        masque = construire_masque(image_hsv)
        contour = extraire_contour_principal(masque)
        hu = calculer_hu(contour)

        print(f"\nClasse : {dossier_classe.name}")
        print(f"Image : {fichier.name}")

        if hu is None:
            print("Échec : aucun contour exploitable.")
            continue

        print(f"Aire du contour : {cv2.contourArea(contour):.2f} px²")

        hu_transformes = transformer_hu(hu)

        print("Moments de Hu bruts :")

        for i, valeur in enumerate(hu, start=1):
            print(f"  Hu{i} = {valeur:.12e}")

        print("\nMoments de Hu transformés :")

        for i, valeur in enumerate(hu_transformes, start=1):
            print(f"  Hu{i}' = {valeur:.6f}")

        print(f"\nNombre de moments obtenus : {len(hu)}")


    print("\nTest terminé.")


if __name__ == "__main__":
    main()
