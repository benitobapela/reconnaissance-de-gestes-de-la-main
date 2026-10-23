import sys
from pathlib import Path

import cv2
import joblib
import pandas as pd

from construire_dataset_final import (
    construire_masque,
    extraire_contour_principal,
    extraire_hu_transformes,
)
from preprocess_dataset import lire_image, redimensionner_image


# =========================
# CHEMINS DU PROJET
# =========================

RACINE = Path(__file__).resolve().parent.parent
CHEMIN_MODELE = RACINE / "models" / "knn_pipeline.pkl"


# =========================
# LECTURE DE L'IMAGE
# =========================

def lire_image_hsv(chemin_image):
    try:
        image_bgr, _ = lire_image(Path(chemin_image))
    except Exception as erreur:
        raise ValueError(
            "Image illisible ou format non pris en charge."
        ) from erreur

    # Même redimensionnement que celui utilisé pour le dataset
    image_bgr = redimensionner_image(image_bgr)

    # Conversion BGR vers HSV
    image_hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    return image_hsv


# =========================
# ANALYSE DE L'IMAGE
# =========================

def analyser_image(chemin_image):

    image_hsv = lire_image_hsv(chemin_image)

    # Création du masque de la main
    masque = construire_masque(image_hsv)

    # Extraction du contour principal
    contour = extraire_contour_principal(masque)

    # Extraction des 7 moments de Hu transformés
    hu = extraire_hu_transformes(image_hsv)

    return hu, masque, contour


# =========================
# CHARGEMENT DU MODELE
# =========================

def charger_modele(chemin=CHEMIN_MODELE):

    if not Path(chemin).exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {chemin}"
        )

    modele = joblib.load(chemin)

    return modele


# =========================
# PREPARATION DES DONNEES
# =========================

def preparer_tableau(modele, hu):

    # Si le modèle connaît les noms des colonnes
    if hasattr(modele, "feature_names_in_"):

        colonnes = list(modele.feature_names_in_)

        tableau = pd.DataFrame(
            [hu],
            columns=colonnes
        )

        return tableau

    # Sinon tableau simple 1 ligne × 7 colonnes
    return hu.reshape(1, -1)


# =========================
# PREDICTION
# =========================

def predire(chemin_image, modele=None):

    if modele is None:
        modele = charger_modele()

    # Prétraitement + Hu
    hu, masque, contour = analyser_image(
        chemin_image
    )

    # Préparation des caractéristiques
    X = preparer_tableau(modele, hu)

    # Prédiction avec le pipeline K-NN
    geste = modele.predict(X)[0]

    return geste, masque, contour


# =========================
# TEST DEPUIS LE TERMINAL
# =========================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage : python src/inference.py chemin_image"
        )

        sys.exit(1)

    chemin_image = sys.argv[1]

    try:

        geste, masque, contour = predire(
            chemin_image
        )

        # Création du dossier results
        dossier = RACINE / "results"
        dossier.mkdir(exist_ok=True)

        # Enregistrement du masque
        chemin_masque = (
            dossier / "masque_test.png"
        )

        cv2.imwrite(
            str(chemin_masque),
            masque
        )

        print()
        print("==============================")
        print("RESULTAT DE LA PREDICTION")
        print("==============================")
        print()
        print("Image :", chemin_image)
        print("Geste prédit :", geste)
        print("Masque enregistré :", chemin_masque)
        print()

    except ValueError as erreur:

        print()
        print("Erreur :", erreur)

    except FileNotFoundError as erreur:

        print()
        print("Erreur :", erreur)

    except Exception as erreur:

        print()
        print("Erreur inattendue :", erreur)