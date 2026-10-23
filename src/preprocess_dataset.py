
from pathlib import Path
import csv

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageOps


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Racine du projet : dossier parent de src/
PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# Deux représentations seront enregistrées séparément.
COLOR_DIR = PROCESSED_DIR / "color"
HSV_DIR = PROCESSED_DIR / "hsv"

REPORT_DIR = PROJECT_DIR / "results"
REPORT_FILE = REPORT_DIR / "pretraitement_rapport.csv"

MAX_SIDE = 256

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".heic"}


# Permet à Pillow de lire les fichiers HEIC.
pillow_heif.register_heif_opener()


# ============================================================
# 2. LECTURE ET VALIDATION
# ============================================================

def lire_image(chemin: Path):
    """
    Lit une image JPG, JPEG ou HEIC et la retourne en BGR,
    format couleur utilisé par OpenCV.

    Retourne :
        image_bgr : tableau NumPy à 3 canaux
        mode_source : mode détecté par Pillow
    """
    with Image.open(chemin) as image:
        image = ImageOps.exif_transpose(image)
        mode_source = image.mode

        # Normalisation en RGB pour obtenir 3 canaux couleur.
        image_rgb = image.convert("RGB")
        tableau_rgb = np.array(image_rgb)

    if tableau_rgb.ndim != 3 or tableau_rgb.shape[2] != 3:
        raise ValueError("L'image ne possède pas 3 canaux couleur après conversion.")

    # Passage de RGB vers BGR pour OpenCV.
    image_bgr = cv2.cvtColor(tableau_rgb, cv2.COLOR_RGB2BGR)

    if image_bgr.size == 0:
        raise ValueError("L'image est vide.")

    hauteur, largeur = image_bgr.shape[:2]

    if hauteur <= 0 or largeur <= 0:
        raise ValueError("Dimensions invalides.")

    return image_bgr, mode_source


# ============================================================
# 3. REDIMENSIONNEMENT SANS DEFORMATION
# ============================================================

def redimensionner_image(image_bgr, cote_max=MAX_SIDE):
    """
    Limite le plus grand côté à cote_max pixels,
    conserve les proportions et n'agrandit pas les petites images.
    """
    hauteur, largeur = image_bgr.shape[:2]
    plus_grand_cote = max(hauteur, largeur)

    if plus_grand_cote <= cote_max:
        return image_bgr.copy()

    facteur = cote_max / plus_grand_cote

    nouvelle_largeur = max(1, round(largeur * facteur))
    nouvelle_hauteur = max(1, round(hauteur * facteur))

    image_redimensionnee = cv2.resize(
        image_bgr,
        (nouvelle_largeur, nouvelle_hauteur),
        interpolation=cv2.INTER_AREA
    )

    return image_redimensionnee


# ============================================================
# 4. PREPARATION DES DOSSIERS DE SORTIE
# ============================================================

def preparer_sorties(chemin_source: Path):
    """
    Préserve la structure des sous-dossiers de data/raw.

    Exemple :
        data/raw/Fist/photo.jpg

    Produit :
        data/processed/color/Fist/photo__jpg.png
        data/processed/hsv/Fist/photo__jpg.npy
    """
    chemin_relatif = chemin_source.relative_to(RAW_DIR)
    dossier_relatif = chemin_relatif.parent

    nom_base = (
        f"{chemin_source.stem}__"
        f"{chemin_source.suffix.lstrip('.').lower()}"
    )

    dossier_couleur = COLOR_DIR / dossier_relatif
    dossier_hsv = HSV_DIR / dossier_relatif

    dossier_couleur.mkdir(parents=True, exist_ok=True)
    dossier_hsv.mkdir(parents=True, exist_ok=True)

    chemin_couleur = dossier_couleur / f"{nom_base}.png"
    chemin_hsv = dossier_hsv / f"{nom_base}.npy"

    return chemin_couleur, chemin_hsv


# ============================================================
# 5. TRAITEMENT D'UNE IMAGE
# ============================================================

def traiter_image(chemin_source: Path):
    """
    Lit, vérifie, redimensionne, convertit en HSV et enregistre
    les deux représentations de l'image.
    """
    resultat = {
        "fichier_source": str(chemin_source.relative_to(PROJECT_DIR)),
        "format": chemin_source.suffix.lower(),
        "mode_source": "",
        "dimensions_originales": "",
        "dimensions_finales": "",
        "canaux_hsv": "",
        "fichier_couleur": "",
        "fichier_hsv": "",
        "statut": "",
        "erreur": ""
    }

    try:
        # Lecture et vérification.
        image_bgr, mode_source = lire_image(chemin_source)

        hauteur_originale, largeur_originale = image_bgr.shape[:2]

        resultat["mode_source"] = mode_source
        resultat["dimensions_originales"] = (
            f"{largeur_originale}x{hauteur_originale}"
        )

        # Redimensionnement.
        image_bgr = redimensionner_image(image_bgr)

        hauteur_finale, largeur_finale = image_bgr.shape[:2]

        # Conversion BGR -> HSV.
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        if image_hsv.shape != (hauteur_finale, largeur_finale, 3):
            raise ValueError("Dimensions inattendues après conversion HSV.")

        if image_hsv.dtype != np.uint8:
            raise ValueError("Type de données HSV inattendu.")

        # Préparation des chemins de sortie.
        chemin_couleur, chemin_hsv = preparer_sorties(chemin_source)

        # Enregistrement de l'image couleur en PNG sans perte.
        image_rgb_finale = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2RGB
        )
        Image.fromarray(image_rgb_finale).save(
            chemin_couleur, format="PNG"
        )

        # Enregistrement du tableau HSV sans conversion en image RGB.
        np.save(chemin_hsv, image_hsv)

        resultat["dimensions_finales"] = (
            f"{largeur_finale}x{hauteur_finale}"
        )
        resultat["canaux_hsv"] = image_hsv.shape[2]
        resultat["fichier_couleur"] = str(
            chemin_couleur.relative_to(PROJECT_DIR)
        )
        resultat["fichier_hsv"] = str(
            chemin_hsv.relative_to(PROJECT_DIR)
        )
        resultat["statut"] = "SUCCES"

    except Exception as erreur:
        resultat["statut"] = "ERREUR"
        resultat["erreur"] = str(erreur)

    return resultat


# ============================================================
# 6. TRAITEMENT DU DATASET COMPLET
# ============================================================

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Le dossier des images originales est introuvable : {RAW_DIR}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fichiers = sorted(
        chemin
        for chemin in RAW_DIR.rglob("*")
        if chemin.is_file()
        and chemin.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not fichiers:
        print("Aucune image JPG, JPEG ou HEIC trouvée.")
        return

    resultats = []

    print(f"Images trouvées : {len(fichiers)}")
    print(f"Taille maximale de travail : {MAX_SIDE} px")
    print("Début du prétraitement...\n")

    for numero, chemin in enumerate(fichiers, start=1):
        resultat = traiter_image(chemin)
        resultats.append(resultat)

        if resultat["statut"] == "SUCCES":
            print(
                f"[{numero}/{len(fichiers)}] OK : "
                f"{resultat['fichier_source']} "
                f"({resultat['dimensions_originales']} -> "
                f"{resultat['dimensions_finales']})"
            )
        else:
            print(
                f"[{numero}/{len(fichiers)}] ERREUR : "
                f"{resultat['fichier_source']} - "
                f"{resultat['erreur']}"
            )

    # Rapport CSV pour conserver la trace de chaque traitement.
    champs = [
        "fichier_source",
        "format",
        "mode_source",
        "dimensions_originales",
        "dimensions_finales",
        "canaux_hsv",
        "fichier_couleur",
        "fichier_hsv",
        "statut",
        "erreur"
    ]

    with REPORT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as fichier_csv:
        redacteur = csv.DictWriter(
            fichier_csv, fieldnames=champs
        )
        redacteur.writeheader()
        redacteur.writerows(resultats)

    succes = sum(
        resultat["statut"] == "SUCCES"
        for resultat in resultats
    )
    erreurs = len(resultats) - succes

    print("\n========== BILAN DU PRETRAITEMENT ==========")
    print(f"Images trouvées : {len(fichiers)}")
    print(f"Traitements réussis : {succes}")
    print(f"Échecs : {erreurs}")
    print(f"Rapport : {REPORT_FILE}")
    print(f"Images couleur : {COLOR_DIR}")
    print(f"Tableaux HSV : {HSV_DIR}")
    print("\nLes fichiers de data/raw n'ont pas été modifiés.")


if __name__ == "__main__":
    main()
