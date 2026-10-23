
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

DATASET_DIR = Path("data/raw")

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff", ".heic", ".heif"
}


def lire_image(chemin):
    """Lit une image et retourne un tableau de pixels RGB."""

    if chemin.suffix.lower() in {".heic", ".heif"}:
        with Image.open(chemin) as image:
            image.load()
            return np.array(image.convert("RGB"))

    image = cv2.imread(str(chemin), cv2.IMREAD_COLOR)

    if image is None:
        return None

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def verifier_integrite():
    total = 0
    illisibles = []
    fichiers_vides = []
    dimensions_invalides = []
    images_uniformes = []
    dimensions = Counter()
    formats = Counter()

    for chemin in sorted(DATASET_DIR.rglob("*")):
        if not chemin.is_file():
            continue

        if chemin.suffix.lower() not in EXTENSIONS:
            continue

        total += 1
        formats[chemin.suffix.lower()] += 1

        if chemin.stat().st_size == 0:
            fichiers_vides.append(str(chemin))
            continue

        try:
            image = lire_image(chemin)

            if image is None or image.size == 0:
                illisibles.append(str(chemin))
                continue

            hauteur, largeur = image.shape[:2]
            dimensions[(largeur, hauteur)] += 1

            if largeur <= 0 or hauteur <= 0:
                dimensions_invalides.append(str(chemin))
                continue

            if largeur < 100 or hauteur < 100:
                dimensions_invalides.append(
                    f"{chemin} ({largeur} x {hauteur})"
                )

            # Une image est dite uniforme si tous ses pixels
            # ont exactement la même valeur.
            if np.all(image == image[0, 0]):
                images_uniformes.append(str(chemin))

        except Exception as erreur:
            illisibles.append(f"{chemin} — {erreur}")

    print("=" * 60)
    print("VÉRIFICATION DE L'INTÉGRITÉ DU DATASET")
    print("=" * 60)

    print(f"\nFichiers image examinés : {total}")
    print(f"Fichiers vides : {len(fichiers_vides)}")
    print(f"Images illisibles : {len(illisibles)}")
    print(f"Dimensions suspectes : {len(dimensions_invalides)}")
    print(f"Images totalement uniformes : {len(images_uniformes)}")

    print("\nRépartition des formats :")
    for extension, nombre in sorted(formats.items()):
        print(f"{extension}: {nombre}")

    print("\nDimensions rencontrées :")
    for (largeur, hauteur), nombre in dimensions.most_common():
        print(f"{largeur} x {hauteur}: {nombre}")

    for titre, liste in [
        ("Fichiers vides", fichiers_vides),
        ("Images illisibles", illisibles),
        ("Dimensions suspectes", dimensions_invalides),
        ("Images uniformes", images_uniformes),
    ]:
        if liste:
            print(f"\n--- {titre} ---")
            for element in liste[:20]:
                print(element)

    print("\nContrôle terminé. Aucun fichier original n'a été modifié.")


if __name__ == "__main__":
    verifier_integrite()
