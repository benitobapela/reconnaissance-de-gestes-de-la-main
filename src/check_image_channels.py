
from pathlib import Path
from collections import Counter

import cv2
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

DATASET_DIR = Path("data/raw")

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff", ".heic", ".heif"
}


def inspecter_canaux():
    distributions_canaux = Counter()
    distributions_dimensions = Counter()
    problemes = []
    exemples = []

    total = 0

    for chemin in sorted(DATASET_DIR.rglob("*")):
        if not chemin.is_file():
            continue

        if chemin.suffix.lower() not in EXTENSIONS:
            continue

        total += 1

        try:
            if chemin.suffix.lower() in {".heic", ".heif"}:
                with Image.open(chemin) as image:
                    image.load()
                    largeur, hauteur = image.size
                    mode = image.mode

                    if mode == "L":
                        nombre_canaux = 1
                    elif mode == "LA":
                        nombre_canaux = 2
                    elif mode == "RGBA":
                        nombre_canaux = 4
                    elif mode == "RGB":
                        nombre_canaux = 3
                    else:
                        nombre_canaux = len(image.getbands())

            else:
                image = cv2.imread(
                    str(chemin),
                    cv2.IMREAD_UNCHANGED
                )

                if image is None:
                    problemes.append(
                        f"Image illisible : {chemin}"
                    )
                    continue

                hauteur, largeur = image.shape[:2]

                if image.ndim == 2:
                    nombre_canaux = 1
                else:
                    nombre_canaux = image.shape[2]

            distributions_canaux[nombre_canaux] += 1
            distributions_dimensions[(largeur, hauteur)] += 1

            if len(exemples) < 10:
                exemples.append({
                    "nom": chemin.name,
                    "classe": chemin.parent.name,
                    "largeur": largeur,
                    "hauteur": hauteur,
                    "canaux": nombre_canaux,
                })

        except Exception as erreur:
            problemes.append(
                f"Erreur : {chemin} — {erreur}"
            )

    print("=" * 60)
    print("VÉRIFICATION DES CANAUX ET DES DIMENSIONS")
    print("=" * 60)

    print(f"\nNombre total de fichiers image : {total}")

    print("\nRépartition selon le nombre de canaux :")
    for canaux, nombre in sorted(distributions_canaux.items()):
        print(f"{canaux} canal(aux) : {nombre} images")

    print("\nNombre de dimensions distinctes :")
    print(len(distributions_dimensions))

    print("\nDimensions les plus fréquentes :")
    for dimensions, nombre in distributions_dimensions.most_common(10):
        largeur, hauteur = dimensions
        print(f"{largeur} x {hauteur} : {nombre} images")

    print("\nExemples de fichiers inspectés :")
    for exemple in exemples:
        print(
            f"{exemple['nom']} | "
            f"Classe : {exemple['classe']} | "
            f"{exemple['largeur']} x {exemple['hauteur']} | "
            f"{exemple['canaux']} canal(aux)"
        )

    print(f"\nNombre de fichiers avec problème : {len(problemes)}")
    for probleme in problemes[:20]:
        print(probleme)

    print("\nVérification terminée.")


if __name__ == "__main__":
    inspecter_canaux()
