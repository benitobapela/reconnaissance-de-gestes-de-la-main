
from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

DATASET_DIR = Path("data/raw")

CLASSES = [
    "Open Palm_Open Hand",
    "Fist",
    "Pointing Up",
    "L-Sign",
    "V-Sign",
]

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff", ".heic", ".heif"
}

NOMBRE_IMAGES = 5
GRAINE = 42


def lire_image(chemin):
    """Lit une image et la retourne en couleurs RGB."""

    if chemin.suffix.lower() in {".heic", ".heif"}:
        with Image.open(chemin) as image:
            return image.convert("RGB").copy()

    image = cv2.imread(str(chemin))

    if image is None:
        return None

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def visualiser_echantillons():
    generateur = random.Random(GRAINE)

    fig, axes = plt.subplots(
        len(CLASSES),
        NOMBRE_IMAGES,
        figsize=(12, 18)
    )

    for ligne, classe in enumerate(CLASSES):
        dossier = DATASET_DIR / classe

        fichiers = sorted(
            fichier
            for fichier in dossier.rglob("*")
            if fichier.is_file()
            and fichier.suffix.lower() in EXTENSIONS
        )

        echantillon = generateur.sample(
            fichiers,
            min(NOMBRE_IMAGES, len(fichiers))
        )

        for colonne in range(NOMBRE_IMAGES):
            axe = axes[ligne, colonne]
            axe.axis("off")

            if colonne >= len(echantillon):
                continue

            chemin = echantillon[colonne]
            image = lire_image(chemin)

            if image is None:
                axe.set_title("Image illisible")
                continue

            axe.imshow(image)
            axe.set_title(chemin.name, fontsize=8)

            if colonne == 0:
                axe.set_ylabel(
                    classe,
                    fontsize=10,
                    rotation=90,
                    labelpad=15
                )

    fig.suptitle(
        "Exemples d'images du dataset HandGesture-5G",
        fontsize=14
    )

    plt.tight_layout()

    sortie = Path("figures") / "echantillons_dataset.png"
    sortie.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(sortie, dpi=120, bbox_inches="tight")
    print(f"Figure enregistrée : {sortie}")

    plt.show()


if __name__ == "__main__":
    visualiser_echantillons()
