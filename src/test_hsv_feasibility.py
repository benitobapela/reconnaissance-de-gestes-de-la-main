
from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import pillow_heif


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DOSSIER_DATASET = Path("data/raw")
DOSSIER_FIGURES = Path("figures")

DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)

random.seed(42)

CLASSES = [
    "Fist",
    "L-Sign",
    "Open Palm_Open Hand",
    "Pointing Up",
    "V-Sign",
]

# Premier seuil exploratoire HSV.
# Attention : les valeurs sont celles attendues par OpenCV :
# H entre 0 et 179 ; S et V entre 0 et 255.
SEUIL_BAS = np.array([0, 20, 40], dtype=np.uint8)
SEUIL_HAUT = np.array([25, 255, 255], dtype=np.uint8)


# --------------------------------------------------
# 2. Lecture des images, y compris les fichiers HEIC
# --------------------------------------------------

pillow_heif.register_heif_opener()


def lire_image(chemin):
    extension = chemin.suffix.lower()

    if extension == ".heic":
        # Pillow lit le HEIC puis fournit une image RGB.
        image_rgb = np.array(
            Image.open(chemin).convert("RGB")
        )
        image_bgr = cv2.cvtColor(
            image_rgb, cv2.COLOR_RGB2BGR
        )
    else:
        image_bgr = cv2.imread(str(chemin))

    if image_bgr is None:
        raise ValueError(
            f"Impossible de lire l'image : {chemin}"
        )

    return image_bgr


# --------------------------------------------------
# 3. Sélection d'une image par classe
# --------------------------------------------------

echantillon = []

extensions = {".jpg", ".jpeg", ".png", ".heic"}

for nom_classe in CLASSES:
    dossier_classe = DOSSIER_DATASET / nom_classe

    fichiers = [
        f for f in dossier_classe.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]

    if not fichiers:
        print(f"Aucune image trouvée dans {dossier_classe}")
        continue

    chemin = random.choice(fichiers)
    echantillon.append((nom_classe, chemin))

print(f"Nombre d'images sélectionnées : {len(echantillon)}")

for classe, chemin in echantillon:
    print(f"{classe} : {chemin.name}")


# --------------------------------------------------
# 4. Conversion HSV et création du masque
# --------------------------------------------------

fig, axes = plt.subplots(
    len(echantillon), 3,
    figsize=(12, 4 * len(echantillon))
)

# Évite les problèmes lorsque l'échantillon ne contient
# qu'une seule ligne.
axes = np.atleast_2d(axes)

for i, (classe, chemin) in enumerate(echantillon):

    image_bgr = lire_image(chemin)

    # Réduction uniquement pour accélérer le test.
    # Cela ne modifie pas les fichiers du dataset.
    hauteur, largeur = image_bgr.shape[:2]
    facteur = min(512 / largeur, 512 / hauteur, 1.0)

    nouvelle_taille = (
        int(largeur * facteur),
        int(hauteur * facteur)
    )

    image_bgr = cv2.resize(
        image_bgr,
        nouvelle_taille,
        interpolation=cv2.INTER_AREA
    )

    # OpenCV attend une image BGR en entrée.
    image_hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    masque = cv2.inRange(
        image_hsv,
        SEUIL_BAS,
        SEUIL_HAUT
    )

    # Appliquer le masque à l'image couleur.
    main_isolee = cv2.bitwise_and(
        image_bgr,
        image_bgr,
        mask=masque
    )

    # OpenCV utilise BGR ; Matplotlib attend RGB.
    image_rgb = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2RGB
    )
    main_rgb = cv2.cvtColor(
        main_isolee, cv2.COLOR_BGR2RGB
    )

    axes[i, 0].imshow(image_rgb)
    axes[i, 0].set_title(f"{classe} — Originale")

    axes[i, 1].imshow(masque, cmap="gray", vmin=0, vmax=255)
    axes[i, 1].set_title("Masque HSV")

    axes[i, 2].imshow(main_rgb)
    axes[i, 2].set_title("Main isolée")

    for j in range(3):
        axes[i, j].axis("off")

plt.suptitle(
    "Test préliminaire de faisabilité de la segmentation HSV",
    fontsize=14
)
plt.tight_layout()

fichier_sortie = DOSSIER_FIGURES / "test_faisabilite_hsv.png"
plt.savefig(fichier_sortie, dpi=150, bbox_inches="tight")
plt.show()

print(f"\nRésultat enregistré dans : {fichier_sortie}")
print("Test terminé.")
