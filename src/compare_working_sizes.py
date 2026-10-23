
from pathlib import Path
import random
import time

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
DOSSIER_RESULTATS = Path("results")

DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Fist",
    "L-Sign",
    "Open Palm_Open Hand",
    "Pointing Up",
    "V-Sign",
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}
TAILLES = [256, 512]

random.seed(42)
pillow_heif.register_heif_opener()


# --------------------------------------------------
# 2. Lecture des images
# --------------------------------------------------

def lire_image(chemin):
    if chemin.suffix.lower() == ".heic":
        image_rgb = np.array(
            Image.open(chemin).convert("RGB")
        )
        image_bgr = cv2.cvtColor(
            image_rgb, cv2.COLOR_RGB2BGR
        )
    else:
        image_bgr = cv2.imread(str(chemin))

    if image_bgr is None:
        raise ValueError(f"Image illisible : {chemin}")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"Image ne possédant pas trois canaux : {chemin}"
        )

    return image_bgr


# --------------------------------------------------
# 3. Redimensionnement en conservant les proportions
# --------------------------------------------------

def redimensionner(image, cote_max):
    hauteur, largeur = image.shape[:2]

    facteur = min(
        cote_max / largeur,
        cote_max / hauteur,
        1.0
    )

    nouvelle_largeur = max(1, round(largeur * facteur))
    nouvelle_hauteur = max(1, round(hauteur * facteur))

    image_redimensionnee = cv2.resize(
        image,
        (nouvelle_largeur, nouvelle_hauteur),
        interpolation=cv2.INTER_AREA
    )

    return image_redimensionnee


# --------------------------------------------------
# 4. Sélection d'une image par classe
# --------------------------------------------------

echantillon = []

for classe in CLASSES:
    dossier = DOSSIER_DATASET / classe

    fichiers = sorted([
        f for f in dossier.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    ])

    if not fichiers:
        print(f"Attention : aucune image trouvée dans {dossier}")
        continue

    chemin = random.choice(fichiers)
    echantillon.append((classe, chemin))

print(f"Images sélectionnées : {len(echantillon)}")


# --------------------------------------------------
# 5. Comparaison visuelle et mesure du temps
# --------------------------------------------------

fig, axes = plt.subplots(
    len(echantillon),
    3,
    figsize=(12, 4 * len(echantillon))
)

axes = np.atleast_2d(axes)
mesures = []

for i, (classe, chemin) in enumerate(echantillon):
    image = lire_image(chemin)

    hauteur_orig, largeur_orig = image.shape[:2]

    # La colonne originale est affichée à une taille
    # réduite pour faciliter la comparaison visuelle.
    # Le fichier original n'est jamais modifié.
    apercu = redimensionner(image, 512)

    images_a_afficher = [
        cv2.cvtColor(apercu, cv2.COLOR_BGR2RGB)
    ]

    mesures.append({
        "classe": classe,
        "fichier": chemin.name,
        "largeur_originale": largeur_orig,
        "hauteur_originale": hauteur_orig,
    })

    for taille in TAILLES:
        image_red = redimensionner(image, taille)

        # Répétitions pour obtenir un temps moyen indicatif.
        repetitions = 10

        debut = time.perf_counter()

        for _ in range(repetitions):
            hsv = cv2.cvtColor(
                image_red, cv2.COLOR_BGR2HSV
            )

        duree_ms = (
            (time.perf_counter() - debut)
            / repetitions
            * 1000
        )

        mesures[-1][f"dimensions_{taille}"] = (
            f"{image_red.shape[1]}x{image_red.shape[0]}"
        )
        mesures[-1][f"conversion_hsv_ms_{taille}"] = round(
            duree_ms, 4
        )

        images_a_afficher.append(
            cv2.cvtColor(image_red, cv2.COLOR_BGR2RGB)
        )

    titres = [
        f"{classe}\nAperçu de l'original",
        f"{classe}\nTaille maximale : 256 px",
        f"{classe}\nTaille maximale : 512 px",
    ]

    for j, image_rgb in enumerate(images_a_afficher):
        axes[i, j].imshow(image_rgb)
        axes[i, j].set_title(titres[j])
        axes[i, j].axis("off")

    print(f"\nClasse : {classe}")
    print(f"Fichier : {chemin.name}")
    print(f"Dimensions originales : {largeur_orig}x{hauteur_orig}")

    for taille in TAILLES:
        print(
            f"Taille max {taille} : "
            f"{mesures[-1][f'dimensions_{taille}']} | "
            f"conversion HSV moyenne : "
            f"{mesures[-1][f'conversion_hsv_ms_{taille}']} ms"
        )

plt.suptitle(
    "Comparaison des tailles de travail pour le prétraitement",
    fontsize=14
)
plt.tight_layout()

fichier_figure = (
    DOSSIER_FIGURES / "comparaison_tailles_pretraitement.png"
)

plt.savefig(fichier_figure, dpi=150, bbox_inches="tight")
plt.show()

import csv

fichier_csv = DOSSIER_RESULTATS / "comparaison_tailles.csv"

with fichier_csv.open(
    "w", newline="", encoding="utf-8-sig"
) as fichier:
    champs = list(mesures[0].keys())
    writer = csv.DictWriter(fichier, fieldnames=champs)
    writer.writeheader()
    writer.writerows(mesures)

print("\nComparaison terminée.")
print(f"Planche enregistrée : {fichier_figure}")
print(f"Mesures enregistrées : {fichier_csv}")
