
from pathlib import Path
from collections import defaultdict
import hashlib
import csv

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

DATASET_DIR = Path("data/raw")
OUTPUT_FILE = Path("results/duplicates_exact.csv")

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff", ".heic", ".heif"
}


def lire_image(chemin):
    """Décode l'image et retourne ses pixels en RGB."""

    if chemin.suffix.lower() in {".heic", ".heif"}:
        with Image.open(chemin) as image:
            image.load()
            return np.array(image.convert("RGB"))

    image = cv2.imread(str(chemin), cv2.IMREAD_COLOR)

    if image is None:
        return None

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def calculer_empreinte(image):
    """Calcule une empreinte à partir des dimensions et des pixels."""

    image = np.ascontiguousarray(image)

    contenu = (
        str(image.shape).encode("utf-8")
        + image.tobytes()
    )

    return hashlib.sha256(contenu).hexdigest()


def rechercher_doublons():
    groupes = defaultdict(list)
    total = 0
    erreurs = []

    fichiers = sorted(
        chemin
        for chemin in DATASET_DIR.rglob("*")
        if chemin.is_file()
        and chemin.suffix.lower() in EXTENSIONS
    )

    print("Analyse des doublons exacts en cours...")

    for chemin in fichiers:
        total += 1

        try:
            image = lire_image(chemin)

            if image is None or image.size == 0:
                erreurs.append(str(chemin))
                continue

            empreinte = calculer_empreinte(image)

            groupes[empreinte].append({
                "chemin": str(chemin),
                "classe": chemin.parent.name,
                "nom": chemin.name,
                "empreinte": empreinte,
            })

        except Exception as erreur:
            erreurs.append(f"{chemin} — {erreur}")

    doublons = [
        groupe
        for groupe in groupes.values()
        if len(groupe) > 1
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as fichier_csv:

        colonnes = [
            "empreinte",
            "nom",
            "classe",
            "chemin",
            "taille_groupe",
        ]

        writer = csv.DictWriter(
            fichier_csv,
            fieldnames=colonnes
        )
        writer.writeheader()

        for groupe in doublons:
            for image in groupe:
                writer.writerow({
                    **image,
                    "taille_groupe": len(groupe),
                })

    nombre_groupes = len(doublons)
    nombre_images_concernees = sum(
        len(groupe) for groupe in doublons
    )

    print("\n" + "=" * 60)
    print("RÉSULTAT DE LA RECHERCHE DE DOUBLONS EXACTS")
    print("=" * 60)

    print(f"Fichiers examinés : {total}")
    print(f"Empreintes distinctes : {len(groupes)}")
    print(f"Groupes de doublons : {nombre_groupes}")
    print(f"Images appartenant à ces groupes : {nombre_images_concernees}")
    print(f"Fichiers non analysés : {len(erreurs)}")

    for numero, groupe in enumerate(doublons, start=1):
        print(f"\nGroupe {numero} ({len(groupe)} images) :")

        for image in groupe:
            print(f"  [{image['classe']}] {image['chemin']}")

    if erreurs:
        print("\nFichiers non analysés :")
        for erreur in erreurs[:20]:
            print(f"  {erreur}")

    print(f"\nRapport enregistré : {OUTPUT_FILE}")
    print("Analyse terminée. Aucun fichier original n'a été modifié.")


if __name__ == "__main__":
    rechercher_doublons()
