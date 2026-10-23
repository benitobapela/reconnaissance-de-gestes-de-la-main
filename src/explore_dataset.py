
from pathlib import Path
from collections import Counter

import cv2
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("data/raw")

CLASSES = [
    "Open Palm_Open Hand",
    "Fist",
    "Pointing Up",
    "L-Sign",
    "V-Sign",
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff", ".heic", ".heif"
}


def explore_dataset():

    records = []

    print("=" * 60)
    print("ANALYSE EXPLORATOIRE DU DATASET")
    print("=" * 60)

    for class_name in CLASSES:

        class_dir = DATASET_DIR / class_name

        if not class_dir.is_dir():
            print(f"[ERREUR] Classe absente : {class_name}")
            continue

        files = sorted(
            file for file in class_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"\n{class_name} : {len(files)} images")

        for file in files:

            try:
                if file.suffix.lower() in {".heic", ".heif"}:
                    # Pillow est utilisé pour lire les HEIC.
                    from PIL import Image
                    from pillow_heif import register_heif_opener

                    register_heif_opener()

                    with Image.open(file) as image:
                        width, height = image.size
                else:
                    image = cv2.imread(str(file))

                    if image is None:
                        print(f"[ILLISIBLE] {file}")
                        continue

                    height, width = image.shape[:2]

                records.append({
                    "image": file.name,
                    "chemin": str(file),
                    "classe": class_name,
                    "largeur": width,
                    "hauteur": height,
                    "format": file.suffix.lower(),
                })

            except Exception as error:
                print(f"[ERREUR] {file}: {error}")

    if not records:
        print("\nAucune image n'a pu être analysée.")
        return

    df = pd.DataFrame(records)

    print("\n" + "=" * 60)
    print("RÉSUMÉ DES DIMENSIONS")
    print("=" * 60)

    print(f"Images analysées : {len(df)}")
    print(f"Largeur minimale : {df['largeur'].min()} px")
    print(f"Largeur maximale : {df['largeur'].max()} px")
    print(f"Hauteur minimale : {df['hauteur'].min()} px")
    print(f"Hauteur maximale : {df['hauteur'].max()} px")

    print("\nDimensions les plus fréquentes :")
    dimensions = Counter(
        zip(df["largeur"], df["hauteur"])
    )

    for (width, height), count in dimensions.most_common(10):
        print(f"{width} x {height} : {count} images")

    print("\nRépartition par classe :")
    print(df["classe"].value_counts().to_string())

    # Sauvegarde du tableau descriptif.
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "dataset_dimensions.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Figure des effectifs.
    counts = df["classe"].value_counts().reindex(CLASSES)

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("Nombre d'images par classe")
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'images")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    figure_path = Path("figures") / "distribution_classes.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"\nTableau sauvegardé : {csv_path}")
    print(f"Figure sauvegardée : {figure_path}")
    print("\nAnalyse terminée.")


if __name__ == "__main__":
    explore_dataset()
