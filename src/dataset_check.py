
from pathlib import Path
from collections import Counter
import cv2


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("data/raw")

# Les noms doivent correspondre aux dossiers réels.
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

# Les formats courants lus directement par OpenCV.
OPENCV_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


# ============================================================
# CONTRÔLE DU DATASET
# ============================================================

def check_dataset():

    print("=" * 60)
    print("CONTRÔLE DU DATASET HANDGESTURE-5G")
    print("=" * 60)

    if not DATASET_DIR.is_dir():
        print(f"\nERREUR : dossier introuvable : {DATASET_DIR.resolve()}")
        return

    total_files = 0
    total_images = 0
    total_readable = 0
    total_unreadable = 0

    extensions = Counter()
    unreadable_files = []
    unsupported_images = []
    missing_classes = []

    print(f"\nDossier : {DATASET_DIR.resolve()}")
    print("\nNOMBRE DE FICHIERS PAR CLASSE")
    print("-" * 45)

    for class_name in CLASSES:

        class_dir = DATASET_DIR / class_name

        if not class_dir.is_dir():
            print(f"[CLASSE ABSENTE] {class_name}")
            missing_classes.append(class_name)
            continue

        # Recherche les fichiers dans le dossier et ses sous-dossiers.
        files = sorted(
            file for file in class_dir.rglob("*")
            if file.is_file()
        )

        image_files = [
            file for file in files
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"{class_name:25s}: {len(image_files)} images")

        total_files += len(files)
        total_images += len(image_files)

        for index, file in enumerate(image_files, start=1):

            extension = file.suffix.lower()
            extensions[extension] += 1

            # OpenCV ne lit pas toujours le format HEIC.
            if extension not in OPENCV_EXTENSIONS:
                unsupported_images.append(str(file))
                continue

            image = cv2.imread(str(file))

            if image is None:
                total_unreadable += 1
                unreadable_files.append(str(file))
            else:
                total_readable += 1

            # Affichage de progression tous les 100 fichiers.
            if index % 100 == 0:
                print(
                    f"  Vérification de {class_name} : "
                    f"{index}/{len(image_files)}"
                )

    # ========================================================
    # RÉSULTATS
    # ========================================================

    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)

    print(f"Total de fichiers dans les classes : {total_files}")
    print(f"Total d'images reconnues            : {total_images}")
    print(f"Images lisibles par OpenCV          : {total_readable}")
    print(f"Images illisibles par OpenCV        : {total_unreadable}")
    print(f"Images à vérifier avec un autre lecteur : "
          f"{len(unsupported_images)}")

    print("\nEXTENSIONS RENCONTRÉES")
    print("-" * 45)

    for extension, count in sorted(extensions.items()):
        print(f"{extension or '[sans extension]':12s}: {count}")

    if missing_classes:
        print("\nDOSSIERS DE CLASSE ABSENTS")
        for name in missing_classes:
            print(f"- {name}")

    if unreadable_files:
        print("\nFICHIERS ILLISIBLES")
        for file in unreadable_files:
            print(f"- {file}")

    if unsupported_images:
        print("\nIMAGES NON TESTÉES PAR OPENCV")
        for file in unsupported_images[:10]:
            print(f"- {file}")

        if len(unsupported_images) > 10:
            print(f"... et {len(unsupported_images) - 10} autres fichiers")

    print("\n" + "=" * 60)
    print("CONTRÔLE TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    check_dataset()
