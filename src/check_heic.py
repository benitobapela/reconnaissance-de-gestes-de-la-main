
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

# Permet à Pillow d'ouvrir les images HEIC/HEIF.
register_heif_opener()

HEIC_DIR = Path("data/raw/V-Sign")

files = sorted(
    file for file in HEIC_DIR.iterdir()
    if file.is_file()
    and file.suffix.lower() in {".heic", ".heif"}
)

print("=" * 55)
print("VERIFICATION DES IMAGES HEIC")
print("=" * 55)

print(f"\nImages HEIC/HEIF trouvees : {len(files)}")

valid = 0
invalid = []

for file in files:
    try:
        with Image.open(file) as image:
            image.verify()

        # Réouverture pour lire les dimensions.
        with Image.open(file) as image:
            width, height = image.size

        print(f"[OK] {file.name} - {width} x {height}")
        valid += 1

    except Exception as error:
        print(f"[ERREUR] {file.name} - {error}")
        invalid.append(file.name)

print("\n" + "=" * 55)
print(f"Images HEIC valides : {valid}")
print(f"Images HEIC invalides : {len(invalid)}")
print("=" * 55)
