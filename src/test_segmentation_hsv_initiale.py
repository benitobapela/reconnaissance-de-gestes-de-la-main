
from pathlib import Path
import csv
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

COLOR_DIR = PROJECT_DIR / "data" / "processed" / "color"
HSV_DIR = PROJECT_DIR / "data" / "processed" / "hsv"

MASK_DIR = PROJECT_DIR / "data" / "processed" / "masks_hsv_initials"
FIGURE_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"

FIGURE_FILE = FIGURE_DIR / "masques_hsv_initiaux.png"
REPORT_FILE = RESULTS_DIR / "evaluation_hsv_initiale.csv"

# Seuils utilisés lors de notre essai préliminaire.
LOWER_HSV = np.array([0, 20, 40], dtype=np.uint8)
UPPER_HSV = np.array([25, 255, 255], dtype=np.uint8)

RANDOM_SEED = 42


# ============================================================
# 2. RECHERCHE DES IMAGES HSV
# ============================================================

def selectionner_images():
    """
    Sélectionne une image HSV par classe, avec une graine fixe.
    Les fichiers HSV doivent être rangés par sous-dossier de classe.
    """
    if not HSV_DIR.exists():
        raise FileNotFoundError(
            f"Dossier HSV introuvable : {HSV_DIR}"
        )

    fichiers = sorted(HSV_DIR.rglob("*.npy"))

    if not fichiers:
        raise FileNotFoundError(
            "Aucun tableau HSV .npy trouvé."
        )

    images_par_classe = {}

    for chemin in fichiers:
        classe = chemin.parent.name
        images_par_classe.setdefault(classe, []).append(chemin)

    generateur = random.Random(RANDOM_SEED)
    selection = []

    for classe in sorted(images_par_classe):
        candidats = images_par_classe[classe]
        chemin = generateur.choice(candidats)
        selection.append((classe, chemin))

    return selection


# ============================================================
# 3. CHARGEMENT ET APPLICATION DES SEUILS
# ============================================================

def creer_masque(chemin_hsv):
    """
    Charge le tableau HSV puis crée le masque binaire initial.
    Aucun traitement morphologique n'est appliqué.
    """
    image_hsv = np.load(chemin_hsv)

    if image_hsv.dtype != np.uint8:
        raise ValueError(
            f"Type HSV inattendu dans {chemin_hsv}"
        )

    if image_hsv.ndim != 3 or image_hsv.shape[2] != 3:
        raise ValueError(
            f"Forme HSV invalide dans {chemin_hsv}: "
            f"{image_hsv.shape}"
        )

    masque = cv2.inRange(
        image_hsv,
        LOWER_HSV,
        UPPER_HSV
    )

    return image_hsv, masque


# ============================================================
# 4. SAUVEGARDE DU MASQUE ET CALCUL DES INDICATEURS
# ============================================================

def analyser_masque(classe, chemin_hsv, masque):
    """
    Sauvegarde le masque et calcule quelques indicateurs descriptifs.
    Ces indicateurs ne constituent pas une mesure de précision.
    """
    chemin_relatif = chemin_hsv.relative_to(HSV_DIR)
    chemin_masque = MASK_DIR / chemin_relatif.with_suffix(".png")

    chemin_masque.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not cv2.imwrite(str(chemin_masque), masque):
        raise IOError(
            f"Impossible d'enregistrer le masque : {chemin_masque}"
        )

    hauteur, largeur = masque.shape
    nombre_pixels = hauteur * largeur
    pixels_blancs = int(np.count_nonzero(masque))

    proportion_blanche = (
        100.0 * pixels_blancs / nombre_pixels
        if nombre_pixels > 0 else 0.0
    )

    contours, _ = cv2.findContours(
        masque.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    aire_max = (
        max(cv2.contourArea(c) for c in contours)
        if contours else 0.0
    )

    proportion_aire_max = (
        100.0 * aire_max / nombre_pixels
        if nombre_pixels > 0 else 0.0
    )

    return {
        "classe": classe,
        "fichier_hsv": str(chemin_relatif),
        "dimensions": f"{largeur}x{hauteur}",
        "pixels_blancs": pixels_blancs,
        "proportion_blanche_pct": round(
            proportion_blanche, 2
        ),
        "nombre_contours_externes": len(contours),
        "aire_plus_grand_contour_pct_image": round(
            proportion_aire_max, 2
        ),
        "fichier_masque": str(
            chemin_masque.relative_to(PROJECT_DIR)
        )
    }


# ============================================================
# 5. CREATION DE LA PLANCHE COMPARATIVE
# ============================================================

def creer_planche(resultats_visuels):
    """
    Produit une figure avec :
    image couleur, masque binaire et main isolée.
    """
    nombre = len(resultats_visuels)

    fig, axes = plt.subplots(
        nombre, 3,
        figsize=(12, 4 * nombre),
        squeeze=False
    )

    for i, element in enumerate(resultats_visuels):
        classe = element["classe"]
        image_bgr = element["image_bgr"]
        masque = element["masque"]

        # Appliquer le masque à l'image couleur.
        main_isolee = cv2.bitwise_and(
            image_bgr,
            image_bgr,
            mask=masque
        )

        image_rgb = cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2RGB
        )
        main_rgb = cv2.cvtColor(
            main_isolee, cv2.COLOR_BGR2RGB
        )

        axes[i, 0].imshow(image_rgb)
        axes[i, 0].set_title(f"{classe} — Image couleur")

        axes[i, 1].imshow(masque, cmap="gray", vmin=0, vmax=255)
        axes[i, 1].set_title("Masque HSV initial")

        axes[i, 2].imshow(main_rgb)
        axes[i, 2].set_title("Image isolée par le masque")

        for j in range(3):
            axes[i, j].axis("off")

    fig.suptitle(
        "Évaluation initiale de la segmentation HSV",
        fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_FILE,
        dpi=160,
        bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# 6. EXECUTION
# ============================================================

def main():
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    selection = selectionner_images()

    resultats_csv = []
    resultats_visuels = []

    print("Évaluation des seuils HSV initiaux")
    print(f"Seuil inférieur : {LOWER_HSV.tolist()}")
    print(f"Seuil supérieur : {UPPER_HSV.tolist()}")
    print(f"Nombre de classes : {len(selection)}\n")

    for classe, chemin_hsv in selection:
        try:
            image_hsv, masque = creer_masque(chemin_hsv)

            chemin_relatif = chemin_hsv.relative_to(HSV_DIR)
            chemin_couleur = (
                COLOR_DIR / chemin_relatif.with_suffix(".png")
            )

            if not chemin_couleur.exists():
                raise FileNotFoundError(
                    f"Image couleur correspondante introuvable : "
                    f"{chemin_couleur}"
                )

            image_bgr = cv2.imread(
                str(chemin_couleur),
                cv2.IMREAD_COLOR
            )

            if image_bgr is None:
                raise ValueError(
                    f"Impossible de lire {chemin_couleur}"
                )

            if image_bgr.shape[:2] != masque.shape:
                raise ValueError(
                    "Dimensions différentes entre l'image couleur "
                    "et le masque."
                )

            indicateurs = analyser_masque(
                classe,
                chemin_hsv,
                masque
            )

            resultats_csv.append(indicateurs)

            resultats_visuels.append({
                "classe": classe,
                "image_bgr": image_bgr,
                "masque": masque
            })

            print(
                f"[OK] {classe} : "
                f"{indicateurs['proportion_blanche_pct']} % "
                f"de pixels blancs, "
                f"{indicateurs['nombre_contours_externes']} "
                f"contour(s) externe(s)"
            )

        except Exception as erreur:
            print(f"[ERREUR] {classe} : {erreur}")

    if not resultats_visuels:
        raise RuntimeError(
            "Aucune image n'a pu être évaluée."
        )

    creer_planche(resultats_visuels)

    champs = list(resultats_csv[0].keys())

    with REPORT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as fichier_csv:
        redacteur = csv.DictWriter(
            fichier_csv,
            fieldnames=champs
        )
        redacteur.writeheader()
        redacteur.writerows(resultats_csv)

    print("\n========== BILAN ==========")
    print(f"Classes évaluées avec succès : {len(resultats_visuels)}")
    print(f"Planche comparative : {FIGURE_FILE}")
    print(f"Rapport CSV : {REPORT_FILE}")
    print(f"Masques individuels : {MASK_DIR}")


if __name__ == "__main__":
    main()
