from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CHEMINS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

HSV_DIR = PROJECT_DIR / "data" / "processed" / "hsv"
MASK_MORPH_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "masks_hsv_morphologie"
)

FIGURE_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"

FIGURE_FILE = (
    FIGURE_DIR
    / "selection_contour_principal.png"
)

REPORT_FILE = (
    RESULTS_DIR
    / "evaluation_contour_principal.csv"
)

RANDOM_SEED = 42

CONFIGURATION_MORPHO = "C_close5_open3"


# ============================================================
# 2. SÉLECTION DES MÊMES IMAGES QUE PRÉCÉDEMMENT
# ============================================================

def selectionner_images():

    fichiers = sorted(
        HSV_DIR.rglob("*.npy")
    )

    groupes = {}

    for fichier in fichiers:
        classe = fichier.parent.name
        groupes.setdefault(
            classe,
            []
        ).append(fichier)

    rng = random.Random(
        RANDOM_SEED
    )

    selections = {}

    for classe in sorted(groupes):
        selections[classe] = rng.choice(
            groupes[classe]
        )

    return selections


# ============================================================
# 3. CHARGEMENT DU MASQUE MORPHOLOGIQUE
# ============================================================

def charger_masque(classe, hsv_path):

    nom_masque = hsv_path.with_suffix(
        ".png"
    ).name

    chemin = (
        MASK_MORPH_DIR
        / CONFIGURATION_MORPHO
        / classe
        / nom_masque
    )

    masque = cv2.imread(
        str(chemin),
        cv2.IMREAD_GRAYSCALE
    )

    if masque is None:
        raise RuntimeError(
            f"Masque introuvable : {chemin}"
        )

    return masque, chemin


# ============================================================
# 4. DÉTECTION DES CONTOURS
# ============================================================

def detecter_contours(masque):

    contours, _ = cv2.findContours(
        masque,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    return contours


# ============================================================
# 5. MESURE D'UN CONTOUR
# ============================================================

def mesurer_contour(
    contour,
    largeur_image,
    hauteur_image
):

    aire = cv2.contourArea(
        contour
    )

    perimetre = cv2.arcLength(
        contour,
        True
    )

    x, y, largeur, hauteur = (
        cv2.boundingRect(contour)
    )

    aire_image = (
        largeur_image
        * hauteur_image
    )

    aire_pct = (
        100.0 * aire / aire_image
        if aire_image > 0
        else 0.0
    )

    centre_x = x + largeur / 2
    centre_y = y + hauteur / 2

    return {
        "aire": aire,
        "aire_pct_image": aire_pct,
        "perimetre": perimetre,
        "x": x,
        "y": y,
        "largeur": largeur,
        "hauteur": hauteur,
        "centre_x": centre_x,
        "centre_y": centre_y,
    }


# ============================================================
# 6. DESSIN DES CONTOURS
# ============================================================

def creer_image_contours(
    masque,
    contours
):

    hauteur, largeur = masque.shape

    image = cv2.cvtColor(
        masque,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # Tous les contours
    # --------------------------------------------------------

    cv2.drawContours(
        image,
        contours,
        -1,
        (150, 150, 150),
        1
    )

    # --------------------------------------------------------
    # Contour principal
    # --------------------------------------------------------

    if contours:

        principal = contours[0]

        cv2.drawContours(
            image,
            [principal],
            -1,
            (255, 255, 255),
            2
        )

        x, y, w, h = cv2.boundingRect(
            principal
        )

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            1
        )

    return image


# ============================================================
# 7. PLANCHE COMPARATIVE
# ============================================================

def creer_planche(resultats):

    classes = sorted(
        resultats.keys()
    )

    fig, axes = plt.subplots(
        len(classes),
        3,
        figsize=(12, 15)
    )

    if len(classes) == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    for ligne, classe in enumerate(classes):

        element = resultats[classe]

        masque = element["masque"]

        contours = element["contours"]

        principal = (
            contours[0]
            if contours
            else None
        )

        # ----------------------------------------------------
        # Masque
        # ----------------------------------------------------

        axes[ligne, 0].imshow(
            masque,
            cmap="gray"
        )

        axes[ligne, 0].set_title(
            f"{classe}\nMasque C"
        )

        axes[ligne, 0].axis("off")

        # ----------------------------------------------------
        # Tous les contours
        # ----------------------------------------------------

        image_tous = cv2.cvtColor(
            masque,
            cv2.COLOR_GRAY2RGB
        )

        for contour in contours:

            cv2.drawContours(
                image_tous,
                [contour],
                -1,
                (150, 150, 150),
                1
            )

        axes[ligne, 1].imshow(
            image_tous
        )

        axes[ligne, 1].set_title(
            f"{len(contours)} contour(s)"
        )

        axes[ligne, 1].axis("off")

        # ----------------------------------------------------
        # Contour principal
        # ----------------------------------------------------

        image_principal = cv2.cvtColor(
            masque,
            cv2.COLOR_GRAY2RGB
        )

        if principal is not None:

            cv2.drawContours(
                image_principal,
                [principal],
                -1,
                (255, 255, 255),
                2
            )

            x, y, w, h = (
                cv2.boundingRect(
                    principal
                )
            )

            cv2.rectangle(
                image_principal,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                1
            )

        axes[ligne, 2].imshow(
            image_principal
        )

        axes[ligne, 2].set_title(
            "Contour principal"
        )

        axes[ligne, 2].axis("off")

    plt.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        FIGURE_FILE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print(
        "DÉTECTION ET SÉLECTION DU CONTOUR PRINCIPAL"
    )
    print("=" * 70)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    selections = selectionner_images()

    resultats = {}

    lignes = []

    for classe, hsv_path in selections.items():

        print(
            f"\nClasse : {classe}"
        )

        masque, chemin_masque = (
            charger_masque(
                classe,
                hsv_path
            )
        )

        contours = detecter_contours(
            masque
        )

        hauteur, largeur = (
            masque.shape
        )

        print(
            f"  Nombre de contours : "
            f"{len(contours)}"
        )

        # ----------------------------------------------------
        # Mesure de tous les contours
        # ----------------------------------------------------

        for rang, contour in enumerate(
            contours,
            start=1
        ):

            mesures = mesurer_contour(
                contour,
                largeur,
                hauteur
            )

            print(
                f"  Contour {rang} : "
                f"aire={mesures['aire']:.1f} px², "
                f"aire_image="
                f"{mesures['aire_pct_image']:.3f} %"
            )

            lignes.append({
                "classe": classe,
                "rang_contour": rang,
                "configuration": (
                    CONFIGURATION_MORPHO
                ),
                "aire_px2": round(
                    mesures["aire"],
                    3
                ),
                "aire_pct_image": round(
                    mesures[
                        "aire_pct_image"
                    ],
                    3
                ),
                "perimetre_px": round(
                    mesures["perimetre"],
                    3
                ),
                "x": mesures["x"],
                "y": mesures["y"],
                "largeur": mesures[
                    "largeur"
                ],
                "hauteur": mesures[
                    "hauteur"
                ],
                "centre_x": round(
                    mesures["centre_x"],
                    3
                ),
                "centre_y": round(
                    mesures["centre_y"],
                    3
                ),
                "fichier_masque": str(
                    chemin_masque.relative_to(
                        PROJECT_DIR
                    )
                ),
            })

        resultats[classe] = {
            "masque": masque,
            "contours": contours,
        }

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    dataframe = pd.DataFrame(
        lignes
    )

    dataframe.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    creer_planche(
        resultats
    )

    print("\n" + "=" * 70)
    print("BILAN")
    print("=" * 70)

    print(
        f"Rapport : {REPORT_FILE}"
    )

    print(
        f"Figure : {FIGURE_FILE}"
    )

    print("\nRésultats :")
    print(
        dataframe.to_string(
            index=False
        )
    )

    print(
        "\nLe contour principal correspond "
        "au contour de plus grande aire."
    )

    print(
        "Vérification visuelle nécessaire "
        "avant validation définitive."
    )


if __name__ == "__main__":
    main()