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

COLOR_DIR = PROJECT_DIR / "data" / "processed" / "color"
HSV_DIR = PROJECT_DIR / "data" / "processed" / "hsv"
MASK_INITIAL_DIR = PROJECT_DIR / "data" / "processed" / "masks_hsv_initials"

MASK_MORPH_DIR = PROJECT_DIR / "data" / "processed" / "masks_hsv_morphologie"

FIGURE_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"

FIGURE_FILE = FIGURE_DIR / "comparaison_morphologie_hsv.png"
REPORT_FILE = RESULTS_DIR / "evaluation_morphologie_hsv.csv"


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_SEED = 42

CONFIGURATIONS = {
    "A_close_3x3": {
        "close_kernel": 3,
        "close_iterations": 1,
        "open_kernel": None,
        "open_iterations": 0,
    },
    "B_close_5x5": {
        "close_kernel": 5,
        "close_iterations": 1,
        "open_kernel": None,
        "open_iterations": 0,
    },
    "C_close5_open3": {
        "close_kernel": 5,
        "close_iterations": 1,
        "open_kernel": 3,
        "open_iterations": 1,
    },
    "D_close7_open3": {
        "close_kernel": 7,
        "close_iterations": 1,
        "open_kernel": 3,
        "open_iterations": 1,
    },
}


# ============================================================
# 3. SÉLECTION DES IMAGES
# ============================================================

def selectionner_images():
    """
    Sélectionne exactement une image par classe.
    Même graine que l'étape 5.1 pour permettre une comparaison directe.
    """

    fichiers = sorted(HSV_DIR.rglob("*.npy"))

    groupes = {}

    for fichier in fichiers:
        classe = fichier.parent.name
        groupes.setdefault(classe, []).append(fichier)

    rng = random.Random(RANDOM_SEED)

    selections = {}

    for classe in sorted(groupes):
        selections[classe] = rng.choice(groupes[classe])

    return selections


# ============================================================
# 4. APPLICATION DE LA MORPHOLOGIE
# ============================================================

def appliquer_morphologie(mask, configuration):
    """
    Applique fermeture puis éventuellement ouverture.
    """

    params = CONFIGURATIONS[configuration]

    resultat = mask.copy()

    # --------------------------------------------------------
    # Fermeture
    # --------------------------------------------------------

    close_kernel_size = params["close_kernel"]

    if close_kernel_size is not None:
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel_size, close_kernel_size)
        )

        resultat = cv2.morphologyEx(
            resultat,
            cv2.MORPH_CLOSE,
            kernel_close,
            iterations=params["close_iterations"]
        )

    # --------------------------------------------------------
    # Ouverture
    # --------------------------------------------------------

    open_kernel_size = params["open_kernel"]

    if open_kernel_size is not None:
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_kernel_size, open_kernel_size)
        )

        resultat = cv2.morphologyEx(
            resultat,
            cv2.MORPH_OPEN,
            kernel_open,
            iterations=params["open_iterations"]
        )

    return resultat


# ============================================================
# 5. MESURES DU MASQUE
# ============================================================

def analyser_masque(mask):

    hauteur, largeur = mask.shape

    pixels_blancs = int(np.count_nonzero(mask))
    total_pixels = hauteur * largeur

    proportion_blanche = (
        100.0 * pixels_blancs / total_pixels
        if total_pixels > 0
        else 0.0
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    nombre_contours = len(contours)

    if contours:
        plus_grand = max(contours, key=cv2.contourArea)
        aire_plus_grand = cv2.contourArea(plus_grand)

        aire_plus_grand_pct = (
            100.0 * aire_plus_grand / total_pixels
        )
    else:
        aire_plus_grand_pct = 0.0

    return {
        "dimensions": f"{largeur}x{hauteur}",
        "pixels_blancs": pixels_blancs,
        "proportion_blanche_pct": round(proportion_blanche, 3),
        "nombre_contours_externes": nombre_contours,
        "aire_plus_grand_contour_pct_image": round(
            aire_plus_grand_pct,
            3
        ),
    }


# ============================================================
# 6. SAUVEGARDE DES MASQUES
# ============================================================

def sauvegarder_masque(mask, classe, configuration, nom_fichier):

    dossier = MASK_MORPH_DIR / configuration / classe
    dossier.mkdir(parents=True, exist_ok=True)

    chemin = dossier / nom_fichier

    succes = cv2.imwrite(str(chemin), mask)

    if not succes:
        raise RuntimeError(
            f"Impossible d'enregistrer le masque : {chemin}"
        )

    return chemin


# ============================================================
# 7. PLANche COMPARATIVE
# ============================================================

def creer_planche(images):

    classes = sorted(images.keys())

    # 5 lignes :
    # une par classe
    #
    # 6 colonnes :
    # couleur
    # initial
    # A
    # B
    # C
    # D

    fig, axes = plt.subplots(
        len(classes),
        6,
        figsize=(18, 15)
    )

    if len(classes) == 1:
        axes = np.expand_dims(axes, axis=0)

    configurations = list(CONFIGURATIONS.keys())

    for ligne, classe in enumerate(classes):

        hsv_path = images[classe]

        # ----------------------------------------------------
        # Image couleur correspondante
        # ----------------------------------------------------

        relatif = hsv_path.relative_to(HSV_DIR)
        couleur_path = COLOR_DIR / relatif.with_suffix(".png")

        image = cv2.imread(str(couleur_path))

        if image is None:
            raise RuntimeError(
                f"Image couleur introuvable : {couleur_path}"
            )

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Masque initial
        # ----------------------------------------------------

        masque_initial_path = (
            MASK_INITIAL_DIR
            / classe
            / hsv_path.with_suffix(".png").name
        )

        masque_initial = cv2.imread(
            str(masque_initial_path),
            cv2.IMREAD_GRAYSCALE
        )

        if masque_initial is None:
            raise RuntimeError(
                f"Masque initial introuvable : "
                f"{masque_initial_path}"
            )

        # ----------------------------------------------------
        # Affichage couleur
        # ----------------------------------------------------

        axes[ligne, 0].imshow(image_rgb)
        axes[ligne, 0].set_title(
            f"{classe}\nOriginal"
        )

        axes[ligne, 0].axis("off")

        # ----------------------------------------------------
        # Affichage masque initial
        # ----------------------------------------------------

        axes[ligne, 1].imshow(
            masque_initial,
            cmap="gray"
        )

        axes[ligne, 1].set_title(
            "Initial"
        )

        axes[ligne, 1].axis("off")

        # ----------------------------------------------------
        # Configurations morphologiques
        # ----------------------------------------------------

        for colonne, configuration in enumerate(
            configurations,
            start=2
        ):

            masque_morph = appliquer_morphologie(
                masque_initial,
                configuration
            )

            axes[ligne, colonne].imshow(
                masque_morph,
                cmap="gray"
            )

            axes[ligne, colonne].set_title(
                configuration.replace("_", "\n")
            )

            axes[ligne, colonne].axis("off")

    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

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
    print("TEST DE MORPHOLOGIE SUR LES MASQUES HSV")
    print("=" * 70)

    MASK_MORPH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    selections = selectionner_images()

    print("\nImages sélectionnées :")

    for classe, fichier in selections.items():
        print(f"  {classe} : {fichier.name}")

    lignes = []

    # --------------------------------------------------------
    # Traitement
    # --------------------------------------------------------

    for classe, hsv_path in selections.items():

        masque_initial_path = (
            MASK_INITIAL_DIR
            / classe
            / hsv_path.with_suffix(".png").name
        )

        masque_initial = cv2.imread(
            str(masque_initial_path),
            cv2.IMREAD_GRAYSCALE
        )

        if masque_initial is None:
            print(
                f"[ERREUR] Masque initial absent : "
                f"{masque_initial_path}"
            )
            continue

        # Mesure du masque initial
        mesures_initiales = analyser_masque(
            masque_initial
        )

        lignes.append({
            "classe": classe,
            "configuration": "Initial",
            **mesures_initiales,
        })

        # Configurations morphologiques
        for configuration in CONFIGURATIONS:

            masque_morph = appliquer_morphologie(
                masque_initial,
                configuration
            )

            nom_fichier = hsv_path.with_suffix(".png").name

            chemin_sortie = sauvegarder_masque(
                masque_morph,
                classe,
                configuration,
                nom_fichier
            )

            mesures = analyser_masque(
                masque_morph
            )

            lignes.append({
                "classe": classe,
                "configuration": configuration,
                **mesures,
                "fichier_masque": str(
                    chemin_sortie.relative_to(PROJECT_DIR)
                ),
            })

    # --------------------------------------------------------
    # Rapport CSV
    # --------------------------------------------------------

    dataframe = pd.DataFrame(lignes)

    dataframe.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Planche comparative
    # --------------------------------------------------------

    creer_planche(selections)

    print("\n" + "=" * 70)
    print("BILAN")
    print("=" * 70)

    print(
        f"Rapport CSV : {REPORT_FILE}"
    )

    print(
        f"Planche comparative : {FIGURE_FILE}"
    )

    print(
        f"Masques : {MASK_MORPH_DIR}"
    )

    print("\nMesures :")
    print(
        dataframe.to_string(index=False)
    )

    print("\nTest terminé.")


if __name__ == "__main__":
    main()