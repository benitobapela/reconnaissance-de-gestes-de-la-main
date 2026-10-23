
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CHEMINS ET PARAMETRES
# ============================================================

RACINE = Path(__file__).resolve().parent.parent

DOSSIER_HSV = RACINE / "data" / "processed" / "hsv"
DOSSIER_COLOR = RACINE / "data" / "processed" / "color"
DOSSIER_RESULTS = RACINE / "results"
DOSSIER_FIGURES = RACINE / "figures"

FICHIER_PREDICTIONS = DOSSIER_RESULTS / "predictions_knn.csv"
FICHIER_SORTIE = DOSSIER_FIGURES / "inspection_segmentation_erreurs.png"
FICHIER_LISTE = DOSSIER_RESULTS / "images_inspectees_segmentation.csv"

HSV_BAS = np.array([0, 20, 40], dtype=np.uint8)
HSV_HAUT = np.array([25, 255, 255], dtype=np.uint8)

NOMBRE_PAR_SENS = 3


# ============================================================
# 2. REPRODUIRE LE MASQUE UTILISE PAR LE MODELE
# ============================================================

def construire_masque(image_hsv):
    masque = cv2.inRange(image_hsv, HSV_BAS, HSV_HAUT)

    noyau_fermeture = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )
    noyau_ouverture = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    masque = cv2.morphologyEx(
        masque, cv2.MORPH_CLOSE, noyau_fermeture
    )
    masque = cv2.morphologyEx(
        masque, cv2.MORPH_OPEN, noyau_ouverture
    )

    return masque


def obtenir_contour_principal(masque):
    contours, _ = cv2.findContours(
        masque.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) <= 0:
        return None

    return contour


# ============================================================
# 3. SELECTIONNER DES ERREURS
# ============================================================

def selectionner_erreurs():
    df = pd.read_csv(FICHIER_PREDICTIONS)

    sens_1 = df[
        (df["classe_reelle"] == "L-Sign")
        & (df["classe_predite"] == "Open Palm_Open Hand")
    ]

    sens_2 = df[
        (df["classe_reelle"] == "Open Palm_Open Hand")
        & (df["classe_predite"] == "L-Sign")
    ]

    # Retirer les doublons exacts connus du rapport de détection.
    fichier_doublons = DOSSIER_RESULTS / "duplicates_exact.csv"
    identifiants_a_exclure = set()

    if fichier_doublons.exists():
        doublons = pd.read_csv(fichier_doublons)

        # Les colonnes du rapport peuvent varier selon le script
        # qui l'a produit. On affiche leurs noms pour vérification.
        print("Colonnes du rapport de doublons :",
              list(doublons.columns))

        # Si le rapport possède une colonne image_id, on l'utilise.
        if "image_id" in doublons.columns:
            identifiants_a_exclure = set(
                doublons["image_id"].dropna().astype(str)
            )

    def choisir(df_sens):
        # Les noms sont triés pour rendre la sélection reproductible.
        candidats = df_sens.sort_values("image_id").copy()

        if identifiants_a_exclure:
            candidats = candidats[
                ~candidats["image_id"].isin(identifiants_a_exclure)
            ]

        # Les fichiers jpeg/jpg portant le même nom de base
        # ne sont pas considérés automatiquement comme indépendants.
        candidats["base"] = (
            candidats["image_id"]
            .str.rsplit("/", n=1)
            .str[-1]
            .str.replace(r"__jpeg\.npy$|__jpg\.npy$", "", regex=True)
        )

        candidats = candidats.drop_duplicates("base")

        return candidats.head(NOMBRE_PAR_SENS)

    selection = pd.concat(
        [choisir(sens_1), choisir(sens_2)],
        ignore_index=True
    )

    if selection.empty:
        raise RuntimeError(
            "Aucune erreur correspondant aux deux confusions trouvée."
        )

    return selection


# ============================================================
# 4. AFFICHER IMAGE, MASQUES ET CONTOUR
# ============================================================

def chemin_image_couleur(image_id):
    # Exemple d'identifiant :
    # L-Sign/IMG_0651__jpeg.npy
    relatif = Path(image_id)
    nom = relatif.name.replace(".npy", ".png")

    return DOSSIER_COLOR / relatif.parent / nom


def afficher_inspection(selection):
    nombre = len(selection)

    fig, axes = plt.subplots(
        nombre, 4,
        figsize=(15, 4 * nombre),
        squeeze=False
    )

    for i, ligne in selection.iterrows():
        image_id = str(ligne["image_id"])
        relatif = Path(image_id)

        fichier_hsv = DOSSIER_HSV / relatif
        fichier_couleur = chemin_image_couleur(image_id)

        image_hsv = np.load(fichier_hsv, allow_pickle=False)
        image_couleur = cv2.imread(str(fichier_couleur))

        if image_couleur is None:
            raise FileNotFoundError(
                f"Image couleur introuvable : {fichier_couleur}"
            )

        image_rgb = cv2.cvtColor(
            image_couleur, cv2.COLOR_BGR2RGB
        )

        masque_initial = cv2.inRange(
            image_hsv, HSV_BAS, HSV_HAUT
        )
        masque_final = construire_masque(image_hsv)
        contour = obtenir_contour_principal(masque_final)

        image_contour = image_rgb.copy()

        if contour is not None:
            cv2.drawContours(
                image_contour,
                [contour],
                -1,
                (255, 0, 0),
                2
            )

        titre = (
            f"Réel : {ligne['classe_reelle']}\n"
            f"Prédit : {ligne['classe_predite']}\n"
            f"{relatif.name}"
        )

        axes[i, 0].imshow(image_rgb)
        axes[i, 0].set_title(titre, fontsize=9)

        axes[i, 1].imshow(masque_initial, cmap="gray")
        axes[i, 1].set_title("Masque HSV initial")

        axes[i, 2].imshow(masque_final, cmap="gray")
        axes[i, 2].set_title("Masque après morphologie")

        axes[i, 3].imshow(image_contour)
        axes[i, 3].set_title("Contour principal")

        for j in range(4):
            axes[i, j].axis("off")

    fig.suptitle(
        "Inspection de la segmentation HSV sur des erreurs K-NN",
        fontsize=14
    )
    fig.tight_layout()

    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FICHIER_SORTIE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    selection.to_csv(FICHIER_LISTE, index=False, encoding="utf-8-sig")

    print("\nInspection terminée.")
    print(f"Figure enregistrée : {FICHIER_SORTIE}")
    print(f"Liste des images : {FICHIER_LISTE}")
    print(f"Nombre d'images inspectées : {len(selection)}")


# ============================================================
# 5. EXECUTION
# ============================================================

def main():
    selection = selectionner_erreurs()

    print("\nImages sélectionnées :")
    print(
        selection[
            ["image_id", "classe_reelle", "classe_predite"]
        ].to_string(index=False)
    )

    afficher_inspection(selection)


if __name__ == "__main__":
    main()
