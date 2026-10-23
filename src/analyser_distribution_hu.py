
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RACINE = Path(__file__).resolve().parent.parent
DOSSIER_HSV = RACINE / "data" / "processed" / "hsv"
DOSSIER_RESULTS = RACINE / "results"
DOSSIER_FIGURES = RACINE / "figures"

SEED = 42
NB_IMAGES_PAR_CLASSE = 30

HSV_BAS = np.array([0, 20, 40], dtype=np.uint8)
HSV_HAUT = np.array([25, 255, 255], dtype=np.uint8)


def extraire_hu(image_hsv):
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

    contours, _ = cv2.findContours(
        masque.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if len(contour) < 3:
        return None

    moments = cv2.moments(contour)

    if abs(moments["m00"]) < 1e-12:
        return None

    hu = cv2.HuMoments(moments).flatten()

    if not np.all(np.isfinite(hu)):
        return None

    epsilon = 1e-30
    hu_log = (
        -np.sign(hu)
        * np.log10(np.maximum(np.abs(hu), epsilon))
    )

    if not np.all(np.isfinite(hu_log)):
        return None

    return hu_log


def main():
    DOSSIER_RESULTS.mkdir(parents=True, exist_ok=True)
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)

    if not DOSSIER_HSV.exists():
        print(f"Dossier introuvable : {DOSSIER_HSV}")
        return

    generateur = np.random.default_rng(SEED)
    lignes = []
    echecs = 0

    dossiers_classes = sorted(
        d for d in DOSSIER_HSV.iterdir() if d.is_dir()
    )

    for dossier_classe in dossiers_classes:
        fichiers = sorted(dossier_classe.glob("*.npy"))

        if not fichiers:
            print(f"Aucun fichier HSV pour {dossier_classe.name}")
            continue

        nombre = min(NB_IMAGES_PAR_CLASSE, len(fichiers))
        indices = generateur.choice(
            len(fichiers), size=nombre, replace=False
        )

        print(
            f"\n{dossier_classe.name} : "
            f"{nombre} images sélectionnées"
        )

        reussites = 0

        for indice in indices:
            fichier = fichiers[int(indice)]

            try:
                image_hsv = np.load(fichier)

                if (
                    image_hsv.ndim != 3
                    or image_hsv.shape[2] != 3
                    or image_hsv.dtype != np.uint8
                ):
                    echecs += 1
                    continue

                hu_log = extraire_hu(image_hsv)

                if hu_log is None:
                    echecs += 1
                    continue

                ligne = {
                    "image_id": fichier.name,
                    "class": dossier_classe.name,
                }

                for i, valeur in enumerate(hu_log, start=1):
                    ligne[f"Hu{i}"] = float(valeur)

                lignes.append(ligne)
                reussites += 1

            except Exception as erreur:
                echecs += 1
                print(f"Échec pour {fichier.name} : {erreur}")

        print(f"Extractions réussies : {reussites}/{nombre}")

    if not lignes:
        print("Aucune extraction valide.")
        return

    df = pd.DataFrame(lignes)
    chemin_csv = DOSSIER_RESULTS / "distribution_hu_echantillon.csv"
    df.to_csv(chemin_csv, index=False, encoding="utf-8-sig")

    colonnes_hu = [f"Hu{i}" for i in range(1, 8)]

    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    axes = axes.flatten()

    classes = sorted(df["class"].unique())

    for i, colonne in enumerate(colonnes_hu):
        donnees = [
            df.loc[df["class"] == classe, colonne].dropna().values
            for classe in classes
        ]

        axes[i].boxplot(donnees, tick_labels=classes)
        axes[i].set_title(f"Distribution de {colonne}")
        axes[i].set_ylabel("Valeur transformée")
        axes[i].tick_params(axis="x", rotation=25)
        axes[i].grid(axis="y", alpha=0.3)

    axes[7].axis("off")
    fig.suptitle(
        "Distribution des moments de Hu transformés par classe",
        fontsize=14
    )
    fig.tight_layout()

    chemin_figure = DOSSIER_FIGURES / "distribution_hu_echantillon.png"
    fig.savefig(chemin_figure, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 55)
    print("ANALYSE TERMINÉE")
    print("=" * 55)
    print(f"Images analysées avec succès : {len(df)}")
    print(f"Échecs d'extraction : {echecs}")
    print(f"CSV : {chemin_csv}")
    print(f"Graphique : {chemin_figure}")
    print("\nNombre d'images valides par classe :")
    print(df.groupby("class").size().to_string())


if __name__ == "__main__":
    main()
