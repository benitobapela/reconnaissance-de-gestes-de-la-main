
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. Définir les chemins et l'ordre des classes
# --------------------------------------------------

RACINE_PROJET = Path(__file__).resolve().parent.parent

FICHIER_PREDICTIONS = (
    RACINE_PROJET / "results" / "predictions_knn.csv"
)

DOSSIER_RESULTATS = RACINE_PROJET / "results"
DOSSIER_FIGURES = RACINE_PROJET / "figures"

FICHIER_CSV = DOSSIER_RESULTATS / "matrice_confusion_knn.csv"
FICHIER_IMAGE = DOSSIER_FIGURES / "matrice_confusion_knn.png"

CLASSES = [
    "Fist",
    "L-Sign",
    "Open Palm_Open Hand",
    "Pointing Up",
    "V-Sign",
]


# --------------------------------------------------
# 2. Charger les prédictions
# --------------------------------------------------

if not FICHIER_PREDICTIONS.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER_PREDICTIONS}"
    )

df = pd.read_csv(FICHIER_PREDICTIONS)

if len(df) != 531:
    raise ValueError(
        f"531 prédictions attendues, mais {len(df)} trouvées."
    )

y_reel = df["classe_reelle"]
y_predit = df["classe_predite"]


# --------------------------------------------------
# 3. Calculer la matrice de confusion
# --------------------------------------------------

matrice = confusion_matrix(
    y_reel,
    y_predit,
    labels=CLASSES,
)

matrice_df = pd.DataFrame(
    matrice,
    index=CLASSES,
    columns=CLASSES,
)

print("=" * 65)
print("ÉTAPE 9.5 — MATRICE DE CONFUSION DU K-NN")
print("=" * 65)

print("\nLignes : classes réelles")
print("Colonnes : classes prédites")
print("\nMatrice de confusion :")
print(matrice_df.to_string())


# --------------------------------------------------
# 4. Vérifier les résultats
# --------------------------------------------------

total_images = int(matrice.sum())
bonnes_predictions = int(matrice.trace())
erreurs = total_images - bonnes_predictions

print("\n--- Vérification ---")
print(f"Total des images         : {total_images}")
print(f"Prédictions correctes    : {bonnes_predictions}")
print(f"Prédictions incorrectes  : {erreurs}")

if total_images != 531:
    raise ValueError(
        "Le total de la matrice doit être égal à 531."
    )

if bonnes_predictions != 462:
    raise ValueError(
        "Le nombre de prédictions correctes ne correspond "
        "pas aux métriques calculées précédemment."
    )

print("[OK] La matrice est cohérente avec les métriques.")


# --------------------------------------------------
# 5. Enregistrer la matrice en CSV
# --------------------------------------------------

DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)
DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)

matrice_df.to_csv(
    FICHIER_CSV,
    encoding="utf-8",
)

print(f"\nMatrice CSV enregistrée : {FICHIER_CSV}")


# --------------------------------------------------
# 6. Créer la figure
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))

image = ax.imshow(matrice, cmap="Blues")

fig.colorbar(image, ax=ax, label="Nombre d'images")

ax.set_title("Matrice de confusion — K-NN")
ax.set_xlabel("Classe prédite")
ax.set_ylabel("Classe réelle")

positions = range(len(CLASSES))

ax.set_xticks(list(positions))
ax.set_yticks(list(positions))

ax.set_xticklabels(CLASSES, rotation=35, ha="right")
ax.set_yticklabels(CLASSES)

# Afficher les nombres dans les cellules
seuil = matrice.max() / 2 if matrice.max() > 0 else 0

for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        valeur = matrice[i, j]

        ax.text(
            j,
            i,
            str(valeur),
            ha="center",
            va="center",
            color="white" if valeur > seuil else "black",
        )

fig.tight_layout()

fig.savefig(
    FICHIER_IMAGE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Figure enregistrée       : {FICHIER_IMAGE}")
print("\nÉtape 9.5 terminée.")
