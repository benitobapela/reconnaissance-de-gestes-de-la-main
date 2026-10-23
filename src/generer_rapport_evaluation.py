
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Définir les chemins
# --------------------------------------------------

RACINE_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_RESULTATS = RACINE_PROJET / "results"

FICHIER_METRIQUES = DOSSIER_RESULTATS / "metriques_knn.csv"
FICHIER_MATRICE = DOSSIER_RESULTATS / "matrice_confusion_knn.csv"
FICHIER_RAPPORT = DOSSIER_RESULTATS / "evaluation_report.txt"


# --------------------------------------------------
# 2. Vérifier les fichiers nécessaires
# --------------------------------------------------

for fichier in [FICHIER_METRIQUES, FICHIER_MATRICE]:
    if not fichier.exists():
        raise FileNotFoundError(f"Fichier introuvable : {fichier}")


# --------------------------------------------------
# 3. Charger les résultats existants
# --------------------------------------------------

metriques = pd.read_csv(FICHIER_METRIQUES)
matrice = pd.read_csv(FICHIER_MATRICE, index_col=0)

if metriques.empty or matrice.empty:
    raise ValueError("Un fichier de résultats est vide.")

if matrice.shape != (5, 5):
    raise ValueError(
        f"La matrice de confusion devrait être 5x5, "
        f"mais sa forme est {matrice.shape}."
    )


# --------------------------------------------------
# 4. Préparer les informations du rapport
# --------------------------------------------------

nombre_total = int(matrice.to_numpy().sum())
nombre_correct = int(matrice.to_numpy().trace())
nombre_erreurs = nombre_total - nombre_correct

lignes = [
    "RAPPORT D'EVALUATION — HANDGESTURE-5G",
    "=" * 55,
    "",
    "1. CONFIGURATION DU MODELE",
    "Algorithme : K-Nearest Neighbors (K-NN)",
    "Standardisation : StandardScaler",
    "Nombre de voisins : 3",
    "Ponderation : distance",
    "Metrique de distance : euclidienne",
    "Caracteristiques : 7 moments de Hu transformes",
    "",
    "2. DONNEES D'EVALUATION",
    f"Nombre d'images evaluees : {nombre_total}",
    f"Predictions correctes : {nombre_correct}",
    f"Predictions incorrectes : {nombre_erreurs}",
    "",
    "3. METRIQUES ENREGISTREES",
    metriques.to_string(index=False),
    "",
    "4. MATRICE DE CONFUSION",
    "Lignes : classes reelles ; colonnes : classes predites.",
    matrice.to_string(),
    "",
    "5. INTERPRETATION",
    f"Exactitude calculee depuis la matrice : "
    f"{nombre_correct / nombre_total:.4f} "
    f"({100 * nombre_correct / nombre_total:.2f} %)",
    "",
    "Les performances observees concernent le jeu de test utilise",
    "dans ce projet. Elles ne garantissent pas les memes resultats",
    "sur des images externes ou dans toutes les conditions.",
    "",
    "Le modele sauvegarde est documente dans README_model.md.",
    "Sa configuration est disponible dans models/model_config.json.",
]

# --------------------------------------------------
# 5. Enregistrer le rapport
# --------------------------------------------------

FICHIER_RAPPORT.write_text(
    "\n".join(lignes) + "\n",
    encoding="utf-8",
)

print("Rapport d'evaluation genere avec succes.")
print(f"Fichier : {FICHIER_RAPPORT}")
print(f"Images evaluees : {nombre_total}")
print(f"Predictions correctes : {nombre_correct}")
print(f"Predictions incorrectes : {nombre_erreurs}")
