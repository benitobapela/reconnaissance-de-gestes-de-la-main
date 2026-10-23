
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# --------------------------------------------------
# 1. Chemins et caractéristiques
# --------------------------------------------------

RACINE = Path(__file__).resolve().parent.parent

FICHIER_FEATURES = RACINE / "data" / "final" / "features.csv"
FICHIER_SPLIT = RACINE / "results" / "split_train_test.csv"
FICHIER_RESULTATS = RACINE / "results" / "comparaison_knn_cv.csv"

COLONNES_HU = [f"Hu{i}" for i in range(1, 8)]


def main():
    print("=" * 65)
    print("ÉTAPE 8.4 — RECHERCHE DU MEILLEUR K-NN")
    print("=" * 65)

    # --------------------------------------------------
    # 2. Charger les caractéristiques et la séparation
    # --------------------------------------------------

    features = pd.read_csv(FICHIER_FEATURES)
    split = pd.read_csv(FICHIER_SPLIT)

    donnees = features.merge(
        split[["image_id", "split", "group_id"]],
        on="image_id",
        how="left",
        validate="one_to_one"
    )

    if donnees["split"].isna().any():
        raise ValueError("Certaines images n'ont pas de séparation.")

    # La recherche se fait uniquement sur l'entraînement.
    train = donnees[donnees["split"] == "train"].copy()

    X_train = train[COLONNES_HU]
    y_train = train["class"]
    groupes_train = train["group_id"]

    print(f"\nImages utilisées pour la validation : {len(train)}")
    print(f"Nombre de groupes : {groupes_train.nunique()}")
    print("Images de test utilisées : 0")

    # --------------------------------------------------
    # 3. Définir le pipeline
    # --------------------------------------------------

    # Le scaler est réappris dans chaque pli :
    # aucune information de validation ne sert à le calculer.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    # Valeurs de k et méthodes de vote à comparer.
    parametres = {
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean"]
    }

    validation = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # --------------------------------------------------
    # 4. Lancer la validation croisée
    # --------------------------------------------------

    recherche = GridSearchCV(
        estimator=pipeline,
        param_grid=parametres,
        scoring="f1_macro",
        cv=validation,
        refit=True,
        n_jobs=-1,
        return_train_score=True,
        error_score="raise"
    )

    recherche.fit(
        X_train,
        y_train,
        groups=groupes_train
    )

    # --------------------------------------------------
    # 5. Enregistrer et afficher les résultats
    # --------------------------------------------------

    resultats = pd.DataFrame(recherche.cv_results_)

    colonnes = [
        "param_knn__n_neighbors",
        "param_knn__weights",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "rank_test_score"
    ]

    resultats = resultats[colonnes].sort_values(
        by=["rank_test_score", "std_test_score"]
    )

    resultats.to_csv(
        FICHIER_RESULTATS,
        index=False,
        encoding="utf-8"
    )

    print("\nMeilleure configuration trouvée :")
    print(recherche.best_params_)

    print(
        "Meilleur F1-score macro moyen en validation croisée : "
        f"{recherche.best_score_:.4f}"
    )

    print("\nTop 10 des configurations :")
    print(resultats.head(10).to_string(index=False))

    print(f"\nComparaison enregistrée : {FICHIER_RESULTATS.relative_to(RACINE)}")
    print("\nRECHERCHE TERMINÉE")


if __name__ == "__main__":
    main()
