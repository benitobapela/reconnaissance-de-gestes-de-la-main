# Étape 8 — Mise en place et sélection du modèle K-NN

## 1. Objectif de l'étape

L'objectif de cette étape est de préparer les données, de mettre en place un classificateur K-NN (*K-Nearest Neighbors*), de rechercher des hyperparamètres adaptés et de retenir une configuration pour la reconnaissance des cinq gestes de la main.

Le modèle utilise les sept moments invariants de Hu extraits précédemment à partir des contours des gestes. Ces caractéristiques constituent le vecteur descriptif de chaque image.

Les cinq classes à reconnaître sont :

- `Fist` : poing fermé ;
- `L-Sign` : geste en forme de L ;
- `Open Palm_Open Hand` : paume ouverte ;
- `Pointing Up` : index pointé vers le haut ;
- `V-Sign` : geste en forme de V.

Cette étape comprend l'inspection du dataset, la séparation des données, la standardisation des caractéristiques, la recherche des hyperparamètres et le choix de la configuration retenue.

## 2. Inspection du dataset

Avant de construire le modèle, le fichier `data/final/features.csv` a été contrôlé afin de vérifier la qualité et la cohérence des données.

### 2.1. Structure du dataset

Le dataset final contient 2 653 lignes et 9 colonnes :

- `image_id` : identifiant de l'image ;
- `class` : classe correspondant au geste ;
- `Hu1` à `Hu7` : sept caractéristiques issues des moments invariants de Hu.

### 2.2. Répartition des classes

| Classe | Nombre d'images | Proportion |
|---|---:|---:|
| Fist | 542 | 20,43 % |
| L-Sign | 537 | 20,24 % |
| Open Palm_Open Hand | 661 | 24,92 % |
| Pointing Up | 405 | 15,27 % |
| V-Sign | 508 | 19,15 % |
| **Total** | **2 653** | **100 %** |

Les effectifs ne sont pas identiques entre les classes. Il est donc pertinent d'utiliser le F1-score macro lors de la comparaison des modèles, car cette métrique accorde la même importance à chaque classe.

### 2.3. Contrôle de qualité

Les vérifications ont confirmé que :

- les 2 653 identifiants sont présents et uniques ;
- aucune valeur manquante n'a été détectée ;
- aucune valeur Hu non numérique ou non finie n'a été détectée ;
- les cinq classes attendues sont présentes.

Le dataset a donc satisfait les contrôles effectués avant la classification.

## 3. Vérification des doublons exacts

Un contrôle des images originales a été réalisé à l'aide d'empreintes permettant de repérer les fichiers contenant exactement les mêmes pixels.

Les résultats sont les suivants :

| Élément vérifié | Résultat |
|---|---:|
| Images examinées | 2 653 |
| Groupes de doublons exacts | 846 |
| Images appartenant à ces groupes | 1 692 |
| Images sans doublon exact détecté | 961 |
| Groupes présents dans plusieurs classes | 0 |

Les 846 groupes correspondent à des paires d'images identiques. Les 1 692 images concernées ont toutes été retrouvées dans `features.csv`.

Cette vérification est importante, car la présence d'une même image dans les données d'entraînement et de test pourrait donner une estimation artificiellement favorable des performances du modèle.

Les images sans doublon exact ont reçu chacune un identifiant de groupe individuel pour la séparation des données.

**Fichier produit :** `results/correspondance_doublons_features.csv`.

## 4. Séparation des données

Le dataset a été séparé en deux ensembles :

- **Entraînement :** 2 122 images, soit 79,98 % ;
- **Test :** 531 images, soit 20,02 %.

La méthode `StratifiedGroupKFold` a été utilisée pour obtenir une séparation approximativement stratifiée, tout en maintenant les images d'un même groupe de doublons dans le même ensemble.

### 4.1. Répartition obtenue

| Classe | Entraînement | Test | Total |
|---|---:|---:|---:|
| Fist | 434 | 108 | 542 |
| L-Sign | 429 | 108 | 537 |
| Open Palm_Open Hand | 529 | 132 | 661 |
| Pointing Up | 324 | 81 | 405 |
| V-Sign | 406 | 102 | 508 |
| **Total** | **2 122** | **531** | **2 653** |

La vérification a confirmé que zéro groupe de doublons est partagé entre l'entraînement et le test.

Le fichier `results/split_train_test.csv` conserve cette affectation afin de la réutiliser dans les étapes suivantes.

## 5. Standardisation des caractéristiques Hu

### 5.1. Pourquoi standardiser ?

Le classificateur K-NN s'appuie sur des distances entre les vecteurs de caractéristiques. Si les caractéristiques présentent des échelles différentes, certaines peuvent influencer davantage le calcul des distances.

La standardisation permet de centrer les caractéristiques autour de zéro et de les ramener à une échelle comparable.

La transformation utilisée est :

\[
z = \frac{x-\mu}{\sigma}
\]

où :

- \(x\) représente la valeur initiale ;
- \(\mu\) représente la moyenne ;
- \(\sigma\) représente l'écart-type ;
- \(z\) représente la valeur standardisée.

### 5.2. Méthode appliquée

La classe `StandardScaler` de Scikit-learn a été utilisée.

Le scaler a été ajusté uniquement sur les données d'entraînement. Les mêmes paramètres ont ensuite été appliqués aux données de test, afin d'éviter toute fuite d'information.

### 5.3. Vérification

Les résultats obtenus sur l'entraînement sont :

- moyennes des sept caractéristiques standardisées : approximativement égales à zéro ;
- écarts-types des sept caractéristiques standardisées : égaux à 1.

Les moyennes calculées sur le test ne sont pas exactement égales à zéro, ce qui est normal puisque les paramètres de standardisation proviennent exclusivement de l'entraînement.

Les deux fichiers produits sont :

- `results/features_train_standardisees.csv` ;
- `results/features_test_standardisees.csv`.

Pour la validation croisée, le scaler a également été intégré dans un pipeline Scikit-learn. Il est ainsi réajusté sur la partie d'entraînement de chaque pli, sans utiliser les données de validation pour calculer les paramètres de standardisation.

## 6. Mise en place du classificateur K-NN

### 6.1. Principe de fonctionnement

K-NN signifie *K-Nearest Neighbors*, ou méthode des \(k\) plus proches voisins.

Lorsqu'une nouvelle image doit être classée, le modèle compare son vecteur de caractéristiques Hu aux vecteurs des images d'entraînement. Il recherche les \(k\) voisins les plus proches, puis attribue une classe selon les classes de ces voisins.

Le paramètre \(k\) représente donc le nombre de voisins utilisés pour prendre la décision.

- Avec \(k=1\), la prédiction dépend d'un seul voisin ;
- avec \(k=3\), elle prend en compte trois voisins ;
- avec \(k=5\), elle prend en compte cinq voisins.

Une valeur faible de \(k\) peut rendre le modèle sensible aux images atypiques. Une valeur plus élevée peut stabiliser la décision, mais aussi inclure des voisins appartenant à des gestes différents.

### 6.2. Paramètres comparés

La recherche a porté sur les configurations suivantes :

| Paramètre | Valeurs testées |
|---|---|
| Nombre de voisins (`n_neighbors`) | 1, 3, 5, 7, 9, 11, 15, 21 |
| Pondération (`weights`) | `uniform`, `distance` |
| Métrique de distance | `euclidean` |
| Métrique de sélection | F1-score macro |
| Validation croisée | 5 plis stratifiés et groupés |

Avec `uniform`, tous les voisins ont le même poids dans le vote.

Avec `distance`, les voisins les plus proches ont davantage d'influence sur la prédiction.

Au total, 16 configurations ont été comparées.

### 6.3. Validation croisée

La recherche a été réalisée exclusivement sur les 2 122 images d'entraînement. Les 531 images du jeu de test n'ont pas été utilisées pour choisir les hyperparamètres.

La validation croisée a été effectuée avec `StratifiedGroupKFold`, en conservant les groupes de doublons ensemble dans chaque séparation. Cette précaution limite le risque qu'une image identique ou son doublon influence simultanément l'apprentissage et la validation.

Le F1-score macro a été utilisé pour comparer les configurations. Il correspond à la moyenne des F1-scores des cinq classes, sans pondération par leur nombre d'images.

## 7. Résultats de la recherche

### 7.1. Meilleures configurations observées

Les principaux résultats sont :

| Configuration | F1-score macro moyen |
|---|---:|
| \(k=1\), `uniform` | 0,8572 |
| \(k=1\), `distance` | 0,8572 |
| \(k=3\), `distance` | 0,8494 |
| \(k=5\), `distance` | 0,8484 |
| \(k=3\), `uniform` | 0,8448 |
| \(k=5\), `uniform` | 0,8380 |

La meilleure configuration selon le score moyen observé est \(k=1\), avec un F1-score macro de 0,8572. Les variantes `uniform` et `distance` obtiennent le même score moyen après arrondi.

La configuration \(k=3\), avec pondération `distance`, obtient un F1-score macro moyen de 0,8494, soit environ 84,94 %.

L'écart entre \(k=1\) et \(k=3\) avec pondération `distance` est d'environ 0,78 point de pourcentage.

### 7.2. Décision concernant la valeur de \(k\)

Bien que \(k=1\) ait obtenu le score moyen le plus élevé, le projet retient **\(k=3\) avec pondération `distance`**.

Ce choix repose sur une préférence méthodologique : prendre en compte plusieurs voisins plutôt que de faire dépendre chaque prédiction d'un seul exemple.

La configuration retenue est donc :

| Paramètre | Valeur retenue |
|---|---|
| Algorithme | K-NN |
| Nombre de voisins | 3 |
| Pondération | `distance` |
| Distance | Euclidienne |
| Prétraitement des caractéristiques | StandardScaler |
| Critère utilisé lors de la recherche | F1-score macro |
| F1-score macro moyen en validation croisée | 0,8494 |

Ce choix constitue un compromis entre le score observé et la volonté de considérer plusieurs voisins lors de la classification.

Il faut cependant préciser que l'utilisation de trois voisins ne garantit pas automatiquement une meilleure robustesse que celle d'un seul voisin. Le choix de \(k=3\) correspond à la stratégie retenue pour ce projet, et non à la configuration ayant obtenu le score moyen maximal pendant la recherche.

## 8. Limites et précautions d'interprétation

Les résultats obtenus à cette étape ne constituent pas encore une mesure définitive des performances du système.

Premièrement, le F1-score macro de 0,8494 est une moyenne calculée pendant la validation croisée sur les données d'entraînement.

Deuxièmement, le score d'entraînement parfait observé pour certaines configurations ne signifie pas que ces modèles classeront parfaitement de nouvelles images.

Troisièmement, les caractéristiques Hu décrivent principalement la forme du contour. Des gestes visuellement proches, des erreurs de segmentation ou des variations de posture peuvent donc conduire à des confusions.

Enfin, le jeu de test doit rester indépendant. Il sera utilisé à l'étape suivante pour évaluer la configuration retenue, et non pour modifier successivement les paramètres jusqu'à obtenir un score favorable.

## 9. Fichiers produits

Les principaux fichiers liés à cette étape sont :

| Fichier | Rôle |
|---|---|
| `src/inspecter_dataset_knn.py` | Contrôle de la structure et de la qualité des caractéristiques |
| `src/verifier_correspondance_doublons.py` | Vérification de la correspondance entre les doublons et les caractéristiques |
| `src/preparer_split_knn.py` | Création de la séparation entraînement/test |
| `src/standardiser_features_knn.py` | Standardisation des caractéristiques |
| `src/rechercher_meilleur_knn.py` | Comparaison des configurations K-NN |
| `results/correspondance_doublons_features.csv` | Correspondance entre les images du rapport de doublons et les identifiants du dataset final |
| `results/split_train_test.csv` | Affectation des images aux ensembles d'entraînement et de test |
| `results/features_train_standardisees.csv` | Caractéristiques standardisées de l'entraînement |
| `results/features_test_standardisees.csv` | Caractéristiques standardisées du test |
| `results/comparaison_knn_cv.csv` | Résultats détaillés de la validation croisée |
| `docs/etape_8_knn.md` | Documentation de l'étape |

## 10. Conclusion

Cette étape a permis de préparer les données pour la classification, de limiter les risques de fuite liés aux doublons exacts, de standardiser les caractéristiques Hu et de comparer 16 configurations K-NN.

La recherche a identifié \(k=1\) comme la configuration ayant obtenu le meilleur F1-score macro moyen. Toutefois, le projet retient \(k=3\), avec pondération par la distance et distance euclidienne, afin que la prédiction tienne compte de plusieurs voisins.

La configuration retenue sera évaluée sur les 531 images du jeu de test à l'étape 9. Cette évaluation permettra de mesurer l'accuracy, la précision, le rappel, le F1-score macro et les confusions entre les cinq gestes.

Le modèle final sera ensuite sauvegardé à l'étape prévue pour la conservation et la réutilisation du modèle.
