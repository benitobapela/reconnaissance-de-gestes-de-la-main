# Étape 9 — Évaluation du modèle K-NN

## 1. Objectif

Cette étape vise à évaluer les performances du modèle K-NN de reconnaissance de gestes de la main à partir de caractéristiques géométriques extraites des contours. L'évaluation mesure la capacité du modèle à reconnaître les cinq classes sur des images qui n'ont pas été utilisées pendant son entraînement.

## 2. Préparation des données

Le jeu de données comprend 2 653 images réparties en cinq classes : Fist, L-Sign, Open Palm_Open Hand, Pointing Up et V-Sign.

La séparation des données a été réalisée précédemment en conservant les groupes d'images identiques dans un seul ensemble, afin de limiter les fuites de données entre l'entraînement et le test.

| Ensemble | Nombre d'images | Proportion |
|---|---:|---:|
| Entraînement | 2 122 | 79,98 % |
| Test | 531 | 20,02 % |
| Total | 2 653 | 100 % |

Les contrôles ont confirmé la présence des sept caractéristiques de Hu, l'absence de valeurs manquantes ou infinies, la cohérence des identifiants et l'absence de groupes communs entre l'entraînement et le test.

## 3. Configuration du modèle

Le modèle évalué utilise l'algorithme K-NN (*K-Nearest Neighbors*).

| Paramètre | Valeur |
|---|---|
| Nombre de voisins | 3 |
| Pondération | Distance (`distance`) |
| Mesure de distance | Euclidienne |
| Standardisation | `StandardScaler` |
| Nombre de caractéristiques | 7 |
| Nombre de classes | 5 |

La standardisation est intégrée dans un pipeline avec K-NN. Les paramètres de standardisation sont calculés uniquement à partir des données d'entraînement, puis appliqués aux données de test.

Le choix de trois voisins constitue une décision méthodologique issue de la recherche par validation croisée. Le choix ne correspond pas au meilleur score moyen observé : k = 1 avait obtenu un F1 macro légèrement supérieur. Le choix de k = 3 privilégie ici une décision reposant sur plusieurs voisins.

## 4. Méthode d'évaluation

Après l'entraînement sur les 2 122 images, le modèle a prédit la classe de chacune des 531 images de test.

Les métriques suivantes ont été calculées :

- **Accuracy** : proportion totale de prédictions correctes.
- **Précision macro** : moyenne des précisions des cinq classes.
- **Rappel macro** : moyenne des rappels des cinq classes.
- **F1-score macro** : moyenne des F1-scores des cinq classes.
- **F1-score pondéré** : moyenne des F1-scores pondérée par le nombre d'images de chaque classe.

Le F1-score macro est particulièrement utile pour examiner les performances sur les différentes classes sans donner davantage de poids aux classes les plus nombreuses.

## 5. Résultats globaux

| Métrique | Résultat |
|---|---:|
| Accuracy | 87,01 % |
| Précision macro | 87,25 % |
| Rappel macro | 86,89 % |
| F1-score macro | 86,99 % |
| F1-score pondéré | 86,99 % |

Sur les 531 images de test, 462 ont été correctement classées et 69 ont été mal classées. L'accuracy obtenue est donc de 87,01 %.

La proximité entre le F1-score macro et le F1-score pondéré suggère que les performances globales ne sont pas fortement modifiées par la pondération des classes. Cela ne signifie toutefois pas que les performances sont identiques pour chaque geste.

## 6. Résultats par classe

| Classe | Précision | Rappel | F1-score | Images de test |
|---|---:|---:|---:|---:|
| Fist | 88,99 % | 89,81 % | 89,40 % | 108 |
| L-Sign | 80,70 % | 85,19 % | 82,88 % | 108 |
| Open Palm_Open Hand | 88,19 % | 84,85 % | 86,49 % | 132 |
| Pointing Up | 90,41 % | 81,48 % | 85,71 % | 81 |
| V-Sign | 87,96 % | 93,14 % | 90,48 % | 102 |

La classe V-Sign obtient le meilleur F1-score, à 90,48 %, et le meilleur rappel, à 93,14 %. Le modèle retrouve ainsi une grande proportion des images appartenant à cette classe.

La classe Fist présente également de bons résultats, avec un F1-score de 89,40 %.

La classe L-Sign obtient le F1-score le plus faible, à 82,88 %. Sa précision de 80,70 % indique que plusieurs images prédites comme L-Sign appartiennent en réalité à d'autres classes.

La classe Pointing Up possède une précision élevée, de 90,41 %, mais le rappel le plus faible, de 81,48 %. Le modèle reconnaît correctement de nombreuses images qu'il prédit comme Pointing Up, mais il ne retrouve pas tous les gestes de cette classe.

## 7. Analyse de la matrice de confusion

La matrice de confusion est enregistrée dans `results/matrice_confusion_knn.csv`, et sa représentation graphique dans `figures/matrice_confusion_knn.png`.

Les lignes correspondent aux classes réelles et les colonnes aux classes prédites.

Les principales confusions observées sont les suivantes :

- 13 images Open Palm_Open Hand ont été classées comme L-Sign.
- 9 images L-Sign ont été classées comme Open Palm_Open Hand.
- 9 images Pointing Up ont été classées comme V-Sign.
- 5 images V-Sign ont été classées comme Pointing Up.
- 7 images Open Palm_Open Hand ont été classées comme Fist.

Les confusions entre Open Palm_Open Hand et L-Sign, ainsi qu'entre Pointing Up et V-Sign, méritent une attention particulière. Elles peuvent être liées à la similarité des contours extraits, aux variations de position ou d'orientation des mains, ou aux limites des moments de Hu pour distinguer ces gestes.

Ces explications restent des hypothèses : elles devront être vérifiées en examinant les images concernées et les masques de segmentation correspondants.

## 8. Limites de l'approche

Plusieurs limites doivent être prises en compte :

1. Les performances dépendent de la qualité de la segmentation de la peau et de l'extraction du plus grand contour.
2. Les sept moments de Hu décrivent certaines propriétés géométriques du contour, mais ne représentent pas tous les détails de la forme de la main.
3. Les gestes dont les contours sont similaires peuvent être confondus.
4. Les résultats obtenus sur ce jeu de test ne garantissent pas les mêmes performances dans des conditions réelles différentes, notamment en présence d'éclairages, d'arrière-plans ou de tons de peau variés.
5. Les doublons exacts ont été pris en compte dans la séparation des données, mais des images différentes peuvent néanmoins représenter des gestes très similaires.
6. Le résultat obtenu avec k = 3 doit être interprété en tenant compte du fait que k = 1 avait obtenu un F1 macro moyen légèrement supérieur pendant la validation croisée.

## 9. Conclusion

Le modèle K-NN, utilisant trois voisins, une pondération par distance euclidienne et sept moments de Hu standardisés, obtient une accuracy de 87,01 % et un F1-score macro de 86,99 % sur les 531 images de test.

Les résultats montrent une capacité encourageante à reconnaître les cinq gestes étudiés. V-Sign et Fist sont relativement bien reconnus, tandis que L-Sign et Pointing Up présentent davantage de difficultés selon les métriques observées.

La matrice de confusion met en évidence des confusions entre certaines classes. Une amélioration pourrait consister à examiner les erreurs de segmentation, à ajouter des caractéristiques complémentaires ou à comparer cette approche à d'autres méthodes de classification.

Ces résultats constituent une évaluation du modèle sur le jeu de test retenu. Ils ne suffisent pas, à eux seuls, à démontrer sa robustesse dans toutes les conditions d'utilisation réelles.