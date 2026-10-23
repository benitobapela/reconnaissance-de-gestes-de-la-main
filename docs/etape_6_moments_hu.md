# Étape 6 — Extraction et analyse des moments de Hu

## 6.1. Objectif de l'étape

L'objectif de cette étape est d'extraire des caractéristiques géométriques à partir des contours des mains segmentées. Ces caractéristiques seront utilisées ultérieurement par le classificateur k-NN pour reconnaître les cinq gestes du dataset HandGesture-5G.

Les moments de Hu constituent sept descripteurs numériques calculés à partir des moments géométriques d'un objet. Ils permettent notamment de caractériser la répartition de la forme dans une image.

Dans notre projet, les moments sont calculés à partir du contour externe principal détecté après la segmentation HSV et les opérations morphologiques.

Les cinq classes étudiées sont :

- Fist ;
- L-Sign ;
- Open Palm_Open Hand ;
- Pointing Up ;
- V-Sign.

## 6.2. Compréhension théorique des moments de Hu

### 6.2.1. Qu'est-ce qu'un moment géométrique ?

Un moment géométrique est une mesure numérique qui décrit certaines propriétés de la répartition d'une forme dans une image.

Les moments permettent notamment d'obtenir des informations sur la position, la taille et la distribution spatiale des pixels d'un objet.

Les moments centraux sont calculés relativement au centre de gravité de la forme. Les moments centraux normalisés permettent ensuite de réduire l'influence de l'échelle.

Les moments de Hu sont construits à partir de ces moments centraux normalisés.

### 6.2.2. Les sept moments de Hu

Les sept moments de Hu, notés Hu1 à Hu7, sont des caractéristiques numériques qui décrivent différentes propriétés géométriques de la forme.

Dans les conditions théoriques usuelles, ils sont invariants à la translation, à la rotation et au changement d'échelle de la forme. En pratique, les résultats peuvent varier à cause de la discrétisation de l'image, des imperfections du contour et de la segmentation.

Les sept moments sont calculés par OpenCV à l'aide de la fonction `cv2.HuMoments()`.

Dans notre projet, nous calculons ces moments à partir des moments géométriques du contour sélectionné.

### 6.2.3. Pourquoi utiliser les moments de Hu ?

Les images du dataset présentent des variations de taille, de position et d'orientation des mains. Les moments de Hu permettent de représenter leur forme par un vecteur de sept nombres plutôt que par l'ensemble des pixels de l'image.

Le vecteur obtenu est :

\[
\mathbf{Hu} =
[Hu_1, Hu_2, Hu_3, Hu_4, Hu_5, Hu_6, Hu_7]
\]

Ce vecteur constitue une représentation compacte de la forme et pourra servir d'entrée au classificateur k-NN.

Il faut cependant noter que les moments décrivent la forme du contour retenu, et non directement le sens sémantique du geste. La qualité de la segmentation et la sélection du contour principal restent donc déterminantes.

## 6.3. Calcul et examen des moments bruts

### 6.3.1. Méthodologie

Le script `src/test_moments_hu.py` a été utilisé pour calculer les moments sur une image sélectionnée dans chacune des cinq classes.

Le traitement appliqué est le suivant :

1. Chargement de l'image HSV prétraitée.
2. Segmentation de la peau à l'aide des seuils HSV retenus.
3. Application d'une fermeture morphologique avec un noyau elliptique de 5 × 5.
4. Application d'une ouverture morphologique avec un noyau elliptique de 3 × 3.
5. Détection des contours externes.
6. Sélection du contour de plus grande aire.
7. Calcul des moments géométriques et des sept moments de Hu.
8. Vérification de la validité des valeurs obtenues.

Les seuils HSV utilisés sont :

- Borne inférieure : `[0, 20, 40]`
- Borne supérieure : `[25, 255, 255]`

### 6.3.2. Résultats obtenus

Le calcul a réussi sur les cinq exemples sélectionnés. Chaque image a produit sept moments de Hu.

| Classe | Image sélectionnée | Aire du contour (px²) |
|---|---|---:|
| Fist | IMG_0090 2__jpg.npy | 14 142,50 |
| L-Sign | IMG_9014__jpeg.npy | 17 939,50 |
| Open Palm_Open Hand | IMG_9394 2__jpg.npy | 20 613,00 |
| Pointing Up | IMG_0908__jpeg.npy | 12 908,00 |
| V-Sign | IMG_0731__jpg.npy | 16 255,00 |

Les aires indiquées correspondent aux contours sélectionnés dans les masques obtenus par segmentation. Elles ne constituent pas une mesure physique de la taille des mains.

### 6.3.3. Valeurs brutes des moments de Hu

Les valeurs obtenues sont présentées ci-dessous en notation scientifique.

| Classe | Hu1 | Hu2 | Hu3 | Hu4 |
|---|---:|---:|---:|---:|
| Fist | 2,12358e-1 | 1,85527e-2 | 8,81835e-5 | 6,49244e-6 |
| L-Sign | 2,16596e-1 | 7,30789e-4 | 4,94894e-3 | 6,83461e-5 |
| Open Palm | 1,98706e-1 | 8,30366e-3 | 2,68794e-4 | 9,41849e-5 |
| Pointing Up | 1,83045e-1 | 1,20746e-3 | 1,61326e-3 | 4,25071e-5 |
| V-Sign | 2,25251e-1 | 1,55275e-2 | 1,16394e-3 | 6,25154e-4 |

| Classe | Hu5 | Hu6 | Hu7 |
|---|---:|---:|---:|
| Fist | 5,37418e-11 | -1,61732e-7 | 1,45756e-10 |
| L-Sign | -6,65514e-9 | 1,40408e-6 | 3,91879e-8 |
| Open Palm | 1,04425e-8 | 8,23795e-6 | 1,07485e-8 |
| Pointing Up | 9,05657e-9 | 3,93258e-7 | -6,47176e-9 |
| V-Sign | 5,32920e-7 | 6,91429e-5 | -1,92463e-8 |

### 6.3.4. Observations sur les valeurs brutes

Les moments de Hu n'ont pas tous des ordres de grandeur comparables. Certains sont relativement élevés, tandis que d'autres sont très proches de zéro.

Les moments Hu5, Hu6 et Hu7 peuvent notamment présenter de très petites valeurs absolues et des signes positifs ou négatifs.

Cette différence d'échelle rend les valeurs brutes moins faciles à comparer directement. Une transformation logarithmique signée est donc appliquée dans la sous-étape suivante.

Ces cinq exemples permettent de vérifier le calcul, mais ne suffisent pas à établir la capacité des moments à distinguer les classes.

## 6.4. Transformation logarithmique signée

### 6.4.1. Problème rencontré

Les valeurs brutes des moments de Hu peuvent varier sur plusieurs ordres de grandeur. Une grande partie des moments peut se situer très près de zéro.

Pour faciliter leur exploitation, nous appliquons une transformation logarithmique signée.

### 6.4.2. Formule utilisée

La transformation est définie par :

\[
Hu_i' =
-\operatorname{sign}(Hu_i)
\log_{10}\left(\max(|Hu_i|,\varepsilon)\right)
\]

avec :

\[
\varepsilon = 10^{-30}
\]

où :

- \(Hu_i\) représente le moment brut ;
- \(Hu_i'\) représente le moment transformé ;
- \(\operatorname{sign}\) conserve le signe du moment ;
- \(\varepsilon\) évite de calculer le logarithme de zéro.

Dans le cas d'un moment exactement nul, le résultat de la transformation reste nul.

### 6.4.3. Implémentation

La fonction utilisée dans `src/test_moments_hu.py` est :

```python
def transformer_hu(hu):
    """Applique la transformation logarithmique signée."""
    epsilon = 1e-30
    hu_transformes = (
        -np.sign(hu)
        * np.log10(np.maximum(np.abs(hu), epsilon))
    )
    return hu_transformes
```

### 6.4.4. Résultats transformés

Les valeurs obtenues sur les cinq images de test sont les suivantes :

| Classe | Hu1' | Hu2' | Hu3' | Hu4' |
|---|---:|---:|---:|---:|
| Fist | 0,672931 | 1,731592 | 4,054613 | 5,187592 |
| L-Sign | 0,664349 | 3,136208 | 2,305488 | 4,165286 |
| Open Palm | 0,701788 | 2,080731 | 3,570580 | 4,026019 |
| Pointing Up | 0,737442 | 2,918126 | 2,792296 | 4,371538 |
| V-Sign | 0,647332 | 1,808898 | 2,934070 | 3,204013 |

| Classe | Hu5' | Hu6' | Hu7' |
|---|---:|---:|---:|
| Fist | 10,269687 | -6,791204 | 9,836374 |
| L-Sign | -8,176843 | 5,852609 | 7,406848 |
| Open Palm | 7,981195 | 5,084181 | 7,968653 |
| Pointing Up | 8,043036 | 6,405322 | -8,188978 |
| V-Sign | 6,273338 | 4,160252 | -7,715652 |

### 6.4.5. Interprétation

La transformation réduit les différences d'ordre de grandeur et rend les valeurs plus lisibles.

Elle conserve le signe des moments bruts non nuls : un moment brut positif donne une valeur transformée positive, tandis qu'un moment brut négatif donne une valeur transformée négative.

Cette transformation ne constitue pas une standardisation statistique. Elle ne garantit pas que les sept caractéristiques auront la même moyenne ou la même dispersion.

Si nécessaire, une standardisation supplémentaire pourra être ajustée sur les seules données d'entraînement lors de la préparation du classificateur. Les mêmes paramètres devront ensuite être appliqués aux données de validation et de test, sans les ajuster sur celles-ci.

## 6.5. Comparaison des moments entre les cinq classes

### 6.5.1. Objectif

Cette sous-étape vise à examiner les distributions des sept moments de Hu transformés dans les cinq classes de gestes.

L'objectif est de déterminer si les valeurs présentent des différences entre classes, d'observer leur dispersion et d'identifier les chevauchements ou les valeurs atypiques.

### 6.5.2. Méthodologie

Le script `src/analyser_distribution_hu.py` a été exécuté sur un échantillon de 30 images par classe, avec une graine aléatoire fixée à 42.

Pour chaque image, le script applique la segmentation HSV, les opérations morphologiques, la sélection du contour externe de plus grande aire, le calcul des sept moments de Hu et leur transformation logarithmique signée.

Les caractéristiques sont enregistrées dans un fichier CSV. Des diagrammes en boîte sont ensuite générés pour comparer les distributions.

Les fichiers produits sont :

- `results/distribution_hu_echantillon.csv`
- `figures/distribution_hu_echantillon.png`

### 6.5.3. Résultats de l'extraction

| Classe | Images sélectionnées | Extractions réussies |
|---|---:|---:|
| Fist | 30 | 30 |
| L-Sign | 30 | 30 |
| Open Palm_Open Hand | 30 | 30 |
| Pointing Up | 30 | 30 |
| V-Sign | 30 | 30 |
| **Total** | **150** | **150** |

Les 150 images sélectionnées ont produit des caractéristiques valides. Aucun échec d'extraction n'a été signalé dans cette analyse.

### 6.5.4. Interprétation des diagrammes

L'observation des diagrammes en boîte indique que les médianes diffèrent entre les classes pour les sept moments. Toutefois, plusieurs distributions se chevauchent.

- **Hu1 :** les médianes se situent à des niveaux différents et les boîtes présentent un chevauchement partiel.
- **Hu2 :** Fist, L-Sign et Open Palm présentent des recouvrements. Pointing Up et V-Sign apparaissent plus bas sur le graphique.
- **Hu3 :** plusieurs classes présentent des distributions qui se chevauchent malgré des différences de médianes.
- **Hu4 :** Fist et Open Palm sont proches et d'autres classes présentent également des recouvrements.
- **Hu5 :** la dispersion est importante et le chevauchement marqué entre plusieurs classes.
- **Hu6 :** certaines classes sont proches, tandis que V-Sign présente une forte dispersion.
- **Hu7 :** plusieurs classes sont très dispersées, alors que Fist apparaît plutôt plus élevé sur le graphique.

Des points isolés sont visibles pour les sept moments. Ils représentent des valeurs atypiques selon la règle du diagramme en boîte, mais ne constituent pas nécessairement des erreurs.

Ces observations suggèrent que les moments de Hu fournissent des informations différentes sur la forme des mains. Cependant, aucun moment ne semble, pris isolément, séparer parfaitement les cinq classes.

Les diagrammes ne permettent pas à eux seuls de déterminer les performances finales du classificateur. Les sept caractéristiques doivent être évaluées conjointement.

### 6.5.5. Résumé statistique des 150 observations

La validation du fichier `results/distribution_hu_echantillon.csv` a produit le résumé suivant :

| Statistique | Hu1 | Hu2 | Hu3 | Hu4 | Hu5 | Hu6 | Hu7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Moyenne | 0,6537 | 2,0663 | 2,8422 | 3,6424 | 3,6002 | 3,2418 | 0,8048 |
| Écart-type | 0,0585 | 0,5281 | 0,6255 | 0,7342 | 6,3192 | 3,7059 | 7,4558 |
| Minimum | 0,3460 | 1,3609 | 1,0566 | 1,9091 | -10,6605 | -8,0300 | -10,6423 |
| Médiane | 0,6579 | 1,9978 | 2,7463 | 3,5276 | 6,2261 | 4,5548 | 5,9640 |
| Maximum | 0,7837 | 5,1026 | 4,8943 | 5,8154 | 11,2004 | 7,2627 | 11,3228 |

Les écarts-types de Hu5, Hu6 et Hu7 sont nettement plus élevés que ceux de Hu1 à Hu4 dans cet échantillon. Cela traduit une dispersion globale plus importante pour ces caractéristiques.

Un écart-type élevé ne signifie toutefois pas automatiquement qu'une caractéristique est inutile. Il faut également examiner sa capacité à distinguer les classes.

### 6.5.6. Limites de l'analyse

L'analyse est exploratoire et porte sur un échantillon de 30 images par classe. Les résultats ne constituent pas une évaluation du classificateur.

Le dataset contient des doublons exacts. Leur présence peut influencer les distributions observées. Pour la suite du projet, les copies d'une même image devront être regroupées avant la séparation entre les ensembles d'entraînement et de test, afin d'éviter les fuites de données.

La sélection du plus grand contour externe peut également retenir un contour imparfait lorsque la segmentation HSV produit des fragments ou des artefacts. Le fait que les extractions aient réussi ne prouve donc pas que toutes les mains ont été segmentées correctement.

### 6.5.7. Conclusion

Les sept moments de Hu produisent des caractéristiques numériques valides sur l'échantillon étudié. Les distributions présentent des différences entre les classes, mais aussi des chevauchements et des valeurs atypiques.

Ces observations justifient de poursuivre avec les sept caractéristiques conjointement, puis de mesurer leur utilité réelle à l'aide du classificateur k-NN sur des données de test indépendantes.

## 6.6. Validation finale et bilan de l'étape

### 6.6.1. Objectif de la validation

Cette dernière sous-étape consiste à vérifier la structure et la validité numérique du fichier exploratoire produit à la sous-étape 6.5, puis à récapituler les résultats et les limites de l'étape.

Le script utilisé est `src/valider_distribution_hu.py`.

### 6.6.2. Résultats de la validation

| Contrôle | Résultat obtenu | Statut |
|---|---|---|
| Nombre de lignes | 150 | Conforme |
| Nombre de colonnes | 9 | Conforme |
| Classes attendues présentes | 5 | Conforme |
| Nombre d'images par classe | 30 | Conforme |
| Identifiants manquants ou classe manquante | Aucun | Conforme |
| Identifiants répétés dans une même classe | Aucun détecté | Conforme |
| Valeurs manquantes ou non numériques dans les moments | 0 | Conforme |
| Valeurs non finies dans les moments | 0 | Conforme |

Le fichier comporte deux colonnes d'identification (`image_id` et `class`) et sept colonnes de caractéristiques (`Hu1` à `Hu7`).

Le résumé statistique a également été produit sans erreur.

### 6.6.3. Bilan technique

Les vérifications effectuées permettent de confirmer que :

- le calcul des sept moments fonctionne sur les exemples testés ;
- la transformation logarithmique signée a été appliquée ;
- l'analyse exploratoire a réussi sur les 150 images sélectionnées ;
- le fichier CSV contient les cinq classes attendues et les sept caractéristiques numériques ;
- aucune valeur manquante, non numérique ou non finie n'a été détectée dans le fichier validé.

Ces résultats valident le fonctionnement technique des scripts pour les données examinées. Ils ne démontrent pas que la segmentation est correcte sur toutes les images du dataset, ni que les caractéristiques sont suffisantes pour obtenir une bonne reconnaissance.

### 6.6.4. Limites et précautions pour la suite

Trois précautions principales sont retenues :

1. **Qualité des contours :** une extraction réussie ne garantit pas que le contour sélectionné représente parfaitement la main.
2. **Doublons :** les doublons exacts doivent être pris en compte avant la séparation des données pour l'entraînement et le test.
3. **Évaluation du modèle :** la capacité des moments de Hu à reconnaître les cinq gestes devra être mesurée avec un classificateur et des données de test indépendantes.

La standardisation éventuelle des caractéristiques devra être ajustée uniquement sur les données d'entraînement afin d'éviter les fuites d'information.

### 6.6.5. Conclusion générale de l'étape 6

L'étape 6 a permis d'étudier les moments de Hu, de les calculer à partir des contours segmentés, de transformer leurs valeurs et d'analyser leur distribution sur un échantillon de 150 images.

Les contrôles techniques du fichier exploratoire ont été validés : les cinq classes sont représentées à raison de 30 images chacune, et les sept caractéristiques ne présentent aucune valeur manquante ou non finie.

Les résultats montrent que les moments de Hu constituent une représentation compacte de la forme des mains. Cependant, les chevauchements observés entre classes et les limites de la segmentation empêchent de conclure à ce stade sur les performances de reconnaissance.

**Statut de l'étape 6 : validée pour l'extraction exploratoire et les contrôles numériques réalisés.** La validation définitive de l'utilité des caractéristiques reste à effectuer lors de l'évaluation du classificateur.

## Fichiers associés à l'étape 6

| Fichier | Rôle |
|---|---|
| `src/test_moments_hu.py` | Calcul des moments bruts et transformés sur un exemple par classe |
| `src/analyser_distribution_hu.py` | Analyse des distributions sur 150 images |
| `src/valider_distribution_hu.py` | Contrôle structurel et numérique du CSV exploratoire |
| `results/distribution_hu_echantillon.csv` | Caractéristiques calculées sur l'échantillon |
| `figures/distribution_hu_echantillon.png` | Diagrammes en boîte des sept moments |
| `docs/06_moments_hu.md` | Documentation de l'étape 6 |

**Fin de l'étape 6 — Extraction et analyse exploratoire des moments de Hu.**