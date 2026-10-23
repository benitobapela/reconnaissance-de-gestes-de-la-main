# Étape 5 — Segmentation HSV et extraction du contour de la main

## 5.1 — Réalisation de la segmentation HSV initiale

### 5.1.1. Objectif

L'objectif de cette étape est d'isoler la main du fond de l'image afin d'obtenir une silhouette exploitable pour les étapes suivantes du système de reconnaissance de gestes.

La segmentation transforme une image couleur en un masque binaire dans lequel les pixels correspondant aux régions détectées comme peau sont représentés en blanc, tandis que les autres pixels sont représentés en noir.

L'approche choisie utilise l'espace de couleur HSV, qui représente la couleur selon trois composantes :

- **H (Hue)** : teinte ;
- **S (Saturation)** : saturation de la couleur ;
- **V (Value)** : luminosité.

Cette représentation est utilisée pour effectuer un seuillage des couleurs susceptibles de correspondre à la peau.

### 5.1.2. Seuils HSV initiaux

Les seuils utilisés pour la première expérimentation sont :

```python
LOWER_HSV = np.array([0, 20, 40], dtype=np.uint8)
UPPER_HSV = np.array([25, 255, 255], dtype=np.uint8)
```

Les pixels dont les composantes HSV se trouvent dans cet intervalle sont retenus dans le masque.

La fonction OpenCV utilisée est `cv2.inRange()`.

À cette étape, aucune opération morphologique n'est appliquée. Le but est d'observer le résultat direct du seuillage avant le nettoyage du masque.

### 5.1.3. Protocole expérimental

Une image a été sélectionnée dans chacune des cinq classes du dataset :

- `Fist` ;
- `L-Sign` ;
- `Open Palm_Open Hand` ;
- `Pointing Up` ;
- `V-Sign`.

La sélection a été réalisée avec une graine aléatoire fixe (`seed = 42`) afin de rendre l'expérience reproductible.

Les critères d'observation sont :

- la conservation de la main ;
- la séparation entre la main et le fond ;
- la conservation des espaces entre les doigts ;
- la présence de trous dans la silhouette ;
- la présence de pixels parasites ;
- le nombre de contours externes ;
- la proportion de pixels blancs ;
- l'aire du plus grand contour.

Les résultats ont été enregistrés dans :

- `figures/masques_hsv_initiaux.png` ;
- `results/evaluation_hsv_initiale.csv`.

### 5.1.4. Observations

**Fist**

La main et le poignet sont largement conservés. Le fond reste majoritairement noir. Des trous apparaissent entre les doigts repliés et quelques lacunes sont visibles près du pouce. La silhouette reste globalement continue.

**L-Sign**

La main et les deux doigts tendus sont bien conservés. Les espaces entre les doigts restent noirs. Un petit trou est visible à l'extrémité de l'index.

**Open Palm_Open Hand**

La paume et les doigts sont largement conservés. Le masque est relativement propre, avec quelques pixels parasites autour du poignet et près du pouce.

**Pointing Up**

Le doigt pointé est conservé, mais la zone du poignet est plus problématique. Le bracelet et une partie de l'avant-bras entraînent des défauts de segmentation, donnant une impression de fragmentation dans cette zone.

**V-Sign**

Les deux doigts sont détectés et l'espace entre eux reste généralement noir. Des trous et des pixels parasites sont néanmoins visibles autour des doigts et près du bord droit de la main.

### 5.1.5. Bilan de la segmentation initiale

La segmentation HSV permet de conserver une grande partie des mains tout en éliminant largement le fond sur les exemples étudiés.

Cependant, les masques présentent encore des trous, des coupures locales et des pixels parasites. Les difficultés les plus visibles concernent `Pointing Up` et `V-Sign`.

Ces résultats sont encourageants, mais ils ne constituent pas une validation sur l'ensemble du dataset : seules cinq images ont été examinées visuellement.

---

## 5.2 — Principe et paramètres du seuillage HSV

Le seuillage HSV consiste à sélectionner les pixels dont les valeurs H, S et V se trouvent dans un intervalle donné.

La fonction `cv2.inRange()` produit un masque binaire :

- pixel blanc (255) : pixel retenu par le seuillage ;
- pixel noir (0) : pixel rejeté.

Le masque obtenu ne garantit pas que tous les pixels blancs correspondent à la main. Des éléments du fond peuvent présenter des couleurs similaires à celles de la peau. Inversement, certaines parties de la main peuvent être rejetées en raison des variations de luminosité, des ombres ou d'autres éléments présents dans l'image.

C'est pourquoi le masque initial est ensuite étudié et nettoyé à l'aide d'opérations morphologiques.

Les seuils HSV initiaux ont été conservés pendant les expérimentations morphologiques afin de comparer l'effet des kernels sans modifier simultanément plusieurs paramètres.

---

## 5.3 — Tester les opérations morphologiques et les kernels

### 5.3.1. Objectif

Cette sous-étape vise à réduire les défauts présents dans les masques initiaux tout en préservant la forme générale de la main et les espaces entre les doigts.

Deux opérations morphologiques ont été étudiées :

- **Fermeture (`MORPH_CLOSE`)** : permet notamment de combler de petites lacunes et de rapprocher des régions blanches proches.
- **Ouverture (`MORPH_OPEN`)** : permet notamment de supprimer de petits éléments blancs isolés.

Le résultat dépend de la taille et de la forme du kernel utilisé. Un kernel trop grand peut modifier la silhouette ou resserrer des espaces qui sont importants pour distinguer les gestes.

### 5.3.2. Configurations testées

Quatre configurations ont été comparées :

| Configuration | Fermeture | Ouverture |
|---|---|---|
| A | Kernel elliptique 3×3 | Aucune |
| B | Kernel elliptique 5×5 | Aucune |
| C | Kernel elliptique 5×5 | Kernel elliptique 3×3 |
| D | Kernel elliptique 7×7 | Kernel elliptique 3×3 |

Chaque opération a été appliquée une fois.

La configuration C correspond à une fermeture avec un kernel elliptique 5×5, suivie d'une ouverture avec un kernel elliptique 3×3.

### 5.3.3. Protocole expérimental

Les quatre configurations ont été appliquées aux mêmes cinq images que celles utilisées pour la segmentation initiale.

Les mesures suivantes ont été calculées :

- nombre de pixels blancs ;
- proportion de pixels blancs ;
- nombre de contours externes ;
- aire du plus grand contour en pourcentage de l'image.

Une comparaison visuelle a également été réalisée pour vérifier la conservation des doigts, la réduction des parasites et l'évolution des trous.

Les résultats sont disponibles dans :

- `results/evaluation_morphologie_hsv.csv` ;
- `figures/comparaison_morphologie_hsv.png`.

Les masques produits sont enregistrés dans `data/processed/masks_hsv_morphologie/`.

### 5.3.4. Résultats quantitatifs

Les valeurs ci-dessous correspondent aux cinq images testées. La proportion blanche est calculée par rapport au nombre total de pixels de l'image. L'aire du plus grand contour est également exprimée en pourcentage de l'image.

**Fist**

| Configuration | Pixels blancs (%) | Contours externes | Aire du plus grand contour (%) |
|---|---:|---:|---:|
| Initial | 35,362 | 1 | 35,711 |
| A | 35,420 | 1 | 35,720 |
| B | 35,602 | 1 | 35,739 |
| C | 35,599 | 1 | 35,738 |
| D | 35,654 | 1 | 35,758 |

Les variations restent faibles. La fermeture améliore légèrement la continuité de la silhouette, mais le trou central reste visible.

**L-Sign**

| Configuration | Pixels blancs (%) | Contours externes | Aire du plus grand contour (%) |
|---|---:|---:|---:|
| Initial | 32,027 | 1 | 31,485 |
| A | 32,036 | 1 | 31,485 |
| B | 32,050 | 1 | 31,496 |
| C | 32,047 | 1 | 31,495 |
| D | 32,056 | 1 | 31,504 |

Les différences sont minimes. Les doigts restent séparés dans les quatre configurations.

**Open Palm_Open Hand**

| Configuration | Pixels blancs (%) | Contours externes | Aire du plus grand contour (%) |
|---|---:|---:|---:|
| Initial | 43,448 | 2 | 42,471 |
| A | 43,459 | 2 | 42,483 |
| B | 43,468 | 2 | 42,489 |
| C | 43,465 | 2 | 42,488 |
| D | 43,695 | 2 | 42,761 |

La configuration D entraîne une modification plus importante de la silhouette. Visuellement, certains espaces étroits entre les doigts semblent davantage resserrés.

**Pointing Up**

| Configuration | Pixels blancs (%) | Contours externes | Aire du plus grand contour (%) |
|---|---:|---:|---:|
| Initial | 10,378 | 3 | 9,653 |
| A | 10,384 | 3 | 9,658 |
| B | 10,397 | 3 | 9,669 |
| C | 10,370 | 3 | 9,667 |
| D | 10,379 | 3 | 9,674 |

Le nombre de contours reste égal à trois dans toutes les configurations. Le défaut de segmentation près du poignet et du bracelet n'est pas corrigé par ces opérations.

**V-Sign**

| Configuration | Pixels blancs (%) | Contours externes | Aire du plus grand contour (%) |
|---|---:|---:|---:|
| Initial | 23,969 | 17 | 23,400 |
| A | 24,110 | 12 | 23,521 |
| B | 24,208 | 13 | 23,598 |
| C | 24,149 | 3 | 23,589 |
| D | 24,197 | 3 | 23,636 |

La configuration C réduit le nombre de contours externes de 17 à 3 sur l'exemple étudié. La configuration D aboutit également à trois contours.

### 5.3.5. Analyse visuelle

Les observations de la planche comparative sont les suivantes :

- **Fist :** les petites coupures sont légèrement réduites, mais le trou central persiste.
- **L-Sign :** les doigts restent séparés dans toutes les configurations.
- **Open Palm_Open Hand :** les espaces entre les doigts restent globalement noirs ; le kernel 7×7 tend à davantage resserrer les espaces étroits.
- **Pointing Up :** le défaut du poignet reste visible malgré les opérations morphologiques.
- **V-Sign :** les configurations comprenant une ouverture réduisent fortement le nombre de petits contours externes ; les deux doigts restent séparés visuellement.

### 5.3.6. Choix de la configuration morphologique

La configuration C, composée d'une fermeture 5×5 suivie d'une ouverture 3×3, est retenue comme **configuration candidate** pour la suite du pipeline.

Ce choix repose sur le compromis observé entre réduction des petits parasites, conservation de la silhouette et maintien des espaces entre les doigts. La configuration D produit également un masque plus propre sur certains aspects, mais son kernel de fermeture plus grand entraîne un lissage plus marqué.

Ce choix reste provisoire, car les essais ont été réalisés sur une seule image par classe. Il devra être vérifié sur un échantillon plus large avant de conclure sur sa robustesse.

---

## 5.4 — Détecter et sélectionner le contour de la main

### 5.4.1. Objectif

Après la segmentation et le nettoyage morphologique, l'objectif est d'extraire le contour correspondant à la silhouette principale de la main.

Cette étape est importante, car les moments de Hu utilisés ultérieurement seront calculés à partir de la forme retenue.

Le principe consiste à détecter les contours externes du masque, à mesurer leur aire et à les classer par ordre décroissant.

### 5.4.2. Méthode utilisée

La configuration morphologique C est utilisée pour produire le masque d'entrée.

La fonction `cv2.findContours()` avec `cv2.RETR_EXTERNAL` et `cv2.CHAIN_APPROX_SIMPLE` permet de détecter les contours externes.

Les contours sont ensuite triés par aire décroissante. Le premier contour de la liste est considéré comme le contour principal candidat.

L'aire du contour est calculée avec `cv2.contourArea()`. Sa boîte englobante est obtenue avec `cv2.boundingRect()`.

Le pipeline est le suivant :

```text
Masque HSV initial
        ↓
Fermeture 5×5
        ↓
Ouverture 3×3
        ↓
Détection des contours externes
        ↓
Tri par aire décroissante
        ↓
Sélection du plus grand contour
        ↓
Vérification visuelle
```

### 5.4.3. Protocole expérimental

Le script `src/test_contour_principal.py` a été exécuté sur les mêmes cinq images.

Pour chaque image, il a enregistré les mesures des contours détectés :

- aire ;
- aire en pourcentage de l'image ;
- périmètre ;
- position ;
- largeur et hauteur de la boîte englobante ;
- coordonnées du centre de cette boîte.

Les résultats sont disponibles dans :

- `results/evaluation_contour_principal.csv` ;
- `figures/selection_contour_principal.png`.

### 5.4.4. Résultats

| Classe | Nombre de contours | Aire du plus grand contour (%) |
|---|---:|---:|
| Fist | 1 | 35,738 |
| L-Sign | 1 | 31,495 |
| Open Palm_Open Hand | 2 | 42,488 |
| Pointing Up | 3 | 9,667 |
| V-Sign | 3 | 23,589 |

Les résultats détaillés sont enregistrés dans le rapport CSV.

### 5.4.5. Vérification visuelle

**Fist**

Le plus grand contour correspond au poing et au poignet. Le petit trou central n'empêche pas la sélection de la silhouette principale.

**L-Sign**

Le contour principal englobe la main et les deux doigts. Le pouce et l'index restent dans une même silhouette.

**Open Palm_Open Hand**

Le contour principal conserve la paume et les doigts. Un petit élément proche du bord inférieur reste séparé et constitue le deuxième contour externe.

**Pointing Up**

Trois contours externes sont détectés. Le contour principal encadre la partie principale de la main, mais les éléments du poignet et du bracelet restent fragmentés en contours distincts. Le contour sélectionné ne représente donc pas nécessairement la totalité de la main jusqu'au poignet.

**V-Sign**

Le contour principal englobe les deux doigts et la main. Deux petits éléments parasites restent séparés et ne sont pas sélectionnés comme contour principal.

### 5.4.6. Limites de la sélection du plus grand contour

La sélection du contour ayant la plus grande aire fonctionne correctement sur les exemples testés pour identifier la silhouette principale de la main.

Cependant, cette méthode ne permet pas de récupérer automatiquement les parties détachées par la segmentation. Dans le cas de `Pointing Up`, les fragments du poignet restent séparés : choisir le plus grand contour ne les réunit pas à la silhouette principale.

Par ailleurs, le nombre de contours externes ne décrit pas tous les défauts possibles du masque. Un trou noir à l'intérieur d'une région blanche n'est pas compté comme un contour externe lorsque `cv2.RETR_EXTERNAL` est utilisé.

La méthode est donc retenue comme règle de sélection candidate, avec une vérification visuelle et une validation ultérieure sur un échantillon plus large.

---

## 5.5 — Vérifier les résultats et conclure l'étape 5

### 5.5.1. Vérification de l'exécution

Les scripts expérimentaux suivants ont été exécutés :

- `src/test_segmentation_hsv_initiale.py` ;
- `src/test_morphologie_hsv.py` ;
- `src/test_contour_principal.py`.

Les sorties attendues ont été produites sans erreur lors des exécutions rapportées :

- masques binaires initiaux ;
- planche comparative des opérations morphologiques ;
- rapport CSV des mesures morphologiques ;
- planche de sélection du contour principal ;
- rapport CSV des mesures des contours.

### 5.5.2. Bilan des résultats

Les expérimentations permettent de retenir les conclusions suivantes :

1. Le seuillage HSV conserve correctement une grande partie de la main sur les cinq exemples sélectionnés.
2. Les masques initiaux comportent des trous, des parasites et des défauts locaux.
3. La fermeture 5×5 suivie d'une ouverture 3×3 constitue une configuration morphologique candidate offrant un compromis intéressant sur les exemples observés.
4. Le plus grand contour correspond à la silhouette principale de la main sur les cinq exemples examinés.
5. Les fragments détachés du poignet dans `Pointing Up` ne sont pas récupérés par la seule sélection du plus grand contour.

### 5.5.3. Limites de validation

Les expérimentations visuelles détaillées de cette étape portent sur cinq images, soit une image par classe. Elles permettent de comparer les méthodes et de repérer les difficultés, mais elles ne suffisent pas à démontrer que les résultats sont robustes sur l'ensemble des 2 653 fichiers du dataset.

En particulier :

- les seuils HSV n'ont pas encore été évalués quantitativement sur tout le dataset ;
- les opérations morphologiques peuvent produire des effets différents selon l'image ;
- le problème du poignet observé sur `Pointing Up` reste à traiter ou à accepter explicitement dans les limites du pipeline ;
- le plus grand contour devra être contrôlé sur un échantillon plus représentatif avant la validation finale.

Aucun score de précision de segmentation n'est revendiqué à ce stade, car aucune comparaison avec des masques de référence annotés n'a été réalisée.

### 5.5.4. Conclusion de l'étape 5

Cette étape a permis de construire et de tester une première chaîne de segmentation de la main à partir de l'espace HSV.

La méthode retenue provisoirement est :

```text
Image prétraitée
        ↓
Représentation HSV
        ↓
Seuillage HSV
        ↓
Masque binaire initial
        ↓
Fermeture morphologique 5×5
        ↓
Ouverture morphologique 3×3
        ↓
Détection des contours externes
        ↓
Sélection du contour de plus grande aire
        ↓
Vérification visuelle
```

Cette chaîne constitue une base pour la prochaine étape, consacrée à l'extraction des caractéristiques géométriques, notamment les moments de Hu.

La segmentation reste une étape perfectible : les résultats devront être vérifiés sur un échantillon plus large et les limites observées devront être prises en compte dans l'interprétation des performances finales du système.
