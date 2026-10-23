# Étape 4 — Prétraitement des images

## 1. Objectif de l'étape

L'objectif de cette étape est de définir et de préparer les images du jeu de données HandGesture-5G afin de les rendre adaptées aux étapes suivantes du système de reconnaissance des gestes de la main.

Le prétraitement doit permettre de disposer d'images lisibles, de dimensions maîtrisées et préparées pour la segmentation dans l'espace colorimétrique HSV.

Cette étape est organisée en quatre sous-étapes :

- **4.1.** Définition des exigences du prétraitement ;
- **4.2.** Comparaison expérimentale des tailles de travail ;
- **4.3.** Définition du pipeline de prétraitement ;
- **4.4.** Implémentation et vérification du prétraitement.

Les sous-étapes 4.1 et 4.2 sont documentées ci-dessous. Les sous-étapes 4.3 et 4.4 seront complétées après la définition et l'exécution du pipeline définitif.

## 2. Sous-étape 4.1 — Définition des exigences du prétraitement

### 2.1. Formats d'images à prendre en charge

L'analyse exploratoire a montré que le jeu de données contient 2 653 fichiers image répartis en trois formats :

| Format | Nombre de fichiers |
|---|---:|
| JPG | 1 758 |
| JPEG | 885 |
| HEIC | 10 |
| **Total** | **2 653** |

Le prétraitement doit donc prendre en charge ces trois formats. Les images JPG et JPEG peuvent être lues avec OpenCV, tandis que les fichiers HEIC nécessitent un mécanisme de lecture adapté, notamment Pillow associé à `pillow-heif`.

### 2.2. Vérification des images

Avant leur traitement, les images doivent être vérifiées afin de s'assurer qu'elles peuvent être lues correctement et qu'elles disposent d'une représentation couleur exploitable.

L'analyse d'intégrité réalisée à l'étape précédente a fourni les résultats suivants :

| Contrôle | Résultat |
|---|---:|
| Fichiers examinés | 2 653 |
| Fichiers vides | 0 |
| Images illisibles | 0 |
| Images de dimensions suspectes selon le seuil utilisé | 0 |
| Images totalement uniformes | 0 |
| Fichiers non analysés | 0 |

Pour ce contrôle, une dimension était considérée comme suspecte lorsqu'au moins un côté de l'image était inférieur à 100 pixels. Ce seuil constitue un critère de vérification utilisé dans le projet et non une règle universelle de qualité d'image.

Ces résultats indiquent que les fichiers examinés ont pu être lus et analysés avec les outils utilisés. Ils ne garantissent cependant pas que toutes les images présentent des conditions visuelles identiques ou une qualité parfaite.

### 2.3. Redimensionnement des images

Les images originales possèdent des dimensions élevées :

| Dimensions originales | Nombre d'images |
|---|---:|
| 6048 × 6048 pixels | 2 292 |
| 3024 × 3024 pixels | 356 |
| 3024 × 4032 pixels | 5 |
| **Total** | **2 653** |

Ces dimensions représentent un coût de traitement potentiellement important, notamment lors de la conversion des couleurs et de la segmentation.

Le prétraitement doit donc réduire la taille des images tout en conservant les proportions de la main. Il ne faut pas imposer systématiquement une image carrée, car cela pourrait déformer les gestes.

La taille de travail sera définie comme une **dimension maximale du plus grand côté**, et non comme une obligation d'obtenir des images carrées. Les images plus petites que la taille retenue ne devront pas être agrandies inutilement.

Pour la réduction, l'interpolation `cv2.INTER_AREA` sera utilisée, car elle convient au redimensionnement des images vers des dimensions plus petites.

### 2.4. Conversion vers l'espace colorimétrique HSV

Après la lecture et le redimensionnement, les images couleur seront converties de l'espace BGR utilisé par OpenCV vers l'espace colorimétrique HSV.

HSV représente la couleur à l'aide de trois composantes :

- **H (Hue)** : la teinte, qui représente la famille de couleur ;
- **S (Saturation)** : la saturation, qui indique l'intensité de la couleur ;
- **V (Value)** : la valeur, qui représente la luminosité.

Cet espace sera utilisé à l'étape suivante pour étudier la segmentation de la peau. La conversion vers HSV fait donc partie de la préparation des données, mais **la segmentation de la peau ne sera pas effectuée dans cette étape**.

### 2.5. Conservation des données originales

Les images du répertoire `data/raw` doivent rester inchangées. Les résultats du prétraitement seront enregistrés séparément afin de préserver les données d'origine et de permettre la reproduction des expériences.

L'organisation définitive des fichiers de sortie sera précisée dans la sous-étape 4.3.

## 3. Sous-étape 4.2 — Comparaison expérimentale des tailles de travail

### 3.1. Objectif de la comparaison

Une comparaison a été réalisée pour déterminer une taille de travail qui préserve suffisamment la forme des gestes tout en réduisant le coût du traitement.

Deux tailles maximales ont été étudiées :

- 256 pixels ;
- 512 pixels.

La comparaison a porté sur cinq images, avec une image sélectionnée dans chacune des cinq classes du jeu de données : Fist, L-Sign, Open Palm, Pointing Up et V-Sign.

Les images ont été redimensionnées en conservant leur rapport d'aspect. Une taille maximale de 256 ou de 512 pixels correspond donc à une limite appliquée au plus grand côté de l'image, et non nécessairement à ses deux dimensions.

Le script utilisé pour cette expérimentation est `src/compare_working_sizes.py`. Les résultats ont été enregistrés dans `results/comparaison_tailles.csv`, et la planche comparative a été générée dans `figures/comparaison_tailles_pretraitement.png`.

### 3.2. Comparaison visuelle

L'observation de la planche comparative montre que les cinq gestes restent reconnaissables à 256 pixels. Les doigts, les espaces entre eux et les contours principaux de la main restent suffisamment visibles pour envisager la poursuite du traitement.

À 512 pixels, certains détails sont légèrement plus fins. Toutefois, la comparaison visuelle réalisée ne met pas en évidence une amélioration suffisamment importante pour justifier, à elle seule, le choix de cette taille plus élevée.

Ces observations concernent uniquement les cinq images sélectionnées pour cette expérimentation. Elles ne constituent pas une validation visuelle exhaustive de toutes les images du jeu de données.

### 3.3. Mesure du temps de conversion HSV

Le script a également mesuré le temps nécessaire à la conversion des images redimensionnées de BGR vers HSV. Les résultats obtenus sont présentés ci-dessous.

| Classe | Temps à 256 px (ms) | Temps à 512 px (ms) |
|---|---:|---:|
| Fist | 0,0830 | 0,3345 |
| L-Sign | 0,1175 | 0,2661 |
| Open Palm | 0,1194 | 0,2696 |
| Pointing Up | 0,0742 | 0,2308 |
| V-Sign | 0,0787 | 0,2744 |
| **Moyenne** | **0,0946** | **0,2751** |

Le temps moyen mesuré est d'environ 0,0946 ms à 256 pixels et de 0,2751 ms à 512 pixels.

Dans cet essai, la conversion à 256 pixels est environ 2,9 fois plus rapide que celle à 512 pixels. Cela représente une réduction d'environ 65,6 % du temps moyen de conversion mesuré.

Ces temps concernent uniquement l'opération de conversion BGR vers HSV, et non l'ensemble du prétraitement. Les mesures ont été réalisées sur cinq images ; elles doivent donc être interprétées comme des résultats préliminaires plutôt que comme une estimation définitive des performances sur l'ensemble du jeu de données.

### 3.4. Choix de la taille de travail

À la suite de la comparaison visuelle et temporelle, la taille maximale de **256 pixels** est retenue pour la suite du projet.

Ce choix repose sur trois éléments :

1. Les cinq gestes restent reconnaissables à cette taille.
2. Les temps de conversion HSV mesurés sont inférieurs à ceux obtenus à 512 pixels pour les cinq images testées.
3. Une image de 256 × 256 pixels contient quatre fois moins de pixels qu'une image de 512 × 512 pixels.

Le dernier point concerne une comparaison de dimensions carrées. Pour les images dont les proportions ne sont pas carrées, le nombre réel de pixels dépendra des dimensions après redimensionnement.

La taille retenue constitue un compromis expérimental entre la préservation de la forme et le coût de traitement. Elle pourra être réévaluée si les résultats ultérieurs de segmentation ou de reconnaissance révèlent une perte importante d'information.

## 4. Sous-étape 4.3 — Définition du pipeline de prétraitement

### 4.3.1. Pipeline retenu

À la suite de l'analyse exploratoire et de la comparaison expérimentale des tailles de travail, le pipeline de prétraitement suivant a été retenu :

1. Lecture de l'image originale au format JPG, JPEG ou HEIC.
2. Vérification de la validité de l'image et de ses dimensions.
3. Conversion de l'image en représentation couleur BGR compatible avec OpenCV.
4. Redimensionnement avec une taille maximale de 256 pixels, en conservant les proportions.
5. Conversion de l'image redimensionnée de BGR vers HSV.
6. Enregistrement des résultats dans des répertoires distincts.
7. Génération d'un rapport CSV permettant de suivre le traitement de chaque fichier.

#### Schéma du pipeline de prétraitement

```text
             ┌───────────────────────────┐
             │      Images originales   │
             │      JPG / JPEG / HEIC    │
             │        data/raw/           │
             └─────────────┬─────────────┘
                           │
                           ▼
             ┌───────────────────────────┐
             │      Lecture des images   │
             │  Pillow / pillow-heif     │
             └─────────────┬─────────────┘
                           │
                           ▼
             ┌───────────────────────────┐
             │   Vérification de validité│
             │  Dimensions et 3 canaux   │
             └─────────────┬─────────────┘
                           │
                           ▼
             ┌───────────────────────────┐
             │   Conversion vers BGR     │
             │  Format couleur OpenCV    │
             └─────────────┬─────────────┘
                           │
                           ▼
             ┌───────────────────────────┐
             │      Redimensionnement   │
             │  Côté maximal : 256 px    │
             │  Proportions conservées   │
             │  Interpolation INTER_AREA │
             └─────────────┬─────────────┘
                           │
                           ▼
             ┌───────────────────────────┐
             │     Conversion BGR → HSV  │
             │  Préparation à la suite   │
             └─────────────┬─────────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │ Résultats obtenus│
                 └────────┬────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
   ┌─────────────────────┐  ┌─────────────────────┐
   │ Image couleur       │  │ Tableau numérique   │
   │ redimensionnée      │  │ HSV                 │
   │ PNG                 │  │ NPY                 │
   └──────────┬──────────┘  └──────────┬──────────┘
              │                        │
              ▼                        ▼
   data/processed/color/    data/processed/hsv/

             ┌───────────────────────────┐
             │       Rapport CSV         │
             │ Statut, dimensions, erreurs│
             └───────────────────────────┘
```

**Remarque :** le schéma présente les opérations réellement prévues par le script. Le rapport CSV est produit en parallèle de la sauvegarde des résultats. La segmentation de la peau n'est pas incluse dans ce pipeline, car elle appartient à l'étape suivante.


La lecture des fichiers HEIC est assurée par Pillow et `pillow-heif`. L'interpolation `cv2.INTER_AREA` est utilisée pour réduire les dimensions des images.

### 4.3.2. Organisation des résultats

Les images originales sont conservées dans `data/raw`. Les résultats sont enregistrés séparément dans les répertoires suivants :

| Répertoire ou fichier | Rôle |
|---|---|
| `data/processed/color/` | Images couleur redimensionnées, enregistrées au format PNG |
| `data/processed/hsv/` | Tableaux numériques HSV enregistrés au format NumPy `.npy` |
| `results/pretraitement_rapport.csv` | Rapport de traitement de chaque image |

La structure des sous-dossiers correspondant aux classes est conservée. Les noms de sortie intègrent le nom et l'extension d'origine afin de limiter les risques de collision entre fichiers possédant le même nom de base.

Le format PNG est utilisé pour conserver une représentation couleur facilement visualisable. Le format `.npy` permet de sauvegarder directement les valeurs numériques HSV, sans les interpréter comme des valeurs RGB.

Cette étape prépare les données pour la segmentation. Aucun masque de peau n'est produit à ce stade.

## 5. Sous-étape 4.4 — Implémentation et vérification du prétraitement

### 4.4.1. Implémentation

Le pipeline a été implémenté dans le script `src/preprocess_dataset.py`.

Le script parcourt récursivement le répertoire `data/raw`, sélectionne les fichiers JPG, JPEG et HEIC, effectue les opérations de prétraitement et enregistre les résultats dans les répertoires de sortie prévus.

Pour chaque image, le rapport CSV conserve notamment :

- le chemin du fichier source ;
- le format et le mode de l'image ;
- les dimensions originales et finales ;
- le nombre de canaux HSV ;
- les chemins des fichiers générés ;
- le statut du traitement et, le cas échéant, le message d'erreur.

La commande utilisée pour lancer le traitement est :

```powershell
python src/preprocess_dataset.py
```

### 4.4.2. Résultats de l'exécution

L'exécution du script sur le jeu de données a produit les résultats suivants :

| Indicateur | Résultat |
|---|---:|
| Images trouvées | 2 653 |
| Traitements réussis | 2 653 |
| Échecs signalés | 0 |
| Taux de réussite déclaré | 100 % |
| Taille maximale de travail | 256 pixels |

Le regroupement des lignes du rapport CSV par statut a confirmé la présence de 2 653 traitements `SUCCES` et aucun traitement déclaré en erreur.

La consultation de plusieurs lignes du rapport a également permis d'observer le passage de dimensions originales de 6048 × 6048 pixels à des dimensions finales de 256 × 256 pixels. Les dernières images HEIC ont aussi été traitées avec succès.

### 4.4.3. Vérification des fichiers générés

Après l'exécution, les répertoires et le rapport suivants ont été vérifiés :

- `data/processed/color/` ;
- `data/processed/hsv/` ;
- `results/pretraitement_rapport.csv`.

Les fichiers de sortie sont présents et l'organisation attendue a été retrouvée. Le script écrit dans les répertoires de sortie et ne contient aucune opération destinée à modifier ou supprimer les fichiers de `data/raw`.

Les résultats disponibles confirment donc la réussite de l'exécution et la production des sorties prévues. Un contrôle automatique exhaustif des dimensions, des formes des tableaux HSV et de l'existence de chaque fichier référencé dans le rapport pourra compléter cette vérification.

### 4.4.4. Limites

La réussite du prétraitement ne signifie pas que les images sont déjà segmentées ni que le système de reconnaissance fonctionne. Les données sont préparées pour les étapes suivantes, mais les seuils HSV, les opérations morphologiques et la sélection du contour de la main restent à définir et à évaluer.

Par ailleurs, les doublons exacts identifiés lors de l'analyse exploratoire n'ont pas été supprimés à cette étape. Leur gestion devra être prise en compte lors de la préparation des ensembles d'entraînement et de test afin d'éviter qu'une même image, ou une copie exacte, apparaisse dans les deux ensembles.

## 6. Bilan de l'étape 4

L'étape de prétraitement a permis de définir puis d'implémenter un pipeline adapté au jeu de données HandGesture-5G. Le choix d'une taille maximale de 256 pixels a été motivé par une comparaison visuelle et par des mesures préliminaires du temps de conversion HSV.

Le script `src/preprocess_dataset.py` a ensuite été exécuté sur les 2 653 images détectées. Les 2 653 traitements ont été déclarés réussis, sans erreur signalée. Les images couleur redimensionnées, les tableaux HSV et le rapport CSV ont été générés dans les répertoires prévus, sans écriture dans `data/raw`.

L'étape 4 est donc **terminée au niveau de l'exécution du pipeline**. Le contrôle automatique exhaustif des propriétés de toutes les sorties reste une vérification complémentaire utile avant de poursuivre.

La prochaine étape consistera à étudier la segmentation de la peau dans l'espace HSV. Elle devra définir et évaluer les seuils de couleur, puis examiner les effets des opérations morphologiques sur les masques obtenus.
