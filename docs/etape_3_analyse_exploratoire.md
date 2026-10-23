# Étape 3 — Analyse exploratoire et validation préliminaire du dataset

## 1. Objectifs de l'étape

L'objectif de cette étape est de comprendre la structure, les caractéristiques et la qualité du dataset **HandGesture-5G** avant de mettre en œuvre le pipeline de reconnaissance des gestes de la main.

Cette analyse vise notamment à :

- étudier la répartition des images entre les cinq classes ;
- examiner les dimensions, les formats et les canaux couleur des images ;
- observer visuellement les teints, les arrière-plans, le cadrage, l'éclairage et la netteté ;
- vérifier l'intégrité des fichiers et rechercher les doublons exacts ;
- évaluer, sur un petit échantillon, la faisabilité d'une segmentation de la peau dans l'espace colorimétrique HSV.

Les résultats obtenus permettront d'identifier les difficultés potentielles et de préparer les choix techniques des étapes suivantes.

## 2. Présentation et répartition du dataset

Le dataset HandGesture-5G utilisé dans ce projet contient initialement **2 653 images couleur**, réparties entre cinq classes de gestes de la main.

### 2.1. Répartition des classes

| Classe | Nombre d'images | Proportion approximative |
|---|---:|---:|
| Open Palm_Open Hand | 661 | 24,9 % |
| Fist | 542 | 20,4 % |
| L-Sign | 537 | 20,2 % |
| V-Sign | 508 | 19,2 % |
| Pointing Up | 405 | 15,3 % |
| **Total** | **2 653** | **100 %** |

La classe `Open Palm_Open Hand` est la plus représentée, avec 661 images, tandis que `Pointing Up` est la moins représentée, avec 405 images.

Le dataset présente donc un déséquilibre modéré entre les classes. Cet écart devra être pris en compte lors de la préparation des données et de l'évaluation du classificateur.

Lors de la constitution des ensembles d'entraînement et de test, une séparation stratifiée pourra être utilisée afin de préserver autant que possible la représentation de chaque classe. Cette séparation devra également tenir compte des doublons exacts identifiés dans la sous-section 5.

**Remarque :** les effectifs présentés ici correspondent aux fichiers initialement recensés dans les dossiers de classes, avant toute déduplication.

## 3. Analyse quantitative des images

### 3.1. Dimensions et résolution

L'analyse automatique des 2 653 images a révélé trois dimensions distinctes.

| Dimensions | Nombre d'images | Proportion approximative |
|---|---:|---:|
| 6048 × 6048 pixels | 2 292 | 86,4 % |
| 3024 × 3024 pixels | 356 | 13,4 % |
| 3024 × 4032 pixels | 5 | 0,2 % |
| **Total** | **2 653** | **100 %** |

La grande majorité des images possède une résolution de 6048 × 6048 pixels. Les images sont donc principalement carrées et de très haute résolution.

Cette résolution constitue une source potentielle de coût de calcul. Les opérations de conversion colorimétrique, de segmentation et de traitement des contours pourraient être plus coûteuses si elles étaient directement appliquées aux images originales.

Un redimensionnement sera donc étudié à l'étape suivante. La taille cible devra représenter un compromis entre le temps de calcul et la conservation des informations utiles à la segmentation et à l'extraction des caractéristiques.

Il faudra également éviter de déformer les proportions des mains. Lorsque les dimensions ne sont pas identiques, un redimensionnement conservant le rapport d'aspect devra être privilégié, sauf justification expérimentale contraire.

### 3.2. Formats des fichiers

L'examen des extensions a identifié les formats suivants :

| Format | Nombre de fichiers |
|---|---:|
| JPEG (`.jpg`) | 1 758 |
| JPEG (`.jpeg`) | 885 |
| HEIC (`.HEIC`) | 10 |
| **Total** | **2 653** |

Le dataset est donc constitué majoritairement de fichiers JPEG. Dix images utilisent le format HEIC, dont la lecture nécessite un traitement spécifique dans l'environnement Python utilisé.

La bibliothèque Pillow, associée à `pillow-heif`, a permis de lire ces fichiers. Pour les opérations réalisées avec OpenCV, les images HEIC doivent être décodées puis converties dans le format couleur attendu par les traitements suivants.

Cette différence de format devra être prise en compte dans le pipeline afin que toutes les images puissent être traitées de manière cohérente.

### 3.3. Vérification des canaux couleur et de la lisibilité

Le script `src/check_image_channels.py` a été utilisé pour examiner les images et vérifier leur représentation couleur.

Les résultats indiquent que les 2 653 images ont été lues et représentées par trois canaux couleur dans les lecteurs utilisés.

Aucun problème de lecture n'a été signalé lors de ce contrôle.

Ces résultats sont compatibles avec une conversion vers l'espace HSV. Il faudra toutefois respecter l'ordre des canaux attendu par la bibliothèque utilisée. En particulier, OpenCV représente généralement les images couleur dans l'ordre BGR, alors que d'autres bibliothèques, comme Pillow, utilisent habituellement l'ordre RGB.

La conversion vers HSV devra donc être effectuée à partir d'une représentation correctement décodée et organisée.

## 4. Analyse visuelle du dataset

L'analyse quantitative ne suffit pas à comprendre les difficultés réelles de segmentation. Une inspection visuelle a donc été réalisée à partir de deux planches d'échantillons.

Une première planche comportait 15 images, soit trois images par classe. Une seconde inspection a porté sur 25 images, soit cinq images par classe.

Ces observations permettent d'identifier des tendances visuelles, sans prétendre décrire chaque image du dataset.

### 4.1. Couleurs de peau

Les images examinées présentent une diversité de teints, allant des teints clairs aux teints foncés.

L'apparence de la peau varie également en fonction des conditions d'éclairage. Une même plage de couleur ne garantit donc pas nécessairement une détection identique dans toutes les images.

Cette diversité constitue un point important pour la segmentation HSV. Les seuils devront être évalués sur plusieurs teints et différentes conditions lumineuses afin de limiter les pertes de pixels appartenant à la main.

### 4.2. Arrière-plans

Les arrière-plans observés sont généralement simples, souvent blancs ou gris clair. Certaines images présentent une légère texture de mur, des ombres ou des zones plus sombres.

Ces arrière-plans relativement simples semblent favorables à une première segmentation fondée sur la couleur.

Cependant, la couleur de certains arrière-plans peut être proche de celle de la peau. Dans ces situations, une partie du fond risque d'être sélectionnée par le masque. À l'inverse, certaines parties de la main pourraient être éliminées lorsque leur apparence colorimétrique sort de l'intervalle choisi.

### 4.3. Contraste

Le contraste entre la main et l'arrière-plan est généralement satisfaisant dans les échantillons examinés.

Il peut néanmoins diminuer lorsque la main est claire sur un fond clair, lorsque l'éclairage est faible ou lorsque des ombres modifient la couleur apparente de la peau.

Ces variations pourraient provoquer des erreurs de segmentation. Elles devront être examinées lors des essais expérimentaux.

### 4.4. Cadrage et orientation

Les mains sont généralement visibles et occupent une partie importante des images.

La taille apparente, la position et l'orientation varient cependant d'une image à l'autre. La paume ou le dos de la main peuvent être visibles selon les cas, et le poignet ou l'avant-bras sont souvent partiellement coupés par le bord inférieur de l'image.

Ces différences devront être prises en compte dans la préparation des images et dans la sélection du contour à traiter.

### 4.5. Netteté et éclairage

Aucun flou important n'a été remarqué sur les planches examinées. Les images observées paraissent globalement suffisamment nettes pour distinguer les silhouettes des mains.

L'éclairage varie toutefois d'une image à l'autre, ce qui peut modifier les composantes de couleur et influencer les résultats de la segmentation.

Ces observations restent limitées aux échantillons inspectés et ne permettent pas d'affirmer que toutes les images possèdent la même qualité visuelle.

### 4.6. Synthèse des observations visuelles

L'examen des 25 images confirme plusieurs caractéristiques importantes :

- les teints sont diversifiés ;
- les arrière-plans sont généralement simples, mais pas toujours uniformes ;
- le contraste est globalement favorable, avec quelques situations plus difficiles ;
- les mains présentent des variations de taille, de position et d'orientation ;
- l'éclairage varie d'une image à l'autre ;
- les silhouettes sont généralement reconnaissables sur les échantillons examinés.

Ces constats suggèrent que la segmentation HSV mérite d'être expérimentée, tout en soulignant la nécessité de vérifier la qualité des masques obtenus.

## 5. Vérification de l'intégrité et recherche des doublons

Plusieurs contrôles ont été réalisés pour examiner l'intégrité des fichiers et rechercher les répétitions exactes dans le dataset.

### 5.1. Contrôles d'intégrité

Le script `src/check_dataset_integrity.py` a examiné les 2 653 fichiers image.

Les résultats obtenus sont les suivants :

| Contrôle | Résultat |
|---|---:|
| Fichiers image examinés | 2 653 |
| Fichiers vides | 0 |
| Images illisibles | 0 |
| Images de dimensions inférieures au seuil de contrôle | 0 |
| Images totalement uniformes | 0 |
| Fichiers non analysés | 0 |

Le contrôle des dimensions suspectes repose sur un seuil de détection des dimensions inférieures à 100 pixels. L'absence d'images signalées signifie qu'aucune image n'a été considérée comme anormalement petite selon ce critère.

De même, aucune image dont tous les pixels possèdent exactement la même valeur n'a été détectée.

Ces résultats sont encourageants : aucun problème de ce type n'a été identifié au cours des vérifications effectuées.

Ils ne garantissent cependant pas l'absence de tout défaut visuel ou sémantique. Ils doivent être interprétés comme les résultats des contrôles automatisés réalisés, et non comme une certification absolue de la qualité de toutes les images.

### 5.2. Recherche des doublons exacts

Une recherche de doublons exacts a ensuite été réalisée avec le script `src/check_duplicates.py`.

La méthode compare les empreintes SHA-256 calculées à partir de la forme et du contenu des pixels décodés des images. Elle permet donc d'identifier des fichiers dont le contenu pixel par pixel est identique, même si leurs noms ou leurs extensions diffèrent.

Les résultats sont les suivants :

| Indicateur | Résultat |
|---|---:|
| Fichiers examinés | 2 653 |
| Empreintes distinctes | 1 807 |
| Groupes de doublons exacts | 846 |
| Images appartenant à ces groupes | 1 692 |
| Fichiers non analysés | 0 |

Les groupes détectés comprennent 1 692 fichiers répartis en 846 groupes de doublons. Chaque groupe correspond à une même empreinte et, selon les résultats obtenus, à deux fichiers identiques. Cela représente 846 copies excédentaires par rapport à une conservation d'une seule occurrence par groupe.

Les 961 autres fichiers ne font partie d'aucun de ces groupes de doublons exacts.

Au total, le dataset contient donc **1 807 contenus d'image distincts selon le critère de comparaison utilisé**, avant toute opération de déduplication.

### 5.3. Vérification des doublons entre classes

Le script `src/analyze_duplicates.py` a permis d'examiner la répartition des doublons entre les classes.

Les résultats indiquent :

- 846 groupes de doublons exacts ;
- 1 692 fichiers concernés ;
- 846 copies excédentaires ;
- aucun groupe de doublons associé à plusieurs classes.

Aucun conflit d'étiquetage n'a donc été détecté parmi les doublons exacts identifiés.

Cette absence de conflit est positive : les copies exactes trouvées ne semblent pas être associées à des classes différentes.

La répartition des fichiers concernés par les groupes de doublons est la suivante :

| Classe | Fichiers appartenant à des groupes de doublons |
|---|---:|
| Open Palm_Open Hand | 444 |
| Fist | 408 |
| L-Sign | 344 |
| V-Sign | 254 |
| Pointing Up | 242 |
| **Total** | **1 692** |

Ces effectifs désignent les fichiers concernés par les groupes de doublons, et non le nombre total de contenus distincts dans chaque classe.

### 5.4. Conséquences pour la préparation des données

La présence de doublons exacts doit être prise en compte avant la séparation des données d'entraînement et de test.

Si deux copies identiques étaient placées dans des ensembles différents, le modèle pourrait être évalué sur une image dont le contenu est déjà présent dans ses données d'entraînement. Cette situation risquerait de produire une estimation trop optimiste de ses performances.

Il faudra donc prévoir une stratégie de déduplication ou de regroupement des images identiques avant la séparation des données. Cette opération devra conserver une seule occurrence de chaque contenu et maintenir une association cohérente avec sa classe.

**Aucun fichier n'a été supprimé à cette étape.** La recherche des doublons a uniquement servi à les identifier et à produire un résumé.

Les fichiers de résultats associés à ces contrôles comprennent notamment `results/duplicates_exact.csv` et `results/duplicates_summary.csv`.

## 6. Test préliminaire de faisabilité de la segmentation HSV

### 6.1. Objectif

À la suite des observations visuelles, un premier test a été réalisé pour vérifier si un seuil HSV simple permet de distinguer approximativement la main de son arrière-plan.

Cette expérience vise uniquement à vérifier la faisabilité apparente de l'approche sur un petit échantillon, avant de concevoir la segmentation définitive.

### 6.2. Protocole expérimental

Cinq images ont été sélectionnées, à raison d'une image par classe :

- Fist ;
- L-Sign ;
- Open Palm_Open Hand ;
- Pointing Up ;
- V-Sign.

La sélection a été effectuée avec une graine aléatoire fixée à 42, ce qui permet de reproduire la sélection dans les mêmes conditions, sous réserve de conserver les mêmes fichiers et la même organisation du dataset.

Les images ont été lues, puis redimensionnées pour les besoins de ce test uniquement, avec une dimension maximale de 512 pixels. Les fichiers originaux n'ont pas été modifiés.

La conversion colorimétrique a été effectuée avec OpenCV, de BGR vers HSV.

Le premier seuil exploratoire utilisé est défini comme suit :

| Composante | Borne inférieure | Borne supérieure |
|---|---:|---:|
| H — Teinte | 0 | 25 |
| S — Saturation | 20 | 255 |
| V — Luminosité | 40 | 255 |

La fonction `cv2.inRange()` a été appliquée pour produire un masque binaire. Les pixels respectant les intervalles ont été représentés en blanc, tandis que les autres ont été représentés en noir.

Le masque a ensuite été appliqué à l'image originale afin de visualiser les pixels conservés.

La planche de résultats a été enregistrée dans le fichier `figures/test_faisabilite_hsv.png`.

### 6.3. Résultats observés

L'inspection visuelle de la planche permet de relever les éléments suivants :

**Conservation de la main.** La majeure partie des mains est conservée dans les cinq exemples. Les silhouettes restent reconnaissables, y compris pour les teints différents.

**Élimination de l'arrière-plan.** L'arrière-plan devient majoritairement noir. Quelques zones parasites restent cependant visibles, notamment près du poignet et du bord droit dans l'exemple du geste en V.

**Continuité des silhouettes.** Les silhouettes sont globalement continues, mais certaines petites zones noires apparaissent à l'intérieur des mains, notamment au niveau des doigts et des plis. Ces défauts sont visibles sur les exemples du poing et du geste en V.

**Diversité des teints.** Les exemples de teints clairs et foncés montrent que le seuil initial conserve une grande partie de la main dans les deux cas. Le masque du geste en V présente toutefois davantage de bruit et de petites zones manquantes.

**Éléments périphériques.** Certains éléments, comme un bracelet, ne sont pas entièrement conservés. Les contours du poignet peuvent également être irréguliers.

### 6.4. Interprétation

Le test initial montre qu'un seuil HSV simple peut produire une séparation visuellement encourageante entre la main et l'arrière-plan sur les cinq images examinées.

Cette observation justifie la poursuite de l'approche HSV dans les étapes suivantes du projet.

Cependant, le masque présente déjà deux types d'erreurs :

- des faux négatifs, lorsque des pixels appartenant à la main sont éliminés ;
- des faux positifs, lorsque des pixels de l'arrière-plan sont sélectionnés.

Ces erreurs pourraient affecter l'extraction du contour et les moments de Hu utilisés par la suite pour représenter la forme de la main.

Le seuil utilisé demeure donc provisoire. Il n'a pas encore fait l'objet d'une comparaison systématique avec d'autres seuils ni d'une évaluation quantitative.

### 6.5. Limites de l'expérience

Cette expérience repose sur cinq images seulement. Les résultats ne permettent pas de conclure sur le comportement de la segmentation pour les 2 653 fichiers initiaux, ni même pour l'ensemble des contenus distincts.

Aucune mesure quantitative de la qualité des masques n'a été calculée. En particulier, aucune vérité terrain de segmentation n'a été utilisée pour mesurer précisément la proportion de peau correctement détectée.

Les observations sont donc visuelles et préliminaires. Elles permettent de juger si l'approche mérite d'être poursuivie, mais pas d'affirmer qu'elle est déjà optimale ou validée pour tout le dataset.

## 7. Conséquences pour la méthode de reconnaissance

À la lumière de l'analyse exploratoire et du test préliminaire, la chaîne de traitement envisagée reste la suivante :

**Image couleur → prétraitement → conversion HSV → segmentation de la peau → masque binaire → extraction du contour → moments de Hu → classification par k-NN.**

Les observations conduisent aux orientations suivantes :

1. **Prétraitement :** étudier un redimensionnement qui limite le coût de calcul tout en conservant les proportions de la main.
2. **Segmentation HSV :** poursuivre l'expérimentation avec le seuil initial comme point de départ, sans le considérer comme définitif.
3. **Qualité des masques :** surveiller les pixels de peau éliminés et les zones d'arrière-plan conservées.
4. **Contours :** vérifier que le contour retenu correspond bien à la main et qu'il n'est pas excessivement affecté par les erreurs du masque.
5. **Préparation des données :** prendre en compte les doublons exacts avant la séparation des ensembles d'entraînement et de test.
6. **Évaluation :** mesurer ultérieurement les performances de la classification à l'aide de métriques adaptées et d'une matrice de confusion.

Ces orientations guideront les étapes suivantes. Elles ne constituent pas encore des résultats de performance du système de reconnaissance.

## 8. Conclusion de l'étape

L'analyse exploratoire a permis de caractériser les 2 653 fichiers initiaux du dataset HandGesture-5G. Les images sont principalement de très haute résolution et sont représentées par trois canaux couleur après lecture.

Les inspections visuelles ont révélé des arrière-plans généralement simples, mais également une diversité de teints, des variations d'éclairage et des différences de cadrage susceptibles d'influencer la segmentation.

Les contrôles d'intégrité n'ont signalé aucun fichier vide, aucune image illisible, aucune image de dimensions inférieures au seuil de contrôle et aucune image totalement uniforme. La recherche de doublons exacts a identifié 846 groupes de doublons, représentant 846 copies excédentaires. Aucun groupe de doublons exacts associé à plusieurs classes n'a été détecté.

Enfin, le test préliminaire HSV réalisé sur cinq images a produit des masques visuellement encourageants : la majeure partie des mains est conservée et les arrière-plans restent majoritairement noirs. Des pertes de pixels et des détections parasites demeurent néanmoins visibles.

**L'étape 3 est considérée comme terminée.** Les résultats obtenus justifient la poursuite de l'approche HSV, sans préjuger de ses performances finales.

L'étape suivante portera sur le prétraitement des images, notamment l'étude du redimensionnement et la préparation des données pour la segmentation. La déduplication devra être planifiée avant la constitution des ensembles d'entraînement et de test.