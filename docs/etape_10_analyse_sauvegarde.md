# Étape 10 — Analyse, discussion et sauvegarde du modèle K-NN

## 10.1 Analyse générale des résultats

L'évaluation finale du modèle K-NN a été réalisée sur un ensemble de test indépendant composé de **531 images**, qui n'ont pas été utilisées pour l'entraînement du modèle.

Le modèle final utilise les paramètres suivants :

- nombre de voisins : **k = 3**
- pondération : **distance**
- métrique : **distance euclidienne**
- standardisation des 7 moments de Hu avant l'application du K-NN.

Les résultats globaux obtenus sont :

| Métrique | Résultat |
|---|---:|
| Accuracy | **87,01 %** |
| Precision macro | **87,25 %** |
| Recall macro | **86,89 %** |
| F1-score macro | **86,99 %** |
| F1-score weighted | **86,99 %** |

Sur les 531 images du test :

- **462 images** ont été correctement classées ;
- **69 images** ont été mal classées.

Le modèle atteint donc une performance globale de **87,01 % d'accuracy** et un **F1-score macro de 86,99 %**.

Les performances varient cependant selon les gestes. Le F1-score obtenu pour chaque classe est :

| Geste | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Fist | 88,99 % | 89,81 % | 89,40 % |
| L-Sign | 80,70 % | 85,19 % | 82,88 % |
| Open Palm_Open Hand | 88,19 % | 84,85 % | 86,49 % |
| Pointing Up | 90,41 % | 81,48 % | 85,71 % |
| V-Sign | 87,96 % | 93,14 % | 90,48 % |

Les principales confusions observées dans la matrice de confusion concernent notamment :

- **Open Palm_Open Hand → L-Sign : 13 images**
- **L-Sign → Open Palm_Open Hand : 9 images**
- **Pointing Up → V-Sign : 9 images**
- **V-Sign → Pointing Up : 5 images**
- **Open Palm_Open Hand → Fist : 7 images**

Ces résultats montrent que le modèle distingue globalement correctement les cinq gestes, mais que certaines formes sont plus difficiles à séparer que d'autres.

Il faut cependant rester prudent dans l'interprétation de ces erreurs. Une confusion entre deux classes ne permet pas, à elle seule, d'affirmer que les moments de Hu sont responsables de l'erreur. Celle-ci peut également être liée à la qualité de la segmentation HSV, à la forme obtenue après extraction du contour, à l'orientation de la main, au fond de l'image ou encore aux limites des caractéristiques utilisées.

---

## 10.2 Geste le mieux reconnu

Le **V-Sign** est le geste le mieux reconnu globalement par le modèle.

Il obtient le meilleur **F1-score avec 90,48 %**, ainsi que le meilleur **rappel avec 93,14 %** parmi les cinq classes.

Sur les **102 images V-Sign** présentes dans l'ensemble de test :

- **95 images** sont correctement classées comme V-Sign ;
- **5 images** sont classées comme Pointing Up ;
- **2 images** sont classées comme L-Sign.

Le rappel de 93,14 % signifie que le modèle reconnaît correctement environ 93 % des véritables V-Sign présents dans l'ensemble de test.

Le V-Sign semble donc présenter une forme suffisamment distinctive dans l'espace des caractéristiques basé sur les moments de Hu pour être relativement bien séparé des autres gestes.

Cependant, cette observation ne permet pas de conclure que le V-Sign est intrinsèquement toujours plus facile à reconnaître. Le résultat dépend également des images présentes dans le dataset, de leur orientation, de leur qualité et de la segmentation obtenue.

À titre de comparaison, **Pointing Up possède la meilleure précision avec 90,41 %**, mais son rappel est seulement de 81,48 %. Il n'est donc pas le meilleur geste globalement reconnu selon le F1-score.

---

## 10.3 Geste le moins bien reconnu

Le **L-Sign** est le geste le moins bien reconnu globalement par le modèle.

Il obtient le plus faible F1-score parmi les cinq classes, avec une valeur de **82,88 %**.

Ses performances sont les suivantes :

| Métrique | Résultat |
|---|---:|
| Precision | **80,70 %** |
| Recall | **85,19 %** |
| F1-score | **82,88 %** |
| Support | **108 images** |

Sur les **108 images L-Sign** présentes dans l'ensemble de test :

- **92 images** sont correctement classées comme L-Sign ;
- **16 images** sont mal classées.

Les erreurs de classification sont réparties comme suit :

| Classe réelle | Classe prédite | Nombre |
|---|---|---:|
| L-Sign | Open Palm_Open Hand | **9** |
| L-Sign | Fist | 3 |
| L-Sign | Pointing Up | 2 |
| L-Sign | V-Sign | 2 |

La principale difficulté concerne donc la confusion entre **L-Sign et Open Palm_Open Hand**. Neuf images L-Sign sont classées comme Open Palm_Open Hand.

Cette confusion apparaît également dans le sens inverse : **13 images Open Palm_Open Hand sont classées comme L-Sign**. Cela montre que ces deux classes sont particulièrement difficiles à séparer pour le modèle utilisé.

Le L-Sign possède également une précision plus faible que les autres classes (**80,70 %**). Cela signifie qu'une partie des images prédites comme L-Sign appartiennent en réalité à d'autres classes.

Cependant, ces résultats ne permettent pas d'affirmer que les moments de Hu sont à eux seuls responsables de cette difficulté. Les erreurs peuvent également être influencées par la qualité de la segmentation HSV, l'apparence des contours après traitement, l'orientation des mains, les variations des images ou les limites générales de la représentation utilisée.

Une inspection des images et des masques de segmentation sera donc nécessaire avant de formuler une conclusion sur l'origine de ces confusions.

## 10.4 Analyse des principales confusions

L'analyse de la matrice de confusion montre que les erreurs de classification ne sont pas réparties uniformément entre les cinq classes.

Les principales confusions observées sont :

| Classe réelle | Classe prédite | Nombre |
|---|---|---:|
| Open Palm_Open Hand | L-Sign | **13** |
| L-Sign | Open Palm_Open Hand | **9** |
| Pointing Up | V-Sign | **9** |
| V-Sign | Pointing Up | **5** |
| Open Palm_Open Hand | Fist | **7** |

Deux couples de classes présentent particulièrement des confusions réciproques.

### L-Sign ↔ Open Palm_Open Hand

Cette paire représente au total **22 erreurs** :

- 13 Open Palm_Open Hand sont classées comme L-Sign ;
- 9 L-Sign sont classées comme Open Palm_Open Hand.

Il s'agit de la principale zone de confusion du modèle.

Cette difficulté est cohérente avec le fait que le L-Sign est la classe ayant obtenu le plus faible F1-score (**82,88 %**).

### Pointing Up ↔ V-Sign

Cette paire représente au total **14 erreurs** :

- 9 Pointing Up sont classées comme V-Sign ;
- 5 V-Sign sont classées comme Pointing Up.

Le V-Sign reste néanmoins la classe la mieux reconnue globalement, avec un F1-score de **90,48 %** et un rappel de **93,14 %**.

### Open Palm_Open Hand ↔ Fist

Une autre confusion notable concerne Open Palm_Open Hand et Fist :

- 7 images Open Palm_Open Hand sont classées comme Fist ;
- 4 images Fist sont classées comme Open Palm_Open Hand.

Cette confusion est moins importante que les deux précédentes, mais elle montre que certaines images peuvent produire des représentations de forme relativement proches après les étapes de segmentation et d'extraction des moments de Hu.

### Interprétation prudente

Les confusions observées indiquent que certaines classes sont plus difficiles à séparer que d'autres dans l'espace des caractéristiques utilisé par le K-NN.

Cependant, ces résultats ne permettent pas d'attribuer directement les erreurs aux seuls moments de Hu. Plusieurs facteurs peuvent intervenir, notamment :

- la qualité de la segmentation HSV ;
- les variations d'orientation et de position de la main ;
- les différences entre les silhouettes obtenues ;
- les informations de forme perdues lors de la représentation par sept moments de Hu ;
- les caractéristiques propres aux images du dataset.

Une inspection visuelle des images mal classées et de leurs masques de segmentation sera donc nécessaire avant de déterminer si la segmentation ou la représentation des formes contribue effectivement aux principales confusions.

## 10.5 Influence du choix de k

Le choix du nombre de voisins `k` influence directement le fonctionnement du classifieur K-NN.

Lors de la recherche d'hyperparamètres, plusieurs valeurs de `k` ont été testées sur l'ensemble d'entraînement à l'aide d'une validation croisée stratifiée et groupée à 5 plis.

Les meilleurs résultats obtenus sont les suivants :

| k | Pondération | F1-score macro moyen |
|---:|---|---:|
| 1 | uniform | **85,72 %** |
| 1 | distance | **85,72 %** |
| 3 | distance | **84,94 %** |
| 5 | distance | **84,84 %** |
| 7 | distance | **84,70 %** |
| 3 | uniform | **84,48 %** |
| 9 | distance | **84,18 %** |
| 11 | distance | **83,85 %** |
| 5 | uniform | **83,80 %** |

Le meilleur résultat moyen de validation croisée a donc été obtenu avec **k = 1**, avec un F1-score macro moyen de **85,72 %**.

Cependant, le modèle final retenu utilise **k = 3** avec une pondération par distance et une distance euclidienne.

Ce choix est volontaire. Avec `k = 1`, la décision repose uniquement sur le voisin le plus proche. Le modèle peut ainsi être davantage influencé par un exemple particulier du jeu d'entraînement.

Avec `k = 3`, plusieurs voisins participent à la décision. La pondération `distance` permet en outre de donner davantage d'importance aux voisins les plus proches.

Le choix de `k = 3` constitue donc un compromis entre la performance mesurée et une décision reposant sur plusieurs exemples voisins.

Il est important de préciser que **k = 3 n'a pas été choisi parce qu'il obtenait le meilleur score de validation croisée**. Au contraire, `k = 1` obtenait un score légèrement supérieur.

L'écart entre les deux configurations est d'environ **0,78 point de pourcentage** de F1 macro :

**85,72 % − 84,94 % = 0,78 point.**

Cette différence reste relativement faible, mais elle doit être signalée pour assurer une présentation transparente des résultats expérimentaux.

Enfin, le jeu de test final de **531 images n'a pas été utilisé pour sélectionner `k`**. La recherche d'hyperparamètres a été réalisée uniquement sur les données d'entraînement avec une validation croisée groupée. Le jeu de test a ensuite été conservé pour l'évaluation finale du modèle.

Le choix de `k = 3` doit donc être considéré comme un choix méthodologique du projet et non comme le résultat de la meilleure performance brute obtenue pendant la recherche d'hyperparamètres.

## 10.6 Les moments de Hu sont-ils suffisants ?

Les sept moments de Hu constituent la représentation utilisée pour décrire la forme de la main.

Les résultats montrent qu'ils permettent d'obtenir une performance globale de **87,01 % d'accuracy**, ce qui indique qu'ils contiennent une information utile pour distinguer les cinq gestes.

Cependant, ils ne permettent pas une séparation parfaite des classes puisque **69 erreurs** sont observées sur les 531 images de test.

Les moments de Hu décrivent principalement des propriétés globales de la forme et présentent donc certaines limites lorsqu'il faut distinguer des gestes dont les silhouettes sont proches.

**À compléter :**
- analyser les confusions ;
- déterminer si certaines formes semblent difficiles à séparer ;
- discuter les informations perdues par une représentation uniquement basée sur les moments de Hu ;
- éventuellement proposer des caractéristiques complémentaires.

---

### 10.7. La segmentation HSV préserve-t-elle la forme de la main ?

#### Objectif

Cette analyse vise à vérifier visuellement si la segmentation HSV conserve suffisamment la silhouette de la main pour permettre l’extraction des moments de Hu.

Pour cette inspection, six images mal classées par le modèle K-NN ont été sélectionnées parmi les confusions entre `L-Sign` et `Open Palm_Open Hand`. Pour chaque image, quatre éléments ont été comparés : l’image couleur, le masque HSV initial, le masque après opérations morphologiques et le contour principal utilisé pour calculer les moments de Hu.

#### Observations visuelles

Sur la planche produite, les doigts sont généralement conservés et le contour principal suit globalement la silhouette de la main. Les erreurs de classification observées ne semblent donc pas dues, dans tous les cas, à une perte complète des doigts.

- **Images de la classe réelle `L-Sign`, prédites `Open Palm_Open Hand` :** le pouce et l’index restent visibles, et le contour principal englobe globalement la main. Cependant, certaines images sont cadrées près des bords ; des parties du contour rejoignent le bord inférieur ou droit. Une image présente également des fragments et du bruit près du poignet, ce qui peut indiquer l’inclusion de pixels parasites ou d’ombres dans le masque.

- **Images de la classe réelle `Open Palm_Open Hand`, prédites `L-Sign` :** les doigts écartés et leurs espaces sont généralement bien conservés. Les masques paraissent propres et le contour principal correspond visuellement à la main. Une image présente toutefois quelques pixels parasites près du bord inférieur droit.

#### Interprétation

Ces observations suggèrent que la segmentation HSV et les opérations morphologiques préservent globalement la silhouette de la main dans les exemples inspectés. Elles montrent néanmoins certaines limites : le cadrage peut couper une partie de la main, tandis que des pixels parasites peuvent modifier localement le contour.

Par ailleurs, certaines erreurs de classification apparaissent alors que la silhouette semble correctement extraite. La segmentation ne suffit donc pas, à elle seule, à expliquer les confusions observées. Les sept moments de Hu décrivent des propriétés globales de la forme et peuvent ne pas représenter suffisamment certains détails permettant de distinguer deux gestes proches.

Il faut cependant rester prudent : cette inspection porte uniquement sur six images mal classées. Elle ne permet pas de conclure que la segmentation est satisfaisante sur l’ensemble des 2 653 images, ni de déterminer précisément la cause de chaque erreur.

#### Conclusion

La segmentation HSV semble globalement préserver la forme de la main dans les exemples examinés, mais certains défauts de cadrage et quelques pixels parasites subsistent. Les confusions entre `L-Sign` et `Open Palm_Open Hand` montrent également qu’une silhouette bien extraite ne garantit pas une classification correcte.

Pour approfondir cette analyse, il serait utile d’inspecter davantage d’images, notamment des exemples correctement classés et des cas présentant des défauts de segmentation. Une comparaison expérimentale entre les moments de Hu seuls et d’autres caractéristiques géométriques permettrait ensuite de déterminer si les erreurs proviennent en partie de la représentation des gestes.

**Figure associée :** `figures/inspection_segmentation_erreurs.png`.


### 10.8. Limites du modèle

Malgré les résultats obtenus, notre système de reconnaissance des gestes de la main présente plusieurs limites qu’il est important de prendre en compte pour interpréter correctement ses performances.

#### 10.8.1. Limites liées au dataset

Le modèle a été entraîné et évalué sur un dataset de 2 653 images réparties en cinq classes. Bien que les cinq gestes soient représentés, le nombre d’images et leur diversité peuvent limiter la capacité du modèle à reconnaître des exemples très différents de ceux utilisés pendant son apprentissage.

Par exemple, des variations d’éclairage, d’arrière-plan, d’orientation de la main ou de cadrage peuvent modifier l’apparence de la silhouette et affecter les caractéristiques extraites. De plus, les informations disponibles ne permettent pas d’affirmer que le dataset couvre toutes les personnes et toutes les conditions possibles d’acquisition.

#### 10.8.2. Limites de la segmentation HSV

La segmentation repose sur des seuils fixes dans l’espace de couleurs HSV. Cette méthode est relativement simple à mettre en œuvre, mais elle peut rencontrer des difficultés lorsque la couleur de la main varie selon l’éclairage ou lorsque l’arrière-plan contient des couleurs proches de celles de la peau.

Les opérations morphologiques permettent de nettoyer certains défauts du masque, mais elles ne corrigent pas toutes les erreurs. La sélection du plus grand contour externe suppose également que ce contour représente la main. Si un élément parasite est plus grand que la main, ou si plusieurs parties de la main sont séparées dans le masque, le contour sélectionné peut ne pas représenter correctement le geste.

L’inspection visuelle de six images mal classées a montré que la silhouette était généralement conservée, malgré quelques défauts de cadrage et des pixels parasites. Toutefois, cet examen qualitatif ne suffit pas à évaluer la qualité de la segmentation sur l’ensemble du dataset.

#### 10.8.3. Limites des moments de Hu

Les sept moments de Hu permettent de représenter la forme globale d’un objet à l’aide de sept valeurs numériques. Cette représentation est compacte et facilite l’apprentissage d’un classificateur comme le K-NN.

Cependant, cette compression entraîne une perte d’informations. Certaines différences locales entre les doigts, leur position ou leur espacement peuvent être insuffisamment représentées par ces sept caractéristiques. Deux gestes différents peuvent ainsi produire des vecteurs de caractéristiques relativement proches.

Les confusions observées entre `L-Sign` et `Open Palm_Open Hand`, ainsi qu’entre `Pointing Up` et `V-Sign`, sont compatibles avec cette hypothèse. Néanmoins, les résultats actuels ne permettent pas d’affirmer que les moments de Hu sont l’unique cause de ces erreurs. Des défauts de segmentation ou d’autres facteurs peuvent également intervenir.

#### 10.8.4. Limites de l’algorithme K-NN

Le K-NN classe un nouvel exemple en fonction de ses voisins les plus proches dans l’espace des caractéristiques. Ses performances dépendent donc de la qualité des caractéristiques, de la distance utilisée et du choix du nombre de voisins \(k\).

Dans notre projet, le modèle final utilise \(k=3\), une pondération par distance et la distance euclidienne, après standardisation des sept caractéristiques de Hu.

La recherche par validation croisée a cependant montré que \(k=1\) obtenait un F1-score macro moyen légèrement supérieur à celui de \(k=3\). Le choix de \(k=3\) constitue donc une décision méthodologique du projet, et non la preuve d’une supériorité expérimentale sur \(k=1\).

Par ailleurs, le K-NN ne détermine pas lui-même si les caractéristiques représentent correctement les différences entre les gestes. Lorsque deux classes possèdent des caractéristiques proches, il peut attribuer un geste à la mauvaise classe.

#### 10.8.5. Limites de l’évaluation

L’évaluation finale a été réalisée sur 531 images de test, distinctes des groupes d’images utilisés pour l’entraînement. La séparation par groupes a permis d’éviter qu’un même groupe d’images dupliquées soit présent à la fois dans les ensembles d’entraînement et de test.

Le modèle a obtenu une accuracy de 87,01 % et un F1-score macro de 86,99 %. Ces résultats décrivent ses performances sur le jeu de test utilisé, mais ne garantissent pas les mêmes performances sur toutes les nouvelles images.

Une évaluation complémentaire sur des images prises dans des conditions différentes serait nécessaire pour mieux mesurer sa capacité de généralisation.

#### 10.8.6. Pistes d’amélioration

Plusieurs améliorations pourraient être envisagées :

- **Améliorer la segmentation :** tester des méthodes plus robustes aux variations d’éclairage et aux arrière-plans complexes.
- **Enrichir les caractéristiques :** combiner les moments de Hu avec des caractéristiques géométriques comme le rapport d’aspect, la densité ou le nombre de doigts détectés.
- **Comparer plusieurs classificateurs :** évaluer d’autres algorithmes en conservant la même séparation des données et une procédure de comparaison équitable.
- **Étudier les erreurs :** examiner davantage d’images mal classées pour distinguer les problèmes de segmentation de ceux liés à la représentation des gestes.
- **Évaluer la généralisation :** tester le modèle sur de nouvelles images acquises dans des conditions différentes.

Ces améliorations devront être évaluées expérimentalement. Il ne suffit pas de supposer qu’une méthode plus complexe sera nécessairement meilleure.

#### Conclusion

Le système développé démontre qu’il est possible de reconnaître cinq gestes de la main à partir d’une segmentation HSV, de l’extraction des sept moments de Hu et d’un classificateur K-NN. Les résultats obtenus sont encourageants, avec une accuracy de 87,01 % et un F1-score macro de 86,99 %.

Cependant, les performances restent dépendantes de la qualité de la segmentation, de la capacité des moments de Hu à distinguer les gestes et du fonctionnement du K-NN. Le modèle doit donc être considéré comme une solution expérimentale dont les limites sont identifiées, et non comme un système infaillible. Des expérimentations complémentaires permettront de déterminer quelles améliorations apporteraient un gain réel.

Plusieurs limites devront être prises en compte :

1. **Représentation limitée**  
   Le modèle utilise uniquement les sept moments de Hu et ne bénéficie pas d'autres informations visuelles.

2. **Dépendance à la segmentation HSV**  
   La qualité du contour dépend directement des seuils HSV et des opérations morphologiques utilisées.

3. **Sensibilité aux formes similaires**  
   Certaines classes présentent des confusions importantes.

4. **Dataset spécifique**  
   Les performances obtenues sont valables dans les conditions du dataset HandGesture-5G et ne garantissent pas les mêmes résultats sur de nouvelles images prises dans des conditions différentes.

5. **Choix du modèle**  
   Le K-NN est un modèle relativement simple et peut être limité lorsque les classes deviennent plus complexes à séparer.

6. **Conditions d'acquisition**  
   Les changements d'éclairage, de position, d'orientation, de distance ou de fond peuvent modifier la segmentation et donc les caractéristiques utilisées par le modèle.

---

### 10.9. Conclusion générale de l’analyse

Cette étape a permis d’analyser les performances du modèle K-NN, d’identifier les gestes les mieux et les moins bien reconnus, d’étudier les principales confusions et d’examiner les limites de la segmentation HSV et des moments de Hu.

Le modèle final, configuré avec \(k=3\), une pondération par distance et la distance euclidienne, a obtenu une accuracy de **87,01 %** et un F1-score macro de **86,99 %** sur les 531 images de test. Sur ces images, 462 prédictions étaient correctes et 69 étaient incorrectes.

L’analyse par classe a montré que `V-Sign` était le geste le mieux reconnu selon le F1-score, avec 90,48 %, tandis que `L-Sign` présentait le F1-score le plus faible, à 82,88 %. Les principales confusions concernaient les classes `L-Sign` et `Open Palm_Open Hand`, ainsi que `Pointing Up` et `V-Sign`.

L’inspection visuelle de six images mal classées a suggéré que la segmentation HSV préservait généralement la silhouette de la main dans les exemples examinés, malgré certains défauts de cadrage et quelques pixels parasites. Toutefois, cette observation ne permet pas de valider la segmentation sur l’ensemble du dataset ni d’attribuer toutes les erreurs à une cause précise.

Le projet montre ainsi qu’une approche fondée sur la segmentation HSV, les sept moments de Hu et le K-NN permet d’obtenir des résultats encourageants pour la reconnaissance de cinq gestes de la main. Néanmoins, les limites des caractéristiques utilisées et les erreurs observées indiquent que des améliorations restent possibles.

Enfin, le modèle devra être sauvegardé avec son prétraitement et sa configuration afin de garantir la reproductibilité des prédictions et de faciliter son intégration dans la suite du projet. Toute amélioration future devra être évaluée expérimentalement sur une procédure de test rigoureuse, afin de vérifier qu’elle apporte un gain réel.

**La prochaine étape consiste donc à sauvegarder le pipeline complet et à documenter son utilisation pour son intégration dans l’application finale.**

### 10.10 Sauvegarde du modèle

Après l'évaluation, le pipeline K-NN final a été sauvegardé dans le fichier `models/knn_pipeline.pkl`. Cette sauvegarde comprend le `StandardScaler` ajusté sur les données d'entraînement ainsi que le classifieur K-NN configuré avec trois voisins, une pondération par distance et une distance euclidienne.

Le fichier `models/model_config.json` documente les paramètres du classifieur, l'ordre des sept caractéristiques de Hu, la transformation logarithmique signée et les principales étapes du prétraitement. Le fichier `README_model.md` explique les conditions nécessaires au chargement et à la réutilisation du modèle.

Un rapport récapitulatif des résultats a également été généré dans `results/evaluation_report.txt`. Il complète les fichiers CSV contenant les métriques, la matrice de confusion et les prédictions.

Enfin, le pipeline sauvegardé a été rechargé à l'aide de Joblib, puis utilisé pour prédire les classes des 531 images du jeu de test. Les nouvelles prédictions ont été comparées aux prédictions initiales : **531 prédictions étaient identiques et aucune différence n'a été observée**.

Cette vérification confirme que la sauvegarde conserve le comportement du pipeline évalué. Elle ne constitue toutefois pas une nouvelle mesure de ses performances de généralisation.

La sauvegarde et le rechargement du modèle sont donc validés.
