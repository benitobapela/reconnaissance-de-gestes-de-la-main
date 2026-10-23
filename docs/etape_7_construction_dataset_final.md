# Étape 7 — Construction et validation du dataset final

## 7.1. Objectif

Cette étape consiste à construire le dataset final de caractéristiques destiné à la classification des gestes de la main par l'algorithme k-NN.

À partir des images HSV prétraitées, nous extrayons sept moments de Hu par image. Ces caractéristiques sont ensuite associées à l'étiquette de leur classe et enregistrées dans un fichier CSV.

Le livrable principal est `data/final/features.csv`.

## 7.2. Méthodologie

La construction du dataset suit les opérations suivantes :

1. Parcourir les cinq dossiers de classes contenant les images HSV prétraitées.
2. Charger chaque image HSV et vérifier sa structure.
3. Segmenter les régions de peau avec les seuils HSV inférieurs `[0, 20, 40]` et supérieurs `[25, 255, 255]`.
4. Appliquer une fermeture morphologique avec un noyau elliptique de 5 × 5, puis une ouverture avec un noyau elliptique de 3 × 3.
5. Détecter les contours externes et sélectionner le contour de plus grande aire.
6. Calculer les sept moments de Hu à partir du contour retenu.
7. Appliquer la transformation logarithmique signée afin de réduire les écarts d'échelle entre les caractéristiques.
8. Associer les caractéristiques à l'identifiant de l'image et à sa classe.
9. Enregistrer les résultats dans un fichier CSV et contrôler leur validité.

La transformation utilisée est :

\[
H_i'=-\operatorname{sign}(H_i)
\log_{10}\left(\max(|H_i|,\varepsilon)\right)
\]

où \(H_i\) représente le moment de Hu initial et \(\varepsilon=10^{-30}\) protège le calcul lorsque la valeur absolue est très proche de zéro.

## 7.3. Structure du dataset

Le fichier `features.csv` contient neuf colonnes :

- `image_id` : identifiant de l'image, comprenant son chemin relatif au dossier HSV ;
- `class` : classe du geste ;
- `Hu1` à `Hu7` : sept moments de Hu après transformation logarithmique signée.

Chaque ligne correspond à une image dont l'extraction des caractéristiques a réussi.

## 7.4. Résultats de la construction

La construction a été exécutée avec le script `src/construire_dataset_final.py`.

Les résultats obtenus sont présentés dans le tableau suivant :

| Classe | Images traitées | Échecs |
|---|---:|---:|
| Fist | 542 | 0 |
| L-Sign | 537 | 0 |
| Open Palm_Open Hand | 661 | 0 |
| Pointing Up | 405 | 0 |
| V-Sign | 508 | 0 |
| **Total** | **2 653** | **0** |

Les 2 653 images ont été traitées avec succès. Le fichier final contient donc 2 653 lignes de données et neuf colonnes.

## 7.5. Validation du dataset

Le script `src/valider_dataset_final.py` a été utilisé pour vérifier le fichier produit.

Les contrôles effectués ont confirmé :

- la présence des neuf colonnes attendues ;
- la présence des 2 653 lignes ;
- la conformité des effectifs des cinq classes ;
- l'absence d'identifiants manquants ou dupliqués ;
- l'absence de valeurs manquantes ;
- l'absence de caractéristiques non numériques ou non finies.

La validation s'est terminée avec le message `VALIDATION RÉUSSIE`.

Ces contrôles confirment la cohérence structurelle et numérique du fichier. Ils ne garantissent toutefois pas que chaque masque HSV représente parfaitement la main : la qualité de la segmentation reste une limite méthodologique à prendre en compte dans l'interprétation des performances futures.

## 7.6. Livrable et conclusion

Le livrable de cette étape est :

`data/final/features.csv`

Ce fichier constitue la base de travail pour l'étape de classification. Chaque image est représentée par un vecteur de sept caractéristiques numériques, accompagné de sa classe réelle.

Le dataset est prêt pour la préparation de l'entraînement et de l'évaluation du modèle k-NN. La séparation entre les données d'entraînement et de test devra notamment tenir compte des doublons exacts identifiés lors de l'analyse exploratoire, afin d'éviter une fuite de données entre les deux ensembles.
