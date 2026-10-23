
from pathlib import Path
import pandas as pd

FICHIER = Path("results/duplicates_exact.csv")

df = pd.read_csv(FICHIER)

if df.empty:
    print("Aucun doublon exact trouvé.")
    raise SystemExit

# Une ligne par groupe d'images identiques
groupes = df.groupby("empreinte")

print("\n=== RÉSUMÉ DES DOUBLONS ===")
print(f"Groupes de doublons : {len(groupes)}")
print(f"Fichiers concernés : {len(df)}")
print(f"Copies excédentaires : {len(df) - len(groupes)}")

# Vérifier si un même contenu apparaît dans plusieurs classes
groupes_classes = groupes["classe"].nunique()
conflits = groupes_classes[groupes_classes > 1]

print("\n=== DOUBLONS ENTRE CLASSES ===")
print(f"Groupes associés à plusieurs classes : {len(conflits)}")

if len(conflits) > 0:
    for empreinte in conflits.index:
        groupe = groupes.get_group(empreinte)
        print("\nGroupe concerné :")
        print(groupe[["classe", "chemin"]].to_string(index=False))
else:
    print("Aucun conflit de classe parmi les doublons détectés.")

# Répartition des fichiers concernés par classe
print("\n=== FICHIERS CONCERNÉS PAR CLASSE ===")
print(df.groupby("classe").size().sort_values(ascending=False))

# Enregistrer le résumé des groupes
resume = groupes.agg(
    nombre_fichiers=("chemin", "size"),
    nombre_classes=("classe", "nunique"),
    classes=("classe", lambda x: ", ".join(sorted(set(x))))
).reset_index()

resume.to_csv(
    "results/duplicates_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nRésumé enregistré dans results/duplicates_summary.csv")
