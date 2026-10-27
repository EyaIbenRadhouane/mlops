import mlflow  # on réutilise le module MLflow pour LIRE les résultats

# search_runs renvoie tous les runs de l'expérience sous forme de tableau pandas.
# order_by trie les lignes : "metrics.f1 DESC" = f1 décroissant (meilleur en premier)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"],
)

# Vérification : l'expérience existe-t-elle et contient-elle des runs ?
if runs.empty:
    raise SystemExit("Aucun run trouvé dans l'expérience 'churn-prediction'.")

# Colonnes à afficher : nom du run, deux paramètres, deux métriques
colonnes = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1",
]

# On ne garde que les colonnes réellement présentes (évite un KeyError)
colonnes = [c for c in colonnes if c in runs.columns]

print(runs[colonnes].to_string(index=False))  # index=False : sans numéros de ligne

# La première ligne du tableau trié est le meilleur run
meilleur = runs.iloc[0]
print()
# [:8] : on n'affiche que les 8 premiers caractères de l'identifiant du run
print(
    "Meilleur run :",
    meilleur.get("tags.mlflow.runName", "?"),
    "| id :",
    meilleur["run_id"][:8],
    "...",
)