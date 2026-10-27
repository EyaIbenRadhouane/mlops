import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# 1. Retrouver le run « essai-2 » (le modèle le plus faible du TP3)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="attributes.run_name = 'essai-2'",
    order_by=["attributes.start_time DESC"],  # le plus récent d'abord
)

if runs.empty:
    # Diagnostic : on affiche ce qui existe réellement dans cette base
    toutes = mlflow.search_runs(experiment_names=["churn-prediction"])
    if toutes.empty:
        raise SystemExit(
            "L'expérience 'churn-prediction' est vide ou absente de mlflow.db.\n"
            "Les essais du TP3 ont probablement été enregistrés sans "
            "mlflow.set_tracking_uri('sqlite:///mlflow.db')."
        )
    noms = toutes["tags.mlflow.runName"].tolist()
    raise SystemExit(f"Aucun run 'essai-2'. Runs disponibles : {noms}")

run_id = runs.iloc[0]["run_id"]
print("Run essai-2 trouvé, id :", run_id[:8])

# 2. L'enregistrer : cela crée une NOUVELLE VERSION de churn-model
v = mlflow.register_model(f"runs:/{run_id}/model", "churn-model")
print("Nouvelle version créée :", v.version)

# 3. Où pointe « champion » aujourd'hui ?
actuel = client.get_model_version_by_alias("churn-model", "champion")
print("Champion actuel : version", actuel.version)

# 4. Déplacer l'alias vers la nouvelle version, puis le remettre sur la bonne
client.set_registered_model_alias("churn-model", "champion", v.version)
print("Après déplacement : champion -> version",
      client.get_model_version_by_alias("churn-model", "champion").version)

client.set_registered_model_alias("churn-model", "champion", actuel.version)
print("Après retour : champion -> version",
      client.get_model_version_by_alias("churn-model", "champion").version)