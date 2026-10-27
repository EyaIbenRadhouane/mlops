import mlflow
import mlflow.sklearn  # pour recharger un modèle scikit-learn
import pandas as pd

# 1. Retrouver le meilleur run (même logique que lire_runs.py)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="attributes.status = 'FINISHED'",  # ignore les runs échoués
    order_by=["metrics.f1 DESC"],
)
if runs.empty:
    raise SystemExit("Aucun run terminé trouvé dans 'churn-prediction'.")

run_id = runs.iloc[0]["run_id"]  # identifiant du meilleur run

# 2. Recharger le modèle. "runs:/<id>/model" = le dossier 'model' de ce run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
print("Modèle rechargé :", type(model).__name__)
print("Nombre de variables attendues :", model.n_features_in_)

# 3. Créer un nouveau client (valeurs au choix)
# Les colonnes d'abonnement sont déjà transformées en 0/1 (True/False) :
# ici : abonnement Standard (Premium = False)
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True,
}])

# 4. Garder uniquement les colonnes connues du modèle, dans le bon ordre
# (selon le run choisi, la colonne salaire est utilisée ou non)
if hasattr(model, "feature_names_in_"):
    attendues = list(model.feature_names_in_)
    manquantes = [c for c in attendues if c not in client.columns]
    if manquantes:
        raise SystemExit(f"Colonnes attendues par le modèle mais absentes : {manquantes}")
    client = client[attendues]
else:
    # Modèle entraîné sur un tableau numpy : on ne peut pas vérifier les noms
    if client.shape[1] != model.n_features_in_:
        raise SystemExit(
            f"Le modèle attend {model.n_features_in_} variables, "
            f"mais le client en a {client.shape[1]}."
        )

# 5. Prédire : 1 = le client va partir, 0 = il reste
prediction = model.predict(client)[0]
print("Le client va-t-il partir ? ->", "OUI" if prediction == 1 else "NON")