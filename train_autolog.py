import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("churn_data.csv")

X = pd.get_dummies(
    df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
    drop_first=True,
)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # garde la même proportion de départs
)

mlflow.set_experiment("churn-prediction")

# Active l'enregistrement automatique pour scikit-learn : MLflow note tout seul
# les paramètres du modèle, des métriques d'entraînement et le modèle lui-même.
mlflow.sklearn.autolog(serialization_format="cloudpickle")

with mlflow.start_run(run_name="essai-auto"):
    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)  # l'enregistrement automatique se déclenche au fit()

    # Métriques sur le jeu de test, avec les noms attendus par lire_runs.py
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)

    print("accuracy test :", round(accuracy, 3))
    print("f1 test       :", round(f1, 3))