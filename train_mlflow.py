# ===== 1. Importer les bibliothèques =====
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MÊME base que deplacer_alias.py et load_champion.py (sinon les runs sont introuvables)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 2. Charger les données (une seule fois) =====
df = pd.read_csv("churn_data.csv")  # le tableau de 500 clients
y = df["churn"]                     # la réponse à prédire : 1 = part, 0 = reste

# ===== 3. Fonction : préparer les données pour une liste de colonnes =====
def preparer(colonnes):
    # transforme le texte 'abonnement' en colonnes 0/1 (si présente)
    X = pd.get_dummies(df[colonnes], drop_first=True)
    # 80 % apprentissage / 20 % test, même découpage à chaque fois (random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ===== 4. Fonction : lancer UN essai et l'enregistrer dans MLflow =====
def lancer_essai(nom, n_estimators, max_depth, colonnes):
    X_train, X_test, y_train, y_test = preparer(colonnes)

    with mlflow.start_run(run_name=nom):
        model = RandomForestClassifier(
            n_estimators=n_estimators,  # nombre d'arbres
            max_depth=max_depth,        # profondeur maximale des arbres
            random_state=42,            # graine : résultat reproductible
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # Métriques calculées une seule fois
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        # --- Paramètres ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))

        # --- Métriques ---
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # --- Artifact : le modèle (MLflow 2.x : remplacer name= par artifact_path=) ---
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
        )

        print(nom, "-> accuracy :", round(accuracy, 3), "| f1 :", round(f1, 3))

# ===== 5. Programme principal =====
if __name__ == "__main__":
    mlflow.set_experiment("churn-prediction")

    TOUT = ["age", "salaire", "abonnement", "anciennete", "consommation"]

    lancer_essai("essai-1", 100, 5, TOUT)    # 100 arbres, profondeur 5
    lancer_essai("essai-2", 50, 3, TOUT)     # modèle plus simple
    lancer_essai("essai-3", 100, 5, ["age", "abonnement", "anciennete", "consommation"])  # sans salaire