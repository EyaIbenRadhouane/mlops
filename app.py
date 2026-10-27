# ===== 1. Importer les bibliothèques =====
from typing import Literal            # pour limiter les valeurs possibles d'un champ

import mlflow                         # charger le modèle depuis le Registry
import mlflow.sklearn
import pandas as pd                   # fabriquer le tableau attendu par le modèle
from fastapi import FastAPI           # le cadre de travail de l'API
from pydantic import BaseModel, Field # décrire et valider les données reçues

# ===== 2. Créer l'application =====
app = FastAPI(title="API Churn")      # le titre s'affiche dans la page /docs

# ===== 3. Charger le champion UNE SEULE FOIS, au démarrage =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")

# ===== 4. Décrire les données attendues (validées automatiquement) =====
class Client(BaseModel):
    age: int = Field(ge=0, le=120)    # entier, borné
    salaire: float = Field(ge=0)
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float = Field(ge=0)
    consommation: float = Field(ge=0)

# ===== 5. Route /health =====
@app.get("/health")
def health():
    return {"status": "ok"}

# ===== 6. Route /predict =====
@app.post("/predict")
def predict(client: Client):
    donnees = client.model_dump()

    # a) encodage explicite de l'abonnement ("Basique" = référence : les deux colonnes valent 0,
    #    comme avec drop_first=True à l'entraînement)
    abonnement = donnees.pop("abonnement")
    donnees["abonnement_Standard"] = abonnement == "Standard"
    donnees["abonnement_Premium"] = abonnement == "Premium"

    # b) tableau d'une ligne
    ligne = pd.DataFrame([donnees])

    # c) mêmes colonnes que le modèle, dans le même ordre
    #    (si le champion n'utilise pas « salaire », la colonne est ignorée)
    ligne = ligne.reindex(columns=list(model.feature_names_in_), fill_value=0)

    # d) prédire : 1 = part, 0 = reste
    resultat = int(model.predict(ligne)[0])
    proba = float(model.predict_proba(ligne)[0][1])  # probabilité de départ

    return {
        "churn": resultat,
        "probabilite_depart": round(proba, 3),
        "message": "Le client va partir" if resultat == 1 else "Le client reste",
    }