import requests  # envoyer des requêtes HTTP depuis Python

URL = "http://localhost:8000"  # l'adresse de notre API
TIMEOUT = 10                   # secondes : évite d'attendre indéfiniment


def afficher(titre, r):
    """Affiche le code HTTP et le JSON de la réponse."""
    print(titre, "->", r.status_code, r.json())


try:
    # 1. L'API est-elle en vie ?
    r = requests.get(URL + "/health", timeout=TIMEOUT)
    print("GET /health ->", r.status_code, r.json())

    # 2. Un client à faible consommation et faible ancienneté
    client = {
        "age": 35,
        "salaire": 2500,
        "abonnement": "Standard",
        "anciennete": 1,
        "consommation": 10,
    }
    r = requests.post(URL + "/predict", json=client, timeout=TIMEOUT)  # json= : envoie en JSON
    afficher("POST /predict (client 1)", r)

    # 3. Un client fidèle : forte consommation, forte ancienneté
    client2 = {
        "age": 50,
        "salaire": 4000,
        "abonnement": "Premium",
        "anciennete": 10,
        "consommation": 80,
    }
    r = requests.post(URL + "/predict", json=client2, timeout=TIMEOUT)
    afficher("POST /predict (client 2)", r)

    # 4. Une erreur volontaire : l'âge est du texte
    mauvais = dict(client, age="abc")  # copie de client avec age modifié
    r = requests.post(URL + "/predict", json=mauvais, timeout=TIMEOUT)
    print("POST /predict (age='abc') ->", r.status_code)  # attendu : 422
    detail = r.json().get("detail", [])
    if detail:
        print("Message :", detail[0]["msg"])

except requests.exceptions.ConnectionError:
    raise SystemExit(
        "Impossible de joindre l'API sur " + URL + ".\n"
        "Lance-la d'abord dans un autre terminal : uvicorn main:app --reload"
    )
except requests.exceptions.Timeout:
    raise SystemExit("L'API met trop de temps à répondre (timeout).")