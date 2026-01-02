# tests/test_predict.py
"""
Tests unitaires pour l'endpoint /predict de l'API.

Ce module teste la fonctionnalité principale de l'API: la prédiction de tags.
Les tests vérifient que l'endpoint accepte des requêtes valides et retourne
des réponses au format attendu.

Tests couverts:
- Vérification du statut HTTP 200 pour une requête valide
- Vérification de la structure de la réponse (présence du champ "tags")
- Vérification du type de données retourné (liste de strings)

Note: Ces tests nécessitent que les modèles soient disponibles.
      En environnement CI/CD, les modèles doivent être téléchargés avant les tests.
"""
from fastapi.testclient import TestClient
from api.main import app


def test_predict_response():
    """
    Test de l'endpoint /predict avec une requête simple.
    
    Ce test vérifie:
    1. Que l'endpoint accepte une requête POST avec un texte valide
    2. Que la réponse a un statut HTTP 200 (succès)
    3. Que la réponse contient le champ "tags"
    4. Que "tags" est une liste (peut être vide si le modèle ne trouve rien)
    
    Utilisation du context manager (with) pour garantir:
    - Le déclenchement de l'événement startup() (chargement des modèles)
    - Le déclenchement de l'événement shutdown() (nettoyage)
    
    Args:
        payload: Dictionnaire contenant le texte à analyser
                Format: {"text": "Hello world"}
    """
    # Utilisation du context manager pour déclencher startup/shutdown
    # Important: cela garantit que les modèles sont chargés avant le test
    with TestClient(app) as client:
        payload = {"text": "Hello world"}
        res = client.post("/predict", json=payload)
        
        # Vérification du statut HTTP
        assert res.status_code == 200, "L'endpoint /predict doit retourner 200"
        
        # Vérification de la structure de la réponse
        data = res.json()
        assert "tags" in data, "La réponse doit contenir le champ 'tags'"
        assert isinstance(data["tags"], list), "Le champ 'tags' doit être une liste"