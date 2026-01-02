"""
Tests unitaires pour l'endpoint /health de l'API.

Ce module teste la fonctionnalité de vérification de santé de l'API

Tests couverts:
- Vérification que l'endpoint répond avec le statut HTTP 200
- Vérification que la réponse contient le champ "status" avec la valeur "ok"
"""
from fastapi.testclient import TestClient
from api.main import app

# Client de test FastAPI qui simule les requêtes HTTP
# Permet de tester l'API sans avoir besoin d'un serveur en cours d'exécution
client = TestClient(app)


def test_health():
    """
    Test de l'endpoint /health.
    
    Vérifie que:
    1. L'endpoint répond avec un statut HTTP 200 (succès)
    2. La réponse JSON contient le champ "status" avec la valeur "ok"
    
    Ce test est critique pour:
    - Le monitoring automatisé (Heroku, Azure, etc.)
    - Les health checks
    - La validation du déploiement CI/CD
    """
    res = client.get("/health")
    assert res.status_code == 200, "L'endpoint /health doit retourner 200"
    assert res.json()["status"] == "ok", "Le statut doit être 'ok'"