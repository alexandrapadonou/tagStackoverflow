# api/main.py
"""
Module principal de l'API FastAPI pour la prédiction de tags StackOverflow.

Ce module expose une API REST permettant de prédire les tags appropriés
pour une question StackOverflow en se basant sur son texte (titre et corps).

Architecture:
    - FastAPI pour la gestion des endpoints REST
    - Pydantic pour la validation des données d'entrée/sortie
    - InferenceService pour l'exécution du modèle ML
    - model_fetch pour le téléchargement automatique des modèles si nécessaire

Endpoints:
    - GET /health: Vérification de l'état de l'API et du répertoire des modèles
    - POST /predict: Prédiction de tags à partir d'un texte

Variables d'environnement:
    - MODEL_DIR: Répertoire contenant les artefacts du modèle (défaut: "models")
    - MODEL_BLOB_URL: URL du blob storage Azure contenant le zip des modèles
"""
import os
from fastapi import FastAPI
from api.inference import InferenceService
from api.model_fetch import ensure_models
from api.schemas import PredictRequest, PredictResponse

# Configuration des chemins et URLs via variables d'environnement
# Permet la flexibilité entre environnement local et cloud (Heroku, Azure, etc.)
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_BLOB_URL = os.getenv("MODEL_BLOB_URL", "https://tagmodel.blob.core.windows.net/modellight/model_artifacts.zip?sp=rw&st=2025-12-24T11:13:49Z&se=2026-01-09T19:28:49Z&spr=https&sv=2024-11-04&sr=b&sig=Y1DIEEL%2Blg88ujVpN3DiGHbpO1IQaIxeqcaxy%2FqfUWc%3D")


# Initialisation de l'application FastAPI
app = FastAPI(
    title="StackOverflow Tagger API",
    version="1.0.0",
    description="API de prédiction de tags pour les questions StackOverflow basée sur un modèle ML"
)

# Instance globale du service d'inférence
# Sera chargée au démarrage de l'application via l'événement startup
svc = InferenceService(MODEL_DIR)


@app.on_event("startup")
def startup():
    """
    Fonction exécutée au démarrage de l'API.
    
    Séquence d'initialisation:
    1. Vérification de santé basique
    2. Téléchargement des modèles si absents (via MODEL_BLOB_URL)
    3. Chargement des artefacts du modèle en mémoire
    
    Cette fonction garantit que l'API est prête à servir des requêtes
    avant d'accepter du trafic.
    """
    health()
    # Télécharge et extrait les modèles si nécessaire
    ensure_models(MODEL_DIR, MODEL_BLOB_URL)
    # Charge les artefacts en mémoire (vectorizer, estimator, mlb)
    svc.load()


@app.get("/health")
def health():
    """
    Endpoint de vérification de santé de l'API.
    
    Utilisé pour:
    - Vérifier que l'API est opérationnelle
    - Monitoring et health checks (Heroku, load balancers, etc.)
    - Debugging: confirmer le répertoire de modèles utilisé
    
    Returns:
        dict: Statut de l'API et chemin du répertoire des modèles
            {
                "status": "ok",
                "model_dir": "models"
            }
    """
    return {"status": "ok", "model_dir": MODEL_DIR}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Endpoint principal de prédiction de tags.
    
    Traite une requête contenant le texte d'une question StackOverflow
    et retourne une liste de tags prédits par le modèle ML.
    
    Processus:
    1. Validation automatique de la requête via Pydantic
    2. Vectorisation du texte via le vectorizer chargé
    3. Prédiction via le modèle ML
    4. Sélection des tags selon topk et threshold
    5. Conversion des indices en noms de tags via MultiLabelBinarizer
    
    Args:
        req: Requête validée contenant le texte et paramètres optionnels
        
    Returns:
        PredictResponse: Réponse contenant la liste des tags prédits
        
    Raises:
        HTTPException: Si le modèle n'est pas chargé ou en cas d'erreur d'inférence
    """
    tags = svc.predict_tags(req.text, topk=req.topk, threshold=req.threshold)
    return PredictResponse(tags=tags)