# api/schemas.py
"""
Schémas Pydantic pour la validation des requêtes et réponses de l'API.

Ces schémas garantissent:
- La validation automatique des données d'entrée
- La sérialisation/désérialisation JSON
- La cohérence des types entre client et serveur

Tous les schémas de l'API sont centralisés dans ce module pour faciliter
la maintenance et éviter la duplication de code.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    """
    Schéma de validation pour les requêtes de prédiction de tags.
    
    Ce schéma valide automatiquement:
    - Que le texte n'est pas vide (min_length=1)
    - Que topk est entre 1 et 50 si fourni
    - Que threshold est un nombre valide si fourni
    
    Attributes:
        text: Texte de la question StackOverflow à analyser
        topk: Nombre maximum de tags à retourner (optionnel, remplace la valeur par défaut)
        threshold: Seuil de probabilité minimum pour un tag (optionnel, remplace la valeur par défaut)
    """
    text: str = Field(..., min_length=1, description="Texte de la question StackOverflow")
    topk: Optional[int] = Field(None, ge=1, le=50, description="Override top-k if provided")
    threshold: Optional[float] = Field(None, ge=-1e9, le=1e9, description="Override threshold if provided")


class PredictResponse(BaseModel):
    """
    Schéma de réponse pour les prédictions de tags.
    
    Attributes:
        tags: Liste des tags prédits par le modèle, triés par pertinence décroissante
    """
    tags: List[str] = Field(..., description="Liste des tags prédits")