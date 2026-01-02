# api/inference.py
"""
Service d'inférence pour la prédiction de tags StackOverflow.

Ce module encapsule toute la logique d'inférence du modèle ML:
- Chargement des artefacts (vectorizer, estimator, MultiLabelBinarizer)
- Vectorisation du texte d'entrée
- Prédiction via le modèle
- Post-traitement pour sélectionner les tags selon topk et threshold
- Conversion des indices en noms de tags

Le service est conçu pour être compatible avec différents types d'estimateurs:
- Modèles avec predict_proba() (probabilités)
- Modèles avec decision_function() (scores de décision)
- Modèles avec predict() uniquement (prédictions binaires)
"""
import json
import os
import joblib
import numpy as np


class InferenceService:
    """
    Service centralisé pour l'inférence du modèle de prédiction de tags.
    
    Cette classe gère le cycle de vie complet de l'inférence:
    1. Chargement des artefacts depuis le disque
    2. Vectorisation du texte d'entrée
    3. Prédiction via le modèle ML
    4. Sélection et formatage des tags prédits
    
    Attributes:
        model_dir: Chemin vers le répertoire contenant les artefacts du modèle
        vectorizer: Vectoriseur TF-IDF (ou similaire) pour transformer le texte
        estimator: Modèle de classification multi-label (LinearSVC, etc.)
        mlb: MultiLabelBinarizer pour encoder/décoder les tags
        topk: Nombre par défaut de tags à retourner
        threshold: Seuil de probabilité par défaut pour inclure un tag
    """
    
    def __init__(self, model_dir: str):
        """
        Initialise le service d'inférence.
        
        Args:
            model_dir: Chemin vers le répertoire contenant les artefacts du modèle
                      (vectorizer.joblib, estimator.joblib, mlb.joblib, config.json)
        """
        self.model_dir = model_dir
        self.vectorizer = None  # Sera chargé via load()
        self.estimator = None   # Sera chargé via load()
        self.mlb = None         # Sera chargé via load()
        self.topk = 5           # Valeur par défaut, sera écrasée par config.json
        self.threshold = 0.0    # Valeur par défaut, sera écrasée par config.json

    def load(self):
        """
        Charge tous les artefacts du modèle depuis le disque.
        
        Artefacts requis:
        - vectorizer.joblib: Vectoriseur TF-IDF pour transformer le texte
        - estimator.joblib: Modèle de classification multi-label
        - mlb.joblib: MultiLabelBinarizer pour encoder/décoder les tags
        - config.json: Configuration contenant topk et threshold par défaut
        
        Cette méthode doit être appelée avant toute prédiction.
        
        Returns:
            self: Permet le chaînage de méthodes
            
        Raises:
            FileNotFoundError: Si un artefact requis est manquant
            json.JSONDecodeError: Si config.json est invalide
        """
        # Construction des chemins vers les artefacts
        vec_path = os.path.join(self.model_dir, "vectorizer.joblib")
        est_path = os.path.join(self.model_dir, "estimator.joblib")
        mlb_path = os.path.join(self.model_dir, "mlb.joblib")
        cfg_path = os.path.join(self.model_dir, "config.json")

        # Vérification de l'existence de tous les fichiers requis
        for p in [vec_path, est_path, mlb_path, cfg_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing model artifact: {p}")

        # Chargement des artefacts sérialisés
        self.vectorizer = joblib.load(vec_path)
        self.estimator = joblib.load(est_path)
        self.mlb = joblib.load(mlb_path)

        # Chargement de la configuration (topk, threshold)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.topk = int(cfg.get("topk", 5))
        self.threshold = float(cfg.get("threshold", 0.0))
        return self

    def _scores(self, X_vec):
        """
        Extrait les scores de prédiction du modèle selon son type.
        
        Cette méthode standardise les différences d’implémentation entre plusieurs types de modèles
        - Modèles probabilistes (RandomForest, LogisticRegression): predict_proba()
        - Modèles à marge (SVM): decision_function()
        - Modèles binaires uniquement: predict()
        
        Args:
            X_vec: Matrice de features vectorisées (format sparse ou dense)
            
        Returns:
            np.ndarray: Matrice de scores (probabilités, scores de décision, ou prédictions binaires)
                       Shape: (n_samples, n_classes)
        """
        # Priorité 1: Modèles avec probabilités (RandomForest, LogisticRegression, etc.)
        if hasattr(self.estimator, "predict_proba"):
            scores = self.estimator.predict_proba(X_vec)
        # Priorité 2: Modèles avec fonction de décision (SVM, etc.)
        elif hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(X_vec)
            # Gestion du cas où decision_function retourne un vecteur 1D (classification binaire)
            if getattr(scores, "ndim", 1) == 1:
                scores = scores.reshape(-1, 1)
        else:
            # Fallback: Prédiction binaire directe (pas de scores continus)
            return self.estimator.predict(X_vec)
        return scores

    def predict_tags(self, text: str, topk: int | None = None, threshold: float | None = None):
        """
        Prédit les tags appropriés pour un texte donné.
        
        Pipeline complet:
        1. Vectorisation du texte via TF-IDF
        2. Prédiction des scores pour chaque tag
        3. Sélection des tags selon topk (top-K tags) et threshold (seuil minimum)
        4. Conversion des indices en noms de tags via MultiLabelBinarizer
        
        Args:
            text: Texte de la question StackOverflow (titre + corps concaténés)
            topk: Nombre maximum de tags à retourner (optionnel, utilise la valeur par défaut si None)
            threshold: Seuil de probabilité minimum pour inclure un tag (optionnel)
        
        Returns:
            list[str]: Liste des tags prédits, triés par pertinence décroissante
            
        Raises:
            AttributeError: Si le modèle n'a pas été chargé (load() non appelé)
        """
        # Utilisation des valeurs par défaut si non spécifiées
        if topk is None:
            topk = self.topk
        if threshold is None:
            threshold = self.threshold

        # Étape 1: Vectorisation du texte
        # Transforme le texte brut en vecteur de features TF-IDF
        X_vec = self.vectorizer.transform([text])
        
        # Étape 2: Prédiction des scores
        scores = self._scores(X_vec)

        # Étape 3: Conversion en prédictions binaires
        # Si les scores sont déjà binaires (0 ou 1), on les utilise directement
        if isinstance(scores, np.ndarray) and scores.dtype in (np.int32, np.int64) or set(np.unique(scores)).issubset({0, 1}):
            y_bin = scores.astype(int)
        else:
            # Conversion des scores continus en sélection de tags
            scores = np.asarray(scores)
            
            # Sélection du top-K: on garde les K tags avec les scores les plus élevés
            topk_idx = np.argsort(-scores, axis=1)[:, :topk]  # Indices triés par score décroissant
            y_topk = np.zeros_like(scores, dtype=int)
            rows = np.arange(scores.shape[0])[:, None]
            y_topk[rows, topk_idx] = 1  # Marque les top-K tags à 1
            
            # Application du seuil: on garde aussi les tags au-dessus du threshold
            if threshold is not None:
                y_thr = (scores >= threshold).astype(int)
                # Union des deux critères: top-K OU au-dessus du threshold
                y_bin = np.maximum(y_topk, y_thr)
            else:
                y_bin = y_topk

        # Étape 4: Conversion des indices en noms de tags
        # Trouve les indices des tags sélectionnés (valeur = 1)
        idx = np.where(y_bin[0] == 1)[0]
        # Convertit les indices en noms de tags via le MultiLabelBinarizer
        return self.mlb.classes_[idx].tolist()