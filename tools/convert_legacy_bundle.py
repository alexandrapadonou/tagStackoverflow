# tools/convert_legacy_bundle.py
"""
Script de conversion d'un bundle legacy de modèle en artefacts unitaires.

Ce script permet de migrer un modèle sérialisé sous forme de bundle monolithique
(typiquement créé lors de la phase d'expérimentation dans un notebook Jupyter)
vers le format d'artefacts unitaires attendu par l'API de production.

Contexte:
    Lors de l'expérimentation, le modèle a été sérialisé comme un objet unique
    (TaggerModel) contenant vectorizer, estimator, mlb, topk, threshold.
    L'API de production attend des fichiers séparés pour une meilleure modularité.

Utilisation:
    python tools/convert_legacy_bundle.py <legacy_bundle_path> <out_dir>

Exemple:
    python tools/convert_legacy_bundle.py model_bundle.joblib models/

Artefacts produits:
    - vectorizer.joblib: Vectoriseur TF-IDF
    - estimator.joblib: Modèle de classification
    - mlb.joblib: MultiLabelBinarizer
    - config.json: Configuration (topk, threshold)

Note importante:
    La classe TaggerModel doit être redéfinie EXACTEMENT comme elle l'était
    lors de la sérialisation, car joblib/pickle nécessite que la classe soit
    disponible pour désérialiser l'objet.
"""
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

# === IMPORTANT ===
# Re-définir la classe EXACTE (nom + attributs) qui a été picklée depuis le notebook.
# Cette classe doit correspondre EXACTEMENT à celle utilisée lors de la sérialisation,
# sinon joblib ne pourra pas désérialiser le bundle legacy.
class TaggerModel:
    """
    Classe legacy du modèle de prédiction de tags.
    
    Cette classe représente le format utilisé lors de l'expérimentation.
    Elle est redéfinie ici uniquement pour permettre la désérialisation du bundle legacy.
    
    Attributes:
        vectorizer: Vectoriseur TF-IDF pour transformer le texte
        estimator: Modèle de classification multi-label
        mlb: MultiLabelBinarizer pour encoder/décoder les tags
        topk: Nombre par défaut de tags à retourner
        threshold: Seuil de probabilité par défaut
    """
    
    def __init__(self, vectorizer, estimator, mlb, topk=5, threshold=0.0):
        """
        Initialise le modèle
        
        Args:
            vectorizer: Vectoriseur TF-IDF
            estimator: Modèle de classification
            mlb: MultiLabelBinarizer
            topk: Nombre par défaut de tags (défaut: 5)
            threshold: Seuil par défaut (défaut: 0.0)
        """
        self.vectorizer = vectorizer
        self.estimator = estimator
        self.mlb = mlb
        self.topk = int(topk)
        self.threshold = threshold

    def _scores(self, X):
        """
        Extrait les scores de prédiction selon le type d'estimateur.
        
        Args:
            X: Matrice de features vectorisées
            
        Returns:
            np.ndarray: Scores de prédiction
        """
        if hasattr(self.estimator, "predict_proba"):
            scores = self.estimator.predict_proba(X)
        elif hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(X)
            if getattr(scores, "ndim", 1) == 1:
                scores = scores.reshape(-1, 1)
        else:
            scores = self.estimator.predict(X)
        return scores

    def predict_topk_binary(self, X_vec):
        """
        Prédit les tags en sélectionnant le top-K et en appliquant le threshold.
        
        Args:
            X_vec: Matrice de features vectorisées
            
        Returns:
            np.ndarray: Prédictions binaires (0 ou 1) pour chaque tag
        """
        scores = self._scores(X_vec)

        # Si les scores sont déjà binaires, on les retourne directement
        if scores.dtype == int or set(np.unique(scores)).issubset({0, 1}):
            return scores.astype(int)

        # Sélection du top-K
        topk_idx = np.argsort(-scores, axis=1)[:, : self.topk]
        Y_topk = np.zeros_like(scores, dtype=int)
        rows = np.arange(scores.shape[0])[:, None]
        Y_topk[rows, topk_idx] = 1

        # Application du threshold si défini
        if self.threshold is not None:
            Y_thr = (scores >= self.threshold).astype(int)
            return np.maximum(Y_topk, Y_thr)

        return Y_topk

    def predict(self, texts):
        """
        Prédit les tags pour une liste de textes.
        
        Args:
            texts: Texte unique, liste de textes, ou Series pandas
            
        Returns:
            list: Liste de listes de tags (une liste par texte)
        """
        # Normalisation de l'entrée en liste
        if isinstance(texts, (pd.Series, np.ndarray)):
            texts = texts.tolist()
        if isinstance(texts, str):
            texts = [texts]

        # Vectorisation et prédiction
        X_vec = self.vectorizer.transform(texts)
        Y_bin = self.predict_topk_binary(X_vec)

        # Conversion des indices en noms de tags
        tags = []
        for row in Y_bin:
            idx = np.where(row == 1)[0]
            tags.append(self.mlb.classes_[idx].tolist())
        return tags


def main():
    """
    Fonction principale du script de conversion.
    
    Cette fonction:
    1. Valide les arguments de ligne de commande
    2. Charge le bundle legacy depuis le disque
    3. Extrait les composants (vectorizer, estimator, mlb, config)
    4. Sauvegarde chaque composant dans un fichier séparé
    
    Args (via sys.argv):
        legacy_bundle_path: Chemin vers le fichier .joblib contenant le bundle legacy
        out_dir: Répertoire de destination pour les artefacts unitaires
        
    Raises:
        SystemExit: Si le nombre d'arguments est incorrect
        FileNotFoundError: Si le bundle legacy n'existe pas
        AttributeError: Si le bundle n'a pas les attributs attendus
    """
    # Validation des arguments de ligne de commande
    if len(sys.argv) != 3:
        print("Usage: python tools/convert_legacy_bundle.py <legacy_bundle_path> <out_dir>")
        print("Example: python tools/convert_legacy_bundle.py model_bundle.joblib models/")
        sys.exit(1)

    legacy_path = sys.argv[1]
    out_dir = sys.argv[2]
    
    # Création du répertoire de sortie s'il n'existe pas
    os.makedirs(out_dir, exist_ok=True)

    # Injection de TaggerModel dans __main__ pour satisfaire le pickle
    # joblib/pickle cherche la classe dans le module __main__ lors de la désérialisation
    import __main__
    __main__.TaggerModel = TaggerModel

    print(f"Loading legacy bundle: {legacy_path}")
    # Chargement du bundle legacy
    # Cette opération nécessite que TaggerModel soit disponible dans __main__
    bundle = joblib.load(legacy_path)

    # Extraction des composants du bundle
    # bundle est maintenant une instance TaggerModel (ou objet compatible)
    vectorizer = bundle.vectorizer
    estimator = bundle.estimator
    mlb = bundle.mlb
    
    # Extraction de la configuration avec valeurs par défaut si absentes
    config = {
        "topk": int(getattr(bundle, "topk", 5)),
        "threshold": float(getattr(bundle, "threshold", 0.0))
    }

    # Sauvegarde de chaque composant dans un fichier séparé
    # Format compatible avec l'API de production
    print(f"Exporting artifacts to: {out_dir}")
    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.joblib"))
    joblib.dump(estimator, os.path.join(out_dir, "estimator.joblib"))
    joblib.dump(mlb, os.path.join(out_dir, "mlb.joblib"))
    
    # Sauvegarde de la configuration en JSON (lisible et modifiable)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("Done. Exported:")
    print(" - vectorizer.joblib")
    print(" - estimator.joblib")
    print(" - mlb.joblib")
    print(" - config.json")
    print(f"\nThese files are ready to be used by the API in: {out_dir}")


if __name__ == "__main__":
    main()