# api/inference.py
import json
import os
import joblib
import numpy as np

class InferenceService:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.vectorizer = None
        self.estimator = None
        self.mlb = None
        self.topk = 5
        self.threshold = 0.0

    def load(self):
        vec_path = os.path.join(self.model_dir, "vectorizer.joblib")
        est_path = os.path.join(self.model_dir, "estimator.joblib")
        mlb_path = os.path.join(self.model_dir, "mlb.joblib")
        cfg_path = os.path.join(self.model_dir, "config.json")

        for p in [vec_path, est_path, mlb_path, cfg_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing model artifact: {p}")

        self.vectorizer = joblib.load(vec_path)
        self.estimator = joblib.load(est_path)
        self.mlb = joblib.load(mlb_path)

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.topk = int(cfg.get("topk", 5))
        self.threshold = float(cfg.get("threshold", 0.0))
        return self

    def _scores(self, X_vec):
        if hasattr(self.estimator, "predict_proba"):
            scores = self.estimator.predict_proba(X_vec)
        elif hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(X_vec)
            if getattr(scores, "ndim", 1) == 1:
                scores = scores.reshape(-1, 1)
        else:
            # fallback : prédiction binaire dure
            return self.estimator.predict(X_vec)
        return scores

    def predict_tags(self, text: str, topk: int | None = None, threshold: float | None = None):
        if topk is None:
            topk = self.topk
        if threshold is None:
            threshold = self.threshold

        X_vec = self.vectorizer.transform([text])
        scores = self._scores(X_vec)

        # Si déjà binaire
        if isinstance(scores, np.ndarray) and scores.dtype in (np.int32, np.int64) or set(np.unique(scores)).issubset({0, 1}):
            y_bin = scores.astype(int)
        else:
            scores = np.asarray(scores)
            topk_idx = np.argsort(-scores, axis=1)[:, :topk]
            y_topk = np.zeros_like(scores, dtype=int)
            rows = np.arange(scores.shape[0])[:, None]
            y_topk[rows, topk_idx] = 1

            if threshold is not None:
                y_thr = (scores >= threshold).astype(int)
                y_bin = np.maximum(y_topk, y_thr)
            else:
                y_bin = y_topk

        idx = np.where(y_bin[0] == 1)[0]
        return self.mlb.classes_[idx].tolist()