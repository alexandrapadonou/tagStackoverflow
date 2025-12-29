# tools/convert_legacy_bundle.py
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

# === IMPORTANT ===
# Re-définir la classe EXACTE (nom + attributs) qui a été picklée depuis le notebook.
class TaggerModel:
    def __init__(self, vectorizer, estimator, mlb, topk=5, threshold=0.0):
        self.vectorizer = vectorizer
        self.estimator = estimator
        self.mlb = mlb
        self.topk = int(topk)
        self.threshold = threshold

    def _scores(self, X):
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
        scores = self._scores(X_vec)

        if scores.dtype == int or set(np.unique(scores)).issubset({0, 1}):
            return scores.astype(int)

        topk_idx = np.argsort(-scores, axis=1)[:, : self.topk]
        Y_topk = np.zeros_like(scores, dtype=int)
        rows = np.arange(scores.shape[0])[:, None]
        Y_topk[rows, topk_idx] = 1

        if self.threshold is not None:
            Y_thr = (scores >= self.threshold).astype(int)
            return np.maximum(Y_topk, Y_thr)

        return Y_topk

    def predict(self, texts):
        if isinstance(texts, (pd.Series, np.ndarray)):
            texts = texts.tolist()
        if isinstance(texts, str):
            texts = [texts]

        X_vec = self.vectorizer.transform(texts)
        Y_bin = self.predict_topk_binary(X_vec)

        tags = []
        for row in Y_bin:
            idx = np.where(row == 1)[0]
            tags.append(self.mlb.classes_[idx].tolist())
        return tags


def main():
    if len(sys.argv) != 3:
        print("Usage: python tools/convert_legacy_bundle.py <legacy_bundle_path> <out_dir>")
        sys.exit(1)

    legacy_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    # Injecter TaggerModel dans __main__ pour satisfaire le pickle
    import __main__
    __main__.TaggerModel = TaggerModel

    print(f"Loading legacy bundle: {legacy_path}")
    bundle = joblib.load(legacy_path)

    # bundle est maintenant une instance TaggerModel (ou objet compatible)
    vectorizer = bundle.vectorizer
    estimator = bundle.estimator
    mlb = bundle.mlb
    config = {"topk": int(getattr(bundle, "topk", 5)),
              "threshold": float(getattr(bundle, "threshold", 0.0))}

    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.joblib"))
    joblib.dump(estimator, os.path.join(out_dir, "estimator.joblib"))
    joblib.dump(mlb, os.path.join(out_dir, "mlb.joblib"))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("Done. Exported:")
    print(" - vectorizer.joblib")
    print(" - estimator.joblib")
    print(" - mlb.joblib")
    print(" - config.json")


if __name__ == "__main__":
    main()