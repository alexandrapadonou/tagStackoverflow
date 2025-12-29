import os
import zipfile
import requests
from pathlib import Path

REQUIRED_FILES = ["vectorizer.joblib", "estimator.joblib", "mlb.joblib", "config.json"]

def _all_present(model_dir: str) -> bool:
    return all(Path(model_dir, f).exists() for f in REQUIRED_FILES)

def ensure_models(model_dir: str, model_zip_url: str, *, timeout_connect=10, timeout_read=60):
    """
    Télécharge et extrait model_artifacts.zip dans model_dir si nécessaire.
    - Streaming (pas de gros buffer mémoire)
    - Timeouts explicites
    - Logs stdout visibles dans App Service log stream
    """
    os.makedirs(model_dir, exist_ok=True)

    if _all_present(model_dir):
        print(f"[model_fetch] Models already present in {model_dir}")
        return

    if not model_zip_url:
        raise RuntimeError("MODEL_BLOB_URL is empty and models are missing.")

    zip_path = Path(model_dir) / "model_artifacts.zip"
    tmp_extract_dir = Path(model_dir) / "_extract_tmp"

    print(f"[model_fetch] Downloading model zip from URL (len={len(model_zip_url)})")
    print(f"[model_fetch] Saving to: {zip_path}")

    # Téléchargement en streaming
    with requests.get(model_zip_url, stream=True, timeout=(timeout_connect, timeout_read)) as r:
        r.raise_for_status()
        total = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
                    if total % (10 * 1024 * 1024) < (1 * 1024 * 1024):  # log ~tous les ~10MB
                        print(f"[model_fetch] Downloaded ~{total/1024/1024:.1f} MB...")

    print(f"[model_fetch] Download complete: {total/1024/1024:.2f} MB")
    print(f"[model_fetch] Extracting zip...")

    # Extraction atomique (extrait dans un tmp puis move)
    if tmp_extract_dir.exists():
        # nettoyage
        for p in tmp_extract_dir.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted(tmp_extract_dir.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()
        tmp_extract_dir.rmdir()

    tmp_extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_extract_dir)

    # Certains zips contiennent un sous-dossier, on gère ça
    # On cherche les fichiers attendus quelque part dans tmp_extract_dir.
    found = {}
    for fname in REQUIRED_FILES:
        matches = list(tmp_extract_dir.rglob(fname))
        if not matches:
            raise RuntimeError(f"[model_fetch] Missing {fname} inside zip.")
        found[fname] = matches[0]

    # Move vers model_dir
    for fname, src in found.items():
        dst = Path(model_dir) / fname
        if dst.exists():
            dst.unlink()
        src.replace(dst)

    # Cleanup
    try:
        zip_path.unlink()
    except Exception:
        pass

    # Cleanup tmp dir
    for p in tmp_extract_dir.rglob("*"):
        if p.is_file():
            p.unlink()
    for p in sorted(tmp_extract_dir.rglob("*"), reverse=True):
        if p.is_dir():
            p.rmdir()
    tmp_extract_dir.rmdir()

    if not _all_present(model_dir):
        raise RuntimeError("[model_fetch] Extraction finished but required files are not present.")

    print(f"[model_fetch] Models ready in {model_dir}")