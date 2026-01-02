"""
Module de téléchargement et extraction des artefacts de modèle depuis un blob storage.

Ce module gère le téléchargement automatique des modèles lors du déploiement cloud,
permettant de ne pas inclure les artefacts (potentiellement volumineux) dans le repository.

Fonctionnalités:
- Téléchargement en streaming (économie mémoire)
- Timeouts explicites pour éviter les blocages
- Extraction atomique (dossier temporaire puis déplacement)
- Validation stricte des fichiers requis
- Logs visibles dans les logs de l'App Service

Utilisation:
    Appelé automatiquement au démarrage de l'API si les modèles sont absents.
    Peut aussi être utilisé manuellement pour préparer l'environnement local.
"""
import os
import zipfile
import requests
from pathlib import Path

# Liste des fichiers requis pour que le modèle soit considéré comme complet
REQUIRED_FILES = ["vectorizer.joblib", "estimator.joblib", "mlb.joblib", "config.json"]


def _all_present(model_dir: str) -> bool:
    """
    Vérifie si tous les fichiers requis sont présents dans le répertoire.
    
    Args:
        model_dir: Chemin vers le répertoire contenant les artefacts
        
    Returns:
        bool: True si tous les fichiers REQUIRED_FILES existent, False sinon
    """
    return all(Path(model_dir, f).exists() for f in REQUIRED_FILES)

def ensure_models(model_dir: str, model_zip_url: str, *, timeout_connect=10, timeout_read=60):
    """
    Télécharge et extrait les artefacts du modèle depuis une URL si absents localement.
    
    Cette fonction est idempotente: si les modèles sont déjà présents, elle ne fait rien.
    Sinon, elle télécharge le zip depuis l'URL fournie et l'extrait dans model_dir.
    
    Caractéristiques:
    - Streaming: téléchargement par chunks pour éviter la surcharge mémoire
    - Timeouts explicites: évite les blocages en cas de réseau lent
    - Extraction atomique: extraction dans un dossier temporaire puis déplacement
    - Validation stricte: vérifie la présence de tous les fichiers requis
    - Logs: messages visibles dans les logs de l'App Service (Heroku, Azure, etc.)
    
    Args:
        model_dir: Répertoire de destination pour les artefacts du modèle
        model_zip_url: URL complète du blob storage contenant le zip des modèles
        timeout_connect: Timeout pour la connexion initiale (secondes, défaut: 10)
        timeout_read: Timeout pour la lecture des données (secondes, défaut: 60)
        
    Raises:
        RuntimeError: Si l'URL est vide et que les modèles sont manquants
        RuntimeError: Si un fichier requis est absent après extraction
        requests.RequestException: En cas d'erreur de téléchargement
        zipfile.BadZipFile: Si le fichier zip est corrompu
    """
    # Création du répertoire de destination s'il n'existe pas
    os.makedirs(model_dir, exist_ok=True)

    # Vérification: si tous les fichiers sont déjà présents, on ne fait rien
    if _all_present(model_dir):
        print(f"[model_fetch] Models already present in {model_dir}")
        return

    # Validation: l'URL doit être fournie si les modèles sont absents
    if not model_zip_url:
        raise RuntimeError("MODEL_BLOB_URL is empty and models are missing.")

    # Chemins pour le téléchargement et l'extraction
    zip_path = Path(model_dir) / "model_artifacts.zip"
    tmp_extract_dir = Path(model_dir) / "_extract_tmp"

    print(f"[model_fetch] Downloading model zip from URL (len={len(model_zip_url)})")
    print(f"[model_fetch] Saving to: {zip_path}")

    # Téléchargement en streaming pour économiser la mémoire
    # Utilise des chunks de 1MB au lieu de charger tout le fichier en mémoire
    with requests.get(model_zip_url, stream=True, timeout=(timeout_connect, timeout_read)) as r:
        r.raise_for_status()  # Lève une exception si le statut HTTP n'est pas 200
        total = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB par chunk
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
                    # Log de progression tous les ~10MB pour suivre le téléchargement
                    if total % (10 * 1024 * 1024) < (1 * 1024 * 1024):
                        print(f"[model_fetch] Downloaded ~{total/1024/1024:.1f} MB...")

    print(f"[model_fetch] Download complete: {total/1024/1024:.2f} MB")
    print(f"[model_fetch] Extracting zip...")

    # Extraction atomique: on extrait dans un dossier temporaire puis on déplace
    # Cela évite de corrompre les fichiers existants en cas d'erreur
    if tmp_extract_dir.exists():
        # Nettoyage préventif du dossier temporaire s'il existe déjà
        for p in tmp_extract_dir.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted(tmp_extract_dir.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()
        tmp_extract_dir.rmdir()

    tmp_extract_dir.mkdir(parents=True, exist_ok=True)

    # Extraction du zip dans le dossier temporaire
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_extract_dir)

    # Gestion de la structure du zip: certains zips contiennent un sous-dossier
    # On cherche récursivement les fichiers requis dans tmp_extract_dir
    found = {}
    for fname in REQUIRED_FILES:
        matches = list(tmp_extract_dir.rglob(fname))
        if not matches:
            raise RuntimeError(f"[model_fetch] Missing {fname} inside zip.")
        found[fname] = matches[0]  # Prend le premier match trouvé

    # Déplacement atomique des fichiers vers model_dir
    # Remplace les fichiers existants s'ils sont présents
    for fname, src in found.items():
        dst = Path(model_dir) / fname
        if dst.exists():
            dst.unlink()  # Supprime l'ancien fichier s'il existe
        src.replace(dst)  # Déplace le nouveau fichier

    # Nettoyage du fichier zip téléchargé
    try:
        zip_path.unlink()
    except Exception:
        pass  # Ignore les erreurs de suppression (fichier peut être verrouillé)

    # Nettoyage du dossier temporaire d'extraction
    for p in tmp_extract_dir.rglob("*"):
        if p.is_file():
            p.unlink()
    for p in sorted(tmp_extract_dir.rglob("*"), reverse=True):
        if p.is_dir():
            p.rmdir()
    tmp_extract_dir.rmdir()

    # Validation finale: vérifie que tous les fichiers sont bien présents
    if not _all_present(model_dir):
        raise RuntimeError("[model_fetch] Extraction finished but required files are not present.")

    print(f"[model_fetch] Models ready in {model_dir}")