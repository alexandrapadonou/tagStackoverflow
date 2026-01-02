# StackOverflow Tagger API

## 1. Objectif et périmètre

Ce projet implémente une **API de prédiction de tags StackOverflow** basée sur un modèle de machine learning multi-label entraîné en amont.

Le périmètre couvre :

* L’exposition du modèle sous forme d’API REST
* La gestion des artefacts modèles (chargement, versionnement, portabilité)
* L’inférence temps réel
* Les tests automatisés
* Le déploiement Cloud avec Heroku
* Les fondations nécessaires à une démarche **MLOps industrialisable**

L’entraînement du modèle n’est volontairement **pas inclus** dans ce repository afin de séparer :

* le cycle *training / experimentation*
* du cycle *serving / production*
<br>

## 2. Principes MLOps appliqués

### 2.1 Séparation des responsabilités

| Composant | Responsabilité                                 |
| --------- | ---------------------------------------------- |
| `tools/`  | Conversion / préparation des artefacts modèles |
| `models/` | Stockage des artefacts prêts pour l’inférence  |
| `api/`    | Exposition du modèle, logique d’inférence      |
| `tests/`  | Validation fonctionnelle et non-régression     |
| CI/CD     | Déploiement automatisé                         |

Cette séparation permet :

* une meilleure maintenabilité
* une évolution indépendante des briques

<br>

### 2.2 Gestion des artefacts modèles

Le modèle est matérialisé par **4 artefacts obligatoires** :

* `vectorizer.joblib`
* `estimator.joblib`
* `mlb.joblib`
* `config.json`

Deux modes sont supportés :

1. **Artefacts locaux** (présents dans `models/`)
2. **Artefacts distants** téléchargés dynamiquement via URL (Blob Storage)

La vérification d’intégrité est systématique avant chargement.

<br>

### 2.3 Portabilité et reproductibilité

* Les artefacts sont sérialisés avec `joblib`
* Les paramètres d’inférence sont externalisés dans `config.json`
* Les dépendances sont explicites (`requirements.txt`)

<br>

## 3. Architecture technique

### 3.1 Vue globale

```
Client (Streamlit)
  │
  ▼
FastAPI
  │
  ├─ Validation (Pydantic)
  ├─ Vectorisation texte
  ├─ Inference modèle
  └─ Post-processing
  │
  ▼
Réponse JSON (tags)
```

<br>

### 3.2 Détail des composants clés

#### `InferenceService` (`api/inference.py`)

Responsabilités :

* Chargement des artefacts
* Abstraction du type de modèle (probabiliste ou non)
* Harmonisation des scores
* Logique métier de sélection des tags
<br>

#### `model_fetch.py`

Brique clé pour la production Cloud :

* Téléchargement **streaming** (pas de surcharge mémoire)
* Timeouts explicites (connect / read)
* Extraction atomique (dossier temporaire)
* Validation stricte des fichiers attendus

<br>

#### `convert_legacy_bundle.py` (tools)

Script de **migration de modèles legacy**.

Il permet de convertir un bundle legacy sérialisé (pickle/joblib) en artefacts unitaires compatibles avec l’API de production.

##### **Contexte :**

Lors de la phase d’expérimentation, le modèle a été sérialisé sous la forme d’un objet monolithique (`TaggerModel`) incluant :

* le vectoriseur
* le modèle de classification
* le `MultiLabelBinarizer`
* les paramètres d’inférence (`topk`, `threshold`)
<br>

##### **Artefacts produits :**

Le script génère exactement les fichiers attendus par l’API :

* `vectorizer.joblib`
* `estimator.joblib`
* `mlb.joblib`
* `config.json`

Ces fichiers sont déposés dans le dossier cible (`models/`).

<br>

## 4. API et contrat d’interface

### Endpoints exposés

| Endpoint   | Méthode | Rôle                            |
| ---------- | ------- | ------------------------------- |
| `/health`  | GET     | Vérification de l’état de l’API |
| `/predict` | POST    | Prédiction de tags              |

<br>

## 5. Lancement du projet en local

### Prérequis

* Python ≥ 3.10
* `pip` ou `pipenv`
* Les artefacts modèles présents dans `models/`
  *(ou une URL valide définie via `MODEL_BLOB_URL`)*

<br>

### Installation des dépendances

```bash
pip install -r requirements.txt
```


### Lancement de l’API

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
<br>


### Lancement de streamlit en local

A la racine du projet lancer la commande.

```bash
streamlit run streamlit_app/app.py
```


<br>

## 6. Qualité, tests et auditabilité

* Tests unitaires des endpoints
* Vérification du chargement du modèle
* Tests de bout en bout simples sur l’inférence

Objectifs :

* Détecter les bugs
* Sécuriser les déploiements CI/CD
* Fournir des preuves de bon fonctionnement

<br>

## 7. Déploiement et CI/CD

### Déploiement Cloud

* Compatible Heroku
* Démarrage idempotent
* Aucun secret hardcodé

### CI/CD

* GitHub Actions
* Déploiement automatique après validation
* Tests exécutés avant mise en production

<br>

## 8. Sécurité et conformité

* Aucune donnée personnelle stockée
* Aucun identifiant utilisateur traité
* Traitement en mémoire uniquement
* Conforme aux principes RGPD (minimisation des données)

<br>

## 9. Conclusion

Ce projet respecte :

* les bonnes pratiques MLOps
* la séparation entraînement / inférence
* la robustesse Cloud
* la traçabilité technique attendue en contexte professionnel