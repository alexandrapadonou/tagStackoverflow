"""
Application Streamlit pour la démonstration interactive de l'API de prédiction de tags.

Cette application fournit une interface utilisateur simple pour:
- Saisir le titre et le corps d'une question StackOverflow
- Configurer le nombre de tags à retourner (topk)
- Visualiser les tags prédits par le modèle
- Vérifier l'état de santé de l'API

Utilisation:
    streamlit run streamlit_app/app.py

Configuration:
    - TAGGER_API_URL: URL de l'API (défaut: http://127.0.0.1:8000)
"""
import os
import requests
import streamlit as st

# Configuration de la page Streamlit
st.set_page_config(page_title="StackOverflow Tagger", layout="centered")

# URL de l'API configurable via variable d'environnement
# Permet de pointer vers l'API locale ou une API déployée (Heroku, Azure, etc.)
API_URL = os.getenv("TAGGER_API_URL", "http://127.0.0.1:8000")


st.title("StackOverflow Tags — Démo locale")

# --- Section: Entrées utilisateur ---
# Permet à l'utilisateur de saisir le titre et le corps de sa question
title = st.text_input(
    "Title",
    placeholder="Ex: How to use tfidf with LinearSVC?",
    help="Saisissez le titre de votre question StackOverflow"
)

body = st.text_area(
    "Body",
    height=220,
    placeholder="Describe your issue, context, code, errors, etc.",
    help="Saisissez le corps détaillé de votre question"
)

# Paramètre topk: nombre de tags à retourner
# Exposé à l'utilisateur pour permettre la personnalisation
topk = st.number_input(
    "Number of tags to return (Top-K)",
    min_value=1,
    max_value=50,
    value=5,
    step=1,
    help="Nombre maximum de tags à retourner (entre 1 et 50)"
)


def build_text(title: str, body: str) -> str:
    """
    Concatène le titre et le corps en un texte unique pour l'API.
    
    Cette fonction prépare le texte à envoyer à l'API en combinant
    le titre et le corps de la question. Si l'un des deux est vide,
    elle retourne l'autre. Si les deux sont présents, elle les sépare
    par deux retours à la ligne pour une meilleure lisibilité.
    
    Args:
        title: Titre de la question (peut être vide)
        body: Corps de la question (peut être vide)
        
    Returns:
        str: Texte concaténé prêt à être envoyé à l'API
             Format: "Titre\n\nCorps" si les deux sont présents
                     Sinon: titre ou corps (celui qui est présent)
    """
    title = (title or "").strip()
    body = (body or "").strip()

    if title and body:
        return f"{title}\n\n{body}"
    return title or body

# --- Section: Action de prédiction ---
# Bouton déclenchant l'appel à l'API pour obtenir les tags prédits
if st.button("Predict tags", type="primary"):
    # Construction du texte final à partir du titre et du corps
    final_text = build_text(title, body)

    # Validation: au moins le titre ou le corps doit être fourni
    if not final_text:
        st.warning("Please provide at least a title or a body.")
    else:
        # Préparation de la requête pour l'API
        payload = {
            "text": final_text,
            "topk": int(topk)   # threshold non exposé à l'utilisateur (utilise la valeur par défaut)
        }

        try:
            # Appel à l'endpoint /predict de l'API
            r = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30  # Timeout de 30 secondes pour éviter les blocages
            )
            r.raise_for_status()  # Lève une exception si le statut HTTP n'est pas 200
            
            # Extraction des tags depuis la réponse JSON
            tags = r.json().get("tags", [])

            # Affichage des résultats
            if tags:
                st.success("Suggested tags:")
                st.write(tags)  # Affiche la liste des tags prédits
            else:
                st.info("No tags suggested by the model.")
        except requests.exceptions.RequestException as e:
            # Gestion des erreurs réseau (timeout, connexion refusée, etc.)
            st.error(f"API call failed: {e}")
        except Exception as e:
            # Gestion des autres erreurs inattendues
            st.error(f"Unexpected error: {e}")

# --- Section: Health check de l'API ---
# Permet de vérifier que l'API est accessible et opérationnelle
with st.expander("API health check"):
    """
    Cette section permet de vérifier l'état de santé de l'API.
    Utile pour le debugging et la vérification de la connectivité.
    """
    try:
        # Appel à l'endpoint /health
        r = requests.get(f"{API_URL}/health", timeout=50)
        r.raise_for_status()
        # Affichage de la réponse JSON (statut et répertoire des modèles)
        st.json(r.json())
    except requests.exceptions.RequestException as e:
        # Erreur si l'API n'est pas accessible
        st.error(f"API not reachable: {e}")
    except Exception as e:
        # Autres erreurs
        st.error(f"Error checking API health: {e}")
