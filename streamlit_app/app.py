import os
import requests
import streamlit as st

st.set_page_config(page_title="StackOverflow Tagger", layout="centered")

API_URL = os.getenv("TAGGER_API_URL", "http://127.0.0.1:8000")


st.title("StackOverflow Tags — Démo locale")

# --- Entrées utilisateur ---
title = st.text_input(
    "Title",
    placeholder="Ex: How to use tfidf with LinearSVC?"
)

body = st.text_area(
    "Body",
    height=220,
    placeholder="Describe your issue, context, code, errors, etc."
)

# Top-K peut rester exposé (ou être figé aussi si tu préfères)
topk = st.number_input(
    "Number of tags to return (Top-K)",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

def build_text(title: str, body: str) -> str:
    """
    Concatène Title + Body en un texte unique pour l'API.
    """
    title = (title or "").strip()
    body = (body or "").strip()

    if title and body:
        return f"{title}\n\n{body}"
    return title or body

# --- Action ---
if st.button("Predict tags", type="primary"):
    final_text = build_text(title, body)

    if not final_text:
        st.warning("Please provide at least a title or a body.")
    else:
        payload = {
            "text": final_text,
            "topk": int(topk)   # threshold NON exposé
        }

        try:
            r = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )
            r.raise_for_status()
            tags = r.json().get("tags", [])

            if tags:
                st.success("Suggested tags:")
                st.write(tags)
            else:
                st.info("No tags suggested by the model.")
        except Exception as e:
            st.error(f"API call failed: {e}")

# --- Health check ---
with st.expander("API health check"):
    try:
        r = requests.get(f"{API_URL}/health", timeout=50)
        st.json(r.json())
    except Exception as e:
        st.error(f"API not reachable: {e}")
