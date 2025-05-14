# app.py

import streamlit as st

# ⚠️ set_page_config DOIT être la toute première commande Streamlit
st.set_page_config(page_title="Outils SEO", layout="wide")

from tools import fusion, semantic_analyzer, similarity, serp_checker

st.sidebar.title("🧰 Suite d'outils SEO")
tool = st.sidebar.selectbox(
    "Choisissez un outil",
    [
        "CSV Fusionner",
        "Semantic Analyzer",
        "Similarity",
        "SERP Checker (Google vs Bing)"
    ]
)

if tool == "CSV Fusionner":
    fusion.run()
elif tool == "Semantic Analyzer":
    semantic_analyzer.run()
elif tool == "Similarity":
    similarity.run()
elif tool == "SERP Checker (Google vs Bing)":
    serp_checker.run()
