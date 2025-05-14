import streamlit as st

# ⚠️ set_page_config DOIT être la toute première commande Streamlit
st.set_page_config(page_title="Outils SEO et RPG", layout="wide")

from tools import fusion, semantic_analyzer, similarity, serp_checker, character_generator

st.sidebar.title("🧰 Suite d'outils SEO et RPG")
tool = st.sidebar.selectbox(
    "Choisissez un outil",
    [
        "CSV Fusionner",
        "Semantic Analyzer",
        "Similarity",
        "SERP Checker (Google vs Bing)",
        "Générateur de personnage RPG"
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
elif tool == "Générateur de personnage RPG":
    character_generator.run()
