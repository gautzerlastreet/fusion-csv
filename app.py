# app.py

import streamlit as st

# ⚠️ set_page_config DOIT être la toute première commande Streamlit
st.set_page_config(page_title="Outils SEO", layout="wide")

from tools import fusion, semantic_analyzer
from tools.brief_generator import generate_content_brief_interface

st.sidebar.title("🧰 Suite d'outils SEO")
tool = st.sidebar.selectbox(
    "Choisissez un outil",
    ["CSV Fusionner", "Semantic Analyzer"]
)

if tool == "CSV Fusionner":
    fusion.run()
elif tool == "Semantic Analyzer":
    semantic_analyzer.run()
