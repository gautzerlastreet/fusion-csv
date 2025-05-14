import streamlit as st

# Doit être la toute première commande Streamlit
st.set_page_config(page_title="Outils SEO", layout="wide")

# Ensuite les autres imports
from tools import fusion, semantic_analyzer

# Interface principale
st.sidebar.title("🧰 Suite d'outils SEO")
tool = st.sidebar.selectbox(
    "Choisissez un outil",
    ["CSV Fusionner", "Semantic Analyzer"]
)

if tool == "CSV Fusionner":
    fusion.run()
elif tool == "Semantic Analyzer":
    semantic_analyzer.run()
