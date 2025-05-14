import streamlit as st
from tools import fusion, semantic_analyzer
from tools.brief_generator import generate_content_brief_interface

st.set_page_config(page_title="Outils SEO", layout="wide")

st.sidebar.title("🧰 Suite d'outils SEO")
tool = st.sidebar.selectbox("Choisissez un outil", ["CSV Fusionner", "Semantic Analyzer", "Générer un brief SEO"])

if tool == "CSV Fusionner":
    fusion.run()
elif tool == "Semantic Analyzer":
    semantic_analyzer.run()
elif tool == "Générer un brief SEO":
    generate_content_brief_interface()
