import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Fusion CSV", layout="centered")

st.title("📁 Fusion simplifiée de CSV")
st.markdown("Dépose plusieurs fichiers CSV encodés en **UTF-16 avec tabulation**.")

uploaded_files = st.file_uploader(
    "Fichiers CSV",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file, encoding='utf-16', sep='\t')
            dfs.append(df)
        except Exception as e:
            st.warning(f"{file.name} : Erreur de lecture → {e}")

    if len(dfs) >= 2:
        fusion = pd.concat(dfs, ignore_index=True)
        csv_output = fusion.to_csv(index=False, sep='\t', encoding='utf-8-sig')

        st.download_button(
            label="📥 Télécharger le fichier fusionné",
            data=csv_output,
            file_name="fusion.csv",
            mime="text/csv"
        )
    elif len(dfs) == 1:
        st.info("Un seul fichier valide, rien à fusionner.")
    else:
        st.error("Aucun fichier valide n’a pu être traité.")
