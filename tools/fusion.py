import streamlit as st
import pandas as pd

def run():
    st.title("📁 Fusionner plusieurs fichiers CSV")
    uploaded_files = st.file_uploader("Fichiers CSV (UTF-16 avec tabulation)", type="csv", accept_multiple_files=True)

    if uploaded_files:
        dfs = []
        for file in uploaded_files:
            try:
                df = pd.read_csv(file, encoding='utf-16', sep='\t')
                dfs.append(df)
            except Exception as e:
                st.warning(f"{file.name} : Erreur → {e}")

        if len(dfs) >= 2:
            fusion = pd.concat(dfs, ignore_index=True)
            output = fusion.to_csv(index=False, sep='\t', encoding='utf-8-sig')
            st.download_button("📥 Télécharger le CSV fusionné", data=output, file_name="fusion.csv", mime="text/csv")
        else:
            st.info("Ajoutez au moins deux fichiers pour fusionner.")
