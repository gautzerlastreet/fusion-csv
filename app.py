import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Fusion CSV", layout="centered")

st.title("🧩 Fusionneur de fichiers CSV")

uploaded_files = st.file_uploader(
    "Dépose plusieurs fichiers CSV avec le même format", 
    type="csv", 
    accept_multiple_files=True
)

if uploaded_files:
    try:
        # Lire tous les fichiers dans des DataFrames
        dfs = [pd.read_csv(file) for file in uploaded_files]

        # Vérifier que les colonnes sont identiques
        colonnes = dfs[0].columns.tolist()
        for df in dfs:
            if list(df.columns) != colonnes:
                st.error("Tous les fichiers doivent avoir les mêmes colonnes et dans le même ordre.")
                st.stop()

        # Fusionner les fichiers
        fusion = pd.concat(dfs, ignore_index=True)

        # Afficher un aperçu
        st.success(f"{len(uploaded_files)} fichiers fusionnés avec succès !")
        st.dataframe(fusion.head())

        # Préparer le fichier à télécharger
        buffer = io.StringIO()
        fusion.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Télécharger le fichier fusionné",
            data=buffer,
            file_name="fusion.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Erreur lors du traitement : {e}")
