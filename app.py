import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Fusionneur de fichiers CSV", layout="centered")

st.title("🧩 Fusionneur de fichiers CSV")
st.markdown("Dépose plusieurs fichiers CSV avec le même format (colonnes identiques)")

uploaded_files = st.file_uploader(
    "Drag and drop files here",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    dfs = []
    colonnes = None
    erreur_encodage = False

    for file in uploaded_files:
        try:
            df = pd.read_csv(file, encoding='utf-8')
            encodage_utilisé = 'utf-8'
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file, encoding='ISO-8859-1')
                encodage_utilisé = 'ISO-8859-1'
            except Exception as e:
                st.error(f"Erreur lors de l'ouverture du fichier {file.name} : {e}")
                erreur_encodage = True
                continue

        # Vérifier que les colonnes sont cohérentes
        if colonnes is None:
            colonnes = df.columns.tolist()
        elif df.columns.tolist() != colonnes:
            st.error(f"Les colonnes du fichier {file.name} ne correspondent pas aux autres fichiers.")
            st.stop()

        dfs.append(df)
        st.info(f"✅ Fichier **{file.name}** chargé avec encodage : `{encodage_utilisé}`")

    if not erreur_encodage and len(dfs) > 1:
        fusion = pd.concat(dfs, ignore_index=True)

        st.success(f"🎉 {len(dfs)} fichiers fusionnés avec succès. Aperçu :")
        st.dataframe(fusion.head())

        # Export CSV
        buffer = io.StringIO()
        fusion.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="📥 Télécharger le fichier fusionné",
            data=buffer,
            file_name="fusion.csv",
            mime="text/csv"
        )
    elif len(dfs) == 1:
        st.warning("Vous devez importer **au moins deux fichiers** pour les fusionner.")
