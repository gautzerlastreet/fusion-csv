import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Fusionneur de fichiers CSV", layout="centered")

st.title("🧩 Fusionneur de fichiers CSV")
st.markdown("Dépose plusieurs fichiers CSV avec la **même ligne d’en-tête**")

uploaded_files = st.file_uploader(
    "Dépose les fichiers ici",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    dfs = []
    lignes_totales = 0
    erreurs = []

    for i, file in enumerate(uploaded_files):
        filename = file.name
        df = None

        try:
            df = pd.read_csv(file, encoding='utf-16', sep='\t')
        except Exception:
            erreurs.append(f"❌ {filename} : fichier illisible (UTF-16 + tabulation attendus)")
            continue

        if df.empty or df.columns.size <= 1:
            erreurs.append(f"⚠️ {filename} : lu mais vide ou mal structuré")
            continue

        dfs.append(df)
        lignes_totales += len(df)
        st.success(f"✅ {filename} chargé avec succès → {len(df)} lignes")

    if len(dfs) >= 2:
        fusion = pd.concat(dfs, ignore_index=True)
        st.success(f"🎉 {len(dfs)} fichiers fusionnés → {len(fusion)} lignes totales")
        st.dataframe(fusion.head())

        csv_output = fusion.to_csv(index=False, sep="\t")
        st.download_button(
            label="📥 Télécharger le fichier fusionné",
            data=csv_output,
            file_name="fusion.csv",
            mime="text/csv"
        )
    elif len(dfs) == 1:
        st.info("Un seul fichier valide, rien à fusionner.")
    else:
        st.error("Aucun fichier exploitable.")

    for err in erreurs:
        st.warning(err)
