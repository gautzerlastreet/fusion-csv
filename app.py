import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Fusionneur CSV simple", layout="centered")

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

        # Test lecture encodage/séparateur
        for enc in ['utf-8', 'ISO-8859-1', 'utf-16']:
            for sep in [',', ';']:
                file.seek(0)
                try:
                    df = pd.read_csv(file, encoding=enc, sep=sep)
                    if df.columns.size > 1 or df.shape[0] > 0:
                        break
                except:
                    continue
            if df is not None and not df.empty:
                break

        if df is None or df.empty:
            erreurs.append(f"❌ {filename} : fichier vide ou illisible")
            continue

        # Ne garder que les lignes de données à partir du 2e fichier
        if i > 0:
            df = df.iloc[1:] if df.columns.equals(dfs[0].columns) else df

        dfs.append(df)
        lignes_totales += len(df)
        st.success(f"✅ {filename} chargé ({len(df)} lignes)")

    # Fusion et export
    if len(dfs) >= 2:
        fusion = pd.concat(dfs, ignore_index=True)
        st.success(f"🎉 {len(uploaded_files)} fichiers fusionnés → {len(fusion)} lignes totales")
        st.dataframe(fusion.head())

        csv_output = fusion.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger le fichier fusionné",
            data=csv_output,
            file_name="fusion.csv",
            mime="text/csv"
        )
    elif len(dfs) == 1:
        st.info("Un seul fichier valide, rien à fusionner.")
    else:
        st.error("Aucun fichier valide trouvé.")

    for err in erreurs:
        st.warning(err)
