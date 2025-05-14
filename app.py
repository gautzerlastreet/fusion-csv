import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Fusion de CSV", layout="centered")

st.title("🧩 Fusionneur de fichiers CSV")
st.markdown("Dépose plusieurs fichiers CSV avec le **même format** (colonnes identiques, même ordre)")

uploaded_files = st.file_uploader(
    "Dépose tes fichiers ici",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    dfs = []
    erreurs = []
    colonnes_ref = None

    for file in uploaded_files:
        df = None
        filename = file.name

        # Tentatives de lecture avec différents encodages / séparateurs
        for enc in ['utf-8', 'ISO-8859-1']:
            for sep in [',', ';']:
                file.seek(0)
                try:
                    df = pd.read_csv(file, encoding=enc, sep=sep)
                    if df.empty or df.columns.size == 0:
                        df = None
                        continue
                    break
                except:
                    continue
            if df is not None:
                break

        if df is None:
            erreurs.append(f"❌ {filename} : fichier vide ou illisible.")
            continue

        # Vérifier colonnes identiques à la première
        if colonnes_ref is None:
            colonnes_ref = df.columns.tolist()
        elif df.columns.tolist() != colonnes_ref:
            erreurs.append(f"⚠️ {filename} : colonnes différentes de la référence.")
            continue

        dfs.append(df)
        st.success(f"✅ {filename} chargé avec succès ({len(df)} lignes)")

    # Affichage des erreurs
    for err in erreurs:
        st.warning(err)

    # Fusionner
    if len(dfs) >= 2:
        fusion = pd.concat(dfs, ignore_index=True)

        # Nettoyage
        st.markdown("### 🧼 Options de nettoyage")
        if st.checkbox("Supprimer les lignes dupliquées"):
            fusion.drop_duplicates(inplace=True)
        if st.checkbox("Supprimer les lignes entièrement vides"):
            fusion.dropna(how="all", inplace=True)

        # Aperçu
        st.success(f"🎉 {len(dfs)} fichiers fusionnés avec succès. Résultat : {len(fusion)} lignes")
        st.dataframe(fusion.head())

        # Export
        st.markdown("### 📤 Exporter le fichier fusionné")
        export_format = st.selectbox("Format de téléchargement :", ["CSV", "Excel (.xlsx)"])

        if export_format == "CSV":
            buffer = io.StringIO()
            fusion.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="📥 Télécharger en CSV",
                data=buffer.getvalue(),
                file_name="fusion.csv",
                mime="text/csv"
            )
        else:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                fusion.to_excel(writer, index=False, sheet_name="Fusion")
            buffer.seek(0)
            st.download_button(
                label="📥 Télécharger en Excel",
                data=buffer,
                file_name="fusion.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.error("❌ Au moins 2 fichiers valides sont requis pour la fusion.")
