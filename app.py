import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Fusionneur de fichiers CSV", layout="centered")

st.title("🧩 Fusionneur de fichiers CSV")
st.markdown("Dépose plusieurs fichiers CSV avec le même format (mêmes colonnes OU une colonne commune pour fusion)")

uploaded_files = st.file_uploader(
    "Drag and drop files here",
    type="csv",
    accept_multiple_files=True
)

fusion_mode = st.radio("Mode de fusion :", ["Fusion verticale (concat)", "Fusion horizontale (merge sur colonne)"])

if uploaded_files:
    dfs = []
    colonnes = None
    erreurs = []

    for file in uploaded_files:
        loaded = False
        filename = file.name

        # Tentatives de lecture avec encodage + séparateurs différents
        try:
            df = pd.read_csv(file, encoding='utf-8')
            encodage = "utf-8"
            sep = ","
            loaded = True
        except:
            file.seek(0)
            try:
                df = pd.read_csv(file, encoding='ISO-8859-1')
                encodage = "ISO-8859-1"
                sep = ","
                loaded = True
            except:
                file.seek(0)
                try:
                    df = pd.read_csv(file, encoding='ISO-8859-1', sep=';')
                    encodage = "ISO-8859-1"
                    sep = ";"
                    loaded = True
                except:
                    pass

        if not loaded or df.empty or df.columns.size == 0:
            erreurs.append(f"❌ {filename} : vide ou illisible.")
            continue

        if fusion_mode == "Fusion verticale (concat)":
            if colonnes is None:
                colonnes = df.columns.tolist()
            elif df.columns.tolist() != colonnes:
                erreurs.append(f"⚠️ {filename} : colonnes différentes")
                continue

        dfs.append(df)
        st.success(f"✅ {filename} chargé ({encodage}, séparateur `{sep}`)")

    # Affichage erreurs
    for err in erreurs:
        st.warning(err)

    if len(dfs) >= 2:
        if fusion_mode == "Fusion verticale (concat)":
            fusion = pd.concat(dfs, ignore_index=True)
        else:
            st.info("💡 Pour fusionner horizontalement, indique le nom de la colonne clé (ex: `URL`, `keyword`, etc.)")
            key = st.text_input("Nom de la colonne clé pour le merge :", value=dfs[0].columns[0])

            try:
                fusion = dfs[0]
                for df in dfs[1:]:
                    fusion = pd.merge(fusion, df, on=key, how='outer')
            except Exception as e:
                st.error(f"Erreur lors du merge horizontal : {e}")
                st.stop()

        # Options de nettoyage
        if st.checkbox("🧼 Supprimer les lignes dupliquées"):
            fusion.drop_duplicates(inplace=True)
        if st.checkbox("🧹 Supprimer les lignes vides (entièrement vides)"):
            fusion.dropna(how="all", inplace=True)

        st.success("✅ Aperçu des données fusionnées :")
        st.dataframe(fusion.head())

        # Choix de l’export
        export_format = st.selectbox("Format d’export :", ["CSV", "Excel (.xlsx)"])

        if export_format == "CSV":
            buffer = io.StringIO()
            fusion.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="📥 Télécharger en CSV",
                data=buffer,
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
        st.info("Veuillez importer au moins 2 fichiers valides.")
