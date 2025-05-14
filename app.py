import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Fusionneur de fichiers CSV", layout="centered")

st.title("🧩 Fusionneur de fichiers CSV")
st.markdown("Dépose plusieurs fichiers CSV avec le même format (colonnes identiques, même ordre)")

uploaded_files = st.file_uploader(
    "Dépose tes fichiers ici",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    dfs = []
    colonnes = None
    erreurs = []

    for file in uploaded_files:
        loaded = False
        filename = file.name
        df = None

        # Essai 1 : utf-8 + ,
        try:
            df = pd.read_csv(file, encoding='utf-8', sep=',')
            loaded = True
            encodage_utilisé = "utf-8"
            separateur = ","
        except Exception:
            pass

        # Essai 2 : ISO-8859-1 + ,
        if not loaded:
            file.seek(0)
            try:
                df = pd.read_csv(file, encoding='ISO-8859-1', sep=',')
                loaded = True
                encodage_utilisé = "ISO-8859-1"
                separateur = ","
            except Exception:
                pass

        # Essai 3 : ISO-8859-1 + ;
        if not loaded:
            file.seek(0)
            try:
                df = pd.read_csv(file, encoding='ISO-8859-1', sep=';')
                loaded = True
                encodage_utilisé = "ISO-8859-1"
                separateur = ";"
            except Exception:
                pass

        # ✅ Essai 4 : utf-16 + ;
        if not loaded:
            file.seek(0)
            try:
                df = pd.read_csv(file, encoding='utf-16', sep=';')
                loaded = True
                encodage_utilisé = "utf-16"
                separateur = ";"
            except Exception:
                pass

        if not loaded or df.empty or df.columns.size == 0:
            erreurs.append(f"❌ {filename} : Fichier vide ou illisible")
            continue

        if colonnes is None:
            colonnes = df.columns.tolist()
        elif df.columns.tolist() != colonnes:
            erreurs.append(f"⚠️ {filename} : Colonnes différentes")
            continue

        dfs.append(df)
        st.success(f"✅ {filename} chargé ({encodage_utilisé}, séparateur `{separateur}`) → {len(df)} lignes")

    # Affichage des erreurs
    for err in erreurs:
        st.warning(err)

    if len(dfs) >= 2:
        fusion = pd.concat(dfs, ignore_index=True)
        st.success(f"🎉 {len(dfs)} fichiers fusionnés avec succès ! Aperçu ci-dessous :")
        st.dataframe(fusion.head())

        # Export CSV
        csv_string = fusion.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger le fichier fusionné",
            data=csv_string,
            file_name="fusion.csv",
            mime="text/csv"
        )

    elif len(dfs) == 1:
        st.info("Un seul fichier valide. Rien à fusionner.")
    else:
        st.error("Aucun fichier valide n’a pu être traité.")
