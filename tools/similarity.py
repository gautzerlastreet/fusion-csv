import streamlit as st
import pandas as pd
import re

# Fonction pour filtrer et formater les mots-clés

def parse_filter_format_keywords(list_str: str, threshold: float):
    if not isinstance(list_str, str):
        return [], 0, 0, 0
    keywords_list = list_str.split(" | ")
    filtered_keywords = []
    total_volume = 0
    total_similarity = 0.0
    count = 0
    for keyword_str in keywords_list:
        match = re.match(r"(.+) \((\d+)\): (\d+\.\d+) %", keyword_str)
        if match:
            keyword, volume, similarity = match.groups()
            volume = int(volume)
            similarity = float(similarity)
            if similarity >= threshold:
                filtered_keywords.append(f"{keyword} ({volume}): {similarity:.2f} %")
                total_volume += volume
                total_similarity += similarity
                count += 1
    avg_similarity = (total_similarity / count) if count > 0 else 0.0
    return filtered_keywords, total_volume, avg_similarity, count


def main_tab():
    st.title("Similarity Refine")
    uploaded_file = st.file_uploader("Choisissez un fichier", type=["xlsx", "xls"])
    if uploaded_file is None:
        return

    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    except ImportError:
        st.error("Le package 'openpyxl' n'est pas installé. Ajoutez 'openpyxl' à votre requirements.txt et relancez l'app.")
        return
    except Exception as e:
        st.error(f"Erreur lecture du fichier Excel: {e}")
        return

    threshold = st.slider(
        "Seuil de similarité (%)", min_value=0, max_value=100, value=40, step=10
    )

    # Appliquer le filtrage
    df[["Filtered Keywords", "Total Volume", "Avg Similarity", "Keyword Count"]] = df.apply(
        lambda x: parse_filter_format_keywords(x["Liste MC et %"], threshold),
        axis=1,
        result_type="expand"
    )

    # Trier et supprimer doublons de mots clés primaire
    df_sorted = df.sort_values(by="Vol. mensuel", ascending=False)
    seen = set()
    rows = []
    for idx, row in df_sorted.iterrows():
        primary = row["Mot-clé"].split(" (")[0]
        if primary in seen:
            continue
        seen.add(primary)
        rows.append(idx)
    df_filtered = df_sorted.loc[rows]

    # Concaténation mots clés secondaires
    df_filtered["Mots clés secondaires"] = df_filtered["Filtered Keywords"].apply(
        lambda lst: " | ".join(lst) if isinstance(lst, list) else ""
    )

    # Renommer et garder uniquement les colonnes demandées
    df_final = df_filtered.rename(columns={
        "Mot-clé": "Mot clé principal",
        "Vol. mensuel": "Volume mot clé principal",
        "Total Volume": "Volume cumulé secondaires",
        "Keyword Count": "Count secondaires"
    })[[
        "Mot clé principal",
        "Volume mot clé principal",
        "Mots clés secondaires",
        "Volume cumulé secondaires",
        "Count secondaires"
    ]]

    # Affichage et export
    st.dataframe(df_final, use_container_width=True)
    output_name = f"similarity_refine_{threshold}.xlsx"
    df_final.to_excel(output_name, index=False)
    with open(output_name, "rb") as f:
        st.download_button(
            "Télécharger le rapport",
            data=f,
            file_name=output_name,
            mime="application/vnd.ms-excel"
        )


def about_tab():
    st.markdown(
        """
        ### À propos
        Cet outil permet de filtrer et d'analyser la similarité des mots clés secondaires
        par rapport aux mots clés primaires, avec export simplifié.
        """
    )


def run():
    tabs = st.tabs(["Main", "About"])
    with tabs[0]:
        main_tab()
    with tabs[1]:
        about_tab()