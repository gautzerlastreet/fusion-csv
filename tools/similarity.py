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
    df[[
        "Filtered Keywords", "Total Volume", "Avg Similarity", "Keyword Count"
    ]] = df.apply(
        lambda x: parse_filter_format_keywords(x["Liste MC et %"], threshold),
        axis=1,
        result_type="expand"
    )

    # Trier par volume mensuel primaire
    df_sorted = df.sort_values(by="Vol. mensuel", ascending=False)

    # Supprimer doublons de mots clés primaire
    unique_secondary = set()
    rows_to_keep = []
    for _, row in df_sorted.iterrows():
        primary = row["Mot-clé"].split(" (")[0]
        if primary in unique_secondary:
            continue
        unique_secondary.add(primary)
        rows_to_keep.append(row.name)
    df_filtered = df_sorted.loc[rows_to_keep]

    # Concaténation mots clés secondaires
    df_filtered["Mots clés secondaires"] = df_filtered["Filtered Keywords"].apply(
        lambda lst: " | ".join(lst) if isinstance(lst, list) else ""
    )

    # Renommer colonnes
    df_final = df_filtered.rename(columns={
        "Mot-clé": "Mot clé principal",
        "Vol. mensuel": "Volume mot clé principal",
        "Total Volume": "Volume cumulé secondaires",
        "Avg Similarity": "% similarité secondaires",
        "Keyword Count": "Count secondaires"
    })

        # Réorganiser colonnes
    base_cols = [
        "Mot clé principal", "Volume mot clé principal", "Mots clés secondaires",
        "Volume cumulé secondaires", "% similarité secondaires", "Count secondaires"
    ]
    other_cols = [c for c in df_final.columns if c not in base_cols]
    cols = base_cols + other_cols
    df_final = df_final[cols]

        # Métriques globales et graphiques côte-à-côte
    total_primary = df_final.shape[0]
    total_secondary = df_final["Count secondaires"].sum()
    total_vol_primary = df_final["Volume mot clé principal"].sum()
    total_vol_secondary = df_final["Volume cumulé secondaires"].sum()

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.metric("Mots clés primaires", total_primary)
        st.metric("Mots clés secondaires", total_secondary)
        st.metric("Volume primaire", total_vol_primary)
        st.metric("Volume secondaire", total_vol_secondary)
    with col2:
        st.markdown("**Nombre de Mots Clés**")
        chart_df1 = pd.DataFrame({
            "Type": ["Primaires", "Secondaires"],
            "Nombre": [total_primary, total_secondary]
        }).set_index("Type")
        st.bar_chart(chart_df1)
    with col3:
        st.markdown("**Volume de Recherche**")
        chart_df2 = pd.DataFrame({
            "Type": ["Primaires", "Secondaires"],
            "Volume": [total_vol_primary, total_vol_secondary]
        }).set_index("Type")
        st.bar_chart(chart_df2)

    # Affichage du DataFrame
    chart_df1 = pd.DataFrame({
        "Type": ["Primaires", "Secondaires"],
        "Nombre": [total_primary, total_secondary]
    }).set_index("Type")
    st.bar_chart(chart_df1)
    chart_df2 = pd.DataFrame({
        "Type": ["Primaires", "Secondaires"],
        "Volume": [total_vol_primary, total_vol_secondary]
    }).set_index("Type")
    st.bar_chart(chart_df2)

    # Affichage du DataFrame
    st.dataframe(df_final, use_container_width=True)

    # Téléchargement
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
        par rapport aux mots clés primaires, avec visualisation et export.
        """
    )


def run():
    tabs = st.tabs(["Main", "About"])
    with tabs[0]:
        main_tab()
    with tabs[1]:
        about_tab()
