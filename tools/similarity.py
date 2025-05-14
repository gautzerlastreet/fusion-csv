import streamlit as st
import pandas as pd
import re

st.set_page_config(
    layout="wide",
    page_title="Similarity Refine",
    page_icon="🥥"
)

# Fonction pour filtrer et formater les mots-clés
def parse_filter_format_keywords(list_str, threshold):
    if not isinstance(list_str, str):
        return [], 0, 0, 0
    
    keywords_list = list_str.split(" | ")
    filtered_keywords = []
    total_volume = 0
    total_similarity = 0
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

    avg_similarity = total_similarity / count if count > 0 else 0
    return filtered_keywords, total_volume, avg_similarity, count

def main_tab():
    st.title("Similarity Refine")

    uploaded_file = st.file_uploader("Choisissez un fichier")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        threshold = st.slider(
            "Entrez le seuil de similarité (%)", min_value=0, max_value=100, value=40, step=10
        )

        df[["Filtered Keywords", "Total Volume", "Avg Similarity", "Keyword Count"]] = df.apply(
            lambda x: parse_filter_format_keywords(x["Liste MC et %"], threshold), axis=1, result_type="expand"
        )

        df_sorted = df.sort_values(by="Vol. mensuel", ascending=False)
        rows_to_remove = []
        unique_secondary_keywords = set()

        # Supprimer les lignes où les mots-clés primaires se dupliquent
        for index, row in df_sorted.iterrows():
            primary_keyword_text = row["Mot-clé"].split(" (")[0]
            if primary_keyword_text in unique_secondary_keywords:
                rows_to_remove.append(index)
            else:
                for keyword in row["Filtered Keywords"]:
                    keyword_text = keyword.split(" (")[0]
                    unique_secondary_keywords.add(keyword_text)

        df_filtered = df_sorted.drop(rows_to_remove)

        # Ajouter une colonne pour les mots-clés secondaires concaténés
        df_filtered["Secondary Keywords Concatenated"] = df_filtered["Filtered Keywords"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else ""
        )

        # Renommer les colonnes existantes
        final_columns = {
            "Mot-clé": "Mot clé principal",
            "Vol. mensuel": "Volume du mots clé principal",
            "Total Volume": "Volume cumulé des mots clés secondaires",
            "Avg Similarity": "% similarité des mots clés secondaires",
            "Keyword Count": "Nombre Mots clés secondaires",
        }
        df_final = df_filtered.rename(columns=final_columns)

        # Insérer la colonne pour les mots-clés secondaires concaténés
        volume_col_index = df_final.columns.get_loc("Volume du mots clé principal")
        df_final.insert(volume_col_index + 1, "Mots clés secondaires", df_filtered["Secondary Keywords Concatenated"])

        # Supprimer les colonnes "Filtered Keywords" et "Secondary Keywords Concatenated"
        df_final = df_final.drop(columns=["Filtered Keywords", "Secondary Keywords Concatenated"])

        # Assurer que "Liste MC et %" est la dernière colonne
        df_final = df_final.reindex(columns=[col for col in df_final.columns if col != "Liste MC et %"] + ["Liste MC et %"])

        # Ajouter des métriques et des graphiques
        total_primary_keywords = len(df_final)
        total_secondary_keywords = df_final["Nombre Mots clés secondaires"].sum()
        total_primary_volume = df_final["Volume du mots clé principal"].sum()
        total_secondary_volume = df_final["Volume cumulé des mots clés secondaires"].sum()

        # Afficher les métriques et des graphiques
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Nombre Mots clés principaux", total_primary_keywords)
            st.metric("Nombre Mots clés secondaires", total_secondary_keywords)
            st.metric("Volume Recherche Mots clés principaux", total_primary_volume)
            st.metric("Volume Recherche Mots clés secondaires", total_secondary_volume)

        with col2:
            st.text("Nombre de Mots Clés")
            data = {
                "Metrics": ["Principaux", "Secondaires"],
                "Values": [total_primary_keywords, total_secondary_keywords],
            }
            st.bar_chart(pd.DataFrame(data).set_index("Metrics"))

        with col3:
            st.text("Volume de Recherche")
            data = {
                "Metrics": ["Principaux", "Secondaires"],
                "Values": [total_primary_volume, total_secondary_volume],
            }
            st.bar_chart(pd.DataFrame(data).set_index("Metrics"))

        # Afficher le DataFrame final
        st.dataframe(df_final)

        # Bouton de téléchargement pour le fichier final
        if st.button("Télécharger les données"):
            output_file_name = f"processed_data_threshold_{threshold}.xlsx"
            df_final.to_excel(output_file_name, index=False)

            with open(output_file_name, "rb") as file:
                st.download_button(
                    "Télécharger Excel",
                    data=file,
                    file_name=output_file_name,
                    mime="application/vnd.ms-excel",
                )

def about_tab():
    st.markdown("""
    This tool helps you refine and analyze keyword similarities for SEO purposes.
    
    ## How to Use
    1. Upload raw similarity file from app.occurus with keywords and their volumes.          
    2. Set the similarity threshold.
    3. Analyze and download the results.
    
    """)

def main():
    tabs = st.tabs(["Main", "About"])
    with tabs[0]:
        main_tab()
    with tabs[1]:
        about_tab()

if __name__ == "__main__":
    main()


    
        # Ajouter l'icône Twitter et le texte en bas à gauche
st.sidebar.markdown(
    """
    <div style="position: fixed; bottom: 10px; left: 20px;">
        <a href="https://github.com/Psimon8" target="_blank" style="text-decoration: none;">
            <img src="https://github.githubassets.com/assets/pinned-octocat-093da3e6fa40.svg" 
                 alt="GitHub Simon le Coz" style="width:20px; vertical-align: middle; margin-right: 5px;">
            <span style="color: white; font-size: 14px;"></span>
        </a>    
        <a href="https://www.linkedin.com/in/simon-le-coz/" target="_blank" style="text-decoration: none;">
            <img src="https://static.licdn.com/aero-v1/sc/h/8s162nmbcnfkg7a0k8nq9wwqo" 
                 alt="LinkedIn Simon Le Coz" style="width:20px; vertical-align: middle; margin-right: 5px;">
            <span style="color: white; font-size: 14px;"></span>
        </a>
        <a href="https://twitter.com/lekoz_simon" target="_blank" style="text-decoration: none;">
            <img src="https://abs.twimg.com/favicons/twitter.3.ico" 
                 alt="Twitter Simon Le Coz" style="width:20px; vertical-align: middle; margin-right: 5px;">
            <span style="color: white; font-size: 14px;">@lekoz_simon</span>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)