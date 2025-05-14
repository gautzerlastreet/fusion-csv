import streamlit as st
import pandas as pd
import numpy as np

def run():
    st.title("🔗 Similarity Checker")
    st.markdown("Comparez deux textes ou CSV pour détecter la similarité de contenus.")

    mode = st.selectbox("Mode", ["Texte", "CSV"])  
    if mode == "Texte":
        txt1 = st.text_area("Texte 1", height=150)
        txt2 = st.text_area("Texte 2", height=150)
        if st.button("Comparer"):
            if not txt1 or not txt2:
                st.error("Veuillez fournir deux textes.")
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer
                vec = TfidfVectorizer().fit_transform([txt1, txt2])
                sim = (vec * vec.T).A[0,1]
                st.metric("Similarité (cosine)", f"{sim:.2f}")
    else:
        csv1 = st.file_uploader("CSV 1", type="csv", key="csv1")
        csv2 = st.file_uploader("CSV 2", type="csv", key="csv2")
        col = st.text_input("Colonne à comparer")
        if st.button("Comparer CSVs"):
            if not csv1 or not csv2 or not col:
                st.error("Veuillez charger deux CSV et indiquer la colonne.")
            else:
                df1 = pd.read_csv(csv1)
                df2 = pd.read_csv(csv2)
                if col not in df1.columns or col not in df2.columns:
                    st.error("Colonne non trouvée dans l'un des CSV.")
                else:
                    texts1 = df1[col].astype(str).tolist()
                    texts2 = df2[col].astype(str).tolist()
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    vec = TfidfVectorizer().fit_transform(texts1 + texts2)
                    mat = (vec * vec.T).A
                    n1, n2 = len(texts1), len(texts2)
                    sim_matrix = mat[:n1, n1:]
                    df_sim = pd.DataFrame(sim_matrix, index=df1.index, columns=df2.index)
                    st.write(df_sim)
                    # Export
                    csv_out = df_sim.to_csv(index=True)
                    st.download_button("Télécharger résultats", csv_out, file_name="similarity.csv")
