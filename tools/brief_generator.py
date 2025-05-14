import streamlit as st
import pandas as pd

def generate_content_brief_interface():
    st.title("🧠 Générateur de brief SEO")
    keyword = st.text_input("Mot-clé cible :")
    uploaded_file = st.file_uploader("Téléversez le fichier d'expressions clés (CSV)", type="csv")
    
    if keyword and uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = df.sort_values(by="% Présence", ascending=False).head(10)

        st.subheader("🔖 Propositions de Titles")
        titles = [
            f"{keyword} : Le guide complet pour optimiser votre stratégie",
            f"Tout savoir sur {keyword} : bonnes pratiques et erreurs à éviter",
            f"Comment réussir avec {keyword} : techniques et conseils",
            f"{keyword.capitalize()} : les clés pour se démarquer en 2024",
            f"{keyword} : les expressions à intégrer pour mieux performer"
        ]
        st.write(titles)

        st.subheader("📌 H1 suggéré")
        st.write(f"{keyword.capitalize()} : ce qu’il faut savoir pour être performant")

        st.subheader("✏️ Meta description")
        st.write(f"Découvrez comment {keyword} peut transformer votre visibilité. Ce guide complet couvre les meilleures pratiques et les erreurs à éviter.")

        st.subheader("🧩 Plan H2 / H3 recommandé")
        top_expr = df['Expression'].tolist()
        st.json({
            "Introduction": [],
            f"Pourquoi {keyword} est essentiel aujourd’hui": [
                "Tendances et chiffres clés",
                "Les erreurs courantes"
            ],
            f"Les meilleures expressions à utiliser pour {keyword}": [
                f"Top 5 : {top_expr[0] if top_expr else ''}",
                f"Comment intégrer {top_expr[1] if len(top_expr) > 1 else ''} dans vos contenus",
                "Exemples concrets d’utilisation"
            ],
            f"Stratégie de contenu autour de {keyword}": [
                "Objectifs SEO et conversions",
                "Formats recommandés"
            ],
            "Conclusion et appel à l'action": []
        })
