# tools/serp_checker.py

import json
import requests
from bs4 import BeautifulSoup
import streamlit as st

# --- USER-AGENT et en-têtes pour éviter le blocage et forcer le français ---
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7"
}

@st.cache_data(ttl=3600, show_spinner=False)
def get_bing_results(query: str) -> list[dict]:
    """Scrape les 10 premiers résultats Bing pour la requête."""
    url = "https://www.bing.com/search"
    params = {"q": query, "count": 10}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=5)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for li in soup.select("li.b_algo")[:10]:
        h2 = li.find("h2")
        if not h2 or not h2.find("a"):
            continue
        link = h2.find("a")["href"]
        title = h2.get_text()
        snippet = li.find("p").get_text() if li.find("p") else ""
        results.append({
            "title":   title,
            "link":    link,
            "snippet": snippet
        })
    return results

def run():
    """Point d’entrée Streamlit pour le SERP Checker (Bing uniquement)."""
    st.header("🅱️ Comparateur SERP Bing (sans API ni Google)")
    query = st.text_input("Entrez un mot-clé", placeholder="ex. “optimisation SEO Python”")
    if st.button("Lancer la recherche") and query:
        with st.spinner("Scraping Bing en cours…"):
            bing_results = get_bing_results(query)

        # Affichage des résultats
        if not bing_results:
            st.warning("Aucun résultat trouvé ou blocage du scraping Bing.")
            return

        st.subheader("🅱️ Bing (10 premiers)")
        for r in bing_results:
            st.markdown(f"**[{r['title']}]({r['link']})**  \n{r['snippet']}")

        # Préparation de la liste d'URLs
        urls = [r["link"] for r in bing_results]
        urls_text = "\n".join(urls)

        # Bouton de téléchargement
        st.download_button(
            label="📥 Télécharger les URLs",
            data=urls_text,
            file_name="bing_urls.txt",
            mime="text/plain"
        )

        # Bouton de copie dans le presse-papier
        # On utilise un petit composant HTML/JS côté client
        urls_json = json.dumps(urls_text)
        copy_button = f"""
            <button
                style="padding:8px 12px; font-size:16px; margin-top:8px;"
                onclick="navigator.clipboard.writeText({urls_json})">
              📋 Copier les URLs
            </button>
        """
        st.markdown(copy_button, unsafe_allow_html=True)
