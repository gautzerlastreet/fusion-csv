# tools/serp_checker.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import streamlit as st

# --- USER-AGENT pour éviter le blocage ---
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

@st.cache_data(ttl=3600, show_spinner=False)
def get_google_results(query: str) -> list[dict]:
    url = f"https://www.google.com/search?q={quote_plus(query)}&hl=fr"
    resp = requests.get(url, headers=HEADERS, timeout=5)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for item in soup.select("div.g")[:10]:
        a = item.find("a")
        h3 = item.find("h3")
        if not (a and h3):
            continue
        snippet_tag = item.find("div", class_="IsZvec") or item.find("span", class_="aCOpRe")
        results.append({
            "title": h3.get_text(),
            "link":  a["href"],
            "snippet": snippet_tag.get_text() if snippet_tag else ""
        })
    return results

@st.cache_data(ttl=3600, show_spinner=False)
def get_bing_results(query: str) -> list[dict]:
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
        p = li.find("p")
        results.append({
            "title":   h2.get_text(),
            "link":    h2.find("a")["href"],
            "snippet": p.get_text() if p else ""
        })
    return results

def run():
    st.header("🔍 Comparateur SERP Google vs Bing (sans API)")
    query = st.text_input("Entrez un mot-clé", placeholder="ex. “optimisation SEO Python”")
    if st.button("Lancer la recherche") and query:
        with st.spinner("Scraping en cours…"):
            google = get_google_results(query)
            bing   = get_bing_results(query)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔎 Google (10 premiers)")
            if not google:
                st.write("Aucun résultat ou blocage du scraping.")
            for r in google:
                st.markdown(f"**[{r['title']}]({r['link']})**  \n{r['snippet']}")
        with col2:
            st.subheader("🅱️ Bing (10 premiers)")
            if not bing:
                st.write("Aucun résultat ou blocage du scraping.")
            for r in bing:
                st.markdown(f"**[{r['title']}]({r['link']})**  \n{r['snippet']}")
