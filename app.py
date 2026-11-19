import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from model import entrainer_et_evaluer

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Predictor")

# --- Choix de l'utilisateur ---
col1, col2 = st.columns(2)
ticker = col1.text_input("Symbole de l'action", "AAPL").upper().strip()
periode = col2.selectbox("Période affichée", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)


# --- Récupération des données (mise en cache pour aller plus vite) ---
@st.cache_data(ttl=3600)
def charger_donnees(ticker):
    # On charge 10 ans : le modèle a besoin de beaucoup d'historique
    return yf.Ticker(ticker).history(period="10y")


@st.cache_data(ttl=3600)
def lancer_modele(ticker):
    return entrainer_et_evaluer(charger_donnees(ticker))


historique = charger_donnees(ticker)

if historique.empty:
    st.error(f"Aucune donnée trouvée pour « {ticker} ». Vérifie le symbole.")
    st.stop()

# Pour le graphique, on ne garde que la période choisie
jours = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260}
data = historique.tail(jours[periode]).copy()

# --- Chiffres clés ---
dernier = historique["Close"].iloc[-1]
veille = historique["Close"].iloc[-2]
variation = (dernier - veille) / veille * 100

c1, c2, c3 = st.columns(3)
c1.metric("Dernier cours", f"{dernier:.2f}", f"{variation:+.2f} %")
c2.metric("Plus haut (période)", f"{data['High'].max():.2f}")
c3.metric("Plus bas (période)", f"{data['Low'].min():.2f}")

# --- Graphique en chandeliers + moyennes mobiles ---
historique["MM20"] = historique["Close"].rolling(20).mean()
historique["MM50"] = historique["Close"].rolling(50).mean()
data = historique.tail(jours[periode])

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index, open=data["Open"], high=data["High"],
    low=data["Low"], close=data["Close"], name="Cours"))
fig.add_trace(go.Scatter(x=data.index, y=data["MM20"], name="Moyenne 20 j"))
fig.add_trace(go.Scatter(x=data.index, y=data["MM50"], name="Moyenne 50 j"))
fig.update_layout(height=500, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, width="stretch")

# --- Prédiction ---
st.header("🔮 Prédiction pour la prochaine séance")

if len(historique) < 300:
    st.warning("Pas assez d'historique pour entraîner le modèle sur cette action.")
    st.stop()

with st.spinner("Entraînement du modèle..."):
    r = lancer_modele(ticker)

proba = r["proba_hausse"]
sens = "📈 Hausse" if proba >= 0.5 else "📉 Baisse"
confiance = max(proba, 1 - proba)

p1, p2 = st.columns(2)
p1.metric("Le modèle penche pour", sens)
p2.metric("Probabilité estimée", f"{confiance:.1%}")

# --- Fiabilité du modèle ---
st.subheader("🧪 Le modèle est-il fiable ?")

f1, f2, f3 = st.columns(3)
f1.metric("Précision du modèle", f"{r['precision_modele']:.1%}")
f2.metric("Stratégie « toujours monte »", f"{r['precision_naive']:.1%}")
ecart = (r["precision_modele"] - r["precision_naive"]) * 100
f3.metric("Avantage du modèle", f"{ecart:+.1f} pts")

st.caption(
    f"Testé sur les {r['nb_jours_test']} séances les plus récentes, "
    "que le modèle n'avait jamais vues pendant son entraînement."
)

# Verdict en langage clair. Avec ~500 jours de test, un écart de moins
# de 3 points peut facilement être dû au hasard.
if ecart >= 3:
    st.success("Le modèle fait nettement mieux que la stratégie naïve sur la période testée. "
               "Ça reste à confirmer sur d'autres périodes.")
elif ecart > 0:
    st.info("Le modèle fait à peine mieux que la stratégie naïve : "
            "l'écart est trop faible pour exclure le hasard.")
else:
    st.warning("Le modèle ne fait pas mieux que la stratégie naïve : "
               "sa prédiction n'a pas de valeur sur cette action.")

with st.expander("Quels indicateurs le modèle utilise-t-il le plus ?"):
    st.bar_chart(r["importance"])

st.divider()
st.caption("⚠️ Projet éducatif. Ceci n'est pas un conseil en investissement.")
