import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Predictor")

# --- Choix de l'utilisateur ---
col1, col2 = st.columns(2)
ticker = col1.text_input("Symbole de l'action", "AAPL").upper()
periode = col2.selectbox("Période", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

# --- Récupération des données (mise en cache pour aller plus vite) ---
@st.cache_data
def charger_donnees(ticker, periode):
    return yf.Ticker(ticker).history(period=periode)

data = charger_donnees(ticker, periode)

if data.empty:
    st.error(f"Aucune donnée trouvée pour « {ticker} ». Vérifie le symbole.")
    st.stop()

# --- Chiffres clés ---
dernier = data["Close"].iloc[-1]
veille = data["Close"].iloc[-2]
variation = (dernier - veille) / veille * 100

c1, c2, c3 = st.columns(3)
c1.metric("Dernier cours", f"{dernier:.2f} $", f"{variation:+.2f} %")
c2.metric("Plus haut (période)", f"{data['High'].max():.2f} $")
c3.metric("Plus bas (période)", f"{data['Low'].min():.2f} $")

# --- Moyennes mobiles ---
data["MM20"] = data["Close"].rolling(20).mean()
data["MM50"] = data["Close"].rolling(50).mean()

# --- Graphique en chandeliers ---
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index, open=data["Open"], high=data["High"],
    low=data["Low"], close=data["Close"], name="Cours"))
fig.add_trace(go.Scatter(x=data.index, y=data["MM20"], name="Moyenne 20 j"))
fig.add_trace(go.Scatter(x=data.index, y=data["MM50"], name="Moyenne 50 j"))
fig.update_layout(height=550, xaxis_rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

st.caption("⚠️ Projet éducatif. Ceci n'est pas un conseil en investissement.")
