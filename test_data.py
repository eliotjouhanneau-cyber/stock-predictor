import yfinance as yf

# Telecharge 1 an d'historique pour Apple
data = yf.download("AAPL", period="1y")

print(data.tail())
print(f"{len(data)} jours recuperes")
