import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def creer_features(data):
    """Transforme les cours bruts en indicateurs que le modèle peut utiliser."""
    df = pd.DataFrame(index=data.index)

    # Rendement : de combien le cours a bougé (en %) sur 1, 5 et 10 jours
    df["rendement_1j"] = data["Close"].pct_change(1)
    df["rendement_5j"] = data["Close"].pct_change(5)
    df["rendement_10j"] = data["Close"].pct_change(10)

    # Écart entre le cours et ses moyennes mobiles
    df["ecart_mm20"] = data["Close"] / data["Close"].rolling(20).mean() - 1
    df["ecart_mm50"] = data["Close"] / data["Close"].rolling(50).mean() - 1

    # Volatilité : à quel point le cours s'agite sur 20 jours
    df["volatilite_20j"] = df["rendement_1j"].rolling(20).std()

    # Volume du jour comparé au volume moyen
    df["volume_relatif"] = data["Volume"] / data["Volume"].rolling(20).mean()

    # RSI (14 jours) : entre 0 et 100, > 70 = "suracheté", < 30 = "survendu"
    variation = data["Close"].diff()
    hausse = variation.clip(lower=0).rolling(14).mean()
    baisse = (-variation.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + hausse / baisse)

    return df


def creer_cible(data):
    """1 si le cours de clôture de DEMAIN est plus haut qu'aujourd'hui, sinon 0."""
    return (data["Close"].shift(-1) > data["Close"]).astype(int)


def entrainer_et_evaluer(data):
    features = creer_features(data)
    cible = creer_cible(data)

    # Le dernier jour n'a pas encore de "demain" : on le garde pour la prédiction
    dernier_jour = features.iloc[[-1]]

    # On retire les lignes incomplètes (début de l'historique) et le dernier jour
    tableau = features.join(cible.rename("cible")).iloc[:-1].dropna()
    X = tableau.drop(columns="cible")
    y = tableau["cible"]

    # Séparation dans le temps : on apprend sur le passé, on teste sur la fin.
    # (Jamais de mélange aléatoire, sinon le modèle "voit le futur".)
    coupure = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:coupure], X.iloc[coupure:]
    y_train, y_test = y.iloc[:coupure], y.iloc[coupure:]

    modele = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=42
    )
    modele.fit(X_train, y_train)

    precision_modele = accuracy_score(y_test, modele.predict(X_test))

    # Référence naïve : toujours prédire "ça monte"
    precision_naive = y_test.mean()

    # Prédiction pour demain
    proba_hausse = modele.predict_proba(dernier_jour)[0][1]

    importance = pd.Series(modele.feature_importances_, index=X.columns)

    return {
        "precision_modele": precision_modele,
        "precision_naive": precision_naive,
        "proba_hausse": proba_hausse,
        "nb_jours_test": len(X_test),
        "importance": importance.sort_values(ascending=False),
    }


if __name__ == "__main__":
    import yfinance as yf

    ticker = "AAPL"
    data = yf.Ticker(ticker).history(period="10y")
    print(f"{len(data)} jours récupérés pour {ticker}\n")

    r = entrainer_et_evaluer(data)

    print(f"Précision du modèle      : {r['precision_modele']:.1%}")
    print(f"Précision 'toujours monte': {r['precision_naive']:.1%}")
    print(f"(testé sur {r['nb_jours_test']} jours que le modèle n'a jamais vus)\n")
    print(f"Probabilité de hausse demain : {r['proba_hausse']:.1%}\n")
    print("Indicateurs les plus utiles au modèle :")
    print(r["importance"].round(3).to_string())
