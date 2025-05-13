
import pandas as pd
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Chargement des données
    X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()

    # Chargement du modèle entraîné
    model = joblib.load("models/final_model.pkl")

    # Prédictions
    y_pred = model.predict(X_test)

    # Calcul des scores
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Sauvegarde des métriques
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump({"mse": mse, "r2": r2}, f, indent=4)

    # Sauvegarde des prédictions
    os.makedirs("data", exist_ok=True)
    predictions_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    predictions_df.to_csv("data/predictions.csv", index=False)

    print(f"Évaluation terminée: MSE={mse:.4f}, R²={r2:.4f}")

if __name__ == "__main__":
    main()