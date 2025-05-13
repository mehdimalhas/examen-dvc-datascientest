

import pandas as pd
import joblib
import os

def main():
    # Chargement des données
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    # Chargement du meilleur modèle (non encore entraîné)
    model = joblib.load("models/best_model.pkl")

    # Entraînement du modèle avec les meilleurs hyperparamètres
    model.fit(X_train, y_train)

    # Sauvegarde du modèle entraîné
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/final_model.pkl")
    print("Modèle final entraîné et sauvegardé.")

if __name__ == "__main__":
    main()