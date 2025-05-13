

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def main():
    # Chargement des données
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()  # .ravel() pour éviter les erreurs de dimension

    # Modèle et grille d'hyperparamètres
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring="neg_mean_squared_error", verbose=1)
    grid.fit(X_train, y_train)

    # Sauvegarde du meilleur modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(grid.best_estimator_, "models/best_model.pkl")

    print(f"Meilleur modèle sauvegardé avec score {grid.best_score_}")
    print(f"Meilleurs paramètres : {grid.best_params_}")

if __name__ == "__main__":
    main()