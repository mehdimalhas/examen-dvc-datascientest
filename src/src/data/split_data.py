
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    # Charger les données brutes
    raw_data_path = "data/raw/raw.csv"
    df = pd.read_csv(raw_data_path)

    # Supposer que la dernière colonne est la cible : silica_concentrate
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer le dossier processed s’il n’existe pas
    os.makedirs("data/processed", exist_ok=True)

    # Sauvegarder les jeux de données
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("Données divisées et sauvegardées dans data/processed.")

if __name__ == "__main__":
    main()
