
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def main():
    input_dir = "data/processed"
    output_dir = "data/processed"

    # Chargement des jeux de données
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))

    # Sélection des colonnes numériques uniquement
    num_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    X_train_num = X_train[num_cols]
    X_test_num = X_test[num_cols]

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    # Reconstruire les DataFrames avec colonnes d’origine
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=num_cols, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=num_cols, index=X_test.index)

    # (Optionnel) concaténer avec les colonnes non numériques si besoin
    # Ici, on garde uniquement les colonnes numériques normalisées :
    X_train_scaled_df.to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled_df.to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False)

    print("Données numériques normalisées et sauvegardées.")

if __name__ == "__main__":
    main()
