import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"):
    print("Pobieranie zbioru UCI Appliances Energy Prediction...")
    df = pd.read_csv(url)

    print("Rozpoczynam Feature Engineering...")
    df['date'] = pd.to_datetime(df['date'])

    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    columns_to_drop = ['date', 'hour', 'day_of_week', 'rv1', 'rv2']
    df = df.drop(columns=columns_to_drop)

    target_col = 'Appliances'
    features_cols = [col for col in df.columns if col != target_col]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df[features_cols] = feature_scaler.fit_transform(df[features_cols])
    df[[target_col]] = target_scaler.fit_transform(df[[target_col]])

    return df, feature_scaler, target_scaler


def create_lstm_sequences(df, target_col='Appliances', sequence_length=36, train_split=0.8):
    """
    Przekształca DataFrame w trójwymiarowe macierze zrozumiałe dla PyTorch.
    sequence_length=36 oznacza 6 godzin historii (przy próbkowaniu co 10 minut).
    """
    print(f"\nGenerowanie sekwencji o długości {sequence_length} kroków...")

    # Do wejścia (X) bierzemy CAŁĄ tabelę - model musi widzieć, ile prądu zużywał dom chwilę wcześniej
    data_X = df.values

    # Do wyjścia (y) wyciągamy indeks kolumny docelowej
    target_idx = df.columns.get_loc(target_col)
    data_y = df.values[:, target_idx]

    X, y = [], []

    # Przesuwamy okno po jednym kroku
    for i in range(len(data_X) - sequence_length):
        X.append(data_X[i:(i + sequence_length)])
        y.append(data_y[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Chronologiczny podział na Trening i Test (zostawiamy przyszłość do weryfikacji)
    split_idx = int(len(X) * train_split)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Kształt X_train: {X_train.shape} (Próbki, Długość sekwencji, Cechy)")
    print(f"Kształt y_train: {y_train.shape}")
    print(f"Kształt X_test:  {X_test.shape}")
    print(f"Kształt y_test:  {y_test.shape}")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    processed_df, f_scaler, t_scaler = load_and_preprocess_data()
    X_train, y_train, X_test, y_test = create_lstm_sequences(processed_df)