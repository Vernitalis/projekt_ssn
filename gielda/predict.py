import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
import config

warnings.filterwarnings('ignore')

from data_loader import fetch_and_prepare_data
from model import SP500PredictorLSTM


def predict_next_hour():
    print("Inicjalizacja modułu predykcyjnego (Regresja Wolumenu)...\n")

    df = fetch_and_prepare_data(period="730d")

    features = [
        'Close', 'Volume', 'VIX', 'TNX', 'DXY',
        'SMA_20', 'RSI_14', 'Bollinger_Upper', 'Bollinger_Lower',
        'MACD', 'MACD_Signal', 'Hourly_Return'
    ]
    data_features = df[features].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_features)

    # Skaler Wolumenu dla targetu
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(df[['Volume']].values)

    last_60_hours = data_features[-config.SEQUENCE_LENGTH:]
    last_60_hours_scaled = scaler.transform(last_60_hours)

    X_input = torch.tensor(last_60_hours_scaled, dtype=torch.float32).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INPUT_DIM = X_input.shape[2]
    model = SP500PredictorLSTM(
        input_size=INPUT_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    try:
        model.load_state_dict(torch.load("sp500_lstm_weights.pth", map_location=device))
        print("Pomyślnie załadowano wagi modelu ('sp500_lstm_weights.pth').")
    except FileNotFoundError:
        print("BŁĄD: Nie znaleziono pliku z wagami! Uruchom najpierw train.py.")
        return

    model.eval()

    # Wnioskowanie
    with torch.no_grad():
        X_input = X_input.to(device)
        prediction = model(X_input)

        predicted_scaled = prediction.item()
        predicted_real_volume = target_scaler.inverse_transform([[predicted_scaled]])[0][0]

    print("\n" + "=" * 50)
    print("📊 PREDYKCJA WOLUMENU S&P 500 NA NASTĘPNĄ GODZINĘ SESYJNĄ")
    print("=" * 50)

    aktualny_wolumen = int(df['Volume'].iloc[-1])
    prognoza = int(predicted_real_volume)
    roznica = prognoza - aktualny_wolumen

    print(f"Ostatnio zarejestrowany wolumen (1H): {aktualny_wolumen:,}".replace(",", " "))
    print(f"Prognozowany wolumen (1H):            {prognoza:,}".replace(",", " "))

    if roznica > 0:
        print(f"\nSygnał: WZROST WOLUMENU (+{roznica:,}) 📈".replace(",", " "))
    else:
        print(f"\nSygnał: SPADEK WOLUMENU ({roznica:,}) 📉".replace(",", " "))

    print("=" * 50)


if __name__ == "__main__":
    predict_next_hour()