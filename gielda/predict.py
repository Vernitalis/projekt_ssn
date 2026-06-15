import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

from data_loader import fetch_and_prepare_data
from model import SP500PredictorLSTM


def predict_tomorrow():
    print("Inicjalizacja modułu predykcyjnego (Regresja)...\n")

    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    DROPOUT_RATE = 0.14589616433930247

    df = fetch_and_prepare_data(period="10y")

    # Rekonstrukcja danych z 10 cechami
    features = [
        'Close', 'Volume', 'VIX', 'TNX', 'DXY',
        'SMA_20', 'RSI_14', 'Bollinger_Upper', 'Bollinger_Lower',
        'MACD', 'MACD_Signal', 'Daily_Return'
    ]
    data_features = df[features].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_features)

    last_60_days = data_features[-60:]
    last_60_days_scaled = scaler.transform(last_60_days)

    X_input = torch.tensor(last_60_days_scaled, dtype=torch.float32).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INPUT_DIM = X_input.shape[2]
    model = SP500PredictorLSTM(input_size=INPUT_DIM, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE).to(device)

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

        # Wyciągamy surową wartość (Daily Return)
        predicted_return = prediction.item()

    print("\n" + "=" * 50)
    print("📊 PREDYKCJA S&P 500 NA NASTĘPNY DZIEŃ SESYJNY")
    print("=" * 50)

    # Tłumaczymy ułamek na procenty
    predicted_pct = predicted_return * 100

    print(f"Prognozowany zwrot (Daily Return): {predicted_pct:+.2f}%\n")

    if predicted_return > 0.001:  # Wymagamy minimum +0.1% żeby nazwać to wzrostem
        print("Sygnał: WZROST 📈 (Przewidywana sesja na plusie)")
    elif predicted_return < -0.001:  # Wymagamy minimum -0.1% żeby nazwać to spadkiem
        print("Sygnał: SPADEK 📉 (Przewidywana sesja na minusie)")
    else:
        print("Sygnał: KONSOLIDACJA ➖ (Ruch boczny, brak silnego trendu na jutro)")

    print("=" * 50)
    print("Uwaga: To narzędzie edukacyjne. Rzeczywisty rynek to chaos! :)")


if __name__ == "__main__":
    predict_tomorrow()