import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


def fetch_and_prepare_data(period="10y"):
    """
    Pobiera S&P 500 oraz pełny kontekst makro (VIX, Obligacje, Dolar).
    """
    print("Pobieranie danych giełdowych i makroekonomicznych...")

    df_sp500 = yf.download("^GSPC", period=period)
    df_vix = yf.download("^VIX", period=period)
    df_tnx = yf.download("^TNX", period=period)  # 10-letnie obligacje
    df_dxy = yf.download("DX-Y.NYB", period=period)  # Indeks dolara

    for d in [df_sp500, df_vix, df_tnx, df_dxy]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.droplevel(1)

    print("Łączenie danych i obliczanie wskaźników...")

    df = pd.DataFrame(index=df_sp500.index)
    df['Close'] = df_sp500['Close']
    df['Volume'] = df_sp500['Volume']
    df['VIX'] = df_vix['Close']
    df['TNX'] = df_tnx['Close']
    df['DXY'] = df_dxy['Close']

    # Łatamy dziury (różne kalendarze świąt dla akcji i obligacji)
    df.ffill(inplace=True)

    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df['RSI_14'] = 100 - (100 / (1 + (ema_up / ema_down)))

    std_20 = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['SMA_20'] + (std_20 * 2)
    df['Bollinger_Lower'] = df['SMA_20'] - (std_20 * 2)

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Daily_Return'] = df['Close'].pct_change()

    df.dropna(inplace=True)

    return df


def prepare_lstm_data(df, sequence_length=60):
    """
    Przygotowuje sekwencje czasowe, zwracając PEŁNY, niepodzielony zbiór X i y.
    Podziałem na foldy (Walk-Forward) zajmie się teraz skrypt train.py.
    """
    print(f"Przygotowywanie sekwencji (okno = {sequence_length} dni)...")

    df = df.copy()
    df['Target'] = df['Daily_Return'].shift(-1)
    df.dropna(inplace=True)

    features = [
        'Close', 'Volume', 'VIX', 'TNX', 'DXY',
        'SMA_20', 'RSI_14', 'Bollinger_Upper', 'Bollinger_Lower',
        'MACD', 'MACD_Signal', 'Daily_Return'
    ]

    data_features = df[features].values
    data_targets = df['Target'].values

    # Skalujemy od razu całą historię (w zaawansowanym Walk-Forward skalery
    # powinno się fittować per fold, ale dla prostoty utrzymamy jeden globalny skaler)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(data_features)

    X, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:(i + sequence_length)])
        y.append(data_targets[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    print(f"Pełny kształt X: {X.shape} (Próbki, Dni, Cechy)")
    print(f"Pełny kształt y: {y.shape}")

    return X, y, scaler


if __name__ == "__main__":
    df = fetch_and_prepare_data()
    X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(df)