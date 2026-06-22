import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


def fetch_and_prepare_data(period="720d"):
    print("Pobieranie danych giełdowych i makroekonomicznych (1H)...")

    df_sp500 = yf.download("^GSPC", period=period, interval="1h")
    df_vix = yf.download("^VIX", period=period, interval="1h")
    df_tnx = yf.download("^TNX", period=period, interval="1h")
    df_dxy = yf.download("DX-Y.NYB", period=period, interval="1h")

    for d in [df_sp500, df_vix, df_tnx, df_dxy]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.droplevel(1)
        if not d.empty:
            d.index = d.index.tz_localize(None).round("1h")

    df_sp500 = df_sp500[~df_sp500.index.duplicated(keep='last')]
    df_vix = df_vix[~df_vix.index.duplicated(keep='last')]
    df_tnx = df_tnx[~df_tnx.index.duplicated(keep='last')]
    df_dxy = df_dxy[~df_dxy.index.duplicated(keep='last')]

    df = pd.DataFrame(index=df_sp500.index)
    df['Close'] = df_sp500['Close']
    df['Volume'] = df_sp500['Volume']
    df['VIX'] = df_vix['Close']
    df['TNX'] = df_tnx['Close']
    df['DXY'] = df_dxy['Close']

    df.ffill(inplace=True)
    df.bfill(inplace=True)

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

    df['Hourly_Return'] = df['Close'].pct_change()

    df.dropna(inplace=True)

    return df


def prepare_lstm_data(df, sequence_length=60):
    print(f"Przygotowywanie sekwencji (okno = {sequence_length} godzin)...")
    df = df.copy()

    features = [
        'Close', 'Volume', 'VIX', 'TNX', 'DXY',
        'SMA_20', 'RSI_14', 'Bollinger_Upper', 'Bollinger_Lower',
        'MACD', 'MACD_Signal', 'Hourly_Return'
    ]

    data_features = df[features].values
    data_targets = df[['Volume']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(data_features)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_targets = target_scaler.fit_transform(data_targets).flatten()

    X, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:(i + sequence_length)])
        y.append(scaled_targets[i + sequence_length])  # Model uczy się poprawnego T+1

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler, target_scaler