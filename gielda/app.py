import streamlit as st
import torch
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import warnings
import config

warnings.filterwarnings('ignore')

# Importujemy nasze własne moduły
from data_loader import fetch_and_prepare_data
from model import SP500PredictorLSTM

# 1. Konfiguracja strony
st.set_page_config(page_title="S&P 500 AI Predictor", layout="wide", page_icon="📈")

st.title("🤖 Deep Learning: S&P 500 Predictor")
st.markdown("Witaj w produkcyjnym panelu Twojego modelu **LSTM**.")


# 2. Pobieranie danych (z systemem Cache!)
# @st.cache_data sprawia, że dane pobierają się tylko raz, a nie przy każdym kliknięciu w aplikacji
@st.cache_data(ttl=3600)
def load_market_data():
    return fetch_and_prepare_data(period="10y")


with st.spinner("Pobieranie najświeższych danych z Yahoo Finance..."):
    df = load_market_data()

# 3. Sekcja Wizualizacji Rynku (Plotly)
st.subheader("📊 Aktualna sytuacja na rynku (Ostatnie 100 dni)")
recent_df = df.tail(100)

fig = go.Figure()
# Cena zamknięcia
fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Close'], mode='lines', name='Cena Zamknięcia (Close)',
                         line=dict(color='white')))
# Średnia 20-dniowa
fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['SMA_20'], mode='lines', name='SMA 20',
                         line=dict(color='orange', dash='dash')))
# Wstęgi Bollingera
fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Bollinger_Upper'], mode='lines', name='Bollinger Górna',
                         line=dict(color='gray', width=1)))
fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df['Bollinger_Lower'], mode='lines', name='Bollinger Dolna',
                         line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)'))

fig.update_layout(template="plotly_dark", hovermode="x unified", height=500)
st.plotly_chart(fig, width='stretch')

# 4. Sekcja Predykcji (Inference)
st.subheader("🔮 Prognoza Modelu na jutrzejszą sesję")

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
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT_RATE = 0.14589616433930247
model = SP500PredictorLSTM(
    input_size=config.INPUT_DIM,
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS,
    dropout_rate=config.DROPOUT_RATE
).to(device)

try:
    model.load_state_dict(torch.load("sp500_lstm_weights.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        X_input = X_input.to(device)
        prediction = model(X_input)
        predicted_return = prediction.item()

    predicted_pct = predicted_return * 100

    # Kafelki z metrykami (Metryki Streamlit)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Aktualna Cena S&P 500", value=f"{df['Close'].iloc[-1]:.2f} pkt")

    with col2:
        # Dynamiczny kolor i strzałka
        delta_color = "normal" if predicted_pct > 0 else "inverse"
        st.metric(label="Prognozowany Zwrot (Model LSTM)", value=f"{predicted_pct:+.2f}%",
                  delta=f"{predicted_pct:+.2f}%", delta_color=delta_color)

    with col3:
        if predicted_return > 0.001:
            st.success("Sygnał: **WZROST** 📈")
        elif predicted_return < -0.001:
            st.error("Sygnał: **SPADEK** 📉")
        else:
            st.warning("Sygnał: **KONSOLIDACJA** ➖")

except FileNotFoundError:
    st.error("BŁĄD: Nie znaleziono pliku 'sp500_lstm_weights.pth'. Uruchom najpierw trening modelu (train.py).")

st.caption("Uwaga: To jest projekt edukacyjny z zakresu Deep Learningu. Nie traktuj tego jako porady inwestycyjnej!")