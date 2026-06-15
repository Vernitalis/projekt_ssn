import streamlit as st
import torch
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

import config
from data_loader import load_and_preprocess_data
from model import EnergyPredictorLSTM

# 1. Konfiguracja Dashboardu IoT
st.set_page_config(page_title="Smart Home Energy Predictor", layout="wide", page_icon="⚡")

st.title("⚡ AI Smart Home: Energy Demand Predictor")
st.markdown(
    "Produkcyjny panel telemetryczny systemu prognozowania obciążenia sieci energetycznej budynku (Model **Multivariate LSTM**).")


# 2. Pobieranie i cache'owanie danych czujników
@st.cache_data(ttl=1800)
def get_iot_data():
    return load_and_preprocess_data()


with st.spinner("Pobieranie strumienia danych z czujników UCI IoT..."):
    df, feature_scaler, target_scaler = get_iot_data()

# 3. Wykres historycznego zużycia (Plotly)
st.subheader("📊 Monitor zużycia energii elektrycznej (Ostatnie 150 pomiarów)")
recent_df = df.tail(150)

# Odtwarzamy realne wartości do wyświetlenia na osi Y wykresu
raw_appliances = target_scaler.inverse_transform(df[['Appliances']].tail(150).values)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recent_df.index,
    y=raw_appliances.flatten(),
    mode='lines+markers',
    name='Faktyczne zużycie (Wh)',
    line=dict(color='orange', width=2)
))
fig.update_layout(template="plotly_dark", hovermode="x unified", height=400, yaxis_title="Watogodziny (Wh)")
st.plotly_chart(fig, width='stretch')

# 4. Przygotowanie wektora wejściowego do predykcji
# Do LSTM potrzebujemy dokładnie ostatnich 'SEQUENCE_LENGTH' wierszy
last_window = df.values[-config.SEQUENCE_LENGTH:]
X_input = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)  # Kształt: (1, Sequence, Features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicjalizacja modelu w oparciu o wspólny Config (SSOT)
model = EnergyPredictorLSTM(
    input_size=config.INPUT_DIM,
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS,
    dropout_rate=config.DROPOUT_RATE
).to(device)

st.subheader("🔮 System Prognozowania Obciążenia Lokalu")

try:
    # Ładowanie wag
    model.load_state_dict(torch.load("iot_lstm_weights.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        X_input = X_input.to(device)
        prediction = model(X_input)

        # Inwersja skalowania wyniku, by uzyskać realne Watogodziny
        predicted_wh = target_scaler.inverse_transform(prediction.cpu().numpy())[0][0]

    # Pobieramy ostatnią znaną wartość z licznika
    current_wh = target_scaler.inverse_transform(df[['Appliances']].values)[-1][0]
    difference = predicted_wh - current_wh

    # Wyświetlanie metryk w kafelkach
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Ostatni odczyt z licznika", value=f"{current_wh:.1f} Wh")

    with col2:
        st.metric(
            label="Prognoza AI na kolejne 10 minut",
            value=f"{predicted_wh:.1f} Wh",
            delta=f"{difference:+.1f} Wh",
            delta_color="inverse" if difference > 0 else "normal"  # Większe zużycie to czerwona strzałka ostrzegawcza
        )

    with col3:
        # Klasyfikacja sygnału obciążenia infrastruktury
        if predicted_wh > 250:
            st.error("Status: WYSOKIE OBCIĄŻENIE 🚨 (Ryzyko szczytu poboru)")
        elif predicted_wh > 90:
            st.warning("Status: ŚREDNIE OBCIĄŻENIE ⚠️")
        else:
            st.success("Status: OPTYMALNY POBÓR ✅")

except FileNotFoundError:
    st.error("BŁĄD: Nie znaleziono pliku wag 'iot_lstm_weights.pth'. Uruchom najpierw trening z pliku train.py.")

st.caption("Panel telemetryczny zasilany asymetryczną funkcją straty odporną na niedoszacowania krytyczne.")