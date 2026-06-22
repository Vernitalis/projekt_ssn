import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
import warnings
import joblib
import config

warnings.filterwarnings('ignore')

from data_loader import fetch_and_prepare_data
from model import SP500PredictorLSTM

st.set_page_config(page_title="S&P 500 AI Predictor", layout="wide", page_icon="📈")

st.title("S&P 500 Volume prediction (LSTM)")

@st.cache_data(ttl=3600)
def load_market_data():
    return fetch_and_prepare_data(period="720d")


with st.spinner("Pobieranie najświeższych danych z Yahoo Finance..."):
    df = load_market_data()

features = [
    'Close', 'Volume', 'VIX', 'TNX', 'DXY',
    'SMA_20', 'RSI_14', 'Bollinger_Upper', 'Bollinger_Lower',
    'MACD', 'MACD_Signal', 'Hourly_Return'
]
data_features = df[features].values
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SP500PredictorLSTM(
    input_size=config.INPUT_DIM,
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS,
    dropout_rate=config.DROPOUT_RATE
).to(device)

try:
    scaler = joblib.load("feature_scaler.pkl")
    target_scaler = joblib.load("target_scaler.pkl")
    model.load_state_dict(torch.load("sp500_lstm_weights.pth", map_location=device))
    model.eval()
except FileNotFoundError:
    st.error(
        "BŁĄD: Nie znaleziono plików modelu lub skalerów (*.pkl / *.pth). Uruchom najpierw trening modelu (Wybór 1).")
    st.stop()

# ==========================================
# 1. GENEROWANIE PREDYKCJI HISTORYCZNYCH (300H)
# ==========================================
DISPLAY_PERIOD = 300
needed_rows = DISPLAY_PERIOD + config.SEQUENCE_LENGTH
chart_df = df.tail(needed_rows)

chart_data_features = chart_df[features].values
chart_scaled_features = scaler.transform(chart_data_features)

X_chart = []
for i in range(DISPLAY_PERIOD):
    X_chart.append(chart_scaled_features[i: i + config.SEQUENCE_LENGTH])

X_chart_t = torch.tensor(np.array(X_chart), dtype=torch.float32).to(device)

with torch.no_grad():
    chart_preds_scaled = model(X_chart_t).cpu().numpy()

chart_preds_real = target_scaler.inverse_transform(chart_preds_scaled).flatten()
recent_300_df = df.tail(DISPLAY_PERIOD)

y_true = recent_300_df['Volume'].values
y_pred = chart_preds_real

# ==========================================
# 2. OBLICZANIE METRYK
# ==========================================
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

# Obliczenia pod Directional Accuracy (i Macierz Pomyłek)
true_delta = np.diff(y_true)
pred_delta = y_pred[1:] - y_true[:-1]
dir_acc = np.mean(np.sign(true_delta) == np.sign(pred_delta)) * 100

# Tłumaczymy delty na klasy dla Macierzy Pomyłek: "Wzrost" vs "Spadek"
true_class = np.where(true_delta > 0, "Wzrost", "Spadek")
pred_class = np.where(pred_delta > 0, "Wzrost", "Spadek")
cm = confusion_matrix(true_class, pred_class, labels=["Spadek", "Wzrost"])

# ==========================================
# 3. WIZUALIZACJA GŁÓWNA & BŁĘDY W CZASIE
# ==========================================
st.subheader(f"Szereg czasowy: Model vs Prawdziwy Rynek (Ostatnie {DISPLAY_PERIOD} godzin)")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=recent_300_df.index, y=y_true, mode='lines', name='Faktyczny Wolumen',
                          line=dict(color='deepskyblue', width=1.2)))
fig1.add_trace(go.Scatter(x=recent_300_df.index, y=y_pred, mode='lines', name='Predykcja LSTM',
                          line=dict(color='red', width=1.5, dash='dash')))
fig1.update_layout(template="plotly_dark", hovermode="x unified", height=450, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig1, width='stretch')

st.subheader("Rozbieżność w czasie (Błąd Bezwzględny)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=recent_300_df.index, y=np.abs(y_true - y_pred), mode='lines', name='Wartość Błędu',
                          line=dict(color='orange', width=1.0)))
fig2.update_layout(template="plotly_dark", hovermode="x unified", height=250, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig2, width='stretch')

# ==========================================
# 4. WYKRESY
# ==========================================
st.markdown("---")
st.subheader("Wykresy")
col_diag1, col_diag2 = st.columns(2)

with col_diag1:
    # Wykres A: Rozkład Błędów (Histogram)
    st.write("**Rozkład Błędów(Predykcja - Prawda)**")
    errors = y_pred - y_true
    fig_hist = go.Figure(data=[go.Histogram(x=errors, nbinsx=40, marker_color='mediumpurple')])
    fig_hist.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_hist, width='stretch')

    st.write("**Macierz Pomyłek (Wzrost/Spadek)**")

    fig_cm, ax = plt.subplots(figsize=(5, 4))

    fig_cm.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Wykres B: macierz pomyłek
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Spadek", "Wzrost"],
                yticklabels=["Spadek", "Wzrost"],
                cbar=False, ax=ax,
                annot_kws={"size": 16})

    ax.set_xlabel('Predykcja', color='gray')
    ax.set_ylabel('Prawda', color='gray')
    ax.tick_params(colors='gray')

    st.pyplot(fig_cm)


with col_diag2:
    # Wykres C: Rzeczywiste vs Przewidywane (Scatter Plot)
    st.write("**Zależność Prawda vs Predykcja**")
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())

    fig_scatter = go.Figure()
    # Punkty modelu
    fig_scatter.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers',
                                     marker=dict(color='cyan', opacity=0.6), name='Punkty danych'))
    # Linia idealnej regresji y = x
    fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                                     line=dict(color='red', dash='dash'), name='Regresja Idealna'))

    fig_scatter.update_layout(template="plotly_dark", height=350,
                              xaxis_title="Faktyczny Wolumen", yaxis_title="Przewidywany Wolumen",
                              margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_scatter, width='stretch')

# ==========================================
# 5. METRYKI PROJEKTOWE I PROGNOZA NA T+1
# ==========================================
st.markdown("---")
st.subheader("Metryki wydajności")
m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

with m_col1:
    st.metric(label="MAE", value=f"{mae:,.0f}".replace(",", " "))
with m_col2:
    st.metric(label="MSE", value=f"{mse:,.0f}".replace(",", " "))
with m_col3:
    st.metric(label="RMSE", value=f"{rmse:,.0f}".replace(",", " "))
with m_col4:
    st.metric(label="MAPE", value=f"{mape:.2f}%")
with m_col5:
    st.metric(label="R² Score", value=f"{r2:.4f}")
with m_col6:
    st.metric(label="Directional Acc.", value=f"{dir_acc:.2f}%")

st.subheader("Prognoza na najbliższą godzinę sesyjną")
last_60_hours = data_features[-config.SEQUENCE_LENGTH:]
last_60_hours_scaled = scaler.transform(last_60_hours)
X_future = torch.tensor(last_60_hours_scaled, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    future_pred_scaled = model(X_future).item()
    future_pred_real = target_scaler.inverse_transform([[future_pred_scaled]])[0][0]

c1, c2 = st.columns(2)
with c1:
    aktualny_wolumen = int(df['Volume'].iloc[-1])
    st.metric(label="Aktualny Wolumen (Ostatnia Godzina)", value=f"{aktualny_wolumen:,}".replace(",", " "))
with c2:
    prognozowany_wolumen = int(future_pred_real)
    delta_wolumen = prognozowany_wolumen - aktualny_wolumen
    st.metric(label="Prognozowany Wolumen (Następna Godzina)", value=f"{prognozowany_wolumen:,}".replace(",", " "),
              delta=f"{delta_wolumen:,}".replace(",", " "), delta_color="normal")