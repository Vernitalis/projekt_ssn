import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    confusion_matrix
)
import warnings

warnings.filterwarnings('ignore')

import config
from data_loader import load_and_preprocess_data
from model import EnergyPredictorLSTM

# ==========================================
# 1. Konfiguracja Dashboardu IoT
# ==========================================
st.set_page_config(page_title="Smart Home Energy Predictor", layout="wide", page_icon="⚡")

st.title("⚡ AI Smart Home: Energy Demand Predictor")
st.markdown(
    "Produkcyjny panel telemetryczny systemu prognozowania obciążenia sieci energetycznej budynku (Model **Multivariate LSTM**)."
)


# ==========================================
# 2. Pobieranie i cache'owanie danych czujników
# ==========================================
@st.cache_data(ttl=1800)
def get_iot_data():
    return load_and_preprocess_data()


with st.spinner("Pobieranie strumienia danych z czujników UCI IoT..."):
    df, feature_scaler, target_scaler = get_iot_data()

# ==========================================
# 3. Wykres historycznego zużycia na bieżąco
# ==========================================
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
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. System Wnioskowania (Na Żywo)
# ==========================================
st.subheader("🔮 System Prognozowania Obciążenia Lokalu (T+1)")

# Przygotowanie wektora wejściowego z ostatnich 'SEQUENCE_LENGTH' wierszy
last_window = df.values[-config.SEQUENCE_LENGTH:]
X_input = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicjalizacja modelu
model = EnergyPredictorLSTM(
    input_size=config.INPUT_DIM,
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS,
    dropout_rate=config.DROPOUT_RATE
).to(device)

try:
    # Ładowanie wag
    model.load_state_dict(torch.load("iot_lstm_weights.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        X_input = X_input.to(device)
        prediction = model(X_input)
        # Inwersja skalowania wyniku
        predicted_wh = target_scaler.inverse_transform(prediction.cpu().numpy())[0][0]

    # Pobieramy ostatnią znaną wartość z licznika
    current_wh = target_scaler.inverse_transform(df[['Appliances']].values)[-1][0]
    difference = predicted_wh - current_wh

    # Wyświetlanie metryk bieżącej predykcji
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Ostatni odczyt z licznika", value=f"{current_wh:.1f} Wh")

    with col2:
        st.metric(
            label="Prognoza AI na kolejne 10 minut",
            value=f"{predicted_wh:.1f} Wh",
            delta=f"{difference:+.1f} Wh",
            delta_color="inverse" if difference > 0 else "normal"
        )

    with col3:
        if predicted_wh > 250:
            st.error("Status: WYSOKIE OBCIĄŻENIE 🚨 (Ryzyko szczytu poboru)")
        elif predicted_wh > 90:
            st.warning("Status: ŚREDNIE OBCIĄŻENIE ⚠️")
        else:
            st.success("Status: OPTYMALNY POBÓR ✅")

except FileNotFoundError:
    st.error("BŁĄD: Nie znaleziono pliku wag 'iot_lstm_weights.pth'. Uruchom najpierw trening z pliku train.py.")
    st.stop()

st.caption("Panel telemetryczny zasilany asymetryczną funkcją straty odporną na niedoszacowania krytyczne.")

# ==========================================
# 5. Zaawansowana Diagnostyka i Wykresy (Generowanie dla Prezentacji)
# ==========================================
st.markdown("---")
st.subheader("🔬 Zaawansowana Diagnostyka Modelu (Ewaluacja Historyczna)")

with st.spinner("Przetwarzanie danych historycznych do analizy statystycznej..."):
    # Bierzemy 1000 ostatnich rekordów do weryfikacji metryk w aplikacji
    eval_df = df.tail(1000)
    eval_values = eval_df.values

    # Indeks kolumny targetu (Appliances)
    target_idx = df.columns.get_loc('Appliances')

    X_eval = []
    y_eval = []
    y_prev = []  # Do liczenia kierunku

    for i in range(len(eval_values) - config.SEQUENCE_LENGTH):
        X_eval.append(eval_values[i:i + config.SEQUENCE_LENGTH])
        y_eval.append(eval_values[i + config.SEQUENCE_LENGTH, target_idx])
        y_prev.append(eval_values[i + config.SEQUENCE_LENGTH - 1, target_idx])

    X_eval_t = torch.tensor(np.array(X_eval), dtype=torch.float32).to(device)

    with torch.no_grad():
        preds_eval = model(X_eval_t).cpu().numpy()

    # Przywracanie skali do oryginalnych jednostek (Wh)
    preds_real = target_scaler.inverse_transform(preds_eval).flatten()
    y_real = target_scaler.inverse_transform(np.array(y_eval).reshape(-1, 1)).flatten()
    y_prev_real = target_scaler.inverse_transform(np.array(y_prev).reshape(-1, 1)).flatten()

    # Czas dla osi X (ominięcie pierwszych wartości z powodu okna sekwencji)
    eval_dates = eval_df.index[config.SEQUENCE_LENGTH:]

    # --- Obliczanie Metryk Klasycznych ---
    mae = mean_absolute_error(y_real, preds_real)
    mse = mean_squared_error(y_real, preds_real)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_real, preds_real) * 100
    r2 = r2_score(y_real, preds_real)

    # --- Obliczanie Trafności Kierunkowej (Directional Accuracy) ---
    true_diff = y_real - y_prev_real
    pred_diff = preds_real - y_prev_real

    true_dir = (true_diff > 0).astype(int)
    pred_dir = (pred_diff > 0).astype(int)

    dir_acc = np.mean(true_dir == pred_dir) * 100

    # Wyświetlanie kafelków z metrykami
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("MAE", f"{mae:.2f} Wh")
    m2.metric("MSE", f"{mse:.0f}")
    m3.metric("RMSE", f"{rmse:.2f} Wh")
    m4.metric("MAPE", f"{mape:.2f} %")
    m5.metric("R² Score", f"{r2:.4f}")
    m6.metric("Kierunek (Acc)", f"{dir_acc:.2f} %")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- WYKRES 1: SZEREG CZASOWY (PRAWDZIWE VS PREDYKCJA) ---
    st.markdown("**Porównanie: Rzeczywiste vs Przewidywane Zużycie (Szereg czasowy)**")
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=eval_dates, y=y_real,
        mode='lines', name='Faktyczne Zużycie (Wh)',
        line=dict(color='orange', width=2)
    ))
    fig_ts.add_trace(go.Scatter(
        x=eval_dates, y=preds_real,
        mode='lines', name='Predykcja AI (Wh)',
        line=dict(color='#00BFFF', width=2, dash='dot')
    ))
    fig_ts.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        xaxis_title="Czas",
        yaxis_title="Watogodziny (Wh)",
        height=450
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- WYKRES 2 i 3: SCATTER ORAZ MACIERZ POMYŁEK ---
    col_charts1, col_charts2 = st.columns(2)

    with col_charts1:
        st.markdown("**Zależność: Prawda vs Predykcja (Scatter Plot)**")
        fig_scatter = px.scatter(
            x=y_real, y=preds_real,
            labels={'x': 'Faktyczne Zużycie [Wh]', 'y': 'Prognozowane Zużycie [Wh]'},
            opacity=0.6,
            color_discrete_sequence=['#00BFFF']
        )
        min_val = min(min(y_real), min(preds_real))
        max_val = max(max(y_real), max(preds_real))
        fig_scatter.add_shape(
            type="line", line=dict(dash='dash', color='white'),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        fig_scatter.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_charts2:
        st.markdown("**Macierz Pomyłek Kierunku (Wzrost / Spadek)**")
        cm = confusion_matrix(true_dir, pred_dir)
        labels = ['Spadek', 'Wzrost']

        # Tworzenie wykresu w matplotlib i seaborn
        fig_cm, ax = plt.subplots(figsize=(6, 5))

        # Wymuszenie przezroczystego tła pod kolorystykę Streamlit
        fig_cm.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    cbar_kws={'label': 'Liczba predykcji'})

        # Konfiguracja kolorów tekstu i jawne wymuszenie opisów osi z marginesem (labelpad)
        ax.set_xlabel('Przewidywany Kierunek', color='white', labelpad=12, fontsize=11)
        ax.set_ylabel('Rzeczywisty Kierunek', color='white', labelpad=12, fontsize=11)
        ax.tick_params(colors='white')

        # Konfiguracja kolorystyczna paska bocznego (colorbar)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('Liczba predykcji', color='white', labelpad=10)

        # Kluczowe automatyczne dopasowanie marginesów figury, by zapobiec obcinaniu osi
        fig_cm.tight_layout()

        st.pyplot(fig_cm)