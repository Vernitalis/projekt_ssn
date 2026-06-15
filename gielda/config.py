# ==========================================
# ⚙️ GLOBALNA KONFIGURACJA MODELU (S&P 500)
# ==========================================

# --- 1. Ustawienia Danych ---
SEQUENCE_LENGTH = 60
INPUT_DIM = 12           # Cechy bazowe + Wskaźniki + TNX + DXY

# --- 2. Architektura Sieci (Ostatni wynik z Optuny) ---
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT_RATE = 0.3618952842568224

# --- 3. Ustawienia Treningu (Walk-Forward) ---
LEARNING_RATE = 0.0004813070290170937
BATCH_SIZE = 32
EPOCHS_PER_FOLD = 20
FOLDS = 5

# --- 4. Ustawienia Funkcji Straty ---
PENALTY_MULTIPLIER = 2.5 # Mnożnik kary za pomyłkę kierunku (Custom Loss)