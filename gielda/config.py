# ==========================================
# ⚙️ GLOBALNA KONFIGURACJA MODELU (S&P 500)
# ==========================================

# --- 1. Ustawienia Danych ---
SEQUENCE_LENGTH = 60
INPUT_DIM = 12

# --- 2. Architektura Sieci ---
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT_RATE = 0.13975364373605703

# --- 3. Ustawienia Treningu (Walk-Forward) ---
LEARNING_RATE = 0.002486566420414481
BATCH_SIZE = 16
EPOCHS_PER_FOLD = 25
FOLDS = 5