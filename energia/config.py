# ==========================================
# ⚙️ GLOBALNA KONFIGURACJA MODELU (SMART HOME)
# ==========================================

# --- 1. Ustawienia Danych ---
SEQUENCE_LENGTH = 36     # 36 kroków x 10 min = 6 godzin historii
INPUT_DIM = 30           # 29 zmiennych wejściowych + 1 historyczne zużycie

# --- 2. Architektura Sieci LSTM ---
# Czujniki IoT są mniej zaszumione niż gielda, więc możemy pozwolić sobie na głębszą sieć
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT_RATE = 0.22404960339243019

# --- 3. Ustawienia Treningu (Walk-Forward Validation) ---
LEARNING_RATE = 0.0021002674736182937
BATCH_SIZE = 64          # Większy batch, bo mamy dużo danych (prawie 20 tys. rekordów)
EPOCHS_PER_FOLD = 20
FOLDS = 5

# --- 4. Custom Loss (Energy Penalty) ---
# Na giełdzie karaliśmy za zły kierunek (+/-). W energetyce pomyłka to niedoszacowanie (Under-prediction).
# Jeśli model przewidzi za mało prądu, a dom pobierze dużo, wywali bezpieczniki (lub transformator).
# Kara musi być asymetryczna - mocniej karzemy za niedoszacowanie prądu niż za przeszacowanie.
UNDER_PREDICTION_PENALTY = 2.0