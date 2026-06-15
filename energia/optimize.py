import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import warnings
import config

warnings.filterwarnings('ignore')

from data_loader import load_and_preprocess_data, create_lstm_sequences
from model import EnergyPredictorLSTM
from train import AsymmetricEnergyLoss

print("Pobieranie danych (tylko raz dla Optuny)...")
GLOBAL_DF, _, TARGET_SCALER = load_and_preprocess_data()


def objective(trial):
    # Optuna będzie szukać najlepszych parametrów
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tworzymy sekwencje zgodnie z konfiguracją
    X_full, y_full, _, _ = create_lstm_sequences(
        GLOBAL_DF,
        sequence_length=config.SEQUENCE_LENGTH,
        train_split=1.0  # Całość dla Walk-Forward
    )

    # Inicjalizacja podziału
    tscv = TimeSeriesSplit(n_splits=config.FOLDS)
    fold_maes = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X_full)):
        X_train, X_test = X_full[train_index], X_full[test_index]
        y_train, y_test = y_full[train_index], y_full[test_index]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        model = EnergyPredictorLSTM(
            input_size=config.INPUT_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        ).to(device)

        # Używamy naszej asymetrycznej kary
        criterion = AsymmetricEnergyLoss(under_penalty=config.UNDER_PREDICTION_PENALTY)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Trening - dajemy tylko 5 epok, aby zminimalizować czas trwania Trialu
        for epoch in range(5):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

        # Walidacja (odwracamy z powrotem do fizycznych Watogodzin)
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_t.to(device))
            preds_raw = TARGET_SCALER.inverse_transform(test_predictions.cpu().numpy())
            y_raw = TARGET_SCALER.inverse_transform(y_test_t.cpu().numpy())

            mae = np.abs(preds_raw - y_raw).mean()
            fold_maes.append(mae)

    # Optuna będzie próbowała ZMINIMALIZOWAĆ ten wynik
    return np.mean(fold_maes)


if __name__ == "__main__":
    print("\n--- Rozpoczynam Optyalizację IoT Walk-Forward ---")
    study = optuna.create_study(direction="minimize", study_name="IoT_LSTM_Optimization")

    # Wykonujemy 10 prób (możesz zwiększyć do 30-50, jeśli masz więcej czasu)
    study.optimize(objective, n_trials=10)

    best_trial = study.best_trial
    print("\n" + "=" * 50)
    print("🚀 OPTYMALIZACJA ZAKOŃCZONA")
    print("=" * 50)
    print(f"Najlepsze Średnie MAE w Watogodzinach: {best_trial.value:.2f} Wh")
    print("\nZaktualizuj poniższe wartości w swoim pliku config.py:")
    for key, value in best_trial.params.items():
        print(f"  {key.upper()} = {value}")