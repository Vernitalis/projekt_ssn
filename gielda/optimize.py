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

from data_loader import fetch_and_prepare_data, prepare_lstm_data
from model import SP500PredictorLSTM

print("Pobieranie danych (tylko raz dla Optuny)...")
GLOBAL_DF = fetch_and_prepare_data()


def objective(trial):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pobieramy 4 elementy z Data Loadera
    X_full, y_full, _, _ = prepare_lstm_data(GLOBAL_DF, sequence_length=config.SEQUENCE_LENGTH)
    INPUT_DIM = X_full.shape[2]

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

        model = SP500PredictorLSTM(
            input_size=INPUT_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(10):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_t.to(device))
            mae = torch.abs(test_predictions - y_test_t.to(device)).mean().item()
            fold_maes.append(mae)

    return np.mean(fold_maes)


if __name__ == "__main__":
    print("\n--- Rozpoczynam Walk-Forward Optymalizację ---")
    study = optuna.create_study(direction="minimize", study_name="SP500_LSTM_Optimization")

    study.optimize(objective, n_trials=10)

    best_trial = study.best_trial
    print("\n" + "=" * 50)
    print("🚀 OPTYMALIZACJA ZAKOŃCZONA")
    print("=" * 50)
    print(f"Najlepsze ŚREDNIE MAE (Walk-Forward): {best_trial.value:.6f}")
    print("Zaktualizuj te wartości w config.py:")
    for key, value in best_trial.params.items():
        print(f"  {key.upper()} = {value}")