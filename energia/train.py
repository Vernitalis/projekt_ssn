import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

import config
from data_loader import load_and_preprocess_data, create_lstm_sequences
from model import EnergyPredictorLSTM


# ==========================================
# 1. CUSTOM LOSS: Asymetryczna Kara (Smart Grid)
# ==========================================
class AsymmetricEnergyLoss(nn.Module):
    def __init__(self, under_penalty=2.0):
        super(AsymmetricEnergyLoss, self).__init__()
        self.under_penalty = under_penalty
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred, y_true):
        # Klasyczne MSE
        base_loss = self.mse(y_pred, y_true)

        # Logika: y_true - y_pred > 0 oznacza, że w rzeczywistości było więcej prądu niż zakładał model
        # (Niedoszacowanie)
        penalty_weights = torch.where(y_true > y_pred, self.under_penalty, 1.0)

        weighted_loss = base_loss * penalty_weights
        return torch.mean(weighted_loss)


# ==========================================
# 2. GŁÓWNA PĘTLA Z WALK-FORWARD
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Uruchamiam system na urządzeniu: {device} ---")

    # Pobieranie i ładowanie całego datasetu ze znormalizowanym czasem i czujnikami
    df, feature_scaler, target_scaler = load_and_preprocess_data()

    X_full, y_full, _, _ = create_lstm_sequences(df, sequence_length=config.SEQUENCE_LENGTH, train_split=1.0)

    # Inicjalizacja podziału chronologicznego
    tscv = TimeSeriesSplit(n_splits=config.FOLDS)
    fold_metrics = {'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'R2': [], 'DA': []}

    print("\n" + "=" * 50)
    print("ROZPOCZYNAM WALIDACJĘ WALK-FORWARD (SMART HOME)")
    print("=" * 50)

    for fold, (train_index, test_index) in enumerate(tscv.split(X_full)):
        print(f"\n--- FOLD {fold + 1}/{config.FOLDS} ---")
        print(f"Dane Treningowe: {len(train_index)} cykli | Dane Testowe: {len(test_index)} cykli")

        X_train, X_test = X_full[train_index], X_full[test_index]
        y_train, y_test = y_full[train_index], y_full[test_index]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # Inicjalizacja modelu (SSOT)
        model = EnergyPredictorLSTM(
            input_size=config.INPUT_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout_rate=config.DROPOUT_RATE
        ).to(device)

        # Asymetryczna funkcja straty
        criterion = AsymmetricEnergyLoss(under_penalty=config.UNDER_PREDICTION_PENALTY)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        for epoch in range(config.EPOCHS_PER_FOLD):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

        # Ewaluacja (na zwykłym MAE do odczytu)
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_t.to(device))

            # Cofamy skalowanie do rzeczywistych Watogodzin
            preds_raw = target_scaler.inverse_transform(test_predictions.cpu().numpy())
            y_raw = target_scaler.inverse_transform(y_test_t.cpu().numpy())

            # Obliczanie standardowych metryk
            mae = mean_absolute_error(y_raw, preds_raw)
            mse = mean_squared_error(y_raw, preds_raw)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_raw, preds_raw) * 100
            r2 = r2_score(y_raw, preds_raw)

            # Directional Accuracy dla absolutnych wartości (czy przewidział wzrost/spadek względem t-1)
            # Używamy np.diff aby policzyć różnice między kolejnymi krokami w wektorze
            true_diff = np.diff(y_raw.flatten())
            pred_diff = np.diff(preds_raw.flatten())
            dir_acc = np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100

            fold_metrics['MAE'].append(mae)
            fold_metrics['MSE'].append(mse)
            fold_metrics['RMSE'].append(rmse)
            fold_metrics['MAPE'].append(mape)
            fold_metrics['R2'].append(r2)
            fold_metrics['DA'].append(dir_acc)

            print(f"Zakończono Fold {fold + 1} | MAE: {mae:.2f} Wh | R2: {r2:.4f} | Dir.Acc: {dir_acc:.2f}%")

    print("\n" + "=" * 50)
    print("PODSUMOWANIE WALK-FORWARD (SMART HOME)")
    print("=" * 50)
    print(f"Średnie MAE:  {np.mean(fold_metrics['MAE']):.2f} Wh")
    print(f"Średnie MSE:  {np.mean(fold_metrics['MSE']):.2f}")
    print(f"Średnie RMSE: {np.mean(fold_metrics['RMSE']):.2f} Wh")
    print(f"Średnie MAPE: {np.mean(fold_metrics['MAPE']):.2f}%")
    print(f"Średnie R²:   {np.mean(fold_metrics['R2']):.4f}")
    print(f"Średnie Dir. Acc (Wykrywanie wzrostu/spadku obciążenia): {np.mean(fold_metrics['DA']):.2f}%")

    torch.save(model.state_dict(), "iot_lstm_weights.pth")
    print("Zapisano wagi modelu IoT gotowe do produkcji.")

if __name__ == "__main__":
    train_model()