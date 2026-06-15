import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from data_loader import fetch_and_prepare_data, prepare_lstm_data
from model import SP500PredictorLSTM
import config


# ==========================================
# 1. CUSTOM LOSS: Kierunkowy błąd finansowy
# ==========================================
class DirectionalMSELoss(nn.Module):
    def __init__(self, penalty_multiplier=2.5):
        """
        Krzyżówka MSE z logiką tradingową. Jeśli model pomyli kierunek
        (przewidzi zysk, a będzie strata), błąd jest mnożony przez penalty_multiplier.
        """
        super(DirectionalMSELoss, self).__init__()
        self.penalty_multiplier = penalty_multiplier
        self.mse = nn.MSELoss(reduction='none')  # Chcemy surowy wektor błędów

    def forward(self, y_pred, y_true):
        # Klasyczny błąd średniokwadratowy dla każdej próbki
        base_loss = self.mse(y_pred, y_true)

        # Logika sprawdzania znaków: iloczyn liczb o różnych znakach jest < 0
        # Używamy torch.where: Jeśli y_pred * y_true < 0, daj karę. W przeciwnym razie x 1.0.
        sign_penalty = torch.where(y_pred * y_true < 0, self.penalty_multiplier, 1.0)

        # Mnożymy błąd przez wektor kar i zwracamy średnią całego batcha
        weighted_loss = base_loss * sign_penalty
        return torch.mean(weighted_loss)


# ==========================================
# 2. GŁÓWNA PĘTLA Z WALK-FORWARD VALIDATION
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Uruchamiam system Walk-Forward na urządzeniu: {device} ---")

    df = fetch_and_prepare_data()
    # Używamy config.SEQUENCE_LENGTH
    X_full, y_full, scaler = prepare_lstm_data(df, sequence_length=config.SEQUENCE_LENGTH)

    # Używamy config.FOLDS
    tscv = TimeSeriesSplit(n_splits=config.FOLDS)

    # Do zapisu średnich wyników ze wszystkich epok czasowych
    fold_metrics = {'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'R2': [], 'DA': []}

    print("\n" + "=" * 50)
    print("ROZPOCZYNAM WALIDACJĘ WALK-FORWARD")
    print("=" * 50)

    for fold, (train_index, test_index) in enumerate(tscv.split(X_full)):
        print(f"\n--- FOLD {fold + 1}/{config.FOLDS} ---")
        print(f"Dane Treningowe: {len(train_index)} dni | Dane Testowe: {len(test_index)} dni")

        # Cięcie danych dla konkretnego folda
        X_train, X_test = X_full[train_index], X_full[test_index]
        y_train, y_test = y_full[train_index], y_full[test_index]

        # Konwersja do tensorów
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        model = SP500PredictorLSTM(
            input_size=config.INPUT_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout_rate=config.DROPOUT_RATE
        ).to(device)

        criterion = DirectionalMSELoss(penalty_multiplier=config.PENALTY_MULTIPLIER)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        # --- ZMIANA PĘTLI EPOK ---
        for epoch in range(config.EPOCHS_PER_FOLD):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)  # Nasz Custom Loss z karą za kierunek
                loss.backward()
                optimizer.step()

        # Ewaluacja dla danego Folda
        model.eval()
        with torch.no_grad():
            X_test_device = X_test_t.to(device)
            y_test_device = y_test_t.to(device)

            test_predictions = model(X_test_device)

            # Przerzucamy z GPU do Numpy na CPU
            preds = test_predictions.cpu().numpy()
            trues = y_test_device.cpu().numpy()

            # Obliczanie standardowych metryk
            mae = mean_absolute_error(trues, preds)
            mse = mean_squared_error(trues, preds)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(trues, preds) * 100
            r2 = r2_score(trues, preds)

            # Directional Accuracy dla zwrotów akcji (czy znaki są takie same)
            dir_acc = np.mean(np.sign(trues) == np.sign(preds)) * 100

            fold_metrics['MAE'].append(mae)
            fold_metrics['MSE'].append(mse)
            fold_metrics['RMSE'].append(rmse)
            fold_metrics['MAPE'].append(mape)
            fold_metrics['R2'].append(r2)
            fold_metrics['DA'].append(dir_acc)

            print(f"Zakończono Fold {fold + 1} | MAE: {mae:.4f} | R2: {r2:.4f} | Dir.Acc: {dir_acc:.2f}%")

    print("\n" + "=" * 50)
    print("PODSUMOWANIE WALK-FORWARD (ŚREDNIE Z 5 OKRESÓW)")
    print("=" * 50)
    print(f"Średnie MAE:  {np.mean(fold_metrics['MAE']):.6f}")
    print(f"Średnie MSE:  {np.mean(fold_metrics['MSE']):.6f}")
    print(f"Średnie RMSE: {np.mean(fold_metrics['RMSE']):.6f}")
    print(f"Średnie MAPE: {np.mean(fold_metrics['MAPE']):.2f}%")
    print(f"Średnie R²:   {np.mean(fold_metrics['R2']):.4f}")
    print(f"Średnie Dir. Acc (Trafność kierunku): {np.mean(fold_metrics['DA']):.2f}%")

    torch.save(model.state_dict(), "sp500_lstm_weights.pth")
    print("Zapisano model gotowy do produkcji.")


if __name__ == "__main__":
    train_model()