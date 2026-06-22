import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import joblib

warnings.filterwarnings('ignore')

from data_loader import fetch_and_prepare_data, prepare_lstm_data
from model import SP500PredictorLSTM
import config


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Uruchamiam Trening na urządzeniu: {device} ---")

    df = fetch_and_prepare_data(period="720d")
    X_full, y_full, scaler, target_scaler = prepare_lstm_data(df, sequence_length=config.SEQUENCE_LENGTH)

    split_idx = int(len(X_full) * 0.8)
    X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    y_train, y_val = y_full[:split_idx], y_full[split_idx:]

    print("\n" + "=" * 50)
    print(f"ZBIÓR TRENINGOWY:  {len(X_train)} godzin")
    print(f"ZBIÓR WALIDACYJNY: {len(X_val)} godzin (Służy do Early Stoppingu)")
    print("=" * 50)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = SP500PredictorLSTM(
        input_size=config.INPUT_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    MAX_EPOCHS = 150
    EARLY_STOPPING_PATIENCE = 15
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t.to(device))
            val_loss = criterion(val_preds, y_val_t.to(device)).item()

        scheduler.step(val_loss)
        print(f"Epoka {epoch + 1:03d}/{MAX_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # ZAPISUJEMY WAGI ORAZ ZAMROŻONE SKALERY (Rozwiązanie wycieku danych)
            torch.save(model.state_dict(), "sp500_lstm_weights.pth")
            joblib.dump(scaler, "feature_scaler.pkl")
            joblib.dump(target_scaler, "target_scaler.pkl")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n🛑 EARLY STOPPING ZADZIAŁAŁ! Model przestał się uczyć przez {EARLY_STOPPING_PATIENCE} epok.")
                break

    print("\n" + "=" * 50)
    print("📊 WYNIKI NA ZBIORZE WALIDACYJNYM (NIEWIDZIANE DANE)")
    print("=" * 50)

    model.load_state_dict(torch.load("sp500_lstm_weights.pth"))
    model.eval()
    with torch.no_grad():
        final_preds_scaled = model(X_val_t.to(device)).cpu().numpy()
        trues_scaled = y_val_t.numpy()

        mae = mean_absolute_error(trues_scaled, final_preds_scaled)
        r2 = r2_score(trues_scaled, final_preds_scaled)

        print(f"Końcowe MAE: {mae:.4f} (w skali 0-1)")
        print(f"Końcowe R²:  {r2:.4f} (Bliżej 1.0 to lepiej)")


if __name__ == "__main__":
    train_model()