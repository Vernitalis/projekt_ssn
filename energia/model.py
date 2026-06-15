import torch
import torch.nn as nn


class EnergyPredictorLSTM(nn.Module):
    def __init__(self, input_size=30, hidden_size=64, num_layers=2, dropout_rate=0.2):
        super(EnergyPredictorLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Warstwa LSTM przyjmująca aż 30 czujników naraz (Multivariate)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Regularyzacja zapobiegająca przeuczeniu
        self.dropout = nn.Dropout(dropout_rate)

        # Warstwa wyjściowa (Linear).
        # Zauważ brak jakiejkolwiek funkcji aktywacji (Sigmoid/Tanh) pod nią!
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Inicjalizacja stanów ukrytych zerami
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Przepuszczamy dane przez LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Interesuje nas tylko przewidywanie na podstawie OSTATNIEGO kroku czasowego z naszego okna
        out = out[:, -1, :]

        out = self.dropout(out)
        out = self.fc(out)  # Czysta ciągła wartość (Regresja)

        return out


if __name__ == "__main__":
    # Test Architektury (Sanity check)
    # Symulujemy batch 32 próbek, 36 kroków czasowych (6 godzin), 30 czujników
    dummy_input = torch.randn(32, 36, 30)

    # Inicjalizujemy nasz model Smart Home
    model = EnergyPredictorLSTM(input_size=30)

    # Wykonujemy testowy Forward Pass
    output = model(dummy_input)

    print("=== Test Architektury Modelu IoT ===")
    print(f"Kształt wejścia: {dummy_input.shape} -> (Batch, Długość Okna, Cechy)")
    print(f"Kształt wyjścia: {output.shape} -> Spodziewamy się [32, 1] (Regresja)")
    print(f"Przykładowe surowe wartości dla pierwszych 3 próbek:\n{output[:3].detach().numpy()}")