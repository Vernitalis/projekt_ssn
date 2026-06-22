import torch
import torch.nn as nn


class SP500PredictorLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, dropout_rate=0.2):
        super(SP500PredictorLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Warstwa LSTM
        # batch_first=True oznacza, że pierwszy wymiar to wielkość batcha (co zgadza się z naszymi danymi)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Dodatkowy Dropout przed warstwą klasyfikacyjną
        self.dropout = nn.Dropout(dropout_rate)

        # Warstwa w pełni połączona (Fully Connected)
        # Przekształca wektor ukryty (hidden_size) na 1 wartość wyjściową
        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, x):
        # Inicjalizacja ukrytych stanów (h0, c0) zerami.
        # Zapewniamy, że są na tym samym urządzeniu (CPU/GPU) co dane wejściowe.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Przepuszczamy dane przez LSTM
        # out ma kształt: (batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Interesuje nas tylko predykcja z ostatniego dnia sekwencji.
        # Pobieramy ostatni element z drugiego wymiaru (sequence_length)
        out = out[:, -1, :]

        # Przepuszczamy przez regularyzację i warstwy końcowe
        out = self.dropout(out)
        out = self.fc(out)

        return out


if __name__ == "__main__":
    # Testujemy architekturę dla 10 cech
    dummy_input = torch.randn(32, 60, 12)
    model = SP500PredictorLSTM(input_size=12)
    output = model(dummy_input)

    print("=== Test Architektury Modelu (10 Cech) ===")
    print(f"Kształt wejścia: {dummy_input.shape}")
    print(f"Kształt wyjścia: {output.shape}")