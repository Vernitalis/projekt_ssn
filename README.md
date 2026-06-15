# 🚀 Deep Learning Time Series Forecasting: S&P 500 & Smart Home IoT

Projekt ten zawiera zautomatyzowany rurociąg (pipeline) uczenia maszynowego oparty na architekturze **Multivariate LSTM** (PyTorch). Służy do predykcji wielowymiarowych szeregów czasowych z wykorzystaniem rygorystycznej walidacji czasowej oraz niestandardowych miar błędów (Custom Loss Functions). 

Projekt obejmuje dwa środowiska o różnej charakterystyce:
1. **Zaszumione dane finansowe:** Prognozowanie kierunku oraz zwrotu indeksu S&P 500 w oparciu o kontekst makroekonomiczny (VIX, DXY, TNX) i wskaźniki analizy technicznej.
2. **Cykliczne dane z sensorów IoT:** Prognozowanie fizycznego zapotrzebowania na prąd (Watogodziny) w budynku Smart Home z wykorzystaniem cyklicznego kodowania czasu (Sin-Cos Encoding).

## 🗂 Architektura Projektu (SSOT)

System został zaprojektowany zgodnie z dobrymi praktykami inżynierii oprogramowania (Single Source of Truth). Architektura dzieli się na niezależne moduły:

* `config.py` - Główny plik konfiguracyjny z hiperparametrami sieci, ustawieniami treningu i mnożnikami funkcji kary.
* `data_loader.py` - Skrypt pobierający dane (z API Yahoo Finance lub zbioru UCI) oraz przeprowadzający zaawansowany *Feature Engineering* (np. inżynieria daty, transformacje Min-Max).
* `model.py` - Deklaracja klas i warstw głębokiej sieci rekurencyjnej LSTM napisanej w PyTorch (z usuniętą warstwą aktywacyjną pod regresję ciągłą).
* `train.py` - Główna pętla ucząca z zastosowaniem **Walk-Forward Validation** (rozszerzające się okno czasowe) oraz asymetrycznymi funkcjami straty (*Directional Loss* i *Asymmetric Energy Loss*).
* `optimize.py` - Skrypt automatycznej optymalizacji hiperparametrów za pomocą statystyki bayesowskiej (Optuna).
* `app.py` - Produkcyjny panel telemetryczny / giełdowy uruchamiany w Streamlit, z interaktywnymi wykresami Plotly i predykcjami na żywo.
* `main.py` - Orkiestrator (CLI) spinający wszystkie powyższe moduły w wygodne, interaktywne menu.

## Uruchomienie
```bash
pip install -r requirements.txt
python gielda/main.py || python energia/main.py
```