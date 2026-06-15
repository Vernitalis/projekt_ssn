import sys
import subprocess
import time


def print_header():
    print("\n" + "=" * 60)
    print("🚀 QUANTITATIVE FINANCE ML PIPELINE (S&P 500)")
    print("=" * 60)


def run_optimization():
    print("\n[Uruchamiam moduł Optuna - Poszukiwanie hiperparametrów...]")
    time.sleep(1)
    # Wywołujemy skrypt optymalizacji w osobnym procesie
    subprocess.run([sys.executable, "optimize.py"])


def run_training():
    print("\n[Uruchamiam moduł Treningowy - Walk-Forward Validation...]")
    time.sleep(1)
    # Importujemy i uruchamiamy funkcję z train.py bez tworzenia nowego procesu
    from train import train_model
    train_model()


def run_dashboard():
    print("\n[Uruchamiam aplikację produkcyjną Streamlit...]")
    print("Przeglądarka powinna otworzyć się automatycznie.")
    time.sleep(1)
    # Streamlit musi być uruchomiony przez własny moduł
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def main_menu():
    while True:
        print_header()
        print("Wybierz moduł do uruchomienia:")
        print("  1. 🧠 PEŁNY PIPELINE (Trening Walk-Forward -> Start Dashboardu)")
        print("  2. 🔬 Optymalizacja Modelu (Optuna)")
        print("  3. 🏋️ Tylko Trening Modelu (Zapisanie nowych wag)")
        print("  4. 📊 Tylko Dashboard (Wnioskowanie na żywo)")
        print("  0. ❌ Wyjście")
        print("-" * 60)

        choice = input("Twój wybór (0-4): ").strip()

        if choice == '1':
            try:
                run_training()
                run_dashboard()
            except Exception as e:
                print(f"\n❌ Błąd podczas wykonywania pipeline'u: {e}")
        elif choice == '2':
            run_optimization()
        elif choice == '3':
            run_training()
        elif choice == '4':
            run_dashboard()
        elif choice == '0':
            print("\nZamykanie systemu. Do zobaczenia!")
            sys.exit(0)
        else:
            print("\n⚠️ Nieprawidłowy wybór. Wpisz cyfrę od 0 do 4.")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        # Obsługa wciśnięcia CTRL+C, by aplikacja zamknęła się z klasą
        print("\n\nPrzerwano działanie programu (CTRL+C). Zamykanie...")
        sys.exit(0)