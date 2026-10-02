# Projekt z uczenia maszynowego — szablon

Z tego szablonu tworzysz własne repozytorium projektu przyciskiem **Use this template**.
**Nie rób forka.** Szablon nie zmienia się w trakcie semestru: nowe pliki dostajesz co tydzień
w repozytorium materiałów, a jego `README.md` mówi, gdzie w tym projekcie trafia każdy plik.

## Co jest w środku

```
├── pyproject.toml      # zależności Pythona (PyTorch na procesor)
├── uv.lock             # zablokowane wersje wszystkich pakietów
├── scripts/
│   ├── mlflow_local.sh #   W1: MLflow jako lokalna usługa, bez kontenerów
│   ├── check_stack.py  #   sprawdzenie: przebieg z plikiem i odczyt pliku z powrotem
│   ├── save_state.sh   #   przeniesienie stanu na inny komputer
│   └── restore_state.sh
├── compose.yaml        # od W2: MLflow z PostgreSQL i MinIO w kontenerach
├── docker/mlflow/      # od W2: obraz serwera MLflow z klientem S3
├── configs/            # konfiguracja danych i uczenia
├── src/mlproject/      # Twój kod: data/, models/, training/, evaluation/, tracking.py
├── exercises/          # ćwiczenia z kolejnych tygodni
├── notebooks/          # brudnopis; nie jest oceniany
├── report/             # raport
├── tests/
├── state/              # baza przebiegów i pliki przebiegów; nie trafia do Gita
└── data/               # zbiory danych; nie trafia do Gita
```

## Pierwsze uruchomienie (W1, bez Dockera)

Potrzebujesz: WSL 2 z Ubuntu albo Linuksa, Pythona 3.12 i `uv`. Pracuj w katalogu na dysku Linuksa
(`/home/...`), nie w `/mnt/c/...`.

```bash
cp .env.example .env
uv sync                        # środowisko z pyproject.toml i uv.lock
scripts/mlflow_local.sh        # DRUGI terminal: serwer MLflow, zostaw uruchomiony
uv run python scripts/check_stack.py
```

Jeżeli wszystko działa, `check_stack.py` wypisze (identyfikator przebiegu będzie inny):

```
🏃 View run stack-check at: http://127.0.0.1:5000/#/experiments/1/runs/d4c219698de34b1687d2f2f3178038b9
🧪 View experiment at: http://127.0.0.1:5000/#/experiments/1
tracking URI : http://127.0.0.1:5000
run id       : d4c219698de34b1687d2f2f3178038b9
artifact URI : mlflow-artifacts:/1/d4c219698de34b1687d2f2f3178038b9/artifacts
artifact     : If you can read this, artifacts reach the store.
```

**Na co zwrócić uwagę:** ostatnia linijka potwierdza, że plik został zapisany przez serwer i odczytany
z powrotem. Przebiegi otwierasz w przeglądarce pod `http://127.0.0.1:5000`.

| Gdzie co leży (W1) | Ścieżka |
|---|---|
| baza przebiegów (parametry, metryki) | `state/mlflow.db` (SQLite) |
| pliki przebiegów (wykresy, tabele, modele) | `state/mlartifacts/` |

## Od W2: ten sam MLflow w kontenerach

Od drugich zajęć serwer MLflow działa w kontenerze, przebiegi trafiają do **PostgreSQL**, a pliki
przebiegów do **MinIO** (magazyn zgodny z S3). Instrukcja i polecenia są w materiałach W2.
**Kod się nie zmienia** — identyfikatory plików przebiegów mają tę samą postać `mlflow-artifacts:/...`,
bo w obu wariantach pliki przechodzą przez serwer MLflow.

## Codzienna praca

- **W1:** `scripts/mlflow_local.sh` w osobnym terminalu na czas pracy; kończysz przez Ctrl+C.
- **Od W2:** `docker compose up -d` na start, `docker compose stop` na koniec.
- W każdym skrypcie, który zapisuje przebiegi:

  ```python
  from mlproject.tracking import setup

  setup("nazwa-eksperymentu")
  ```

- Skrypty uruchamiasz przez `uv run`, np. `uv run python scripts/train.py`.
- Notatnik służy do rozglądania się po danych i do odczytu zapisanych przebiegów. Jako jądro wybierz
  środowisko `.venv` projektu. Zmienne z `.env` wczytuje `uv run --env-file .env ...`.

## Przenoszenie stanu na inny komputer

Przebiegi leżą w `state/` na komputerze, przy którym pracujesz. Przy zmianie komputera, na przykład
stanowiska w pracowni, przenosisz je archiwum:

```bash
scripts/save_state.sh /mnt/d            # archiwum w $HOME i kopia na pendrive
scripts/save_state.sh /mnt/d --wipe     # wspólne konto w pracowni: po kopii usuń state/ ze stanowiska
scripts/restore_state.sh /mnt/d/ml-state_2026-10-02_1600.tar.gz
```

- Przed archiwizacją zatrzymaj serwer MLflow (Ctrl+C), inaczej skrypt przerwie pracę z komunikatem.
- Archiwum powstaje najpierw na dysku Linuksa, dopiero potem jest kopiowane na pendrive
  i sprawdzane sumą kontrolną.
- **Nie rozpakowuj archiwum na pendrivie.** Robi to `restore_state.sh` w katalogu projektu.
- `restore_state.sh` odmawia nadpisania istniejącego stanu. Nadpisanie wymaga `--force`.
- Ścieżka pendrive'a w WSL (`/mnt/d`) zależy od litery dysku w Windows.

Utrata `state/` nie przekreśla projektu, jeżeli jest odtwarzalny: skrypty odtwarzają przebiegi.

## Co nie trafia do Gita

`data/`, `state/` i `.env` są w `.gitignore`. Dane, wyniki, modele i hasła nie trafiają do repozytorium.

## Wspólny serwer MLflow

Jeżeli prowadzący udostępni wspólny serwer, zmieniasz w `.env` adres i dane logowania
(`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`). Kod się nie zmienia.

## Gdy coś nie działa

**`Connection refused` przy `127.0.0.1:5000`.** Serwer nie działa: uruchom `scripts/mlflow_local.sh`
w drugim terminalu (od W2: `docker compose up -d`).

**`Address already in use` przy starcie serwera.** Port 5000 zajmuje inny proces, na przykład serwer
z poprzedniego terminala. Znajdź go: `ss -ltnp | grep 5000`.

**`ModuleNotFoundError: No module named 'mlproject'`.** Polecenie bez `uv run` albo przed `uv sync`.

**Pobieranie artefaktu wisi przez wiele minut (od W2).** Skrypt nie importuje `mlproject.tracking`
albo notatnik nie ma zmiennych z `.env`. Serwer MLflow 3.16 z magazynem S3 kieruje klienta
bezpośrednio do MinIO pod adres `minio:9000`, który istnieje tylko wewnątrz sieci Dockera.
Zmienne `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` i `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false`
kierują pobieranie przez serwer MLflow.

**`Some files in state/ belong to another user` (od W2).** `LOCAL_UID` i `LOCAL_GID` w `.env` nie
zgadzają się z wynikiem `id -u` i `id -g`.
