# MLOps: Wine Quality Prediction

**English** | [Українська](README.uk.md)

A containerized ML system that predicts red wine quality (`quality`, scale 0–10) from physicochemical features. It consists of three services: PostgreSQL for data storage, a FastAPI service for training and prediction, and a Streamlit web interface.

The dataset is [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (UCI Machine Learning Repository). A copy of the red wine subset is also available on [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009).

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Data and Preprocessing](#data-and-preprocessing)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)
- [Dataset Citation](#dataset-citation)

## Features

- Three regression models: `linear_regression`, `random_forest`, `gradient_boosting`
- Training on data stored in PostgreSQL, triggered via the API or the UI
- Evaluation with RMSE, MAE and R² on both training and test sets (`test_size=0.2`)
- Prediction from 11 physicochemical features, reusing the min–max parameters computed during training
- Web UI for training, prediction and viewing metrics
- One-command startup with Docker Compose

## Architecture

```text
Streamlit (8501)  →  Models API (8000)  →  PostgreSQL (5432)
                         ↓
           artifacts/*.pkl + metadata
```

| Service | Container | Port | Purpose |
|---|---|---|---|
| `database` | `mlops_database` | 5432 | `wine_data` table, training dataset |
| `models_api` | `mlops_models_api` | 8000 | Training, prediction, metrics |
| `streamlit_app` | `mlops_streamlit` | 8501 | User interface |

## Tech Stack

- **Database:** PostgreSQL
- **Backend:** Python 3.11, FastAPI, scikit-learn, SQLAlchemy
- **Frontend:** Streamlit, Plotly
- **Infrastructure:** Docker, Docker Compose

## Prerequisites

- Docker Desktop
- Docker Compose

The dataset file `database/data/winequality.csv` is already in the repository (`;` delimiter), so no separate download is needed.

## Quick Start

Run all commands from the project root (`mlops-wine-quality-prediction`).

**1. (Optional) Create a local configuration file**

The stack starts with built-in defaults, so you can skip this step. To use your own database credentials, copy the template and edit it (see [Configuration](#configuration)):

```bash
cp .env.example .env
```

On Windows (cmd): `copy .env.example .env`

**2. Build and start the services**

```bash
docker compose up --build
```

This keeps running in the terminal. Open a second terminal for the next steps, or add `-d` to run in the background.

**3. Load the dataset into the database**

Once the containers are up:

- Windows (PowerShell):

  ```powershell
  .\init_database.ps1
  ```

- Linux / macOS (if you get "permission denied", run `bash init_database.sh` instead):

  ```bash
  ./init_database.sh
  ```

- Or manually (any OS):

  ```bash
  docker exec mlops_models_api python scripts/load_data_to_db.py
  ```

If the table is already populated, the script does not insert anything again.

**4. Open the interfaces**

| Interface | URL |
|---|---|
| Web UI | http://localhost:8501 |
| API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| PostgreSQL | `localhost:5432` |

**Stopping**

```bash
docker compose down
```

The PostgreSQL volume is preserved. To also wipe the database data:

```bash
docker compose down -v
```

## Usage

1. Open http://localhost:8501.
2. **Model Training** page (*Тренування моделей*): train at least one model.
3. **Prediction** page (*Прогнозування*): enter the wine's features and get a quality estimate.
4. **Model Metrics** page (*Метрики моделей*): compare RMSE / MAE / R².

Prediction uses the same min–max normalization parameters that were computed during training.

## Configuration

Database credentials are read from environment variables or from a `.env` file next to `docker-compose.yml` (create it from `.env.example`). Every variable has a default, so `.env` is optional.

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_USER` | `mlops_user` | Database user (shared by `database` and `models_api`) |
| `POSTGRES_PASSWORD` | `mlops_password` | Database password |
| `POSTGRES_DB` | `mlops_db` | Database name |

The API container also reads `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD` and `DB_NAME` (filled in by `docker-compose.yml` from the variables above) and `MODELS_DIR` (where trained models are stored; default `artifacts`, i.e. `models_api/artifacts/` on the host).

> **Notes**
>
> - The defaults are meant for the **local learning environment only**. Do not reuse them anywhere else.
> - PostgreSQL applies these credentials only when the database volume is first created. After changing them, reset the volume with `docker compose down -v` and load the dataset again.

Open a `psql` session in the database container:

```bash
docker exec -it mlops_database sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status |
| `GET` | `/models` | `available_models` (all algorithms) and `trained_models` (already saved `.pkl` files) |
| `POST` | `/train/{model_name}` | Train on data from PostgreSQL |
| `POST` | `/predict/{model_name}` | Predict from 11 features |
| `GET` | `/metrics/{model_name}` | Metrics and training date |

`model_name` is one of: `linear_regression`, `random_forest`, `gradient_boosting`.

Every input feature has a valid range (shown in the interactive documentation). Values outside it are rejected with HTTP `422`, and predicting with a model that has not been trained returns `404`.

Interactive documentation is available at http://localhost:8000/docs.

### Example: prediction request

Linux / macOS:

```bash
curl -X POST http://localhost:8000/predict/random_forest \
  -H "Content-Type: application/json" \
  -d '{"fixed_acidity":7.4,"volatile_acidity":0.7,"citric_acid":0.0,"residual_sugar":1.9,"chlorides":0.076,"free_sulfur_dioxide":11,"total_sulfur_dioxide":34,"density":0.9978,"pH":3.51,"sulphate":0.56,"alcohol":9.4}'
```

Windows (cmd):

```bat
curl -X POST http://localhost:8000/predict/random_forest ^
  -H "Content-Type: application/json" ^
  -d "{\"fixed_acidity\":7.4,\"volatile_acidity\":0.7,\"citric_acid\":0.0,\"residual_sugar\":1.9,\"chlorides\":0.076,\"free_sulfur_dioxide\":11,\"total_sulfur_dioxide\":34,\"density\":0.9978,\"pH\":3.51,\"sulphate\":0.56,\"alcohol\":9.4}"
```

> **Note:** in the JSON the sulphate field is called `sulphate`, while in the CSV it is `sulphates`.

## Data and Preprocessing

Expected CSV columns: `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`, `quality`.

The `wine_data` table: `fixed_acidity`, `volatile_acidity`, `citric_acid`, `residual_sugar`, `chlorides`, `free_sulfur_dioxide`, `total_sulfur_dioxide`, `density`, `ph`, `sulphate`, `alcohol`, `quality`.

Preprocessing applied during training:

1. Duplicate removal
2. Outlier removal using the 3σ rule
3. Min–max normalization of features

## Project Structure

```text
mlops-wine-quality-prediction/
├── database/
│   ├── Dockerfile
│   ├── init.sql
│   └── data/
│       └── winequality.csv
├── models_api/
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── requirements.txt
│   ├── app.py
│   ├── artifacts/              # trained models (*.pkl), git-ignored
│   │   └── .gitkeep
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_trainer.py
│   │   └── model_predictor.py
│   ├── scripts/
│   │   └── load_data_to_db.py
│   └── utils/
│       ├── __init__.py
│       └── data_processor.py
├── streamlit_app/
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── requirements.txt
│   └── app.py
├── docker-compose.yml
├── .env.example
├── .gitignore
├── init_database.ps1
├── init_database.sh
├── README.md
└── README.uk.md
```

## Troubleshooting

- **Training fails with an empty or unavailable database:** make sure the dataset was loaded (step 3 of [Quick Start](#quick-start)).
- **Prediction returns 404:** the model must be trained first. Check `GET /models` and look for it in `trained_models`.
- **A port is already in use:** stop whatever is using 5432, 8000 or 8501, or change the port mapping in `docker-compose.yml`.
- **Trained models are missing after an update:** models are now stored in `models_api/artifacts/`. Models saved by earlier versions in `models_api/models/` are not picked up, so train them again. Retraining is also needed after upgrading scikit-learn, because pickled models are tied to the library version.
- **Database login fails after changing the credentials:** PostgreSQL keeps the credentials from the first start. Run `docker compose down -v` and start again.
- **Need a clean slate:** run `docker compose down -v` and start again from [Quick Start](#quick-start).

## Development Notes

`docker-compose.yml` is a development setup: the source folders are bind-mounted into the containers and the API runs with `--reload`, so code changes are applied without rebuilding the images. Trained models therefore appear on the host in `models_api/artifacts/`.

The following are **not** committed (see `.gitignore`):

- `__pycache__/`, `*.pyc`
- `venv/`, `.venv/`
- `.env` (the template `.env.example` is committed)
- trained models: `models_api/artifacts/*` and any `*.pkl`
- IDE and OS service files

The CSV `database/data/winequality.csv` is committed. The `.dockerignore` files keep caches, local environments and trained models out of the Docker build context.

Python dependencies are pinned in `models_api/requirements.txt` and `streamlit_app/requirements.txt`; the Docker images use Python 3.11.

## Dataset Citation

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4):547–553, 2009.

Sources:

- Original: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Mirror: [Red Wine Quality on Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) (shared for convenience; not the original publisher)
