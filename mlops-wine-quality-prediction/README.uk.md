# MLOps: прогнозування якості вина

[English](README.md) | **Українська**

Контейнеризована ML-система, яка прогнозує якість червоного вина (`quality`, шкала 0–10) за фізико-хімічними ознаками. Складається з трьох сервісів: PostgreSQL для зберігання даних, FastAPI для тренування і прогнозу та веб-інтерфейсу на Streamlit.

Датасет — [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (UCI Machine Learning Repository). Копія підмножини з червоним вином також є на [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009).

## Зміст

- [Можливості](#можливості)
- [Архітектура](#архітектура)
- [Стек технологій](#стек-технологій)
- [Вимоги](#вимоги)
- [Швидкий старт](#швидкий-старт)
- [Використання](#використання)
- [Конфігурація](#конфігурація)
- [API](#api)
- [Дані та попередня обробка](#дані-та-попередня-обробка)
- [Структура проєкту](#структура-проєкту)
- [Усунення проблем](#усунення-проблем)
- [Нотатки для розробки](#нотатки-для-розробки)
- [Джерело датасету](#джерело-датасету)

## Можливості

- Три регресійні моделі: `linear_regression`, `random_forest`, `gradient_boosting`
- Тренування на даних із PostgreSQL через API або інтерфейс
- Оцінка за RMSE, MAE та R² на навчальній і тестовій вибірках (`test_size=0.2`)
- Прогноз за 11 фізико-хімічними ознаками з тими самими min–max параметрами, що були пораховані під час тренування
- Веб-інтерфейс для тренування, прогнозу та перегляду метрик
- Запуск однією командою через Docker Compose

## Архітектура

```text
Streamlit (8501)  →  Models API (8000)  →  PostgreSQL (5432)
                         ↓
           artifacts/*.pkl + metadata
```

| Сервіс | Контейнер | Порт | Призначення |
|---|---|---|---|
| `database` | `mlops_database` | 5432 | Таблиця `wine_data`, навчальна вибірка |
| `models_api` | `mlops_models_api` | 8000 | Тренування, прогноз, метрики |
| `streamlit_app` | `mlops_streamlit` | 8501 | Інтерфейс користувача |

## Стек технологій

- **База даних:** PostgreSQL
- **Бекенд:** Python 3.11, FastAPI, scikit-learn, SQLAlchemy
- **Фронтенд:** Streamlit, Plotly
- **Інфраструктура:** Docker, Docker Compose

## Вимоги

- Docker Desktop
- Docker Compose

Файл `database/data/winequality.csv` уже є в репозиторії (роздільник `;`), окремо завантажувати датасет не потрібно.

## Швидкий старт

Усі команди виконуйте з кореня проєкту (`mlops-wine-quality-prediction`).

**1. (Необов'язково) Створіть локальний файл конфігурації**

Стенд запускається і з вбудованими значеннями за замовчуванням, тож цей крок можна пропустити. Щоб використати власні облікові дані БД, скопіюйте шаблон і відредагуйте його (див. [Конфігурація](#конфігурація)):

```bash
cp .env.example .env
```

У Windows (cmd): `copy .env.example .env`

**2. Зберіть і запустіть сервіси**

```bash
docker compose up --build
```

Команда займає термінал. Для наступних кроків відкрийте другий термінал або додайте `-d`, щоб запустити у фоні.

**3. Завантажте датасет у базу даних**

Коли контейнери запустяться:

- Windows (PowerShell):

  ```powershell
  .\init_database.ps1
  ```

- Linux / macOS (якщо з'явиться «permission denied», виконайте `bash init_database.sh`):

  ```bash
  ./init_database.sh
  ```

- Або вручну (будь-яка ОС):

  ```bash
  docker exec mlops_models_api python scripts/load_data_to_db.py
  ```

Якщо таблиця вже заповнена, скрипт повторно нічого не вставляє.

**4. Відкрийте інтерфейси**

| Інтерфейс | Адреса |
|---|---|
| Веб-інтерфейс | http://localhost:8501 |
| API | http://localhost:8000 |
| Документація API (Swagger) | http://localhost:8000/docs |
| PostgreSQL | `localhost:5432` |

**Зупинка**

```bash
docker compose down
```

Том PostgreSQL зберігається. Щоб повністю очистити дані БД:

```bash
docker compose down -v
```

## Використання

1. Відкрийте http://localhost:8501.
2. Сторінка **Тренування моделей**: навчіть хоча б одну модель.
3. Сторінка **Прогнозування**: введіть ознаки вина й отримайте оцінку якості.
4. Сторінка **Метрики моделей**: порівняйте RMSE / MAE / R².

Прогноз використовує ті самі параметри min–max нормалізації, що були пораховані під час тренування.

## Конфігурація

Облікові дані БД читаються зі змінних середовища або з файлу `.env` поруч із `docker-compose.yml` (створіть його з `.env.example`). Кожна змінна має значення за замовчуванням, тож `.env` необов'язковий.

| Змінна | За замовчуванням | Призначення |
|---|---|---|
| `POSTGRES_USER` | `mlops_user` | Користувач БД (спільний для `database` і `models_api`) |
| `POSTGRES_PASSWORD` | `mlops_password` | Пароль БД |
| `POSTGRES_DB` | `mlops_db` | Назва бази |

Контейнер API також читає `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD` і `DB_NAME` (їх підставляє `docker-compose.yml` зі змінних вище) та `MODELS_DIR` (де зберігаються навчені моделі; за замовчуванням `artifacts`, тобто `models_api/artifacts/` на хості).

> **Зверніть увагу**
>
> - Значення за замовчуванням призначені **лише для локального навчального стенда**. Не використовуйте їх деінде.
> - PostgreSQL застосовує ці облікові дані лише при першому створенні тому БД. Після їх зміни скиньте том командою `docker compose down -v` і знову завантажте датасет.

Відкрити сесію `psql` у контейнері бази даних:

```bash
docker exec -it mlops_database sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

## API

| Метод | Шлях | Опис |
|---|---|---|
| `GET` | `/health` | Стан сервісу |
| `GET` | `/models` | `available_models` (усі алгоритми) і `trained_models` (уже збережені `.pkl`) |
| `POST` | `/train/{model_name}` | Навчання на даних з PostgreSQL |
| `POST` | `/predict/{model_name}` | Прогноз за 11 ознаками |
| `GET` | `/metrics/{model_name}` | Метрики та дата тренування |

`model_name`: `linear_regression` | `random_forest` | `gradient_boosting`

Кожна вхідна ознака має допустимий діапазон (він показаний в інтерактивній документації). Значення поза діапазоном відхиляються з HTTP `422`, а прогноз моделлю, яку не навчали, повертає `404`.

Інтерактивна документація: http://localhost:8000/docs.

### Приклад запиту на прогноз

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

> **Примітка:** у JSON поле сульфатів називається `sulphate`, а в CSV — `sulphates`.

## Дані та попередня обробка

Очікувані колонки CSV: `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`, `quality`.

Таблиця `wine_data`: `fixed_acidity`, `volatile_acidity`, `citric_acid`, `residual_sugar`, `chlorides`, `free_sulfur_dioxide`, `total_sulfur_dioxide`, `density`, `ph`, `sulphate`, `alcohol`, `quality`.

Під час тренування виконується:

1. Видалення дублікатів
2. Відсікання викидів за правилом 3σ
3. Min–max нормалізація ознак

## Структура проєкту

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
│   ├── artifacts/              # навчені моделі (*.pkl), ігноруються git
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

## Усунення проблем

- **Тренування завершується помилкою про порожню або недоступну БД:** переконайтеся, що датасет завантажено (крок 3 у [Швидкому старті](#швидкий-старт)).
- **Прогноз повертає 404:** модель спочатку треба навчити. Перевірте `GET /models` і знайдіть її в `trained_models`.
- **Порт уже зайнятий:** зупиніть програму, що використовує 5432, 8000 або 8501, або змініть мапінг портів у `docker-compose.yml`.
- **Навчені моделі зникли після оновлення:** тепер моделі зберігаються в `models_api/artifacts/`. Моделі, збережені старішими версіями в `models_api/models/`, не підхоплюються, тому навчіть їх заново. Повторне тренування потрібне й після оновлення scikit-learn, бо збережені моделі прив'язані до версії бібліотеки.
- **Не вдається увійти в БД після зміни облікових даних:** PostgreSQL зберігає облікові дані з першого запуску. Виконайте `docker compose down -v` і запустіть знову.
- **Потрібен чистий стан:** виконайте `docker compose down -v` і почніть знову зі [Швидкого старту](#швидкий-старт).

## Нотатки для розробки

`docker-compose.yml` — це конфігурація для розробки: каталоги з кодом підключені до контейнерів, а API працює з `--reload`, тому зміни в коді застосовуються без перезбирання образів. Через це навчені моделі з'являються на хості в `models_api/artifacts/`.

У коміт **не** потрапляють (див. `.gitignore`):

- `__pycache__/`, `*.pyc`
- `venv/`, `.venv/`
- `.env` (шаблон `.env.example` комітиться)
- навчені моделі: `models_api/artifacts/*` та будь-які `*.pkl`
- службові файли IDE та ОС

CSV `database/data/winequality.csv` комітиться. Файли `.dockerignore` не пускають у контекст збірки Docker кеші, локальні середовища та навчені моделі.

Залежності Python зафіксовані в `models_api/requirements.txt` і `streamlit_app/requirements.txt`; образи Docker використовують Python 3.11.

## Джерело датасету

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4):547–553, 2009.

Джерела:

- Оригінал: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Дзеркало: [Red Wine Quality на Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) (викладено для зручності; не є оригінальним видавцем)
