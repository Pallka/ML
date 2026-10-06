#!/bin/bash
set -e
echo "Ініціалізація бази даних..."

if ! docker ps --format '{{.Names}}' | grep -qx mlops_database; then
    echo "Контейнер mlops_database не запущений. Спочатку виконайте: docker compose up --build"
    exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx mlops_models_api; then
    echo "Контейнер mlops_models_api не запущений. Спочатку виконайте: docker compose up --build"
    exit 1
fi

if [ ! -f "database/data/winequality.csv" ]; then
    echo "Файл database/data/winequality.csv не знайдено."
    exit 1
fi

echo "Завантаження CSV у PostgreSQL..."
docker exec mlops_models_api python scripts/load_data_to_db.py
echo "База даних ініціалізована."
