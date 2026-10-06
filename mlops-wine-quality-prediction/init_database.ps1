Write-Host "Ініціалізація бази даних..." -ForegroundColor Green

$dbRunning = docker ps --filter "name=mlops_database" --format "{{.Names}}" | Select-String "mlops_database"
$apiRunning = docker ps --filter "name=mlops_models_api" --format "{{.Names}}" | Select-String "mlops_models_api"

if (-not $dbRunning) {
    Write-Host "Контейнер mlops_database не запущений. Спочатку виконайте: docker compose up --build" -ForegroundColor Red
    exit 1
}

if (-not $apiRunning) {
    Write-Host "Контейнер mlops_models_api не запущений. Спочатку виконайте: docker compose up --build" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "database\data\winequality.csv")) {
    Write-Host "Файл database\data\winequality.csv не знайдено." -ForegroundColor Yellow
    exit 1
}

Write-Host "Завантаження CSV у PostgreSQL..." -ForegroundColor Cyan
docker exec mlops_models_api python scripts/load_data_to_db.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "База даних ініціалізована." -ForegroundColor Green
} else {
    Write-Host "Помилка ініціалізації бази даних." -ForegroundColor Red
    exit 1
}
