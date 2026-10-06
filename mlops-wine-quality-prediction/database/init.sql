-- Таблиця навчальних даних про якість червоного вина
CREATE TABLE IF NOT EXISTS wine_data (
    id SERIAL PRIMARY KEY,
    fixed_acidity FLOAT,
    volatile_acidity FLOAT,
    citric_acid FLOAT,
    residual_sugar FLOAT,
    chlorides FLOAT,
    free_sulfur_dioxide FLOAT,
    total_sulfur_dioxide FLOAT,
    density FLOAT,
    ph FLOAT,
    sulphate FLOAT,
    alcohol FLOAT,
    quality INTEGER
);

CREATE INDEX IF NOT EXISTS idx_wine_quality ON wine_data(quality);
CREATE INDEX IF NOT EXISTS idx_wine_alcohol ON wine_data(alcohol);

COMMENT ON TABLE wine_data IS 'Wine Quality Dataset (red wine) для навчання регресійних моделей';
COMMENT ON COLUMN wine_data.quality IS 'Цільова змінна: якість вина (0-10)';
COMMENT ON COLUMN wine_data.sulphate IS 'Сульфати (у датасеті колонка sulphates)';
