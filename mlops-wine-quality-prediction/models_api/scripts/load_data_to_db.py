#!/usr/bin/env python3
"""Завантаження Wine Quality CSV у таблицю wine_data."""
import os
import sys

import pandas as pd
import psycopg2

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "database"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "mlops_user"),
    "password": os.getenv("DB_PASSWORD", "mlops_password"),
    "database": os.getenv("DB_NAME", "mlops_db"),
}

CSV_PATHS = [
    "/data/winequality.csv",
    "/tmp/winequality.csv",
    "../database/data/winequality.csv",
    "database/data/winequality.csv",
]


def load_data():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'wine_data'
            );
        """)
        if not cursor.fetchone()[0]:
            cursor.execute("""
                CREATE TABLE wine_data (
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
            """)
            conn.commit()

        cursor.execute("SELECT COUNT(*) FROM wine_data;")
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"Таблиця вже містить {count} записів. Пропуск завантаження.")
            return

        csv_path = next((path for path in CSV_PATHS if os.path.exists(path)), None)
        if not csv_path:
            raise FileNotFoundError(
                f"Файл winequality.csv не знайдено. Перевірені шляхи: {CSV_PATHS}"
            )

        df = pd.read_csv(csv_path, sep=";", quotechar='"')
        df = df.rename(
            columns={
                "fixed acidity": "fixed_acidity",
                "volatile acidity": "volatile_acidity",
                "citric acid": "citric_acid",
                "residual sugar": "residual_sugar",
                "free sulfur dioxide": "free_sulfur_dioxide",
                "total sulfur dioxide": "total_sulfur_dioxide",
                "sulphates": "sulphate",
            }
        )

        insert_data = [
            (
                row["fixed_acidity"],
                row["volatile_acidity"],
                row["citric_acid"],
                row["residual_sugar"],
                row["chlorides"],
                row["free_sulfur_dioxide"],
                row["total_sulfur_dioxide"],
                row["density"],
                row["pH"],
                row["sulphate"],
                row["alcohol"],
                row["quality"],
            )
            for _, row in df.iterrows()
        ]
        cursor.executemany(
            """
            INSERT INTO wine_data (
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                ph, sulphate, alcohol, quality
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            insert_data,
        )
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM wine_data;")
        print(f"Успішно завантажено {cursor.fetchone()[0]} записів.")
    except Exception as e:
        print(f"Помилка: {e}")
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    load_data()
