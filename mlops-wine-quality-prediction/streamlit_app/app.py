import os

import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="MLOps Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🍷 Система прогнозування якості вина")
st.markdown("---")

st.sidebar.title("Навігація")
page = st.sidebar.radio(
    "Виберіть сторінку:",
    ["Прогнозування", "Тренування моделей", "Метрики моделей", "Інформація"],
)


def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_models():
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("available_models", []), data.get("trained_models", [])
    except requests.RequestException:
        pass
    return [], []


def get_model_metrics(model_name):
    try:
        response = requests.get(f"{API_URL}/metrics/{model_name}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        st.error(f"Помилка: {str(e)}")
    return None


if not check_api_health():
    st.error("API сервіс недоступний. Запустіть docker compose і переконайтеся, що models_api працює.")
    st.stop()

available_models, trained_models = get_models()

if page == "Прогнозування":
    st.header("Прогнозування якості вина")

    if not trained_models:
        st.warning("Немає навчених моделей. Спочатку навчіть модель на сторінці «Тренування моделей».")
    else:
        selected_model = st.selectbox("Модель для прогнозу:", trained_models)
        st.markdown("### Параметри вина")
        col1, col2 = st.columns(2)

        with col1:
            fixed_acidity = st.number_input("Фіксована кислотність", 4.0, 16.0, 7.4, 0.1)
            volatile_acidity = st.number_input("Летюча кислотність", 0.1, 2.0, 0.7, 0.01)
            citric_acid = st.number_input("Лимонна кислота", 0.0, 1.0, 0.0, 0.01)
            residual_sugar = st.number_input("Залишковий цукор", 0.5, 15.0, 1.9, 0.1)
            chlorides = st.number_input("Хлориди", 0.01, 0.6, 0.076, 0.001)
            free_sulfur_dioxide = st.number_input("Вільний діоксид сірки", 1.0, 72.0, 11.0, 1.0)

        with col2:
            total_sulfur_dioxide = st.number_input("Загальний діоксид сірки", 6.0, 289.0, 34.0, 1.0)
            density = st.number_input("Густина", 0.99, 1.004, 0.9978, 0.0001, format="%.4f")
            pH = st.number_input("pH", 2.7, 4.0, 3.51, 0.01)
            sulphate = st.number_input("Сульфати", 0.2, 2.0, 0.56, 0.01)
            alcohol = st.number_input("Алкоголь", 8.0, 15.0, 9.4, 0.1)

        if st.button("Зробити прогноз", type="primary"):
            with st.spinner("Виконується прогнозування..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/{selected_model}",
                        json={
                            "fixed_acidity": fixed_acidity,
                            "volatile_acidity": volatile_acidity,
                            "citric_acid": citric_acid,
                            "residual_sugar": residual_sugar,
                            "chlorides": chlorides,
                            "free_sulfur_dioxide": free_sulfur_dioxide,
                            "total_sulfur_dioxide": total_sulfur_dioxide,
                            "density": density,
                            "pH": pH,
                            "sulphate": sulphate,
                            "alcohol": alcohol,
                        },
                        timeout=10,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result["prediction"]
                        prediction_rounded = result["prediction_rounded"]
                        st.success("Прогноз виконано.")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Прогнозована якість", f"{prediction_rounded}/10")
                        m2.metric("Точне значення", f"{prediction:.2f}")
                        quality_level = (
                            "Відмінна" if prediction_rounded >= 7
                            else "Добра" if prediction_rounded >= 5
                            else "Середня"
                        )
                        m3.metric("Рівень якості", quality_level)
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction_rounded,
                            title={"text": "Якість вина"},
                            gauge={
                                "axis": {"range": [0, 10]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 5], "color": "lightgray"},
                                    {"range": [5, 7], "color": "gray"},
                                    {"range": [7, 10], "color": "lightblue"},
                                ],
                            },
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.error(response.json().get("detail", "Невідома помилка"))
                except requests.RequestException as e:
                    st.error(f"Помилка під час прогнозування: {str(e)}")

elif page == "Тренування моделей":
    st.header("Тренування моделей")
    st.info(
        """
        Доступні моделі:
        - **linear_regression** — лінійна регресія
        - **random_forest** — випадковий ліс
        - **gradient_boosting** — градієнтний бустинг
        """
    )
    if not available_models:
        st.warning("Список моделей недоступний.")
    else:
        selected_model = st.selectbox("Модель для тренування:", available_models)
        if st.button("Навчити модель", type="primary"):
            with st.spinner(f"Тренування {selected_model}..."):
                try:
                    response = requests.post(
                        f"{API_URL}/train/{selected_model}",
                        timeout=300,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Модель навчена.")
                        metadata = result.get("metadata", {})
                        train_metrics = metadata["metrics"]["train"]
                        test_metrics = metadata["metrics"]["test"]
                        c1, c2 = st.columns(2)
                        with c1:
                            st.subheader("Навчальна вибірка")
                            st.metric("RMSE", f"{train_metrics['rmse']:.4f}")
                            st.metric("MAE", f"{train_metrics['mae']:.4f}")
                            st.metric("R²", f"{train_metrics['r2']:.4f}")
                        with c2:
                            st.subheader("Тестова вибірка")
                            st.metric("RMSE", f"{test_metrics['rmse']:.4f}")
                            st.metric("MAE", f"{test_metrics['mae']:.4f}")
                            st.metric("R²", f"{test_metrics['r2']:.4f}")
                        st.caption(f"Дата тренування: {metadata['training_date']}")
                    else:
                        st.error(response.json().get("detail", "Невідома помилка"))
                except requests.RequestException as e:
                    st.error(f"Помилка під час тренування: {str(e)}")

elif page == "Метрики моделей":
    st.header("Метрики моделей")
    if not trained_models:
        st.warning("Немає навчених моделей.")
    else:
        selected_model = st.selectbox("Модель:", trained_models)
        if st.button("Завантажити метрики"):
            metrics = get_model_metrics(selected_model)
            if metrics:
                c1, c2, c3 = st.columns(3)
                c1.metric("Алгоритм", metrics["algorithm"])
                c2.metric("Дата тренування", metrics["training_date"][:10])
                c3.metric("Навчальних зразків", metrics["training_info"]["train_samples"])
                train_metrics = metrics["metrics"]["train"]
                test_metrics = metrics["metrics"]["test"]
                left, right = st.columns(2)
                with left:
                    st.subheader("Навчальна вибірка")
                    fig_train = go.Figure(data=[go.Bar(
                        x=["rmse", "mae", "r2"],
                        y=[train_metrics["rmse"], train_metrics["mae"], train_metrics["r2"]],
                        marker_color="lightblue",
                    )])
                    st.plotly_chart(fig_train, width="stretch")
                    st.write(f"**RMSE:** {train_metrics['rmse']:.4f}")
                    st.write(f"**MAE:** {train_metrics['mae']:.4f}")
                    st.write(f"**R²:** {train_metrics['r2']:.4f}")
                with right:
                    st.subheader("Тестова вибірка")
                    fig_test = go.Figure(data=[go.Bar(
                        x=["rmse", "mae", "r2"],
                        y=[test_metrics["rmse"], test_metrics["mae"], test_metrics["r2"]],
                        marker_color="lightgreen",
                    )])
                    st.plotly_chart(fig_test, width="stretch")
                    st.write(f"**RMSE:** {test_metrics['rmse']:.4f}")
                    st.write(f"**MAE:** {test_metrics['mae']:.4f}")
                    st.write(f"**R²:** {test_metrics['r2']:.4f}")

elif page == "Інформація":
    st.header("Про систему")
    st.markdown("""
    Три контейнеризовані сервіси:

    **База даних (PostgreSQL)** — навчальний набір Wine Quality (червоне вино).

    **Models API (FastAPI)** — тренування і прогноз:
    - Linear Regression
    - Random Forest
    - Gradient Boosting

    **Інтерфейс (Streamlit)** — введення ознак, тренування, метрики.

    **Задача:** регресія, ціль — `quality` (0–10).

    **Ознаки:** fixed acidity, volatile acidity, citric acid, residual sugar,
    chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.

    **Метрики:** RMSE, MAE, R².
    """)
    st.markdown(f"**API URL:** `{API_URL}`")
