from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models.model_trainer import ModelTrainer
from models.model_predictor import ModelPredictor

app = FastAPI(
    title="MLOps Wine Quality API",
    description="API для тренування моделей і прогнозу якості червоного вина",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trainer = ModelTrainer()
predictor = ModelPredictor()


class PredictionInput(BaseModel):
    fixed_acidity: float = Field(..., ge=4.0, le=16.0, description="Фіксована кислотність")
    volatile_acidity: float = Field(..., ge=0.1, le=2.0, description="Летюча кислотність")
    citric_acid: float = Field(..., ge=0.0, le=1.0, description="Лимонна кислота")
    residual_sugar: float = Field(..., ge=0.5, le=15.0, description="Залишковий цукор")
    chlorides: float = Field(..., ge=0.01, le=0.6, description="Хлориди")
    free_sulfur_dioxide: float = Field(..., ge=1.0, le=72.0, description="Вільний діоксид сірки")
    total_sulfur_dioxide: float = Field(..., ge=6.0, le=289.0, description="Загальний діоксид сірки")
    density: float = Field(..., ge=0.99, le=1.004, description="Густина")
    pH: float = Field(..., ge=2.7, le=4.0, description="pH")
    sulphate: float = Field(..., ge=0.2, le=2.0, description="Сульфати")
    alcohol: float = Field(..., ge=8.0, le=15.0, description="Алкоголь")


@app.get("/")
async def root():
    return {
        "message": "MLOps Wine Quality API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "train": "/train/{model_name}",
            "predict": "/predict/{model_name}",
            "metrics": "/metrics/{model_name}"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "models_api"}


@app.get("/models")
async def get_models():
    available = trainer.get_available_models()
    trained = predictor.get_trained_models()
    return {
        "available_models": available,
        "trained_models": trained,
        "count": len(available)
    }


@app.post("/train/{model_name}")
async def train_model(model_name: str):
    try:
        return trainer.train_model(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка тренування: {str(e)}")


@app.post("/predict/{model_name}")
async def predict(model_name: str, input_data: PredictionInput):
    try:
        return predictor.predict(model_name, input_data.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка прогнозування: {str(e)}")


@app.get("/metrics/{model_name}")
async def get_metrics(model_name: str):
    try:
        return predictor.get_metrics(model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка отримання метрик: {str(e)}")
