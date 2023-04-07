from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import DEFAULT_MODEL_PATH
from app.services.model import Model


@asynccontextmanager
async def model_lifespan(application: FastAPI):
    ml_models = {"basic_prediction": _startup_model()}
    application.state.ml_models = ml_models
    yield
    ml_models.clear()


def _startup_model() -> Model:
    model_path = DEFAULT_MODEL_PATH
    model_instance = Model(model_path)  # use joblib or pickle.load, etc.
    return model_instance
