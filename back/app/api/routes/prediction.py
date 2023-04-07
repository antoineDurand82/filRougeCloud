from fastapi import APIRouter
from starlette.requests import Request

from app.models.features import Features

router = APIRouter()


@router.get("/", status_code=200)
def read_root():
    return "test"


@router.post("/predict", name="predict")
async def predict(
        features: Features,
        request: Request):
    model = request.app.state.ml_models["basic_prediction"]
    prediction = model.predict(features.age, features.sex)
    return prediction
