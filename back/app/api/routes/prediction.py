from fastapi import APIRouter
from starlette.requests import Request

from app.models.features import Features

router = APIRouter()


@router.get("/", status_code=200)
def read_root():
    return "test"


@router.post("/generate", name="generate")
async def generate(
        features: Features,
        request: Request):
    model = request.app.state.ml_models["basic_prediction"]
    prediction = model.generate(features.prompt)
    return prediction
