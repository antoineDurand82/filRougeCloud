from pydantic import BaseModel


class Prediction(BaseModel):
    result: str
