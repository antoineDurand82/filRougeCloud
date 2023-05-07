import PIL
from pydantic import BaseModel


class Prediction(BaseModel):
    result: PIL.Image.Image
