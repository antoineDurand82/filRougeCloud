from pydantic import BaseModel


class Features(BaseModel):
    age: int
    sex: int
