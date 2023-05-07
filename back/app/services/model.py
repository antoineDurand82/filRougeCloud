import io
import pickle
import pandas as pd
from diffusers import StableDiffusionPipeline
from fastapi.responses import FileResponse
import torch
import asyncio
import tempfile


class Model:

    def __init__(self, path):
        # self.model = pickle.load(open(path, 'rb'))
        self.model = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
        self.model = self.model.to("cuda")

    def generate(self, prompt) -> dict:
        #  x_test = pd.DataFrame([[param1, param2]])
        # print(self.model)
        image = self.model(prompt).images[0]
        # prob = str(self.model.predict_proba(X)[0].tolist())  # send to list for return
        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as tmp_file:
            image.save(tmp_file.name, format="JPEG")
            return FileResponse(tmp_file.name, media_type='image/jpeg')


    


