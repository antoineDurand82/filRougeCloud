import pickle
import pandas as pd
from diffusers import StableDiffusionPipeline
import torch
import asyncio

class Model:

    def __init__(self, path):
        # self.model = pickle.load(open(path, 'rb'))
        self.model = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
        self.model = self.model.to("cuda")

    def generate(self, prompt) -> dict:
        #  x_test = pd.DataFrame([[param1, param2]])
        # print(self.model)
        image = self.model(prompt, num_inference_steps=2, height=256, width=256).images[0]
        # prob = str(self.model.predict_proba(X)[0].tolist())  # send to list for return
        return {'prediction': 10.0, "confidence": 15.0}
