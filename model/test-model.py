from diffusers import StableDiffusionPipeline
import torch

# model_id = "nitrosocke/Ghibli-Diffusion"
model_id = "./Ghibli-Diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "ghibli style magical princess with golden hair"
image = pipe(prompt).images[0]

image.save("./images/magical_princess.png")

