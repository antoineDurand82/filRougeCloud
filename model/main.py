from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

model_id = "nitrosocke/Ghibli-Diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

x = 3

prompt = ["ghibli style magical princess with golden hair"] * x
images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images



def images_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(w, h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img)
        grid.save(f"astronaut_rides_horse{i}.png")
    return grid

grid = images_grid(images, rows=1, cols=3)

# grid.save(f"astronaut_rides_horse.png")

# image.save("./magical_princess.png")