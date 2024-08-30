from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

base_model_path = "/openbayes/input/input3/stable-diffusion-v1-5"
controlnet_path = "/openbayes/input/input0/cat_controlnet/controlnet"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
).to(torch.device('cuda:0'))

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# memory optimization.
pipe.enable_model_cpu_offload()

control_image = load_image("2.png")
prompt = "a cute cat"

# generate image
generator = torch.manual_seed(0)
image = pipe(
    prompt, num_inference_steps=20, generator=generator, image=control_image
).images[0]
image.save("./output1.png")