from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from io import BytesIO
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.utils import load_image
import torch
from controlnet_aux import HEDdetector
from contextlib import asynccontextmanager
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the models once on startup
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    app.state.hed = HEDdetector.from_pretrained("lllyasviel/Annotators")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=dtype, local_files_only=True
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    app.state.pipe = pipe

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/images", StaticFiles(directory="images"), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def generate_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    hed = request.app.state.hed
    pipe = request.app.state.pipe

    image = hed(image, scribble=True)

    prompt = (
        "photorealistic face portrait, masterpiece, 8K resolution, "
        "strict adherence to input line art reference (white-on-black), "
        "only drawn facial features & accessories, no additional details, "
        "realistic skin pores, soft cinematic lighting, sharp focus"
    )

    neg_prompt = (
        "distorted, deformed anatomy, low-res, blurry, cartoon, missing features, "
        "extra accessories, poor lighting, artifacts, image noise"
    )

    out = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=image,
        num_inference_steps=20,
        guidance_scale=8.5,
        controlnet_conditioning_scale=1.3,
    ).images[0]

    os.makedirs("images", exist_ok=True)
    out_path = "images/transformed_2D_image1.png"
    out.save(out_path)

    return JSONResponse(content={"url": f"/{out_path}"})
