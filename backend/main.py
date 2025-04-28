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
    UniPCMultistepScheduler,
)
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
        "lllyasviel/sd-controlnet-hed", torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    app.state.pipe = pipe

    if device == "cuda":
        pipe.enable_model_cpu_offload()
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

    scribble = hed(image, scribble=True)
    out = pipe("bag", scribble, num_inference_steps=10).images[0]

    os.makedirs("images", exist_ok=True)
    out_path = "images/transformed_2D_image.png"
    out.save(out_path)

    return JSONResponse(content={"url": f"/{out_path}"})
