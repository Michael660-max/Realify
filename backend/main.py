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
    DPMSolverMultistepScheduler,
)
import torch
from controlnet_aux import HEDdetector
from contextlib import asynccontextmanager
import os
from deca import DECA
import numpy as np, os, torch


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

    # DECA
    config = "vendor/deca/configs/deca_cfg.yaml"
    app.state.deca = DECA(config, device=device)

    app.state.hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    scribble = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=dtype,
        local_files_only=True,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=scribble,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    app.state.pipe = pipe

    # Optimize
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


@app.post("/generate_2d")
async def generate_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    form = await request.form()
    user_prompt = form.get("prompt")

    scribble_net = Image.open(BytesIO(contents)).convert("RGB").resize((512, 512))
    hed = request.app.state.hed
    pipe = request.app.state.pipe

    image = hed(scribble_net, scribble=True)

    prompt = ", ".join(
        [
            user_prompt,
            "photorealistic face portrait",
            "masterpiece, 8k resolution",
            "faithful interpretation of input line art",
            "natural eyes, nose, mouth, and ears",
            "soft even studio lighting, minimal shadows",
            "front-facing gaze, centered composition",
            "sharp details, realistic skin",
        ]
    )
    neg_prompt = ", ".join(
        [
            "blurry",
            "low-res",
            "cartoon",
            "deformed anatomy",
            "missing features",
            "artifacts",
            "poor lighting",
        ]
    )

    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=image,
            num_inference_steps=20,
            guidance_scale=8.5,
            controlnet_conditioning_scale=1.1,
        ).images[0]

    os.makedirs("images", exist_ok=True)
    out_path = "images/transformed_2D_image1.png"
    out.save(out_path)

    return JSONResponse(content={"url": f"/{out_path}"})


@app.post("/reconstruct_3d")
async def generate_model(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB").resize((512, 512))
    img_np = np.array(image).astype(np.float32) / 255.0

    deca = app.state.deca
    codedict = deca.encode(img_np)
    opdict = deca.decode(codedict)

    base = os.path.splitext(file.filename)[0]
    obj_path = f"images/{base}.obj"
    albedo_path = f"images/{base}_albedo.png"

    deca.save_obj(obj_path, opdict)
    albedo = (
        (opdict["albedo"][0].cpu().numpy().transpose(1, 2, 0) * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )
    Image.fromarray(albedo).save(albedo_path)
    
    return JSONResponse({
        "obj": f"/images/{base}.obj",
        "mtl": f"/images/{base}.mtl",
        "albedo": f"/images/{base}_albedo.png"
    })
