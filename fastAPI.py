from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from PIL import Image
from sr_model.capsule_srgan import generator
import numpy as np
from sr_model import resolve_single
import cv2 as cv
import io

app = FastAPI()
capsule_model = generator()
capsule_model.load_weights(
    'GeneratorToDeploy/caps_gan_generator.h5')

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    image = file.file
    img = Image.open(image)
    lr = np.array(img)
    sr = resolve_single(capsule_model, lr)
    is_success, buffer = cv.imencode(".png", sr.numpy())
    io_buf = io.BytesIO(buffer)
    return Response(content=io_buf.getvalue(), media_type="image/png")
