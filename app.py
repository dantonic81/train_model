# app.py
import io
import torch
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from model import MNISTModel
from PIL import Image

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

model = MNISTModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

class ImageInput(BaseModel):
    pixels: list  # 28x28 grayscale pixel values [0,255]

@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"[INFO] {request.method} {request.url.path} took {duration:.3f}s")
    return response

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html") as f:
        return f.read()

@app.post("/predict")
def predict(data: ImageInput):
    pixels = np.array(data.pixels).astype(np.float32).reshape(1, 1, 28, 28) / 255.0
    tensor = torch.tensor(pixels)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    return {"prediction": pred}