from turtle import pos
from fastapi import (
    FastAPI,
    Request,
    File,
    UploadFile,
    )
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

import uvicorn
import pose_estimation

app = FastAPI()
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "examples"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
video_path = ""

# http GET
@app.get("/", response_class=HTMLResponse)
def home_view(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...)):
    global video_path
    # save the uploaded file on local
    video_path = str(_save_file(video, path=UPLOAD_DIR, file_name=video.filename))
    #await pose_estimation.load_cam(temp_file)
    print(video.filename, video_path)
    return video_path

@app.get('/run_video')
def run_video():
    print("run video", video_path)
    return StreamingResponse(pose_estimation.load_video(video_path), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get('/run_webcam')
def run_webcam():
    return StreamingResponse(pose_estimation.load_webcam(), media_type="multipart/x-mixed-replace;boundary=frame")

# save the uploaded video in examples folder
def _save_file(uploaded_file, path, file_name):
    temp_file = path / file_name
    # with open(temp_file, "wb") as buffer:
    #     shutil.copyfileobj(uploaded_file.file, buffer)
    return temp_file


if __name__ == "__main__":
    uvicorn.run(app)