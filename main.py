from fastapi import FastAPI, UploadFile, File
from datetime import datetime
from contextlib import asynccontextmanager
from ultralytics import YOLO

import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),          # 터미널
        logging.FileHandler("app.log", encoding="utf-8"),   # 파일
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    global model
    model = YOLO("/models/yolo26x-pose.pt")
    logger.info(f"Service가 시작되었고 모델이 로드되었습니다.")
    yield
    # 종료 시 실행 (cleanup)

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}

# 사용자가 이미지를 입력하면 서버는 이미지를 저장한다.
# 파일 저장 이름 바꾸기(오늘날짜-파일이름) hint: datetime
@app.post("/upload_image")
def save_image(file: UploadFile = File(...)):
    # 파일 이름 설정
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"./images/{now}-{file.filename}"

    # 파일 저장
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "이미지를 저장했습니다.",
        "filename": file_name,
        "time": datetime.now().strftime("%Y%m%d%H%M%S"),
    }
