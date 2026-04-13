from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from datetime import datetime
from contextlib import asynccontextmanager
from ultralytics import YOLO

import shutil
import logging
import time
import io
import os
import json
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 터미널
        logging.FileHandler("app.log", encoding="utf-8"),  # 파일
    ],
)
model = None
reference_poses = {}  # {1: [...], 2: [...], ...}
logger = logging.getLogger(__name__)


def _load_reference_poses() -> dict:
    """final_poses/1.json ~ 11.json 로드"""
    poses = {}
    for i in range(1, 12):
        path = f"./final_poses/{i}.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                poses[i] = json.load(f)
            logger.info(f"[Reference] {path} 로드 완료")
        else:
            logger.warning(f"[Reference] {path} 파일 없음 - 건너맹니다")
    return poses


def _normalize_pose(person_data: dict) -> dict:
    """어깨 중심점을 원점으로, 어깨너비를 스케일 기준으로 정규화"""
    ls = person_data.get("left_shoulder")
    rs = person_data.get("right_shoulder")
    if not ls or not rs:
        return person_data

    cx = (ls["x"] + rs["x"]) / 2
    cy = (ls["y"] + rs["y"]) / 2
    scale = max(abs(rs["x"] - ls["x"]), 1.0)  # 어깨너비

    normalized = {}
    for name, kp in person_data.items():
        normalized[name] = {
            "x": (kp["x"] - cx) / scale,
            "y": (kp["y"] - cy) / scale,
            "conf": kp["conf"],
        }
    return normalized


def _compare_poses(actual: dict, reference: dict, conf_threshold: float = 0.5) -> float:
    """정규화된 키포인트 간 유클리드 거리 기반 유사도 (0.0 ~ 1.0)"""
    scores = []
    for name in actual:
        if name not in reference:
            continue
        a = actual[name]
        r = reference[name]
        if a["conf"] < conf_threshold:
            continue
        dist = np.sqrt((a["x"] - r["x"]) ** 2 + (a["y"] - r["y"]) ** 2)
        # 거리 0 -> 100%, 거리 1.0(어깨너비 만큼) -> 0%
        similarity = max(0.0, 1.0 - dist)
        scores.append(similarity)
    return float(np.mean(scores)) if scores else 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    global model, reference_poses
    model = YOLO("/models/yolo26x-pose.pt")
    reference_poses = _load_reference_poses()
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
    global model

    t_start = time.perf_counter()

    # 파일 이름 설정
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"./images/{now}-{file.filename}"

    # 파일 저장
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    t_saved = time.perf_counter()
    logger.info(f"[랩타임] 파일 수신 및 저장: {(t_saved - t_start) * 1000:.1f}ms")

    if model is None:
        logger.error("모델이 로드되지 않았습니다.")
        return {"error": "모델이 로드되지 않았습니다."}
    else:
        logger.info("모델이 성공적으로 로드되었습니다.")
        # COCO 키포인트 (머리 제외: 5~16)
        # 5:왼어깨, 6:오른어깨, 7:왼팔꿈치, 8:오른팔꿈치, 9:왼손목, 10:오른손목
        # 11:왼엉덩이, 12:오른엉덩이, 13:왼무릎, 14:오른무릎, 15:왼발목, 16:오른발목
        BODY_IDX = list(range(5, 17))  # 로컬 인덱스 0~11
        KEYPOINT_NAMES = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ]

        # 스켈레톤 연결 (로컬 인덱스 기준)
        SKELETON = [
            (0, 1),   # 왼어깨 - 오른어깨
            (0, 2),   # 왼어깨 - 왼팔꿈치
            (1, 3),   # 오른어깨 - 오른팔꿈치
            (2, 4),   # 왼팔꿈치 - 왼손목
            (3, 5),   # 오른팔꿈치 - 오른손목
            (0, 6),   # 왼어깨 - 왼엉덩이
            (1, 7),   # 오른어깨 - 오른엉덩이
            (6, 7),   # 왼엉덩이 - 오른엉덩이
            (6, 8),   # 왼엉덩이 - 왼무릎
            (7, 9),   # 오른엉덩이 - 오른무릎
            (8, 10),  # 왼무릎 - 왼발목
            (9, 11),  # 오른무릎 - 오른발목
        ]

        # Predict with the model
        results = model(file_name)  # predict on the uploaded image

        t_inferred = time.perf_counter()
        logger.info(f"[랩타임] 모델 추론: {(t_inferred - t_saved) * 1000:.1f}ms")

        # 드로잉용 이미지 복사본 로드
        img = cv2.imread(file_name)

        # Access the results
        keypoints_list = []
        for result in results:
            kpts = result.keypoints.data  # x, y, visibility
            body_kpts = kpts[:, BODY_IDX, :]  # 머리 제외 키포인트

            # 포즈 데이터를 이름 포함 딕셔너리로 변환
            for person_kpts in body_kpts:
                person_data = {}
                for name, kp in zip(KEYPOINT_NAMES, person_kpts):
                    person_data[name] = {"x": float(kp[0]), "y": float(kp[1]), "conf": float(kp[2])}
                keypoints_list.append(person_data)

            all_kpts = body_kpts  # 드로잉에 재사용

            # 사람별 키포인트 드로잉
            for person_kpts in all_kpts:
                points = []
                for kp in person_kpts:
                    x, y, conf = int(kp[0]), int(kp[1]), float(kp[2])
                    points.append((x, y) if conf > 0.5 else None)

                # 관절 점 그리기
                for pt in points:
                    if pt is not None:
                        cv2.circle(img, pt, 6, (0, 255, 0), -1)

                # 연결 라인 그리기
                for i, j in SKELETON:
                    if points[i] is not None and points[j] is not None:
                        cv2.line(img, points[i], points[j], (255, 0, 0), 2)

        # 이미지를 JPEG 바이트로 인코딩
        _, encoded = cv2.imencode(".jpg", img)
        image_bytes = io.BytesIO(encoded.tobytes())

        # 추론 결과 이미지 저장
        os.makedirs("./results", exist_ok=True)
        result_path = f"./results/{now}-{file.filename}"
        cv2.imwrite(result_path, img)
        logger.info(f"추론 결과 이미지 저장: {result_path}")

        # 포즈 데이터 JSON 저장
        os.makedirs("./poses", exist_ok=True)
        pose_path = f"./poses/{now}-{file.filename}.json"
        with open(pose_path, "w", encoding="utf-8") as f:
            json.dump(keypoints_list, f, ensure_ascii=False, indent=2)
        logger.info(f"포즈 데이터 저장: {pose_path}")

        # reference pose 와 유사도 비교
        if reference_poses and keypoints_list:
            actual_person = keypoints_list[0]  # 첫 번째 감지된 사람 기준
            actual_norm = _normalize_pose(actual_person)
            logger.info("[자세 비교] ──────────────────────────")
            total_scores = []
            for ref_id in range(1, 12):
                if ref_id not in reference_poses:
                    continue
                ref_data = reference_poses[ref_id]
                # reference JSON이 list면 첫 번째 사람 사용
                ref_person = ref_data[0] if isinstance(ref_data, list) else ref_data
                ref_norm = _normalize_pose(ref_person)
                score = _compare_poses(actual_norm, ref_norm)
                pct = score * 100
                total_scores.append(pct)
                logger.info(f"[자세 비교] {ref_id:2d}번 포즈 유사도: {pct:5.1f}%")
            if total_scores:
                avg = float(np.mean(total_scores))
                best_id = int(np.argmax(total_scores)) + 1
                logger.info(f"[자세 비교] 종합 평균 스코어   : {avg:5.1f}%")
                logger.info(f"[자세 비교] 가장 유사한 포즈   : {best_id}번 ({max(total_scores):.1f}%)")
            logger.info("[자세 비교] ──────────────────────────")

    t_end = time.perf_counter()
    logger.info(f"[랩타임] 드로잉 및 인코딩: {(t_end - t_inferred) * 1000:.1f}ms")
    logger.info(f"[랩타임] 전체 소요시간: {(t_end - t_start) * 1000:.1f}ms")

    return StreamingResponse(image_bytes, media_type="image/jpeg")
