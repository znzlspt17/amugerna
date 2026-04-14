from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
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


class UTF8JSONResponse(JSONResponse):
    """ensure_ascii=False 으로 한글을 그대로 인코딩"""
    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False).encode("utf-8")

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
    skipped = 0
    for name in actual:
        if name not in reference:
            continue
        a = actual[name]
        r = reference[name]
        if a["conf"] < conf_threshold:
            skipped += 1
            continue
        dist = np.sqrt((a["x"] - r["x"]) ** 2 + (a["y"] - r["y"]) ** 2)
        # 거리 0 -> 100%, 거리 1.0(어깨너비 만큼) -> 0%
        similarity = max(0.0, 1.0 - dist)
        scores.append(similarity)
    result = float(np.mean(scores)) if scores else 0.0
    logger.debug(f"[ComparePose] 유효 관절: {len(scores)}개, 저신뢰 제외: {skipped}개, 유사도: {result * 100:.1f}%")
    return result


BODY_IDX = list(range(5, 17))
KEYPOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5),
    (0, 6), (1, 7), (6, 7), (6, 8), (7, 9),
    (8, 10), (9, 11),
]


def _extract_keypoints(file_name: str) -> list[dict]:
    """이미지에서 포즈 키포인트 추출"""
    logger.info(f"[KeypointExtract] 추론 시작 - 입력: {file_name}")
    results = model(file_name)
    logger.info(f"[KeypointExtract] YOLO 추론 완료 - 결과 배치 수: {len(results)}")
    keypoints_list = []
    for result in results:
        kpts = result.keypoints.data
        body_kpts = kpts[:, BODY_IDX, :]
        logger.info(f"[KeypointExtract] 프레임 내 감지된 사람 수: {len(body_kpts)}명")
        for idx, person_kpts in enumerate(body_kpts):
            person_data = {}
            for name, kp in zip(KEYPOINT_NAMES, person_kpts):
                person_data[name] = {"x": float(kp[0]), "y": float(kp[1]), "conf": float(kp[2])}
            valid_kpts = sum(1 for v in person_data.values() if v["conf"] > 0.5)
            logger.info(f"[KeypointExtract] 사람 {idx + 1}번 - 유효 관절: {valid_kpts}/{len(KEYPOINT_NAMES)}개 (conf>0.5)")
            keypoints_list.append(person_data)
    logger.info(f"[KeypointExtract] 추출 완료 - 총 {len(keypoints_list)}명")
    return keypoints_list


def _score_against_references(keypoints_list: list[dict]) -> dict:
    """reference poses 와 유사도 비교 결과 반환"""
    if not reference_poses or not keypoints_list:
        return {}
    actual_norm = _normalize_pose(keypoints_list[0])
    scores = {}
    for ref_id in range(1, 12):
        if ref_id not in reference_poses:
            continue
        ref_data = reference_poses[ref_id]
        ref_person = ref_data[0] if isinstance(ref_data, list) else ref_data
        ref_norm = _normalize_pose(ref_person)
        scores[ref_id] = round(_compare_poses(actual_norm, ref_norm) * 100, 1)
    return scores


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    global model, reference_poses
    logger.info("[Startup] 모델 로드 시작: ./models/yolo26x-pose.pt")
    model = YOLO("./models/yolo26x-pose.pt")
    logger.info("[Startup] 모델 로드 완료")
    reference_poses = _load_reference_poses()
    loaded_ids = sorted(reference_poses.keys())
    logger.info(f"[Startup] 레퍼런스 포즈 {len(reference_poses)}개 로드 완료 - ID: {loaded_ids}")
    logger.info("[Startup] 서비스 준비 완료")
    yield
    logger.info("[Shutdown] 서비스 종료")


app = FastAPI(lifespan=lifespan, default_response_class=UTF8JSONResponse)


@app.get("/")
async def root():
    return {"message": "Hello World"}


# 사용자가 이미지를 입력하면 서버는 이미지를 저장한다.
# 파일 저장 이름 바꾸기(오늘날짜-파일이름) hint: datetime
@app.post("/upload_image")
def save_image(file: UploadFile = File(...)):
    global model

    logger.info(f"[upload_image] 요청 수신 - 파일명: {file.filename}, content-type: {file.content_type}")
    t_start = time.perf_counter()

    # 파일 이름 설정
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"./images/{now}-{file.filename}"
    logger.info(f"[upload_image] 저장 경로: {file_name}")

    # 파일 저장
    os.makedirs("./images", exist_ok=True)
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file_size = os.path.getsize(file_name)
    logger.info(f"[upload_image] 파일 저장 완료 - 크기: {file_size / 1024:.1f} KB")

    t_saved = time.perf_counter()
    logger.info(f"[랩타임] 파일 수신 및 저장: {(t_saved - t_start) * 1000:.1f}ms")

    if model is None:
        logger.error("[upload_image] 모델이 로드되지 않았습니다.")
        return UTF8JSONResponse({"error": "모델이 로드되지 않았습니다."})
    else:
        logger.info("[upload_image] 모델 추론 시작")

        # Predict with the model
        results = model(file_name)

        t_inferred = time.perf_counter()
        logger.info(f"[랩타임] 모델 추론: {(t_inferred - t_saved) * 1000:.1f}ms")

        # 드로잉용 이미지 복사본 로드
        img = cv2.imread(file_name)
        h, w = img.shape[:2]
        logger.info(f"[upload_image] 이미지 로드 완료 - 해상도: {w}x{h}")

        # Access the results
        keypoints_list = _extract_keypoints(file_name)
        logger.info(f"[upload_image] 총 {len(keypoints_list)}명 감지됨 - 스켈레톤 드로잉 시작")

        # 사람별 키포인트 드로잉
        for person_data in keypoints_list:
            points = []
            for name in KEYPOINT_NAMES:
                kp = person_data[name]
                x, y, conf = int(kp["x"]), int(kp["y"]), kp["conf"]
                points.append((x, y) if conf > 0.5 else None)

            visible = sum(1 for pt in points if pt is not None)
            logger.info(f"[upload_image] 드로잉 - 가시 관절: {visible}/{len(KEYPOINT_NAMES)}개")

            # 관절 점 그리기
            for pt in points:
                if pt is not None:
                    cv2.circle(img, pt, 6, (0, 255, 0), -1)

            # 연결 라인 그리기
            drawn_lines = 0
            for i, j in SKELETON:
                if points[i] is not None and points[j] is not None:
                    cv2.line(img, points[i], points[j], (255, 0, 0), 2)
                    drawn_lines += 1
            logger.info(f"[upload_image] 드로잉 완료 - 연결선: {drawn_lines}/{len(SKELETON)}개")

        # 이미지를 JPEG 바이트로 인코딩
        _, encoded = cv2.imencode(".jpg", img)
        image_bytes = io.BytesIO(encoded.tobytes())

        # 추론 결과 이미지 저장
        os.makedirs("./results", exist_ok=True)
        result_path = f"./results/{now}-{file.filename}"
        cv2.imwrite(result_path, img)
        result_size = os.path.getsize(result_path)
        logger.info(f"[upload_image] 결과 이미지 저장 완료: {result_path} ({result_size / 1024:.1f} KB)")

        # 포즈 데이터 JSON 저장
        os.makedirs("./poses", exist_ok=True)
        pose_path = f"./poses/{now}-{file.filename}.json"
        with open(pose_path, "w", encoding="utf-8") as f:
            json.dump(keypoints_list, f, ensure_ascii=False, indent=2)
        logger.info(f"[upload_image] 포즈 JSON 저장 완료: {pose_path} ({len(keypoints_list)}명)")

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


@app.post("/pose_score")
def pose_score(pose_id: int = Form(...), file: UploadFile = File(...)):
    global model

    logger.info(f"[pose_score] 요청 수신 - pose_id: {pose_id}, 파일명: {file.filename}, content-type: {file.content_type}")

    if pose_id not in range(1, 12):
        logger.warning(f"[pose_score] 유효하지 않은 pose_id: {pose_id} (허용 범위: 1~11)")
        return UTF8JSONResponse({"error": f"pose_id는 1~11 사이여야 합니다. 입력값: {pose_id}"})

    t_start = time.perf_counter()

    # 파일 저장
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"./images/{now}-{file.filename}"
    logger.info(f"[pose_score] 저장 경로: {file_name}")
    os.makedirs("./images", exist_ok=True)
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file_size = os.path.getsize(file_name)
    logger.info(f"[pose_score] 파일 저장 완료 - 크기: {file_size / 1024:.1f} KB")

    t_saved = time.perf_counter()
    logger.info(f"[랩타임][pose_score] 파일 저장: {(t_saved - t_start) * 1000:.1f}ms")

    if model is None:
        logger.error("[pose_score] 모델이 로드되지 않았습니다.")
        return UTF8JSONResponse({"error": "모델이 로드되지 않았습니다."})

    logger.info(f"[pose_score] 모델 추론 시작")
    keypoints_list = _extract_keypoints(file_name)

    t_inferred = time.perf_counter()
    logger.info(f"[랩타임][pose_score] 모델 추론: {(t_inferred - t_saved) * 1000:.1f}ms")

    if not keypoints_list:
        logger.warning("[pose_score] 사람을 감지하지 못했습니다.")
        return UTF8JSONResponse({"error": "사람을 감지하지 못했습니다."})

    if pose_id not in reference_poses:
        logger.error(f"[pose_score] {pose_id}번 레퍼런스 포즈 파일이 없습니다.")
        return UTF8JSONResponse({"error": f"{pose_id}번 reference pose 파일이 없습니다."})

    logger.info(f"[pose_score] {pose_id}번 레퍼런스 포즈와 비교 시작")
    actual_norm = _normalize_pose(keypoints_list[0])
    ref_data = reference_poses[pose_id]
    ref_person = ref_data[0] if isinstance(ref_data, list) else ref_data
    ref_norm = _normalize_pose(ref_person)
    score = round(_compare_poses(actual_norm, ref_norm) * 100, 1)
    logger.info(f"[pose_score] 비교 완료 - {pose_id}번 포즈 유사도: {score:.1f}%")

    t_end = time.perf_counter()
    logger.info(f"[랩타임][pose_score] 전체 소요시간: {(t_end - t_start) * 1000:.1f}ms")

    return {
        "pose_id": pose_id,
        "score": score,
    }
