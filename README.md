# amugerna

YOLO26x 기반 전신 포즈 추적 및 레퍼런스 자세 유사도 측정 FastAPI 서버입니다.

---

## 기술 스택

| 항목 | 내용 |
|---|---|
| **언어** | Python 3.12+ |
| **웹 프레임워크** | FastAPI + Uvicorn |
| **포즈 추론** | Ultralytics YOLO26x-pose |
| **배경 제거** | BiRefNet (ZhengPeng7/BiRefNet, HuggingFace) |
| **이미지 처리** | OpenCV (cv2), Pillow |
| **수치 연산** | NumPy |
| **딥러닝 프레임워크** | PyTorch + torchvision, Transformers |
| **패키지 관리** | uv |

---

## 프로젝트 구조

```
amugerna/
├── main.py                  # FastAPI 서버 메인
├── pyproject.toml           # 패키지 의존성 정의
├── ipynb/
│   └── birefnet_img.ipynb   # BiRefNet 배경 제거 & 마스킹 노트북
├── final_poses/             # 레퍼런스 포즈 JSON (1.json ~ 11.json)
├── final_images/            # 원본 레퍼런스 인물 이미지 (1.jpg ~ 11.jpg)
├── final_images_masked/     # BiRefNet 처리 결과 PNG (배경 투명/검정 마스킹)
├── images/                  # 업로드된 원본 이미지 저장
├── results/                 # 스켈레톤 드로잉 결과 이미지 저장
├── poses/                   # 추론된 키포인트 JSON 저장
├── app.log                  # 서버 로그 파일 (UTF-8)
└── .gitignore
```

---

## 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

서버 시작 시 자동으로 수행되는 작업:
- `/models/yolo26x-pose.pt` 모델 로드
- `./final_poses/1.json ~ 11.json` 레퍼런스 포즈 로드

---

## API 엔드포인트

### `GET /`
서버 상태 확인

**응답**
```json
{ "message": "Hello World" }
```

---

### `POST /upload_image`
이미지를 업로드하면 전신 포즈 추론 후 스켈레톤이 그려진 이미지를 반환합니다.
레퍼런스 포즈 1~11번 전체와 유사도를 비교하여 로그에 출력합니다.

**요청** `multipart/form-data`
| 파라미터 | 타입 | 설명 |
|---|---|---|
| `file` | UploadFile | 분석할 이미지 파일 |

**응답** `image/jpeg`  
스켈레톤(관절 점 + 연결선)이 그려진 JPEG 이미지

**저장 결과**
- `./images/{datetime}-{filename}` — 원본 이미지
- `./results/{datetime}-{filename}` — 스켈레톤 드로잉 이미지
- `./poses/{datetime}-{filename}.json` — 키포인트 JSON

**로그 출력 예시**
```
[랩타임] 파일 수신 및 저장: 12.3ms
[랩타임] 모델 추론: 487.5ms
[자세 비교]  1번 포즈 유사도:  82.3%
...
[자세 비교] 종합 평균 스코어   :  67.4%
[자세 비교] 가장 유사한 포즈   : 3번 (91.7%)
[랩타임] 드로잉 및 인코딩: 8.1ms
[랩타임] 전체 소요시간: 510.2ms
```

---

### `POST /pose_score`
특정 레퍼런스 포즈 번호를 지정하여 유사도 점수만 반환합니다.

**요청** `multipart/form-data`
| 파라미터 | 타입 | 설명 |
|---|---|---|
| `pose_id` | int | 비교할 레퍼런스 포즈 번호 (1~11) |
| `file` | UploadFile | 분석할 이미지 파일 |

**응답** `application/json`
```json
{
  "pose_id": 3,
  "score": 91.7
}
```

**에러 응답 예시**
```json
{ "error": "pose_id는 1~11 사이여야 합니다. 입력값: 15" }
{ "error": "사람을 감지하지 못했습니다." }
```

---

## 키포인트 구조

COCO 키포인트 기준, **머리(코/눈/귀) 제외** 12개 관절만 사용합니다.

| 인덱스 | 이름 |
|---|---|
| 0 | left_shoulder |
| 1 | right_shoulder |
| 2 | left_elbow |
| 3 | right_elbow |
| 4 | left_wrist |
| 5 | right_wrist |
| 6 | left_hip |
| 7 | right_hip |
| 8 | left_knee |
| 9 | right_knee |
| 10 | left_ankle |
| 11 | right_ankle |

키포인트 JSON 형식:
```json
[
  {
    "left_shoulder": { "x": 320.1, "y": 150.4, "conf": 0.95 },
    "right_shoulder": { "x": 280.3, "y": 148.7, "conf": 0.93 },
    ...
  }
]
```

---

## 유사도 계산 방식

1. **정규화**: 어깨 중심점을 원점, 어깨너비를 스케일 기준으로 좌표 정규화 (이미지 크기/위치 무관)
2. **거리 계산**: 정규화된 각 관절별 유클리드 거리 계산
3. **유사도 변환**: `similarity = max(0, 1 - distance)` → 백분율 변환
4. **confidence 필터**: 0.5 미만 관절은 비교에서 제외

---

## 레퍼런스 포즈 추가/수정

`./final_poses/` 폴더에 `{번호}.json` 형식으로 저장하면 서버 재시작 시 자동 로드됩니다.
현재 지원 범위: `1.json ~ 11.json`

---

## 로그

- **터미널**: 실시간 출력
- **파일**: `app.log` (UTF-8, 자동 누적)
- `app.*` 패턴은 `.gitignore`에 포함되어 있습니다.

각 단계별 상세 로그를 출력합니다:

| 태그 | 설명 |
|---|---|
| `[Startup]` | 서버 시작 시 모델/레퍼런스 로드 상태 |
| `[upload_image]` | 파일 수신, 해상도, 드로잉 통계, 저장 경로 |
| `[pose_score]` | 파일 수신, 추론, 유사도 비교 결과 |
| `[KeypointExtract]` | 감지된 사람 수, 유효 관절 수 (conf > 0.5) |
| `[자세 비교]` | 레퍼런스별 유사도, 종합 평균, 최고 유사 포즈 |
| `[랩타임]` | 각 처리 단계별 소요 시간 (ms) |

---

## BiRefNet 배경 제거 노트북

`ipynb/birefnet_img.ipynb` 에서 `final_images/` 폴더의 인물 이미지를 처리합니다.

### 처리 파이프라인

1. **누끼 추출**: `ZhengPeng7/BiRefNet` 모델로 인물 세그멘테이션
2. **블랙 마스킹**: 인물 영역 검정(RGB=0), 배경 투명(alpha=0) PNG 생성
3. **좌우 반전**: 필요 시 수평 플립 (`PIL.Image.FLIP_LEFT_RIGHT`)
4. **배경↔사람 반전**: alpha 채널 반전 → 배경 검정, 인물 투명

### 출력 형식

| 픽셀 영역 | R | G | B | A |
|---|---|---|---|---|
| 인물 (마스킹) | 0 | 0 | 0 | 255 |
| 배경 | 0 | 0 | 0 | 0 |

결과는 `final_images_masked/` 폴더에 PNG로 저장됩니다.
