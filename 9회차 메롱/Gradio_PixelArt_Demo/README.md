# 👾 사진 → 귀여운 픽셀아트 변환기 (Image-to-Image)
RTX 5090 서버에서 실행하는 Gradio 기반 Image-to-Image 픽셀아트 데모

# 서버 주소
(서버 실행 후 콘솔에 출력되는 https://xxxx.gradio.live 주소를 여기에 기입)

---

## 구성

### Tab 1 – 픽셀 캐릭터 변환
| 등급 | 모델 | 픽셀 스타일을 만드는 방법 | 속도(5090) |
|------|------|--------------|---|
| 🔵 하 | SDXL-Turbo (512px) | 프롬프트만 | ~1.5초 |
| 🟡 중 | SDXL + Pixel-Art-XL LoRA (1024px) | 프롬프트 + LoRA | ~3~6초 |
| 🔴 상 | SDXL + LoRA + ControlNet Canny (1024px) | 프롬프트 + LoRA + 윤곽선 고정 | ~4~8초 |
| 🟣 최상 | FLUX.1 Kontext + Pixel LoRA (1024px) | 문장 지시 편집 + LoRA | ~40~45초 |

- 사진 업로드 / 웹캠 / 클립보드 입력
- 사진 속 대상(사람·동물·사물·풍경) 선택
- 고정 설정: 스타일 16비트 RPG / 변형 강도 0.25 / 원본 색 유지 / 채도 1.2 (`app.py` 상단 상수)
- 픽셀 해상도 96칸 / 128칸 선택
- 고급 설정: 스텝 수, Guidance, LoRA 강도, ControlNet 강도, Seed
- AI 변환 결과(후처리 전)와 최종 픽셀아트를 나란히 표시
- 원본 픽셀 크기 스프라이트 PNG 다운로드

### Tab 2 – 실험실
- 같은 Seed로 strength 0.3 ~ 0.9 결과를 한 번에 비교 (최상 모델은 LoRA 강도 0 ~ 1.25 비교)
- "AI 없이 픽셀화만" vs "AI + 픽셀화" 비교
- 상 모델 선택 시 ControlNet 입력(Canny 엣지)도 함께 표시

### Tab 3 – 이론 설명
- Image-to-Image 파이프라인, strength 의미
- Text-to-Image(5회차)와 차이
- LoRA / ControlNet / FLUX Kontext / 픽셀화 후처리 원리

---

## 실행 순서

### 1. 가상환경 만들기
```bash
conda create -n pixelart python=3.12 -y
conda activate pixelart
```

### 2. 패키지 설치
```bash
# RTX 5090(Blackwell)은 CUDA 12.8 빌드 torch 필요
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

cd ~/2026_ROK_DCC/9회차\ 메롱/Gradio_PixelArt_Demo
pip install -r requirements.txt
```

### 3. Hugging Face 로그인 (최상 모델용)
FLUX.1-Kontext-dev는 라이선스 동의가 필요한 모델입니다.
1. https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev 접속 → 로그인 → **Agree and access repository** 클릭
2. 같은 계정의 토큰으로 서버에 로그인
   ```bash
   hf auth login
   ```

### 4. 모델 다운로드
```bash
python download_models.py              # 전체 (하/중/상/최상, 약 50GB)
python download_models.py 하 중        # 원하는 등급만
```

| 등급 | 받는 모델 | 크기 |
|---|---|---|
| 하 | `stabilityai/sdxl-turbo` (fp16) | ~6.5GB |
| 중 | `stabilityai/stable-diffusion-xl-base-1.0` (fp16) + `madebyollin/sdxl-vae-fp16-fix` + `nerijs/pixel-art-xl` | ~7GB |
| 상 | 중 + `diffusers/controlnet-canny-sdxl-1.0` (fp16) | +2.4GB |
| 최상 | `black-forest-labs/FLUX.1-Kontext-dev` + `Kontext-Style/Pixel_lora` | ~34GB |

- 저장 위치: `~/.cache/huggingface/hub` (다른 프로젝트와 공유)
- 다운로드를 건너뛰어도 첫 변환 때 자동으로 받지만, 수업 중 대기가 길어지므로 미리 받아두는 것을 권장

### 5. Gradio 서버 실행
```bash
python app.py
```
- 콘솔에 `Running on public URL: https://xxxx.gradio.live` 가 뜨면 그 주소를 교육생에게 공유 (72시간 유효)
- 같은 네트워크에서는 `http://<서버IP>:7861` 로도 접속 가능
- 포트 `7861` 사용 (5회차 데모 7860과 동시 실행 가능)

터미널을 닫아도 계속 돌게 하려면:
```bash
nohup python app.py > app.log 2>&1 &
grep gradio.live app.log      # 공개 주소 확인
pkill -f "python app.py"      # 종료
```

---

## 운영 참고
- 한 번에 하나의 모델만 GPU에 올리고, 다른 등급을 고르면 자동 교체 (교체 시 10~30초)
- 요청은 대기열로 한 명씩 순서대로 처리 → 수업 중에는 🔵 하 / 🟡 중 모델 위주로 쓰는 것을 권장
- 최상(Kontext)은 전체 ~34GB라 CPU 오프로드로 동작 (RAM 여유 40GB 이상 권장)

## 파일
| 파일 | 역할 |
|---|---|
| `app.py` | Gradio UI, 고정 설정·프롬프트(`PIXEL_STYLE`, `SUBJECTS`) 정의 |
| `model_manager.py` | 모델 설정(`MODEL_CONFIGS`), 로딩/교체, img2img 추론 |
| `pixel_utils.py` | 입력 리사이즈, Canny 엣지, 픽셀화 후처리·팔레트 |
| `download_models.py` | 등급별 모델 사전 다운로드 |

## 확장 아이디어
- `PIXEL_STYLE` 문구를 바꿔 다른 스타일 실험 ("포켓몬 도트", "메이플스토리 느낌" 등)
- `FIXED_PALETTES`에 나만의 팔레트 추가
- Kontext-Style의 다른 스타일 LoRA(20여 종)로 교체해보기
- 한 사진으로 여러 표정/포즈 스프라이트 시트 만들기
