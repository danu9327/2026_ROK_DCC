"""
app.py  –  사진 → 귀여운 픽셀아트 (Image-to-Image) Gradio 앱
RTX 5090 서버용 | 군 AI 교육 자료
"""

import random
import tempfile

import gradio as gr
from PIL import Image

from model_manager import manager, MODEL_CONFIGS, KONTEXT_TRIGGER
from pixel_utils import KEEP_PALETTE, canny_image, fit_image, pixelate

# ══════════════════════════════════════════════════════════════════════════════
# 고정 설정
# ══════════════════════════════════════════════════════════════════════════════

PIXEL_STYLE = "16-bit SNES RPG game sprite style, bold dark outline, vibrant colors, simple shading"
STRENGTH = 0.25
GRID_CHOICES = [96, 128]
PALETTE = KEEP_PALETTE
N_COLORS = 16
SATURATION = 1.2

# ══════════════════════════════════════════════════════════════════════════════
# 프롬프트 구성
# ══════════════════════════════════════════════════════════════════════════════

SUBJECTS = {
    "🧑 사람": "a cute chibi person, big head, small body, big sparkly eyes, rosy cheeks",
    "🐶 동물": "a cute chubby little animal, round body, big shiny eyes, tiny paws",
    "🚗 사물/차량": "a cute tiny toy-like object, rounded chunky shapes, small and adorable",
    "🏞️ 풍경": "a cute cozy tiny world, miniature scene, soft rounded shapes",
}

BASE_PROMPT = "pixel art, {subject}, {style}, cute, adorable, kawaii, clean crisp pixels, sharp edges"
NEGATIVE_PROMPT = (
    "realistic, photo, photorealistic, 3d render, blurry, smooth gradients, noise, jpeg artifacts, "
    "scary, creepy, horror, deformed, ugly, text, watermark"
)


def build_prompt(subject: str, model_name: str) -> str:
    if MODEL_CONFIGS[model_name]["kind"] == "kontext":
        # Kontext는 '지시문' 방식. LoRA 학습 문장을 맨 앞에 그대로 둬야 효과가 가장 좋음
        return f"{KONTEXT_TRIGGER} Make it look like {SUBJECTS[subject]}, {PIXEL_STYLE}, cute and adorable."
    return BASE_PROMPT.format(subject=SUBJECTS[subject], style=PIXEL_STYLE)


def _resolve_seed(seed) -> int:
    seed = int(seed)
    return random.randint(0, 2**31 - 1) if seed < 0 else seed


def _save_sprite(sprite: Image.Image) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".png", prefix="pixel_sprite_", delete=False)
    sprite.save(f.name)
    return f.name


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 – 픽셀 캐릭터 변환
# ══════════════════════════════════════════════════════════════════════════════

def convert(image, model_name, subject, steps, guidance, seed, lora_scale, cn_scale, grid):
    if image is None:
        yield "⚠️ 사진을 업로드하세요.", None, None, None, ""
        return

    seed = _resolve_seed(seed)
    prompt = build_prompt(subject, model_name)
    status_msgs = []

    yield "⏳ 대기열 / 모델 준비 중...", None, None, None, ""

    with manager.lock:
        try:
            manager.load(model_name, status_fn=status_msgs.append)
        except Exception as e:
            yield f"❌ 모델 로드 실패: {e}", None, None, None, ""
            return

        status_msgs.append("🎨 AI 이미지 변환 중...")
        try:
            raw = manager.generate(
                image, prompt, NEGATIVE_PROMPT, STRENGTH, steps, guidance, seed,
                lora_scale=lora_scale, controlnet_scale=cn_scale,
            )
        except Exception as e:
            yield f"❌ 이미지 생성 실패: {e}", None, None, None, ""
            return
        vram = manager.vram_info()

    status_msgs.append("🟦 픽셀화 후처리 중...")
    big, sprite = pixelate(raw, grid=int(grid), palette=PALETTE, n_colors=N_COLORS, saturation=SATURATION)
    status_msgs.append(f"✅ 완료! 스프라이트 크기: {sprite.width}×{sprite.height}px")

    log = (
        f"**[사용한 프롬프트]**\n{prompt}\n\n"
        f"**[Seed]** `{seed}` (같은 값을 넣으면 같은 결과 재현)\n\n"
        f"**[상태 로그]**\n" + "\n".join(status_msgs)
    )
    yield log, raw, big, _save_sprite(sprite), vram


def on_model_change(name):
    cfg = MODEL_CONFIGS[name]
    return (
        cfg["description"],
        gr.update(value=cfg["steps"]),
        gr.update(value=cfg["guidance"]),
        gr.update(visible=cfg["kind"] != "turbo"),
        gr.update(visible=cfg["kind"] == "sdxl_cn"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 – 실험실 (변형 강도 비교 / AI 없이 픽셀화 비교)
# ══════════════════════════════════════════════════════════════════════════════

SWEEP_STRENGTHS = [0.3, 0.45, 0.6, 0.75, 0.9]
SWEEP_LORA_SCALES = [0.0, 0.5, 0.75, 1.0, 1.25]   # Kontext는 strength가 없으므로 LoRA 강도로 비교


def strength_sweep(image, model_name, subject, seed, grid):
    if image is None:
        yield "⚠️ 사진을 업로드하세요.", []
        return

    seed = _resolve_seed(seed)
    grid = int(grid)
    cfg = MODEL_CONFIGS[model_name]
    prompt = build_prompt(subject, model_name)
    is_kontext = cfg["kind"] == "kontext"
    sweep = SWEEP_LORA_SCALES if is_kontext else SWEEP_STRENGTHS
    param = "LoRA 강도" if is_kontext else "strength"

    base = fit_image(image, cfg["size"])
    plain, _ = pixelate(base, grid=grid, palette=PALETTE, n_colors=N_COLORS, saturation=SATURATION)
    gallery = [(base, "원본"), (plain, "AI 없이 픽셀화만")]
    if cfg["kind"] == "sdxl_cn":
        gallery.append((canny_image(base), "ControlNet 입력 (Canny 엣지)"))
    yield f"⏳ 모델 준비 중... (Seed `{seed}`)", gallery

    with manager.lock:
        try:
            manager.load(model_name)
        except Exception as e:
            yield f"❌ 모델 로드 실패: {e}", gallery
            return

        for i, v in enumerate(sweep, 1):
            yield f"🎨 {i}/{len(sweep)} 생성 중... ({param}={v})", gallery
            strength, lora_scale = (1.0, v) if is_kontext else (v, 1.0)
            try:
                raw = manager.generate(
                    image, prompt, NEGATIVE_PROMPT, strength, cfg["steps"], cfg["guidance"], seed,
                    lora_scale=lora_scale,
                )
            except Exception as e:
                yield f"❌ 생성 실패 ({param}={v}): {e}", gallery
                return
            big, _ = pixelate(raw, grid=grid, palette=PALETTE, n_colors=N_COLORS, saturation=SATURATION)
            gallery.append((big, f"AI + 픽셀화 | {param} {v}"))

    yield f"✅ 비교 완료! Seed `{seed}` | {manager.vram_info()}", gallery


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI 정의
# ══════════════════════════════════════════════════════════════════════════════

INTRO_MD = """
# 👾 사진 → 귀여운 픽셀아트 변환기

**파이프라인:** 내 사진 → Image-to-Image Diffusion (귀여운 픽셀 스타일로 다시 그리기) → 픽셀화 후처리 (격자 축소 + 각진 확대) → 도트 캐릭터

🎮 **스타일: 16비트 RPG 고정** | 변형 강도 0.25 | 원본 색 유지

| 등급 | 모델 | 픽셀 스타일을 만드는 방법 |
|------|------|--------------|
| 🔵 하 | SDXL-Turbo (512px) | 프롬프트만 |
| 🟡 중 | SDXL + Pixel-Art LoRA (1024px) | 프롬프트 + LoRA |
| 🔴 상 | SDXL + LoRA + ControlNet (1024px) | 프롬프트 + LoRA + 윤곽선 고정 |
| 🟣 최상 | FLUX.1 Kontext + Pixel LoRA (1024px) | 문장 지시 편집 + LoRA |
"""

# 브라우저가 확대 시 픽셀을 뭉개지 않도록
CSS = ".pixelated img { image-rendering: pixelated; }"

MODEL_NAMES = list(MODEL_CONFIGS.keys())
DEFAULT_MODEL = MODEL_NAMES[1]

with gr.Blocks(title="귀여운 픽셀아트 변환기", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(INTRO_MD)

    # ── Tab 1: 픽셀 캐릭터 변환 ─────────────────────────────────────────────
    with gr.Tab("👾 픽셀 캐릭터 변환"):
        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    label="📷 원본 사진",
                    type="pil",
                    sources=["upload", "webcam", "clipboard"],
                    height=360,
                )
                model_dd = gr.Dropdown(MODEL_NAMES, value=DEFAULT_MODEL, label="🧠 이미지 변환 모델")
                model_info = gr.Markdown(MODEL_CONFIGS[DEFAULT_MODEL]["description"])

            with gr.Column(scale=1):
                subject_rd = gr.Radio(list(SUBJECTS.keys()), value="🧑 사람", label="📌 사진 속 대상")

                with gr.Accordion("🟦 픽셀화 설정", open=True):
                    grid_rd = gr.Radio(GRID_CHOICES, value=GRID_CHOICES[0], label="픽셀 해상도 (긴 변의 칸 수, 작을수록 큼직한 도트)")

                with gr.Accordion("⚙️ 고급 설정", open=False):
                    steps_sl = gr.Slider(1, 50, value=MODEL_CONFIGS[DEFAULT_MODEL]["steps"], step=1, label="추론 스텝 수")
                    guidance_sl = gr.Slider(0.0, 12.0, value=MODEL_CONFIGS[DEFAULT_MODEL]["guidance"], step=0.5, label="Guidance Scale")
                    lora_sl = gr.Slider(0.0, 1.5, value=1.0, step=0.1, label="LoRA 강도")
                    cn_sl = gr.Slider(0.0, 1.5, value=0.6, step=0.05, label="ControlNet 강도 (윤곽 고정력)", visible=False)
                    seed_nb = gr.Number(value=-1, label="Seed (-1 = 랜덤)", precision=0)

                convert_btn = gr.Button("🚀 픽셀아트로 변환", variant="primary", size="lg")

        with gr.Row():
            raw_img = gr.Image(label="① AI 변환 결과 (후처리 전)", type="pil", height=420)
            pixel_img = gr.Image(label="② 최종 픽셀아트", type="pil", height=420, elem_classes="pixelated")

        with gr.Row():
            with gr.Column(scale=2):
                log_md = gr.Markdown()
            with gr.Column(scale=1):
                sprite_file = gr.File(label="💾 원본 크기 스프라이트 PNG (게임 제작용)")
                vram_tb = gr.Textbox(label="VRAM", interactive=False)

        model_dd.change(
            on_model_change, model_dd,
            [model_info, steps_sl, guidance_sl, lora_sl, cn_sl],
        )
        convert_btn.click(
            convert,
            inputs=[
                input_img, model_dd, subject_rd,
                steps_sl, guidance_sl, seed_nb, lora_sl, cn_sl,
                grid_rd,
            ],
            outputs=[log_md, raw_img, pixel_img, sprite_file, vram_tb],
        )

    # ── Tab 2: 실험실 ───────────────────────────────────────────────────────
    with gr.Tab("🔬 실험실 (변형 강도 비교)"):
        gr.Markdown(
            """
### 같은 사진, 같은 Seed로 **변형 강도(strength)** 만 바꿔서 비교

- **AI 없이 픽셀화만**: 단순히 모자이크 + 색 줄이기 → 그냥 '깨진 사진' 느낌
- **AI + 픽셀화**: AI가 먼저 귀여운 도트 캐릭터로 다시 그린 뒤 픽셀화 → 진짜 게임 캐릭터 느낌
- strength가 커질수록 원본 형태는 사라지고 AI의 상상력이 커집니다
- 🔴 상 모델은 ControlNet 덕분에 strength가 높아도 윤곽이 유지되는지 확인해보세요
- 🟣 최상(Kontext)은 strength가 없어서 **LoRA 강도 0 ~ 1.25** 를 비교합니다 (0 = LoRA 없이 Kontext 기본 편집)
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                lab_img = gr.Image(label="📷 원본 사진", type="pil", sources=["upload", "webcam", "clipboard"], height=300)
            with gr.Column(scale=1):
                lab_model = gr.Dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="🧠 모델")
                lab_subject = gr.Radio(list(SUBJECTS.keys()), value="🧑 사람", label="📌 대상")
                lab_grid = gr.Radio(GRID_CHOICES, value=GRID_CHOICES[0], label="픽셀 해상도")
                lab_seed = gr.Number(value=42, label="Seed", precision=0)
                lab_btn = gr.Button("🔍 비교 생성", variant="primary")

        lab_status = gr.Markdown()
        lab_gallery = gr.Gallery(label="비교 결과", columns=4, height="auto", elem_classes="pixelated")

        lab_btn.click(
            strength_sweep,
            inputs=[lab_img, lab_model, lab_subject, lab_seed, lab_grid],
            outputs=[lab_status, lab_gallery],
        )

    # ── Tab 3: 이론 설명 ────────────────────────────────────────────────────
    with gr.Tab("📚 이론 설명"):
        gr.Markdown("""
## 🔄 Image-to-Image 픽셀아트 파이프라인

```
내 사진
   │
   ▼
┌──────────────────────┐
│  ① 노이즈 섞기         │  ← strength 만큼 원본을 흐리게 (0.3 = 살짝, 0.9 = 거의 잡음)
└──────────────────────┘
   │
   ▼
┌──────────────────────┐
│  ② Diffusion 복원     │  ← 프롬프트 "cute pixel art..." 방향으로 다시 그림
│   (+ LoRA / ControlNet)│
└──────────────────────┘
   │
   ▼
┌──────────────────────┐
│  ③ 픽셀화 후처리       │  ← 격자 축소 → 각지게 확대
└──────────────────────┘
   │
   ▼
귀여운 도트 캐릭터 👾
```

---

## 🖼️ Text-to-Image vs Image-to-Image

| | Text-to-Image (5회차) | Image-to-Image (9회차) |
|---|---|---|
| 시작점 | 완전한 잡음 | **내 사진 + 약간의 잡음** |
| 입력 | 프롬프트 | 사진 + 프롬프트 |
| 핵심 파라미터 | steps, guidance | **strength** (얼마나 바꿀지) |
| 결과 | 무엇이든 새로 생성 | 원본 구도/색을 살린 변환 |

### strength (변형 강도)
- `0.25` : 원본을 거의 그대로 살리고 스타일만 입힘 ← **이 앱의 고정값**
- `0.6` : 구도는 유지, 그림체는 확 바뀜
- `0.9` : 원본은 힌트 정도, AI가 거의 새로 그림
- 🔬 실험실 탭에서 다른 값들을 직접 비교해볼 수 있습니다

---

## 🔧 LoRA — 픽셀아트 전문가 어댑터
- 기본 SDXL은 '픽셀아트'를 흐릿하게 흉내만 냄
- **Pixel-Art-XL LoRA**: 수많은 도트 그림으로 추가 학습된 수 MB짜리 어댑터
- 🔵 하(프롬프트만) vs 🟡 중(LoRA) 결과를 비교해보면 차이가 확실!

## 🦴 ControlNet — 윤곽선 뼈대 고정
- 원본 사진에서 **엣지(Canny)** 를 추출해 "이 선을 따라 그려라"라고 추가 조건을 줌
- strength를 높여 귀엽게 크게 바꿔도 **포즈와 형태는 유지**

## 🟣 FLUX.1 Kontext — "말로 시키는" 이미지 편집
- 하~상 모델: 원본에 노이즈를 섞은 뒤 다시 그림 → 원본 유지 ↔ 스타일 변화가 **줄다리기**
- Kontext: 원본 사진을 **그대로 참고 자료로 보면서** "Turn this image into the Pixel style." 지시를 수행
- 그래서 strength 설정이 없고, 얼굴·구도를 지키면서 스타일만 바꾸는 데 강함
- **Pixel LoRA**: 원본↔픽셀 스타일 '짝 이미지'로 학습된 어댑터 (같은 방식의 스타일 LoRA가 20여 종)

---

## 🟦 왜 후처리가 필요할까?
AI가 그린 '픽셀아트 스타일' 그림도 확대해보면 경계가 흐릿하고 색이 수천 가지입니다.
진짜 도트 그림의 조건을 코드로 강제합니다:

1. **격자 축소** — 1024px → 96칸 또는 128칸 (칸마다 평균색 1개)
2. **채도·대비 부스트** — 평균내며 흐려진 색을 또렷하게
3. **Nearest Neighbor 확대** — 부드럽게 보간하지 않고 칸을 그대로 키움 → 각진 도트
        """)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.queue(max_size=64)       # 여러 교육생 요청을 순서대로 처리
    demo.launch(
        server_name="0.0.0.0",   # 모든 네트워크 인터페이스에 바인딩
        server_port=7861,        # 5회차 데모(7860)와 동시 실행 가능하도록
        share=True,              # Gradio 공개 URL 생성 (https://xxxx.gradio.live)
        show_error=True,
        inbrowser=False,         # 서버이므로 브라우저 자동 실행 안 함
    )
