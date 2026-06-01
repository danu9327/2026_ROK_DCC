"""
app.py  –  VLM 이미지 생성 실습 Gradio 앱
RTX 5090 서버용 | 군 AI 교육 자료
"""

import os
import threading
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model_manager import manager, TEXT_MODEL_CONFIGS, IMAGE_MODEL_CONFIGS
from finetune import train_lora

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
LORA_OUTPUT_DIR = "./lora_weights"
os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)

# ── 전역 상태 ─────────────────────────────────────────────────────────────────
_training_active = False
_uploaded_pairs: list = []   # [(PIL Image, caption str), ...]
_train_logs: list = []       # [{"step": int, "loss": float}, ...]


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 – 이미지 생성 실습
# ══════════════════════════════════════════════════════════════════════════════

def generate(
    text_model_name: str,
    image_model_name: str,
    user_prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    seed: int,
    use_lora: bool,
):
    if not user_prompt.strip():
        yield (
            "⚠️ 프롬프트를 입력하세요.",
            None,
            None,
        )
        return

    status_msgs = []

    def status(msg: str):
        status_msgs.append(msg)

    # 1) 텍스트 모델 로드
    try:
        manager.load_text_model(text_model_name, status_fn=status)
    except Exception as e:
        yield f"❌ 텍스트 모델 로드 실패: {e}", None, None
        return

    # 2) 프롬프트 확장
    status("🧠 프롬프트 이해 및 확장 중...")
    try:
        enhanced = manager.enhance_prompt(user_prompt)
    except Exception as e:
        yield f"❌ 프롬프트 확장 실패: {e}", None, None
        return

    # 3) 이미지 모델 로드
    try:
        manager.load_image_model(image_model_name, status_fn=status)
    except Exception as e:
        yield f"❌ 이미지 모델 로드 실패: {e}", None, None
        return

    # 4) 이미지 생성
    cfg = IMAGE_MODEL_CONFIGS[image_model_name]
    w, h = cfg["default_size"]
    lora_path = LORA_OUTPUT_DIR if (use_lora and image_model_name.startswith("하")) else None
    if lora_path and not (Path(lora_path) / "adapter_config.json").exists():
        lora_path = None  # 아직 학습 안 됨

    status("🎨 이미지 생성 중...")
    try:
        img = manager.generate_image(
            prompt=enhanced,
            negative_prompt=negative_prompt,
            width=w,
            height=h,
            steps=steps,
            guidance_scale=guidance,
            seed=seed,
            lora_path=lora_path,
        )
    except Exception as e:
        yield f"❌ 이미지 생성 실패: {e}", None, None
        return

    log_text = (
        f"**[원본 프롬프트]**\n{user_prompt}\n\n"
        f"**[확장된 프롬프트 (텍스트 모델 출력)]**\n{enhanced}\n\n"
        f"**[상태 로그]**\n" + "\n".join(status_msgs)
    )

    yield log_text, img, manager.vram_info()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 – 모델 학습 (파인튜닝)
# ══════════════════════════════════════════════════════════════════════════════

def add_image_caption(images, caption: str):
    global _uploaded_pairs
    if not images:
        return _format_data_table(), "⚠️ 이미지를 업로드하세요."
    if not caption.strip():
        return _format_data_table(), "⚠️ 캡션을 입력하세요."

    for img in (images if isinstance(images, list) else [images]):
        pil = Image.open(img) if isinstance(img, str) else img
        _uploaded_pairs.append((pil.convert("RGB"), caption.strip()))

    return _format_data_table(), f"✅ {len(images)}장 추가됨. 총 {len(_uploaded_pairs)}장"


def clear_data():
    global _uploaded_pairs
    _uploaded_pairs = []
    return _format_data_table(), "🗑️ 학습 데이터 초기화됨."


def _format_data_table() -> str:
    if not _uploaded_pairs:
        return "*(데이터 없음)*"
    rows = "\n".join(
        f"| {i+1} | {cap[:60]}{'...' if len(cap)>60 else ''} |"
        for i, (_, cap) in enumerate(_uploaded_pairs)
    )
    return f"| # | 캡션 |\n|---|---|\n{rows}"


def start_training(num_steps: int, lr: float, lora_rank: int):
    global _training_active, _train_logs

    if _training_active:
        return "⚠️ 이미 학습 중입니다.", None

    if len(_uploaded_pairs) < 1:
        return "⚠️ 학습 데이터를 먼저 추가하세요.", None

    _training_active = True
    _train_logs = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    progress_state = {"step": 0, "total": num_steps, "loss": 0.0}

    def progress_cb(step, total, loss):
        _train_logs.append({"step": step, "loss": loss})
        progress_state.update(step=step, total=total, loss=loss)

    def run():
        global _training_active
        try:
            train_lora(
                image_caption_pairs=_uploaded_pairs,
                output_dir=LORA_OUTPUT_DIR,
                num_steps=num_steps,
                learning_rate=lr,
                lora_rank=lora_rank,
                progress_fn=progress_cb,
                device=device,
            )
        finally:
            _training_active = False

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    # 학습이 끝날 때까지 폴링 (Gradio generator)
    while _training_active or progress_state["step"] == 0:
        time.sleep(2)
        step = progress_state["step"]
        total = progress_state["total"]
        loss = progress_state["loss"]
        pct = step / total if total > 0 else 0
        bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
        msg = (
            f"**학습 진행: {step}/{total} 스텝**\n"
            f"`[{bar}]` {pct*100:.1f}%\n"
            f"Loss: `{loss:.5f}`"
        )
        chart = _loss_chart()
        yield msg, chart

    # 완료
    yield "✅ **학습 완료!** 아래에서 생성 결과를 확인하세요.", _loss_chart()


def _loss_chart():
    if not _train_logs:
        return None
    steps = [x["step"] for x in _train_logs]
    losses = [x["loss"] for x in _train_logs]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(steps, losses, color="#4e79a7", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("LoRA Training Loss")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def compare_before_after(test_prompt: str, steps: int, seed: int):
    if not test_prompt.strip():
        yield "⚠️ 테스트 프롬프트를 입력하세요.", None, None
        return

    # SD v1.5 로드 확인
    if manager._image_model_id != "runwayml/stable-diffusion-v1-5":
        status_msgs = []
        manager.load_image_model("하 - SD v1.5", status_fn=lambda m: status_msgs.append(m))

    yield "⏳ 학습 전 이미지 생성 중...", None, None
    try:
        before = manager.generate_image(
            prompt=test_prompt, steps=steps, seed=seed, width=512, height=512
        )
    except Exception as e:
        yield f"❌ 생성 실패: {e}", None, None
        return

    lora_path = LORA_OUTPUT_DIR
    lora_ready = (Path(lora_path) / "adapter_config.json").exists()

    if not lora_ready:
        yield "⚠️ 학습된 LoRA 없음. 학습을 먼저 실행하세요.", before, None
        return

    yield "⏳ 학습 후 이미지 생성 중...", before, None
    try:
        after = manager.generate_image(
            prompt=test_prompt,
            steps=steps,
            seed=seed,
            width=512,
            height=512,
            lora_path=lora_path,
        )
    except Exception as e:
        yield f"❌ LoRA 생성 실패: {e}", before, None
        return

    yield f"✅ 비교 완료! | {manager.vram_info()}", before, after


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI 정의
# ══════════════════════════════════════════════════════════════════════════════

INTRO_MD = """
# 🎨 VLM 이미지 생성 실습

**파이프라인:** 사용자 입력 → 텍스트 이해 모델(LLM) → 상세 영어 프롬프트 → 이미지 생성 모델 → 결과 이미지

| 등급 | 텍스트 이해 모델 | 이미지 생성 모델 |
|------|--------------|--------------|
| 🔵 하 | Phi-3.5-mini (3.8B) | SD v1.5 (512px) |
| 🟡 중 | Qwen2.5-7B | SDXL (1024px) |
| 🔴 상 | Qwen2.5-14B (4-bit) | FLUX.1-schnell (1024px) |
"""

with gr.Blocks(title="VLM 이미지 생성 실습", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO_MD)

    # ── Tab 1: 이미지 생성 ──────────────────────────────────────────────────
    with gr.Tab("🎨 이미지 생성 실습"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 모델 선택")
                text_model_dd = gr.Dropdown(
                    choices=list(TEXT_MODEL_CONFIGS.keys()),
                    value=list(TEXT_MODEL_CONFIGS.keys())[0],
                    label="📝 텍스트 이해 모델",
                )
                text_model_info = gr.Markdown(
                    TEXT_MODEL_CONFIGS[list(TEXT_MODEL_CONFIGS.keys())[0]]["description"]
                )
                image_model_dd = gr.Dropdown(
                    choices=list(IMAGE_MODEL_CONFIGS.keys()),
                    value=list(IMAGE_MODEL_CONFIGS.keys())[0],
                    label="🖼️ 이미지 생성 모델",
                )
                image_model_info = gr.Markdown(
                    IMAGE_MODEL_CONFIGS[list(IMAGE_MODEL_CONFIGS.keys())[0]]["description"]
                )

                # 모델 설명 업데이트
                text_model_dd.change(
                    fn=lambda x: TEXT_MODEL_CONFIGS[x]["description"],
                    inputs=text_model_dd,
                    outputs=text_model_info,
                )
                image_model_dd.change(
                    fn=lambda x: IMAGE_MODEL_CONFIGS[x]["description"],
                    inputs=image_model_dd,
                    outputs=image_model_info,
                )

            with gr.Column(scale=2):
                gr.Markdown("### 프롬프트 입력")
                user_prompt = gr.Textbox(
                    label="📌 원하는 이미지 설명 (한국어 OK)",
                    placeholder="예: 벚꽃이 피는 봄날, 한복을 입은 여성이 강가에 서있는 모습",
                    lines=3,
                )
                neg_prompt = gr.Textbox(
                    label="🚫 부정 프롬프트 (원하지 않는 요소)",
                    placeholder="blurry, bad anatomy, watermark, low quality",
                    lines=2,
                )
                with gr.Row():
                    use_lora_cb = gr.Checkbox(
                        label="🔧 학습된 LoRA 적용 (하 모델 전용)",
                        value=False,
                    )
                    vram_text = gr.Textbox(label="VRAM", interactive=False, scale=0)

                with gr.Accordion("⚙️ 고급 설정", open=False):
                    steps_sl = gr.Slider(10, 50, value=20, step=1, label="추론 스텝 수")
                    guidance_sl = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
                    seed_nb = gr.Number(value=-1, label="Seed (-1 = 랜덤)", precision=0)

                gen_btn = gr.Button("🚀 이미지 생성", variant="primary", size="lg")

        with gr.Row():
            prompt_log = gr.Markdown(label="처리 과정 로그")
            output_img = gr.Image(label="🖼️ 생성된 이미지", type="pil", height=512)

        gen_btn.click(
            fn=generate,
            inputs=[
                text_model_dd, image_model_dd,
                user_prompt, neg_prompt,
                steps_sl, guidance_sl, seed_nb,
                use_lora_cb,
            ],
            outputs=[prompt_log, output_img, vram_text],
        )

    # ── Tab 2: 모델 학습 ────────────────────────────────────────────────────
    with gr.Tab("🔧 모델 학습 (LoRA 파인튜닝)"):
        gr.Markdown(
            """
### SD v1.5 (하 모델) LoRA 파인튜닝 실습

**목표:** 기본 모델이 생성하지 못하는 개념(특정 캐릭터, 스타일 등)을
이미지-캡션 쌍 학습을 통해 생성할 수 있게 만들어봅니다.

**학습 데이터 작성 팁:**
- 이미지 5~20장 권장
- 캡션에 **고유 트리거 단어**를 포함하세요 (예: `sks character` / `mydog`)
- 예시 캡션: `"a photo of sks character, smiling"`
- 생성 시 같은 트리거 단어를 사용하면 학습된 개념이 나옵니다
            """
        )

        with gr.Row():
            # 데이터 추가 패널
            with gr.Column():
                gr.Markdown("#### 1️⃣ 학습 데이터 추가")
                upload_imgs = gr.File(
                    label="이미지 업로드 (여러 장 선택 가능)",
                    file_count="multiple",
                    file_types=["image"],
                )
                caption_input = gr.Textbox(
                    label="캡션 (위 이미지 모두에 동일하게 적용)",
                    placeholder="a photo of sks character, standing in a park",
                )
                with gr.Row():
                    add_btn = gr.Button("➕ 데이터 추가", variant="secondary")
                    clear_btn = gr.Button("🗑️ 초기화", variant="stop")

                data_table = gr.Markdown("*(데이터 없음)*", label="현재 학습 데이터")
                add_status = gr.Textbox(label="상태", interactive=False)

                add_btn.click(add_image_caption, [upload_imgs, caption_input], [data_table, add_status])
                clear_btn.click(clear_data, [], [data_table, add_status])

            # 학습 설정 패널
            with gr.Column():
                gr.Markdown("#### 2️⃣ 학습 설정 및 실행")
                train_steps = gr.Slider(50, 1000, value=500, step=50, label="학습 스텝 수")
                train_lr = gr.Number(value=5e-5, label="학습률 (Learning Rate)", precision=6)
                train_rank = gr.Slider(2, 16, value=4, step=2, label="LoRA Rank (높을수록 표현력↑, VRAM↑)")

                gr.Markdown(
                    "> 💡 **스텝 수 가이드:** 200 (빠른 체험) / 500 (균형) / 1000 (높은 품질)"
                )

                train_btn = gr.Button("🏋️ 학습 시작", variant="primary", size="lg")
                train_status = gr.Markdown("*(학습 전)*")
                loss_chart = gr.Plot(label="📉 Loss 곡선")

                train_btn.click(
                    fn=start_training,
                    inputs=[train_steps, train_lr, train_rank],
                    outputs=[train_status, loss_chart],
                )

        gr.Markdown("---")
        gr.Markdown("#### 3️⃣ 학습 전/후 비교")

        with gr.Row():
            compare_prompt = gr.Textbox(
                label="테스트 프롬프트 (트리거 단어 포함)",
                placeholder="a photo of sks character, holding a sword",
                scale=3,
            )
            compare_steps = gr.Slider(10, 50, value=20, step=5, label="스텝", scale=1)
            compare_seed = gr.Number(value=42, label="Seed", precision=0, scale=1)

        compare_btn = gr.Button("🔍 학습 전/후 비교 생성", variant="primary")
        compare_status = gr.Markdown()

        with gr.Row():
            before_img = gr.Image(label="⬅️ 학습 전 (기본 SD v1.5)", type="pil", height=400)
            after_img = gr.Image(label="➡️ 학습 후 (LoRA 적용)", type="pil", height=400)

        compare_btn.click(
            fn=compare_before_after,
            inputs=[compare_prompt, compare_steps, compare_seed],
            outputs=[compare_status, before_img, after_img],
        )

    # ── Tab 3: 이론 설명 ─────────────────────────────────────────────────────
    with gr.Tab("📚 이론 설명"):
        gr.Markdown("""
## 🔄 VLM 이미지 생성 파이프라인

```
사용자 입력 (한국어)
        │
        ▼
┌─────────────────────┐
│   텍스트 이해 모델   │  ← LLM (Phi/Qwen)
│   (프롬프트 확장)    │
└─────────────────────┘
        │  상세 영어 프롬프트
        ▼
┌─────────────────────┐
│   이미지 생성 모델   │  ← Diffusion Model (SD/SDXL/FLUX)
│   (이미지 생성)      │
└─────────────────────┘
        │
        ▼
     결과 이미지
```

---

## 📝 텍스트 이해 모델이란?

Large Language Model(LLM)이 사용자의 **짧고 불완전한 텍스트**를 이해하여
이미지 생성에 최적화된 **상세한 영어 프롬프트**로 변환합니다.

| 입력 | → | 출력 |
|------|---|------|
| "봄 풍경" | → | "A serene spring landscape with cherry blossoms in full bloom, soft morning light, photorealistic, 8k, detailed..." |

---

## 🖼️ 이미지 생성 모델이란? (Diffusion Model)

1. **순방향 과정**: 원본 이미지에 조금씩 노이즈를 추가 → 완전한 잡음
2. **역방향 과정**: 텍스트 조건을 받아 잡음에서 이미지를 복원

```
잡음  →  →  →  →  →  원본 이미지
(텍스트 프롬프트가 방향을 안내)
```

---

## 🔧 LoRA 파인튜닝이란?

- **문제**: 기본 모델은 특정 캐릭터나 개인화된 개념을 생성 못 함
- **해결**: 작은 어댑터 레이어(LoRA)만 학습 → 전체 모델 재학습 대비 **99% 파라미터 절약**

```
기본 모델 가중치 (고정, 수십 GB)
        +
LoRA 어댑터 (학습, 수 MB)
        =
커스텀 이미지 생성 가능!
```

### LoRA Rank란?
- **낮은 Rank (2~4)**: 파라미터 적음, 빠른 학습, 약한 표현력
- **높은 Rank (8~16)**: 파라미터 많음, 느린 학습, 강한 표현력
        """)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # 모든 네트워크 인터페이스에 바인딩
        server_port=7860,
        share=True,              # Gradio 공개 URL 생성 (https://xxxx.gradio.live)
        show_error=True,
        inbrowser=False,         # 서버이므로 브라우저 자동 실행 안 함
    )
