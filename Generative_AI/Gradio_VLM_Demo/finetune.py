"""
finetune.py
SD v1.5에 LoRA를 적용해 커스텀 이미지-캡션 쌍으로 파인튜닝
"하" 모델이 학습을 통해 새로운 개념을 생성할 수 있게 된다는 것을 시연
"""

import gc
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
from typing import Callable, List, Optional, Tuple

SD15_ID = "runwayml/stable-diffusion-v1-5"


# ── Dataset ──────────────────────────────────────────────────────────────────

class ImageCaptionDataset(Dataset):
    """업로드된 (PIL Image, caption) 쌍을 학습용 텐서로 변환"""

    def __init__(self, pairs: List[Tuple[Image.Image, str]], tokenizer, size: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image, caption = self.pairs[idx]
        pixel_values = self.transform(image.convert("RGB"))
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": input_ids}


# ── Training ─────────────────────────────────────────────────────────────────

def train_lora(
    image_caption_pairs: List[Tuple[Image.Image, str]],
    output_dir: str,
    num_steps: int = 500,
    learning_rate: float = 5e-5,
    lora_rank: int = 4,
    progress_fn: Optional[Callable[[int, int, float], None]] = None,
    device: str = "cuda",
) -> List[dict]:
    """
    SD v1.5의 UNet에 LoRA 어댑터를 추가하고 학습합니다.
    저장 형식은 diffusers 호환 포맷 (pytorch_lora_weights.safetensors)

    Args:
        image_caption_pairs: [(PIL Image, "caption"), ...] 리스트
        output_dir:          학습된 LoRA 가중치 저장 경로
        num_steps:           총 학습 스텝 수
        learning_rate:       학습률 (권장: 5e-5 ~ 1e-5)
        lora_rank:           LoRA rank (낮을수록 파라미터 수 적음)
        progress_fn:         (현재 스텝, 전체 스텝, loss) 콜백
        device:              'cuda' or 'cpu'

    Returns:
        학습 로그 [{"step": int, "loss": float}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    # 컴포넌트 로드
    tokenizer = CLIPTokenizer.from_pretrained(SD15_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        SD15_ID, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        SD15_ID, subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        SD15_ID, subfolder="unet"
    ).to(device)
    scheduler = DDPMScheduler.from_pretrained(SD15_ID, subfolder="scheduler")

    # LoRA 설정 (UNet attention 레이어에만 적용)
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_cfg)

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]

    # 코사인 LR 스케줄러 + 웜업으로 안정적 학습
    warmup_steps = min(50, num_steps // 10)
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=num_steps,
        pct_start=warmup_steps / num_steps,
        anneal_strategy="cos",
    )

    # 데이터셋
    dataset = ImageCaptionDataset(image_caption_pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    logs = []
    step = 0
    unet.train()

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            pv = batch["pixel_values"].to(device, dtype=torch.float16)
            ids = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pv).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                t = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                ).long()
                noisy = scheduler.add_noise(latents, noise, t)
                enc_hidden = text_encoder(ids)[0]

            noise_pred = unet(
                noisy.float(), t, enc_hidden.float()
            ).sample

            loss = F.mse_loss(noise_pred, noise.float())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            lr_scheduler.step()

            step += 1
            log_entry = {"step": step, "loss": round(loss.item(), 5)}
            logs.append(log_entry)

            if progress_fn:
                progress_fn(step, num_steps, loss.item())

    # ── LoRA 가중치 저장 (PEFT 포맷: adapter_config.json + adapter_model.safetensors)
    unet.save_pretrained(output_dir)

    # 메모리 정리
    del text_encoder, vae, unet, optimizer, lr_scheduler
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return logs
