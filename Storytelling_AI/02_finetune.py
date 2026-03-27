"""
🎓 02_finetune.py - LoRA 파인튜닝 (한 파일에 끝!)

실행: python 02_finetune.py
옵션: python 02_finetune.py --epochs 1 --quick   (빠른 테스트용)

사용 데이터셋:
  - jojo0217/korean_safe_conversation (26K 일상대화)
  - heegyu/open-korean-instructions (한국어 챗봇 통합 데이터)

GPU VRAM 6GB면 충분합니다. (4-bit 양자화 사용)
학습 완료 후 ./my_storyteller/ 에 모델이 저장됩니다.
"""
import argparse
import os
import re

import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# =====================================================================
# 설정
# =====================================================================
MODEL_NAME = "skt/ko-gpt-trinity-1.2B-v0.5"   # 한국어 GPT 1.2B
OUTPUT_DIR = "./my_storyteller"
MAX_LENGTH = 512


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--quick", action="store_true", help="데이터 1000개만 사용 (빠른 테스트)")
    p.add_argument("--no_4bit", action="store_true", help="양자화 끄기 (VRAM 많을 때)")
    return p.parse_args()


# =====================================================================
# 1. 데이터 준비
# =====================================================================
def prepare_data(tokenizer, quick=False):
    """대화 데이터 → 학습용 텍스트로 변환"""
    print("\n📦 데이터 다운로드 중...")

    all_texts = []

    # --- 데이터셋 1: 한국어 안전 대화 (26K Q&A) ---
    print("  [1/2] korean_safe_conversation 로딩...")
    safe_conv = load_dataset("jojo0217/korean_safe_conversation", split="train")
    cols = safe_conv.column_names

    # 컬럼명 자동 감지
    q_col = next((c for c in ["instruction", "Q", "question"] if c in cols), cols[0])
    a_col = next((c for c in ["output", "A", "answer"] if c in cols), cols[1] if len(cols) > 1 else cols[0])

    for item in safe_conv:
        q = str(item[q_col]).strip()
        a = str(item[a_col]).strip()
        if q and a:
            text = f"### 질문: {q}\n### 답변: {a}{tokenizer.eos_token}"
            all_texts.append(text)

    print(f"       → {len(all_texts):,}개 로드")

    # --- 데이터셋 2: Open Korean Instructions ---
    print("  [2/2] open-korean-instructions 로딩...")
    oki = load_dataset("heegyu/open-korean-instructions", split="train")
    oki_count = 0

    for item in oki:
        raw = item.get("text", "").strip()
        if not raw:
            continue

        # <usr>, <bot>, <sys> 토큰을 프롬프트 형식으로 변환
        text = raw
        text = re.sub(r"<sys>\s*", "### 시스템: ", text)
        text = re.sub(r"<usr>\s*", "### 질문: ", text)
        text = re.sub(r"<bot>\s*", "### 답변: ", text)
        text = text.strip() + tokenizer.eos_token
        all_texts.append(text)
        oki_count += 1

    print(f"       → {oki_count:,}개 로드")
    print(f"  📊 총 {len(all_texts):,}개 학습 텍스트")

    # --- 셔플 & 자르기 ---
    import random
    random.seed(42)
    random.shuffle(all_texts)

    if quick:
        all_texts = all_texts[:1000]
        print(f"  ⚡ 빠른 모드: {len(all_texts)}개만 사용")

    # --- Dataset 구성 & 토큰화 ---
    dataset = Dataset.from_dict({"text": all_texts})

    def tokenize(ex):
        tokens = tokenizer(
            ex["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = dataset.map(tokenize, remove_columns=["text"], batched=False)
    dataset.set_format("torch")

    # 90/10 분할
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  학습: {len(split['train']):,} / 검증: {len(split['test']):,}")

    return split["train"], split["test"]


# =====================================================================
# 2. 모델 로드 + LoRA 적용
# =====================================================================
def load_model(use_4bit=True):
    """모델 & 토크나이저 로드"""
    print(f"\n🤖 모델 로딩: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit 양자화 (VRAM 절약)
    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("  → 4-bit 양자화 ON (VRAM 절약)")

    # dtype 키 호환 (최신 transformers는 dtype, 이전은 torch_dtype)
    import transformers
    dtype_key = "dtype" if transformers.__version__ >= "4.46" else "torch_dtype"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        **{dtype_key: torch.float16},
        device_map="auto",
        trust_remote_code=True,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA 어댑터 부착 (전체 파라미터의 ~1%만 학습)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  → LoRA 적용: 학습 파라미터 {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# =====================================================================
# 3. 학습
# =====================================================================
def train(model, tokenizer, train_ds, val_ds, args):
    """HuggingFace Trainer로 간단하게 학습"""
    print(f"\n🏋️ 학습 시작! (에폭: {args.epochs}, 배치: {args.batch_size})")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    # 저장
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ 학습 완료! 모델 저장 위치: {OUTPUT_DIR}")
    print(f"   다음 단계: python 03_story_game.py --checkpoint {OUTPUT_DIR}")


# =====================================================================
# 메인
# =====================================================================
if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        print("⚠️  GPU가 없습니다. 파인튜닝은 GPU가 필요합니다.")
        print("   GPU 없이 체험하려면: python 03_story_game.py")
        exit(1)

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"    VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    use_4bit = not args.no_4bit
    model, tokenizer = load_model(use_4bit=use_4bit)
    train_ds, val_ds = prepare_data(tokenizer, quick=args.quick)
    train(model, tokenizer, train_ds, val_ds, args)
