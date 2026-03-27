"""
🎮 03_story_game.py - 대화형 스토리텔링 게임

실행:
    python 03_story_game.py                              # 베이스 모델로 바로 플레이
    python 03_story_game.py --checkpoint ./my_storyteller # 파인튜닝된 모델로 플레이
    python 03_story_game.py --model skt/kogpt2-base-v2   # 가벼운 모델 (저사양 PC)

게임 모드:
    🆓 자유 모드   - 자유롭게 대화하며 이야기 만들기
    📖 장르 모드   - 장르를 골라서 그에 맞는 이야기 생성
    🔄 릴레이 모드 - AI와 한 문장씩 번갈아 쓰며 이야기 완성하기
"""
import argparse
import random
import re
import textwrap
from collections import deque

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =====================================================================
# 설정
# =====================================================================
DEFAULT_MODEL = "skt/ko-gpt-trinity-1.2B-v0.5"

GENRES = {
    "1": ("판타지", "마법과 신비로운 모험의 세계"),
    "2": ("로맨스", "설레는 만남과 사랑 이야기"),
    "3": ("미스터리", "의문의 사건과 숨겨진 진실"),
    "4": ("SF", "미래 기술과 우주 탐험"),
    "5": ("일상", "소소하지만 따뜻한 하루"),
    "6": ("공포", "등골이 오싹해지는 이야기"),
    "7": ("역사", "조선시대 배경 시대극"),
    "8": ("모험", "미지의 세계로 떠나는 여정"),
}

# 장르별 이야기 시작 문장 (릴레이 모드용)
STORY_STARTERS = {
    "판타지": [
        "어느 날, 평범한 대학생이 도서관 지하에서 빛나는 문을 발견했다.",
        "왕국의 마지막 마법사가 눈을 떴을 때, 세상은 이미 변해 있었다.",
    ],
    "로맨스": [
        "비 오는 버스 정류장에서, 같은 우산을 집으려 손이 겹쳤다.",
        "매일 같은 카페에서 마주치던 그 사람이 오늘은 내 옆자리에 앉았다.",
    ],
    "미스터리": [
        "새벽 3시, 아무도 없어야 할 연구실에서 불이 켜져 있었다.",
        "10년 전 사라진 친구에게서 문자가 왔다. '나 지금 네 뒤에 있어.'",
    ],
    "SF": [
        "2150년, 인류 최초의 외계 신호가 서울 하늘에서 수신되었다.",
        "AI가 감정을 갖게 된 그날, 모든 로봇이 동시에 울기 시작했다.",
    ],
    "일상": [
        "동네 빵집 할머니가 오늘따라 빵을 하나 더 넣어주셨다.",
        "회사를 그만두고 제주도행 비행기를 예약했다. 편도로.",
    ],
    "공포": [
        "이사 온 집의 거울에 내가 아닌 다른 사람이 비쳤다.",
        "엘리베이터가 없는 층에서 문이 열렸다.",
    ],
    "역사": [
        "정조대왕의 밀명을 받은 젊은 무관이 한양 성문을 나섰다.",
        "임진왜란 전야, 이름 없는 도공이 비밀 서신을 전달받았다.",
    ],
    "모험": [
        "지도에 없는 섬의 좌표가 적힌 낡은 편지를 발견했다.",
        "백두산 정상에서 고대 동굴 입구가 드러났다.",
    ],
}

SYSTEM_PROMPTS = {
    "자유": "당신은 한국어 스토리텔러입니다. 사용자와 자연스럽게 대화하며 이야기를 만들어가세요.",
    "장르": "당신은 {genre} 장르 전문 스토리텔러입니다. {desc} 분위기로 이야기를 이어가세요.",
    "릴레이": "당신은 릴레이 소설 작가입니다. 사용자가 쓴 문장에 이어서 정확히 1~2문장만 추가하세요.",
}


# =====================================================================
# 모델 로드
# =====================================================================
def _model_load_kwargs():
    """환경에 맞는 모델 로드 옵션 생성"""
    kwargs = {"trust_remote_code": True}

    # torch_dtype vs dtype: 최신 transformers는 dtype 사용
    import transformers
    dtype_key = "dtype" if hasattr(transformers, "__version__") and transformers.__version__ >= "4.46" else "torch_dtype"
    kwargs[dtype_key] = torch.float16 if torch.cuda.is_available() else torch.float32

    # device_map="auto"는 accelerate 필요 → 없으면 수동 .to(device)
    try:
        import accelerate  # noqa: F401
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
    except ImportError:
        pass  # accelerate 없으면 device_map 생략, 아래서 수동 이동

    return kwargs


def load_model(model_name=DEFAULT_MODEL, checkpoint=None):
    """모델 로드 (파인튜닝 체크포인트 or 베이스 모델)"""
    load_kwargs = _model_load_kwargs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint:
        print(f"🤖 파인튜닝 모델 로딩: {checkpoint}")
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model = PeftModel.from_pretrained(base, checkpoint)
        model = model.merge_and_unload()
    else:
        print(f"🤖 베이스 모델 로딩: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # device_map 없이 로드한 경우 수동 이동
    if "device_map" not in load_kwargs and device == "cuda":
        model = model.to(device)

    model.eval()
    print(f"  → 디바이스: {device}")
    return model, tokenizer


# =====================================================================
# 텍스트 생성
# =====================================================================
@torch.inference_mode()
def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.85):
    """프롬프트를 받아 텍스트 생성"""
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.92,
        top_k=50,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # 후처리: 불완전한 문장 자르기
    for sep in ["### ", "사용자:", "질문:"]:
        if sep in text:
            text = text[:text.index(sep)]

    # 마지막 완전한 문장까지만
    last = max(text.rfind(c) for c in ".!?")
    if last > 10:
        text = text[:last + 1]

    return text.strip() or "..."


# =====================================================================
# 게임 모드 1: 자유 모드
# =====================================================================
def play_free(model, tokenizer):
    """자유롭게 대화하며 이야기 만들기"""
    print("\n🆓 자유 모드! 아무 말이나 해보세요.")
    print("   (예: '옛날 옛적에 용감한 고양이가 살았는데...')")
    print("   /reset = 대화 초기화 | /quit = 나가기\n")

    history = deque(maxlen=6)  # 최근 3턴 유지

    while True:
        user = input("🧑 나: ").strip()
        if not user:
            continue
        if user in ["/quit", "/종료", "q"]:
            break
        if user in ["/reset", "/초기화"]:
            history.clear()
            print("🔄 대화 초기화!\n")
            continue

        history.append(f"사용자: {user}")

        prompt = (
            f"### 설정: {SYSTEM_PROMPTS['자유']}\n\n"
            f"### 대화:\n" + "\n".join(history) + "\n스토리텔러:"
        )

        response = generate(model, tokenizer, prompt)
        history.append(f"스토리텔러: {response}")

        wrapped = textwrap.fill(response, width=60)
        print(f"\n📖 AI: {wrapped}\n")


# =====================================================================
# 게임 모드 2: 장르 모드
# =====================================================================
def play_genre(model, tokenizer):
    """장르를 선택하고 그에 맞는 이야기 생성"""
    print("\n📖 장르를 선택하세요:")
    for key, (name, desc) in GENRES.items():
        print(f"   {key}. {name} - {desc}")

    choice = input("\n번호 입력: ").strip()
    if choice not in GENRES:
        print("잘못된 선택!")
        return

    genre, desc = GENRES[choice]
    print(f"\n🎭 [{genre}] 모드 시작! 이야기의 시작을 입력하세요.")
    print(f"   /quit = 나가기\n")

    system = SYSTEM_PROMPTS["장르"].format(genre=genre, desc=desc)
    history = deque(maxlen=6)

    while True:
        user = input("🧑 나: ").strip()
        if not user:
            continue
        if user in ["/quit", "/종료", "q"]:
            break

        history.append(f"사용자: {user}")

        prompt = (
            f"### 설정: {system}\n\n"
            f"### 대화:\n" + "\n".join(history) + "\n스토리텔러:"
        )

        response = generate(model, tokenizer, prompt)
        history.append(f"스토리텔러: {response}")

        wrapped = textwrap.fill(response, width=60)
        print(f"\n📖 [{genre}] AI: {wrapped}\n")


# =====================================================================
# 게임 모드 3: 릴레이 모드
# =====================================================================
def play_relay(model, tokenizer):
    """AI와 한 문장씩 번갈아 쓰며 이야기 완성"""
    print("\n🔄 릴레이 모드! AI와 번갈아가며 이야기를 만듭니다.")
    print("   장르를 선택하면 첫 문장을 AI가 제안해줍니다.\n")

    for key, (name, _) in GENRES.items():
        print(f"   {key}. {name}")

    choice = input("\n번호 입력 (Enter=랜덤): ").strip()
    if choice in GENRES:
        genre = GENRES[choice][0]
    else:
        genre = random.choice(list(GENRES.values()))[0]
    print(f"\n🎭 장르: [{genre}]")

    starters = STORY_STARTERS.get(genre, STORY_STARTERS["판타지"])
    story = [random.choice(starters)]
    print(f"\n📖 이야기 시작:\n   '{story[0]}'")
    print(f"\n   이어서 한 문장을 써주세요! (10턴 후 자동 종료)\n")

    for turn in range(10):
        user_line = input(f"  [{turn+1}/10] 🧑 내 차례: ").strip()
        if not user_line:
            continue
        if user_line in ["/quit", "/종료", "q"]:
            break
        story.append(user_line)

        story_so_far = " ".join(story)
        prompt = (
            f"### 설정: {SYSTEM_PROMPTS['릴레이']}\n"
            f"### 장르: {genre}\n\n"
            f"### 이야기:\n{story_so_far}\n\n"
            f"### 다음 문장:"
        )

        ai_line = generate(model, tokenizer, prompt, max_tokens=80, temperature=0.9)
        sentences = re.split(r'(?<=[.!?])\s+', ai_line)
        ai_line = " ".join(sentences[:2])

        story.append(ai_line)
        print(f"         🤖 AI 차례: {ai_line}\n")

    # 완성된 이야기 출력
    print("\n" + "=" * 60)
    print("  📕 완성된 이야기")
    print("=" * 60)
    full_story = " ".join(story)
    print(textwrap.fill(full_story, width=55, initial_indent="  ", subsequent_indent="  "))
    print("=" * 60)

    save = input("\n💾 이야기를 파일로 저장할까요? (y/n): ").strip().lower()
    if save == "y":
        fname = f"story_{genre}_{random.randint(1000,9999)}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"[{genre}] 릴레이 스토리\n\n{full_story}\n")
        print(f"  → 저장 완료: {fname}")


# =====================================================================
# 메인 메뉴
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="파인튜닝 체크포인트 경로")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="베이스 모델 이름")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.checkpoint)

    while True:
        print("\n" + "=" * 50)
        print("  🎭 한국어 스토리텔링 AI")
        print("=" * 50)
        print("  1. 🆓 자유 모드  - 자유롭게 대화")
        print("  2. 📖 장르 모드  - 장르별 이야기")
        print("  3. 🔄 릴레이 모드 - 번갈아 쓰기")
        print("  q. 종료")
        print("=" * 50)

        choice = input("\n  선택: ").strip().lower()

        if choice == "1":
            play_free(model, tokenizer)
        elif choice == "2":
            play_genre(model, tokenizer)
        elif choice == "3":
            play_relay(model, tokenizer)
        elif choice in ["q", "/종료"]:
            print("\n👋 다음에 또 만나요!")
            break
        else:
            print("  1, 2, 3, q 중에서 선택해주세요.")


if __name__ == "__main__":
    main()
