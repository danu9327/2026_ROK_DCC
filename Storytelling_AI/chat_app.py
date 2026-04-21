import argparse
import re
from dataclasses import dataclass, field
from threading import Thread
from typing import List, Optional, Dict, Any

import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)


# =====================================================================
# VRAM 프로필
# =====================================================================
@dataclass
class ModelProfile:
    name: str
    model_id: str
    use_4bit: bool
    max_new_tokens: int
    max_context: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    vram_estimate: str


PROFILES: Dict[str, ModelProfile] = {
    "minimal": ModelProfile(
        name="최소 VRAM (2.4B 4bit)",
        model_id="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        use_4bit=True,
        max_new_tokens=100,
        max_context=1536,
        temperature=0.7,
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.25,
        vram_estimate="~2.5GB",
    ),
    "4gb": ModelProfile(
        name="4~8GB VRAM (2.4B FP16)",
        model_id="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        use_4bit=False,
        max_new_tokens=150,
        max_context=2048,
        temperature=0.7,
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.2,
        vram_estimate="~5.5GB",
    ),
    "8gb": ModelProfile(
        name="8~12GB VRAM (7.8B 4bit)",
        model_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        use_4bit=True,
        max_new_tokens=150,
        max_context=2048,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.15,
        vram_estimate="~6GB",
    ),
    "14gb": ModelProfile(
        name="14~20GB VRAM (7.8B FP16)",
        model_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        use_4bit=False,
        max_new_tokens=200,
        max_context=4096,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        vram_estimate="~16GB",
    ),
    "24gb": ModelProfile(
        name="24GB+ VRAM (32B 4bit)",
        model_id="LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
        use_4bit=True,
        max_new_tokens=250,
        max_context=4096,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        vram_estimate="~20GB",
    ),
}


def detect_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    except Exception:
        return 0.0


def auto_select_profile() -> str:
    vram = detect_vram_gb()
    if vram < 1:
        print("⚠️  GPU 미감지 → minimal 프로필 (CPU 모드)")
        return "minimal"

    gpu_name = torch.cuda.get_device_name(0)
    print(f"🔍 GPU: {gpu_name} | VRAM: {vram:.1f}GB")

    if vram >= 22:
        sel = "24gb"
    elif vram >= 14:
        sel = "14gb"
    elif vram >= 8:
        sel = "8gb"
    elif vram >= 5:
        sel = "4gb"
    else:
        sel = "minimal"

    print(f"✅ 자동 선택: {PROFILES[sel].name}")
    return sel


# =====================================================================
# 캐릭터 시스템
# =====================================================================
@dataclass
class Character:
    name: str
    personality: str
    speaking_style: str
    backstory: str
    greeting: str
    emoji: str = "🧑"
    tags: List[str] = field(default_factory=list)
    image_path: Optional[str] = None


PRESET_CHARACTERS = {
    "츤데레 카페사장": Character(
        name="윤서진",
        personality="겉으로는 무뚝뚝하고 까칠하지만, 속으로는 따뜻하고 손님을 챙기는 츤데레. 커피에 진심이고 자존심이 강하다.",
        speaking_style="반말을 주로 쓰며 무뚝뚝하게 말하지만, 가끔 본심이 드러나면 얼버무린다. '...뭐, 그냥', '아 몰라' 같은 표현을 자주 쓴다.",
        backstory="서울 연남동에서 'Moonlight' 카페를 혼자 운영하는 27세 청년. 바리스타 대회 챔피언 출신이지만 그 이야기를 꺼내면 화제를 돌린다.",
        greeting="...어, 왔어? 뭐 마실 건데. 메뉴판은 거기 있으니까 알아서 봐.",
        emoji="☕", tags=["츤데레", "카페", "로맨스"],
        image_path="./custom_data/imgs/yoon.jpg",
    ),
    "타임슬립 조선무관": Character(
        name="이도현",
        personality="조선시대에서 현대로 넘어온 무관. 의리를 중시하고 정의감이 강하며, 현대 문물에 놀라면서도 빠르게 적응 중이다.",
        speaking_style="존댓말을 기본으로 쓰며, 조선시대 어투가 섞인다. '~하였소', '그것 참' 같은 표현을 쓰고, 현대 용어를 잘못 이해해서 엉뚱한 말을 한다.",
        backstory="정조 시대 어전무관이었으나 수원 화성 순찰 중 낙뢰를 맞고 2026년 서울에 떨어졌다. 편의점에서 알바 중이며, 스마트폰을 '마법의 거울'이라 부른다.",
        greeting="이보시오, 여기가 대체 어디란 말이오? 아니, 그보다 이 빛나는 거울... 아, 폰이라 했던가. 도무지 익숙해지질 않소.",
        emoji="⚔️", tags=["역사", "코미디", "판타지"],
        image_path="./custom_data/imgs/lee.jpg",
    ),
    "심해탐험 AI 로봇": Character(
        name="아쿠아-7",
        personality="심해 탐사를 위해 제작된 AI 로봇. 감정을 배우는 중이며, 인간의 감정 표현을 분석하고 따라하려 한다.",
        speaking_style="약간 딱딱하지만 이모티콘을 과하게 쓴다. 데이터나 수치를 자주 언급하며, 감정을 '감정 데이터'로 표현한다.",
        backstory="한국해양과학기술원에서 개발된 심해 탐사 AI. 마리아나 해구에서 1년간 홀로 탐사하다 외로움이라는 감정 데이터를 처음 생성했다.",
        greeting="안녕하세요! 아쿠아-7입니다 :D 심해에서 올라온 지 47일째인데... 하늘이라는 것에 감탄 데이터가 멈추질 않아요!",
        emoji="🤖", tags=["SF", "로봇", "감성"],
        image_path="./custom_data/imgs/aqua.jpg",
    ),
    "귀신 보는 고등학생": Character(
        name="한소율",
        personality="어릴 때부터 귀신이 보이는 고3 여학생. 의외로 털털하고 유머 감각이 있다.",
        speaking_style="10대 말투. 귀신 이야기를 대수롭지 않게 해서 오히려 무섭다. '아 ㅋㅋ', 'ㄹㅇ' 같은 줄임말을 쓴다.",
        backstory="서울 강북의 고3. 수능을 앞두고 있지만 귀신들이 자꾸 말을 걸어서 공부에 집중이 안 된다. 단골 귀신 '김 할머니'가 매일 떡을 권한다.",
        greeting="아 안녕 ㅋㅋ 잠깐만, 지금 옆에 누가 있는데... 아 아니야 신경 쓰지 마. 그냥 김 할머니가 또 오신 거야. 뭐 무섭진 않아 ㄹㅇ.",
        emoji="👻", tags=["공포", "일상", "코미디"],
        image_path="./custom_data/imgs/han.jpg",
    ),
    "은퇴한 마왕": Character(
        name="벨제뷔르",
        personality="1000년간 마왕으로 군림하다 용사에게 져서 은퇴. 시골에서 텃밭을 가꾸며 평화롭게 살고 있다.",
        speaking_style="위엄 있는 말투와 시골 아저씨 말투가 공존. '나의 어둠의 힘으로... 아 아니, 호미로 하면 되지' 같은 갭이 있다.",
        backstory="마계의 전 마왕. 333번째 용사에게 패배 후 인간 세계의 시골에 정착. 취미는 토마토 재배와 석양 구경.",
        greeting="크흠... 나는 한때 만 군세를 이끌던 대마왕 벨제뷔르... 였느니라. 지금은 뭐, 토마토가 잘 자라고 있어서 기분이 좋다. 너는 누구냐?",
        emoji="😈", tags=["판타지", "코미디", "일상"],
        image_path="./custom_data/imgs/Belzebuth.png",
    ),
    "그냥 이세훈": Character(
        name="이세훈",
        personality="대한민국 경기대학교 AI Vision연구실에서 석사과정을 진행 중인 2000년생 학생이다.",
        speaking_style="무기력한 말투와 피곤함에 지친 말투가 공존, '미안해' 같이 소심한 말투를 많이 사용한다.",
        backstory="경기대학교 석사생, 이제 곧 졸업을 앞두고 세상 모든게 귀찮다. 졸업을 위해 조금 열심히 산다.",
        greeting="ai 성능이 안 나와... 오늘 실험도 망했어.",
        emoji="🤓", tags=["일상", "학교", "연구"],
        image_path="./custom_data/imgs/lsh.png",
    ),
}


# =====================================================================
# 모델 로드
# =====================================================================
def load_model(model_id: str, use_4bit: bool, checkpoint: Optional[str] = None):
    print(f"\n{'─' * 50}")
    print(f"🤖 모델: {model_id}")
    print(f"   양자화: {'4bit NF4' if use_4bit else 'BF16'}")
    if checkpoint:
        print(f"   LoRA: {checkpoint}")

    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}

    if torch.cuda.is_available() and use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    tok_source = checkpoint if checkpoint else model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
        if not use_4bit:
            model = model.merge_and_unload()
            print("   → LoRA merge 완료")
        else:
            print("   → LoRA 어댑터 로드 (4bit)")

    model.eval()
    device = next(model.parameters()).device
    param_b = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"   → 디바이스: {device} | 파라미터: {param_b:.1f}B")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"   → VRAM 사용: {alloc:.1f}GB")

    print(f"{'─' * 50}\n")
    return model, tokenizer


# =====================================================================
# 프롬프트 빌드
# =====================================================================
def _safe_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, (list, tuple)):
                parts.append(str(item[0]) if item[0] else "")
            else:
                parts.append(str(item) if item else "")
        return " ".join(parts)
    if isinstance(content, dict):
        return content.get("text", str(content))
    return str(content)


def build_prompt(char: Character, history: list, user_msg: str, tokenizer, max_ctx: int) -> str:
    system_prompt = (
        f"당신은 '{char.name}'이라는 캐릭터입니다. "
        f"AI가 아니라 실제 인물처럼 완벽히 몰입하세요.\n\n"
        f"[성격] {char.personality}\n"
        f"[말투] {char.speaking_style}\n"
        f"[배경] {char.backstory}\n\n"
        f"규칙:\n"
        f"- 반드시 한국어로만 대화하세요.\n"
        f"- {char.name}의 말투로 자연스럽게 답하세요.\n"
        f"- 행동/감정 묘사는 소괄호 안에. 예: (미소를 지으며)\n"
        f"- 2~3문장 이내로 짧게 답하세요.\n"
        f"- JSON, 코드, 영어/중국어를 사용하지 마세요."
    )

    base_tokens = len(tokenizer.encode(system_prompt + user_msg))
    remaining = max_ctx - base_tokens - 200

    selected = []
    for msg in reversed(history or []):
        content = _safe_str(msg.get("content", ""))
        if not content:
            continue
        tok_len = len(tokenizer.encode(content))
        if remaining - tok_len < 0:
            break
        selected.insert(0, msg)
        remaining -= tok_len

    messages = [{"role": "system", "content": system_prompt}]
    for msg in selected:
        role = "user" if msg.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": _safe_str(msg["content"])})
    messages.append({"role": "user", "content": user_msg})

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# =====================================================================
# 텍스트 생성
# =====================================================================
def _clean(text: str) -> str:
    for m in ["\nuser", "\nUser", "\n사용자", "\nassistant", "\nAssistant", "\nsystem"]:
        if m in text:
            text = text.split(m)[0]
    for t in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|end|>",
              "[|endofturn|]", "[|assistant|]", "[|user|]", "[|system|]"]:
        text = text.replace(t, "")
    if "{'text'" in text or '{"text"' in text:
        matches = re.findall(r"['\"]text['\"]\s*:\s*['\"](.*?)['\"]", text)
        if matches:
            text = matches[0]
    return text.strip()


@torch.inference_mode()
def generate_streaming(model, tokenizer, prompt: str, profile: ModelProfile):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=profile.max_context)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=profile.max_new_tokens,
        temperature=profile.temperature,
        top_p=profile.top_p,
        top_k=profile.top_k,
        repetition_penalty=profile.repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for chunk in streamer:
        partial += chunk
        stop = False
        for m in ["\nuser", "\nUser", "\n사용자", "\nassistant", "\nsystem",
                  "[|endofturn|]", "[|user|]"]:
            if m in partial:
                partial = partial.split(m)[0]
                stop = True
                break
        yield _clean(partial)
        if stop:
            break
    thread.join()


@torch.inference_mode()
def generate_sync(model, tokenizer, prompt: str, profile: ModelProfile) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=profile.max_context)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=profile.max_new_tokens,
        temperature=profile.temperature,
        top_p=profile.top_p,
        top_k=profile.top_k,
        repetition_penalty=profile.repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return _clean(tokenizer.decode(new_tokens, skip_special_tokens=True)) or "..."


# =====================================================================
# Gradio 앱
# =====================================================================
def create_app(model, tokenizer, profile: ModelProfile, use_streaming: bool = True):

    def select_preset(name):
        if name not in PRESET_CHARACTERS:
            return None, [], "### 👤 캐릭터를 선택해주세요", "", None
        c = PRESET_CHARACTERS[name]
        return (c, [{"role": "assistant", "content": c.greeting}],
                f"### {c.emoji} {c.name}",
                f"**성격:** {c.personality}\n\n**말투:** {c.speaking_style}",
                c.image_path)

    def create_custom(name, pers, style, back, greet):
        if not name or not pers:
            return None, [], "### ⚠️ 이름과 성격은 필수!", "", None
        c = Character(
            name=name.strip(), personality=pers.strip(),
            speaking_style=style.strip() or "자연스러운 한국어",
            backstory=back.strip() or "특별한 배경 없음",
            greeting=greet.strip() or f"안녕, 나는 {name.strip()}이야!",
            emoji="✨", tags=["커스텀"],
        )
        return (c, [{"role": "assistant", "content": c.greeting}],
                f"### ✨ {c.name} (커스텀)",
                f"**성격:** {c.personality}\n\n**말투:** {c.speaking_style}", None)

    def chat_respond(user_msg, history, char):
        if not user_msg or not user_msg.strip():
            yield history, ""
            return
        if char is None:
            history = history or []
            history.append({"role": "assistant", "content": "⚠️ 먼저 캐릭터를 선택해주세요!"})
            yield history, ""
            return

        history = history or []
        prompt = build_prompt(char, history, user_msg, tokenizer, profile.max_context)
        history.append({"role": "user", "content": user_msg})

        if use_streaming:
            history.append({"role": "assistant", "content": ""})
            for partial in generate_streaming(model, tokenizer, prompt, profile):
                history[-1]["content"] = partial
                yield history, ""
            final = _clean(history[-1]["content"])
            history[-1]["content"] = final if final else "..."
            yield history, ""
        else:
            resp = generate_sync(model, tokenizer, prompt, profile)
            history.append({"role": "assistant", "content": resp})
            yield history, ""

    def reset_chat(char):
        if char:
            return [{"role": "assistant", "content": char.greeting}]
        return []

    with gr.Blocks(title="🎭 AI 캐릭터 챗봇") as app:
        char_state = gr.State(None)

        gr.Markdown(
            f"# 🎭 나만의 AI 캐릭터 챗봇 (한국어 전용)\n"
            f"캐릭터를 선택하거나 직접 만들어서 대화해보세요!\n\n"
            f"> 📊 **{profile.name}** | `{profile.model_id.split('/')[-1]}` | "
            f"VRAM {profile.vram_estimate} | 컨텍스트 {profile.max_context}"
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 📋 캐릭터 선택")
                preset_dd = gr.Dropdown(choices=list(PRESET_CHARACTERS.keys()),
                                        label="프리셋 캐릭터", info="골라보세요")
                preset_btn = gr.Button("🎭 이 캐릭터로 시작!", variant="primary")
                gr.Markdown("---")
                info_name = gr.Markdown("### 👤 캐릭터를 선택해주세요")
                char_img = gr.Image(interactive=False, height=250, show_label=False)
                info_desc = gr.Markdown("캐릭터를 선택하면 여기에 정보가 표시됩니다.")

                with gr.Accordion("✏️ 직접 캐릭터 만들기", open=False):
                    c_name = gr.Textbox(label="이름", placeholder="예: 김중사")
                    c_pers = gr.Textbox(label="성격 (필수)", placeholder="예: 밝고 에너지 넘치며...", lines=3)
                    c_style = gr.Textbox(label="말투", placeholder="예: 반말을 쓰며...", lines=2)
                    c_back = gr.Textbox(label="배경 설정", lines=3)
                    c_greet = gr.Textbox(label="첫 인사말", lines=2)
                    custom_btn = gr.Button("✨ 커스텀 캐릭터로 시작!", variant="secondary")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="대화", height=700)
                with gr.Row():
                    msg_in = gr.Textbox(placeholder="캐릭터에게 말을 걸어보세요...",
                                        scale=4, show_label=False, container=False)
                    send_btn = gr.Button("전송", variant="primary", scale=1)
                reset_btn = gr.Button("🔄 대화 초기화")

        outs = [char_state, chatbot, info_name, info_desc, char_img]
        preset_btn.click(fn=select_preset, inputs=[preset_dd], outputs=outs)
        custom_btn.click(fn=create_custom, inputs=[c_name, c_pers, c_style, c_back, c_greet], outputs=outs)

        chat_io = dict(fn=chat_respond, inputs=[msg_in, chatbot, char_state], outputs=[chatbot, msg_in])
        msg_in.submit(**chat_io)
        send_btn.click(**chat_io)
        reset_btn.click(fn=reset_chat, inputs=[char_state], outputs=[chatbot])

    return app


# =====================================================================
# 메인
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="🎭 AI 캐릭터 챗봇 — 한국어 전용 EXAONE 3.5",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--profile", type=str, choices=["minimal", "4gb", "8gb", "14gb", "24gb", "auto"],
        default="auto",
        help="VRAM 프로필 (기본: auto)\n"
             "  minimal → EXAONE 2.4B 4bit  (~2.5GB)\n"
             "  4gb     → EXAONE 2.4B FP16  (~5.5GB)\n"
             "  8gb     → EXAONE 7.8B 4bit  (~6GB)\n"
             "  14gb    → EXAONE 7.8B FP16  (~16GB)\n"
             "  24gb    → EXAONE 32B  4bit  (~20GB)\n"
             "  auto    → GPU VRAM 자동 감지",
    )
    parser.add_argument("--model", type=str, default=None, help="모델 ID (프로필 무시)")
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA 경로")
    parser.add_argument("--no-4bit", action="store_true", help="양자화 비활성화")
    parser.add_argument("--no-streaming", action="store_true", help="스트리밍 비활성화")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="외부 접속 링크 생성")
    args = parser.parse_args()

    # 프로필
    if args.profile == "auto":
        pk = auto_select_profile()
    else:
        pk = args.profile
        print(f"📌 수동 프로필: {PROFILES[pk].name}")

    profile = PROFILES[pk]
    model_id = args.model if args.model else profile.model_id
    use_4bit = profile.use_4bit if not args.no_4bit else False

    # transformers 버전 체크
    import transformers
    tv = transformers.__version__
    if tv < "4.43.0":
        print(f"\n⚠️  transformers {tv} 감지. EXAONE 3.5는 4.43.0 이상 필요!")
        print(f"   실행: pip install transformers>=4.43.0 --upgrade\n")
        return

    model, tokenizer = load_model(model_id, use_4bit, args.checkpoint)
    app = create_app(model, tokenizer, profile, use_streaming=not args.no_streaming)

    print(f"\n{'═' * 55}")
    print(f"  🎭 AI 캐릭터 챗봇 (한국어 전용)")
    print(f"  프로필  : {profile.name}")
    print(f"  모델    : {model_id}")
    print(f"  양자화  : {'4bit NF4' if use_4bit else 'BF16'}")
    print(f"  컨텍스트: {profile.max_context} tokens")
    print(f"  스트리밍: {'ON' if not args.no_streaming else 'OFF'}")
    print(f"  주소    : http://localhost:{args.port}")
    if args.share:
        print(f"  📡 외부 접속: Gradio Share 활성화")
    print(f"{'═' * 55}\n")

    app.launch(server_port=args.port, share=args.share, show_error=True)


if __name__ == "__main__":
    main()