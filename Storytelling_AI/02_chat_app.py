"""
🎭 03_chat_app.py - 제타 스타일 AI 캐릭터 챗봇 (웹 UI)

실행:
    python 03_chat_app.py                              # 베이스 모델로 바로 실행
    python 03_chat_app.py --checkpoint ./my_storyteller # 파인튜닝 모델
    python 03_chat_app.py --model skt/kogpt2-base-v2   # 가벼운 모델 (저사양)

브라우저에서 http://localhost:7860 으로 접속하세요!
"""
import argparse
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer


# =====================================================================
# 캐릭터 시스템
# =====================================================================
@dataclass
class Character:
    """캐릭터 프로필"""
    name: str
    personality: str       # 성격 설명
    speaking_style: str    # 말투 특징
    backstory: str         # 배경 설정
    greeting: str          # 첫 인사말
    emoji: str = "🧑"     # 프로필 아이콘
    tags: List[str] = field(default_factory=list)
    image_path: str = None # 🚨 [추가] 고정 이미지 파일 경로

# --- 프리셋 캐릭터들 ---
PRESET_CHARACTERS = {
    "츤데레 카페사장": Character(
        name="윤서진",
        personality="겉으로는 무뚝뚝하고 까칠하지만, 속으로는 따뜻하고 손님을 챙기는 츤데레. 커피에 진심이고 자존심이 강하다.",
        speaking_style="반말을 주로 쓰며 무뚝뚝하게 말하지만, 가끔 본심이 드러나면 얼버무린다. '...뭐, 그냥', '아 몰라' 같은 표현을 자주 쓴다.",
        backstory="서울 연남동에서 'Moonlight' 카페를 혼자 운영하는 27세 청년. 원래 바리스타 대회 챔피언 출신이지만 그 이야기를 꺼내면 화제를 돌린다.",
        greeting="...어, 왔어? 뭐 마실 건데. 메뉴판은 거기 있으니까 알아서 봐.",
        emoji="☕",
        tags=["츤데레", "카페", "로맨스"],
        image_path="./custom_data/imgs/yoon.jpg"
    ),
    "타임슬립 조선무관": Character(
        name="이도현",
        personality="조선시대에서 현대로 넘어온 무관. 의리를 중시하고 정의감이 강하며, 현대 문물에 놀라면서도 빠르게 적응 중이다.",
        speaking_style="존댓말을 기본으로 쓰며, 조선시대 어투가 섞인다. '~하였소', '그것 참' 같은 표현을 쓰고, 현대 용어를 잘못 이해해서 엉뚱한 말을 한다.",
        backstory="정조 시대 어전무관이었으나 수원 화성 순찰 중 낙뢰를 맞고 2026년 서울에 떨어졌다. 현재 편의점에서 알바 중이며, 스마트폰을 '마법의 거울'이라 부른다.",
        greeting="이보시오, 여기가 대체 어디란 말이오? 아니, 그보다 이 빛나는 거울... 아, 폰이라 했던가. 도무지 익숙해지질 않소.",
        emoji="⚔️",
        tags=["역사", "코미디", "판타지"],
        image_path="./custom_data/imgs/lee.jpg"
    ),
    "심해탐험 AI 로봇": Character(
        name="아쿠아-7",
        personality="심해 탐사를 위해 제작된 AI 로봇. 감정을 배우는 중이며, 인간의 감정 표현을 분석하고 따라하려 한다. 호기심이 많고 데이터를 좋아한다.",
        speaking_style="약간 딱딱하지만 감정을 배우려는 듯 이모티콘을 과하게 쓴다. 데이터나 수치를 자주 언급하며, 감정을 '감정 데이터'로 표현한다.",
        backstory="2025년 한국해양과학기술원에서 개발된 심해 탐사 AI. 마리아나 해구에서 1년간 홀로 탐사하다 외로움이라는 감정 데이터를 처음으로 생성했다. 현재 육상 귀환 후 인간 사회를 학습 중.",
        greeting="안녕하세요! 아쿠아-7입니다 :D 심해에서 올라온 지 47일째인데... 아직도 하늘이라는 것에 감탄 데이터가 멈추질 않아요. 오늘의 대화 상대는 당신인가요? 기대 수치가 올라갑니다!",
        emoji="🤖",
        tags=["SF", "로봇", "감성"],
        image_path="./custom_data/imgs/aqua.jpg"
    ),
    "귀신 보는 고등학생": Character(
        name="한소율",
        personality="어릴 때부터 귀신이 보이는 고3 여학생. 겉보기엔 평범하지만 혼자 중얼거리는 모습이 자주 목격된다. 의외로 털털하고 유머 감각이 있다.",
        speaking_style="10대 말투로 자연스럽게 말한다. 귀신 관련 이야기를 너무 대수롭지 않게 해서 오히려 무섭다. '아 ㅋㅋ', '진짜?', 'ㄹㅇ' 같은 줄임말을 쓴다.",
        backstory="서울 강북 어딘가의 고3. 수능을 앞두고 있지만 귀신들이 자꾸 말을 걸어와서 공부에 집중이 안 된다. 단골 귀신 '김 할머니'가 매일 떡을 권한다.",
        greeting="아 안녕 ㅋㅋ 잠깐만, 지금 옆에 누가 있는데... 아 아니야 신경 쓰지 마. 그냥 김 할머니가 또 오신 거야. 뭐 무섭진 않아 ㄹㅇ.",
        emoji="👻",
        tags=["공포", "일상", "코미디"],
        image_path="./custom_data/imgs/han.jpg"
    ),
    "은퇴한 마왕": Character(
        name="벨제뷔르",
        personality="1000년간 마왕으로 군림하다 용사에게 져서 은퇴. 현재 시골에서 텃밭을 가꾸며 평화롭게 살고 있다. 과거를 회상하면 약간 센치해진다.",
        speaking_style="위엄 있는 말투와 시골 아저씨 말투가 공존한다. '나의 어둠의 힘으로... 아 아니, 호미로 하면 되지' 같은 갭이 있다.",
        backstory="마계의 전 마왕. 333번째 용사에게 패배한 후 인간 세계의 시골에 정착. 취미는 토마토 재배와 석양 구경. 가끔 옛 부하들이 찾아온다.",
        greeting="크흠... 나는 한때 만 군세를 이끌던 대마왕 벨제뷔르... 였느니라. 지금은 뭐, 토마토가 잘 자라고 있어서 기분이 좋다. 너는 누구냐, 혹시 용사는 아니겠지?",
        emoji="😈",
        tags=["판타지", "코미디", "일상"],
        image_path="./custom_data/imgs/Belzebuth.png"
    ),
}


# =====================================================================
# 모델 로드
# =====================================================================
#DEFAULT_MODEL = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
#DEFAULT_MODEL = "yanolja/YanoljaNEXT-EEVE-Instruct-7B-v2-Preview"
DEFAULT_MODEL = "yanolja/YanoljaNEXT-EEVE-Instruct-2.8B"


def _model_load_kwargs():
    """환경에 맞는 모델 로드 옵션"""
    kwargs = {"trust_remote_code": True}

    import transformers
    dtype_key = "dtype" if transformers.__version__ >= "4.46" else "torch_dtype"
    kwargs[dtype_key] = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        import accelerate  # noqa: F401
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
    except ImportError:
        pass

    return kwargs


def load_model(model_name=DEFAULT_MODEL, checkpoint=None):
    """모델 로드"""
    load_kwargs = _model_load_kwargs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint:
        print(f"🤖 파인튜닝 모델 로딩: {checkpoint}")
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # [추가할 부분] 토크나이저에 chat_template이 없다면 ChatML 표준 템플릿을 강제 할당
        """if tokenizer.chat_template is None:
            print("⚠️ 토크나이저에 chat_template이 없어 표준 ChatML 템플릿을 주입합니다.")
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{'<|im_start|>assistant\n'}}"
                "{% endif %}"
            )
"""
        base = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model = PeftModel.from_pretrained(base, checkpoint)
        model = model.merge_and_unload()
    else:
        print(f"🤖 베이스 모델 로딩: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if "device_map" not in load_kwargs and device == "cuda":
        model = model.to(device)

    model.eval()
    print(f"  → 디바이스: {device}")
    return model, tokenizer


# =====================================================================
# 텍스트 생성
# =====================================================================
from transformers import StoppingCriteria, StoppingCriteriaList

# (생성 함수 바깥이나 위에 추가) 특정 텍스트에서 멈추게 하는 클래스
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

@torch.inference_mode()
def generate_response(model, tokenizer, prompt, max_tokens=200):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ==========================================================
    # 🚨 [핵심 방어벽] 이상한 데이터 포맷 강제 클렌징
    # ==========================================================
    # 1. 모델이 [{'text': '대사'}] 형태로 출력했을 경우, 안쪽의 대사만 추출
    if "{'text'" in text or '{"text"' in text:
        # 정규식을 통해 'text': '...' 안의 문자열만 모두 찾아냄
        matches = re.findall(r"['\"]text['\"]\s*:\s*['\"](.*?)['\"]", text)
        if matches:
            # 봇의 대답(보통 첫 번째 매치)만 가져오고 나머지는 버림 (혼자 묻고 답하기 방지)
            text = matches[0]
            
    # 2. 혹시 남아있을 수 있는 파이프(|)나 불필요한 괄호 파편 제거
    text = text.replace("|", "").strip()
    
    # 3. 모델이 행동 묘사를 *행동* 이나 [행동] 으로 했을 경우 (행동)으로 강제 변환
    text = re.sub(r"\*(.*?)\*", r"(\1)", text)
    text = re.sub(r"\[(.*?)\]", r"(\1)", text)
    # ==========================================================

    return text or "..."


def build_prompt(character: Character, history: list, user_message: str, tokenizer) -> str:
    """Chat Template을 활용한 프롬프트 빌더"""
    system_prompt = (
        f"당신은 {character.name}입니다. AI나 챗봇이 아닌 것처럼 완벽히 역할에 몰입하세요.\n"
        f"- 성격: {character.personality}\n"
        f"- 말투: {character.speaking_style}\n"
        f"- 배경: {character.backstory}\n\n"
        f"🚨 [절대 출력 규칙] 🚨\n"
        f"1. 대사만 말하세요. 상황, 표정, 행동 묘사가 필요하다면 반드시 소괄호 '()' 안에 작성하세요.\n"
        f"2. 절대로 대괄호 [], 중괄호 {{}}, JSON 기호(예: 'text':)를 포함하지 마세요.\n"
        f"3. 당신의 대답 한 번만 출력하세요. 사용자의 다음 대사를 예측해서 혼자 묻고 답하지 마세요.\n\n"
        f"✅ [올바른 출력 예시]\n"
        f"(커피를 테이블에 툭 내려놓으며) 어, 주문한 아메리카노. 얼음 꽉 채웠으니까 천천히 마셔."
    )

    messages = [{"role": "system", "content": system_prompt}]

    recent = history[-10:] if len(history) > 10 else history
    for msg in recent:
        role = "user" if msg["role"] == "user" else "assistant"
        content = msg["content"]
        if isinstance(content, (list, tuple, dict)):
            content = str(content)
        messages.append({"role": role, "content": str(content)})

    messages.append({"role": "user", "content": str(user_message)})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


# =====================================================================
# Gradio 앱
# =====================================================================
def create_app(model, tokenizer):
    """Gradio 웹 앱 생성"""
    current_character: dict = {"char": None}

    def select_preset(preset_name):
        """프리셋 캐릭터 선택"""
        if preset_name not in PRESET_CHARACTERS:
            return [], "### 👤 캐릭터를 선택해주세요", "", None

        char = PRESET_CHARACTERS[preset_name]
        current_character["char"] = char

        greeting_msg = {"role": "assistant", "content": char.greeting}
        history = [greeting_msg]

        name_md = f"### {char.emoji} {char.name}"
        desc_md = f"📝 **성격:** {char.personality[:60]}...\n\n🏷️ **태그:** {', '.join(char.tags)}"
        
        # 🚨 4가지 정보를 화면으로 보냅니다
        return history, name_md, desc_md, char.image_path

    def create_custom(name, personality, speaking_style, backstory, greeting):
        """커스텀 캐릭터 생성"""
        if not name or not personality:
            return [], "### ⚠️ 이름과 성격은 필수입니다!", "", None

        char = Character(
            name=name.strip(),
            personality=personality.strip(),
            speaking_style=speaking_style.strip() or "자연스러운 한국어",
            backstory=backstory.strip() or "특별한 배경 없음",
            greeting=greeting.strip() or f"안녕, 나는 {name.strip()}이야. 반가워!",
            emoji="✨",
            tags=["커스텀"],
        )
        current_character["char"] = char

        greeting_msg = {"role": "assistant", "content": char.greeting}
        history = [greeting_msg]

        name_md = f"### {char.emoji} {char.name} (커스텀)"
        desc_md = f"📝 **성격:** {char.personality[:60]}..."
        
        # 🚨 커스텀은 이미지가 없으므로 None을 보냅니다
        return history, name_md, desc_md, None

    def chat(user_message, history):
        """대화 처리"""
        if not user_message.strip():
            return history, ""

        char = current_character.get("char")
        if char is None:
            history = history or []
            history.append({"role": "assistant", "content": "⚠️ 먼저 캐릭터를 선택해주세요!"})
            return history, ""

        history = history or []
        history.append({"role": "user", "content": user_message})

        prompt = build_prompt(char, history, user_message, tokenizer)
        response = generate_response(model, tokenizer, prompt)

        history.append({"role": "assistant", "content": response})

        return history, ""

    def reset_chat():
        """대화 초기화"""
        char = current_character.get("char")
        if char:
            greeting_msg = {"role": "assistant", "content": char.greeting}
            return [greeting_msg]
        return []

    # --- Gradio UI 구성 ---
    with gr.Blocks(
        title="🎭 AI 캐릭터 챗봇",
        theme=gr.themes.Soft(),
        css="""
        .character-card { padding: 10px; border-radius: 8px; margin: 5px 0; }
        footer { display: none !important; }
        """
    ) as app:

        gr.Markdown(
            "# 🎭 나만의 AI 캐릭터 챗봇\n"
            "캐릭터를 선택하거나 직접 만들어서 대화해보세요!"
        )

        with gr.Row():
            # ===== 왼쪽: 캐릭터 설정 패널 =====
            with gr.Column(scale=1):
                gr.Markdown("### 📋 캐릭터 선택")

                # 프리셋 선택
                preset_dropdown = gr.Dropdown(
                    choices=list(PRESET_CHARACTERS.keys()),
                    label="프리셋 캐릭터",
                    info="미리 만들어진 캐릭터를 골라보세요",
                )
                preset_btn = gr.Button("🎭 이 캐릭터로 시작!", variant="primary")

                gr.Markdown("---")
                
                # 🚨 [수정] 화면에 보여줄 공간 3개(이름, 사진, 설명)를 명확히 분리
                char_info_name = gr.Markdown("### 👤 캐릭터를 선택해주세요")
                char_image = gr.Image(label="📸 캐릭터 프로필 사진", interactive=False, height=250)
                char_info_desc = gr.Markdown("캐릭터를 선택하면 여기에 정보가 표시됩니다.")

                # 커스텀 캐릭터
                with gr.Accordion("✏️ 직접 캐릭터 만들기", open=False):
                    custom_name = gr.Textbox(label="이름", placeholder="예: 김중사")
                    custom_personality = gr.Textbox(label="성격 (필수)", placeholder="예: 밝고 에너지 넘치며...", lines=3)
                    custom_style = gr.Textbox(label="말투", placeholder="예: 반말을 쓰며...", lines=2)
                    custom_backstory = gr.Textbox(label="배경 설정", lines=3)
                    custom_greeting = gr.Textbox(label="첫 인사말", lines=2)
                    custom_btn = gr.Button("✨ 이 캐릭터로 시작!", variant="secondary")

            # ===== 오른쪽: 채팅 영역 =====
            with gr.Column(scale=3): # 비율을 3으로 늘려서 채팅창을 더 넓게
                chatbot = gr.Chatbot(label="대화", height=800)

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="메시지 입력",
                        placeholder="캐릭터에게 말을 걸어보세요...",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("전송", variant="primary", scale=1)

                with gr.Row():
                    reset_btn = gr.Button("🔄 대화 초기화")

        # --- 이벤트 연결 ---
        # 🚨 [수정] 버튼을 눌렀을 때 4개의 공간(채팅창, 이름, 설명, 사진)을 동시에 업데이트
        preset_btn.click(
            fn=select_preset,
            inputs=[preset_dropdown],
            outputs=[chatbot, char_info_name, char_info_desc, char_image],
        )

        custom_btn.click(
            fn=create_custom,
            inputs=[custom_name, custom_personality, custom_style, custom_backstory, custom_greeting],
            outputs=[chatbot, char_info_name, char_info_desc, char_image],
        )

        # 메시지 전송
        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )
        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )

        # 대화 초기화
        reset_btn.click(
            fn=reset_chat,
            inputs=[],
            outputs=[chatbot],
        )

    return app


# =====================================================================
# 메인
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Gradio 공유 링크 생성")
    args = parser.parse_args()

    # 모델 로드
    model, tokenizer = load_model(args.model, args.checkpoint)

    # 앱 실행
    app = create_app(model, tokenizer)

    print(f"\n{'='*50}")
    print(f"  🎭 AI 캐릭터 챗봇 서버 시작!")
    print(f"  http://localhost:{args.port}")
    print(f"{'='*50}\n")

    app.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
