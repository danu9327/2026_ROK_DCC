# 나만의 AI 캐릭터 챗봇 만들기

캐릭터 기반 대화형 AI 챗봇을 만들어 봅시다!
캐릭터의 성격, 말투, 세계관을 설정하고 웹 채팅으로 대화합니다.

## 🏃 빠른 시작

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. LLM의 주요 알고리즘 실습!
Lab1_Transformer_LLM 다운로드 받아서 Colab으로!

# 4. LLM의 효율적인 학습방법 비교 !
Lab2_Training_Methods 다운로드 받아서 Colab으로!
```

"""

## 모델: LG AI Research EXAONE 3.5 (한국어-영어 이중언어 전용)

# VRAM 프로필:
    (안됨)  minimal → EXAONE 2.4B (4bit)    ~2.5GB   ← 4GB VRAM 이하
            4gb     → EXAONE 2.4B (FP16)    ~5.5GB   ← 6~8GB VRAM
    (안됨)  8gb     → EXAONE 7.8B (4bit)    ~6GB     ← 8~12GB VRAM
            14gb    → EXAONE 7.8B (FP16)    ~16GB    ← 14~20GB VRAM
    (안됨)  24gb    → EXAONE 32B  (4bit)    ~20GB    ← 24GB+ VRAM

# 실행:
    python chat_app.py                      # 서버용 실행 코드
    python chat_app.py --share              # 서버용 실행 코드 외부 접속 허용
"""

## 🎮 주요 기능
- **프리셋 캐릭터**: 츤데레 카페사장, 타임슬립 조선무관, 심해탐험 AI 등
- **커스텀 캐릭터**: 이름/성격/말투/배경을 직접 설정
- **대화 기억**: 최근 대화 맥락을 유지하며 자연스러운 대화

## 💻 환경
- Python 3.9+
- GPU: VRAM 6GB+ 권장
- CPU: 추론(채팅)만 가능 (매우매우 느리지만 동작)

## 접속서버

1. RTX 4090 서버 주소:
2. RTX 5090-1 서버 주소:
3. RTX 5090-2 서버 주소:
4. RTX 5090-3 서버 주소:
