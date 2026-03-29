# 🎭 나만의 AI 캐릭터 챗봇 만들기

캐릭터 기반 대화형 AI 챗봇을 만들어 봅시다!
캐릭터의 성격, 말투, 세계관을 설정하고 웹 채팅으로 대화합니다.

## 🏃 빠른 시작

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 데이터 구경 (5분)
python 01_explore_data.py

# 4. 웹 챗봇 실행!
python 02_chat_app.py
```

> 💡 GPU 없어도 `02_chat_app.py`는 베이스 모델로 바로 실행 가능!

## 📁 파일 구조
```
01_explore_data.py  ← 데이터셋 탐색
02_chat_app.py      ← 🎮 웹 캐릭터 챗봇 (메인!)
```

## 🎮 주요 기능
- **프리셋 캐릭터**: 츤데레 카페사장, 타임슬립 조선무관, 심해탐험 AI 등
- **커스텀 캐릭터**: 이름/성격/말투/배경을 직접 설정
- **웹 채팅 UI**: 브라우저에서 바로 대화 (Gradio)
- **대화 기억**: 최근 대화 맥락을 유지하며 자연스러운 대화

## 💻 환경
- 많이 무거운 모델 VRAM 최소 24GB "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
- 좀 무거운 모델 VRAM 최소 18GB "yanolja/YanoljaNEXT-EEVE-Instruct-7B-v2-Preview"
- 가벼운 모델 VRAM 최소 8GB "yanolja/YanoljaNEXT-EEVE-Instruct-2.8B"

## 💻 환경
- Python 3.9+
- GPU: 파인튜닝 시 VRAM 6GB+ 권장
- CPU: 추론(채팅)만 가능 (느리지만 동작)