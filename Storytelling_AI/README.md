# 🎭 나만의 한국어 스토리텔링 AI 만들기

한국어 대화 데이터로 나만의 이야기꾼 AI를 만들어 봅시다!

## 🏃 빠른 시작

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 데이터 구경하기 (5분)
python 01_explore_data.py

# 3. 파인튜닝 (GPU 있으면 30분~1시간)
python 02_finetune.py

# 4. 내 AI와 이야기하기!
python 03_story_game.py
```

> 💡 GPU 없어도 `03_story_game.py`는 베이스 모델로 바로 체험 가능!

## 📁 파일 구조
```
01_explore_data.py  ← 데이터셋 탐색 & 재미있는 대화 구경
02_finetune.py      ← LoRA 파인튜닝 (한 파일에 끝)
03_story_game.py    ← 🎮 대화형 스토리 게임 (메인!)
```

## 🎮 게임 모드
- **자유 모드**: 아무 말이나 던지면 AI가 이야기를 이어감
- **장르 모드**: 판타지/로맨스/미스터리 등 장르 선택
- **릴레이 모드**: AI와 번갈아가며 한 문장씩 이야기 만들기

## 💻 환경
- Python 3.9+
- GPU: 파인튜닝 시 VRAM 6GB+ 권장 (4-bit 양자화)
- CPU: 추론(게임 플레이)만 가능
