# 🎓 8회차 자유주제 스타터 템플릿 6종

## 📁 파일 구성

| # | 파일 | 주제 | 핵심 기술 |
|---|---|---|---|
| 1 | `01_photo_character.ipynb` | 사진 → 캐릭터화 | Stable Diffusion XL Turbo |
| 2 | `02_text_rpg.ipynb` | 텍스트 RPG (엔딩 있음) | LLM + 상태 관리 |
| 3 | `03_receipt_ocr.ipynb` | 영수증 OCR + 가계부 | EasyOCR + LLM 파싱 |
| 4 | `04_pdf_qa.ipynb` | PDF 요약 + Q&A | RAG (ChromaDB + LLM) |
| 5 | `05_kakao_analyzer.ipynb` | 카톡 대화 분석 | 정규식 파싱 + LLM 인사이트 |
| 6 | `06_restaurant_recommender.ipynb` | 맛집 추천 | 룰 필터 + LLM 추천 |

## 🚀 학생 이용 방법

1. **카테고리 선택**: 위 6개 중 본인이 만들고 싶은 거 하나
2. **Colab에서 열기**: 해당 .ipynb를 본인 Google Drive에 업로드 후 더블클릭
3. **GPU 활성화** (1, 3번만): 메뉴 → 런타임 → 런타임 유형 변경 → GPU (T4)
4. **`SERVER_URL` 입력** (1번 제외): 강사가 알려준 cloudflared URL
5. **셀 위에서 아래로 실행** (Shift+Enter 또는 Ctrl+F9 "모두 실행")
6. **마지막 셀에 나오는 `https://*.gradio.live` URL 클릭** → 본인 앱 열림

## ⚠️ 강사 준비물

- 학생들에게 cloudflared URL 공유
URL: https://reducing-behind-tube-analysis.trycloudflare.com
## 🛠️ 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| 1번 모델 로딩 실패 | GPU 미활성 | 런타임 유형 변경 |
| 5090 LLM 연결 안 됨 | URL 오타/터널 죽음 | 강사에게 새 URL 요청 |
| Gradio share URL 안 뜸 | 첫 실행 1분 대기 | 잠시 기다리기 |
| 한글 깨짐 | 폰트 미설치 | 노트북에 폰트 설치 셀 포함됨 |

## 💡 진행 팁

- 워밍업으로 7회차 결과물 회상 후 "이번엔 본인 아이디어로!" 선언
- 6개 중 못 정하는 학생을 위한 빠른 결정 도구:
  - "내가 매일 쓸 거 같다" → 3, 4, 5
  - "SNS에 자랑하고 싶다" → 1, 2
  - "친구한테 보내고 싶다" → 2, 6
- 1시간 코딩 + 30분 시연 + 30분 마무리 권장
