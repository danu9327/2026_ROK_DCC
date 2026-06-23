# 🎓 8회차 수업 운영 가이드 (강사용)

> **버전**: A안 — 매 수업마다 새 cloudflared URL을 학생들에게 안내하는 방식  
> **대상**: 5090 PC에서 Ollama + cloudflared로 LLM 서버를 학생들에게 노출  
> **소요**: 수업 시작 30분 전부터 종료 직후까지

---

## 🗺️ 한눈에 보기

```
[수업 30분 전] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. 5090 PC 전원/네트워크 확인
  2. 터미널 #1 - Ollama 상태 점검 (열었다 닫아도 OK)
  3. 터미널 #2 - cloudflared 터널 시작 (수업 끝까지 유지 ⚠️)
  4. URL 메모 / 슬라이드에 입력

[수업 시작] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  5. 학생들에게 URL 공유
  6. 첫 5분 동안 연결 안 되는 학생 챙기기

[수업 중] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  7. (선택) 터미널 #3 - GPU/Ollama 모니터링
  8. 터미널 #2 죽었나 가끔 확인 (트러블슈팅 섹션 참조)

[수업 종료] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  9. 터미널 #2 종료 (Ctrl+C)
  10. (선택) Ollama 서비스 정지
```

---

## ⚙️ 사전 준비 — 1회만 (이미 완료했으면 스킵)

### 0-1. Ollama 설치 확인

```bash
ollama --version
```

버전이 나오면 설치된 것. 안 나오면:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 0-2. systemd 환경 설정

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_NUM_PARALLEL=8"
Environment="OLLAMA_KEEP_ALIVE=24h"
Environment="OLLAMA_ORIGINS=*"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 0-3. 모델 다운로드 (~5GB)

```bash
ollama pull qwen2.5:7b-instruct
```

### 0-4. cloudflared 설치

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
cloudflared --version
```

---

## 🕒 수업 30분 전 — 준비 작업

### 단계 1. 5090 PC 점검

- [ ] PC 켜져 있고 절전 모드 해제
- [ ] 유선 인터넷 연결 확인 (Wi-Fi보다 안정적)
- [ ] 화면 절전 OFF (장시간 idle 시 sleep 안 되도록)
  ```bash
  # Ubuntu에서 화면 잠금/절전 일시 비활성화
  systemd-inhibit --what=idle --who="class" --why="lecture" sleep 4h &
  ```

### 단계 2. 터미널 #1 — Ollama 상태 점검 (5분 소요)

**새 터미널 창 1개 열기** (Ctrl+Alt+T 또는 GUI에서)

```bash
# 2-1. 서비스 살아있나
sudo systemctl status ollama
```

화면에 **`Active: active (running)`** 보이면 OK.

만약 죽어있거나 `inactive`면:
```bash
sudo systemctl restart ollama
sleep 3
sudo systemctl status ollama
```

```bash
# 2-2. 모델이 로드 가능한지
curl -s http://localhost:11434/api/tags | grep qwen2.5
```

`qwen2.5:7b-instruct` 보이면 OK.

```bash
# 2-3. 실제 추론 워밍업 (첫 호출은 항상 느림, 30초 정도)
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b-instruct",
    "messages": [{"role":"user","content":"안녕 한 문장으로 자기소개"}],
    "max_tokens": 80
  }'
```

JSON 응답이 오면 워밍업 완료. **수업 중 학생 첫 요청이 빠릅니다.**

> 💡 **이 터미널 #1은 점검 끝나면 닫아도 됩니다.** 

### 단계 3. 터미널 #2 — cloudflared 터널 시작 ⚠️ 핵심

**새 터미널 창을 또 하나 열기** (반드시 새 창 — #1과 별개)

```bash
cloudflared tunnel --url http://localhost:11434
```

10-20초 후 콘솔에 박스 형태로 URL이 출력됩니다:

```
2026-06-XX INF +-----------------------------------------------------+
2026-06-XX INF |  Your quick Tunnel has been created!                |
2026-06-XX INF |  https://random-words-here.trycloudflare.com        |
2026-06-XX INF +-----------------------------------------------------+
```

**🚨 이 터미널 창은 절대로 닫지 마세요. 수업 끝까지 켜둬야 합니다.**

- 창을 닫으면 → 터널 죽음 → 학생 502 에러
- Ctrl+C 누르면 → 터널 죽음
- 노트북 덮으면 → 절전 가능성 → 터널 죽음
- 화면 보호기 → 보통은 OK, 그래도 비활성화 권장

### 단계 4. URL 검증 (1분)

**새 터미널 #3** 열거나 폰 브라우저로:

```bash
curl https://random-words-here.trycloudflare.com/v1/models
```

JSON으로 모델 목록 응답이 오면 외부 접근 OK.

> ⚠️ **만약 잘 안 되면**: cloudflared 막 띄운 직후 30초간은 propagation 시간이 필요할 수 있습니다. 1분 기다린 후 재시도.

### 단계 5. URL 공유 준비

발급된 URL을 메모:
```
SERVER_URL: https://random-words-here.trycloudflare.com
MODEL_NAME: qwen2.5:7b-instruct
```

공유 채널 사전 준비:
- 슬라이드 첫 장에 미리 입력 또는
- 단톡방/슬랙에 미리 메시지 작성해두기 또는
- 칠판에 큰 글씨로 적을 수 있게 준비

---

## 🎬 수업 시작 — 첫 5분

### 단계 6. 학생들에게 안내

다음 형식으로 공유 (단톡방/슬랙/슬라이드):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 8회차 LLM 서버 정보
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SERVER_URL = "https://random-words-here.trycloudflare.com"
MODEL = "qwen2.5:7b-instruct"

⚠️ URL은 수업 종료 시 무효화됩니다
⚠️ URL 끝에 슬래시(/) 붙이지 마세요
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 단계 7. 연결 안 되는 학생 챙기기 (3-5분)

학생들이 위 URL을 본인 노트북에 붙여넣고 첫 셀을 실행할 때 자주 발생:

| 학생 에러 | 원인 | 해결 |
|---|---|---|
| `Connection refused` | URL 오타 | 복사 다시 확인 |
| `Name or service not known` | URL 앞뒤 공백 | 노트북에 `.strip()` 들어있음. 다시 실행 |
| `502 Bad Gateway` | cloudflared 죽음 | 본인 터미널 #2 확인 |
| `403 Forbidden` | Cloudflare 봇 차단 | 노트북에 User-Agent 위장 코드 있는지 확인 |
| `404 Not Found` | URL 끝에 `/` 또는 path 잘못 | URL 끝 정리 |

---

## 🖥️ 수업 중 — 모니터링 (선택)

### 단계 7 (선택). 터미널 #3 — GPU 부하 모니터링

학생들 30명 정도 동시에 LLM 호출하면 GPU 부하 걱정될 때:

**새 터미널 #3 열기**:
```bash
watch -n 2 nvidia-smi
```

체크 포인트:
- **VRAM 사용량**: 30GB 넘으면 위험 (전체 32GB)
- **GPU 사용률**: 100% 지속이면 큐 쌓이는 중 → 학생 응답 느려짐

추가로 Ollama 자체 로그 보고 싶으면 **터미널 #4**:
```bash
journalctl -u ollama -f
```

> 💡 **터미널 #3, #4는 강사가 안 봐도 되면 안 열어도 됩니다.** 학생 응답이 갑자기 느려지면 그때 열어서 확인.

### 단계 8. 터미널 #2 (cloudflared) 가끔 확인

cloudflared 창을 가끔 들여다보세요. 정상이면 다음과 같은 로그만 흐릅니다:
```
2026-06-XX INF Updated to new configuration
```

다음과 같은 게 뜨면 트러블 신호:
```
ERR Serve tunnel error: ...
WRN connection lost
```

→ 보통 자동 재연결되지만, **빨간 에러가 1분 이상 지속되면 [트러블슈팅 섹션](#-트러블슈팅) 참조**.

---

## 🏁 수업 종료

### 단계 9. cloudflared 정리

**터미널 #2**에서:
```bash
Ctrl + C    # 정상 종료
```

cloudflared가 깨끗하게 종료됩니다. URL은 이제 무효화 됨.

### 단계 10 (선택). Ollama 정지

부하 줄이고 싶거나 PC 다른 용도로 쓰려면:
```bash
sudo systemctl stop ollama
```

다시 켜고 싶을 땐:
```bash
sudo systemctl start ollama
```

---

## 🚨 트러블슈팅

### 증상 A — 수업 중 학생 다수가 502 에러

**원인 1**: cloudflared 터널이 죽음  
**확인**: 터미널 #2 보기. 박스로 발급됐던 URL이 더 이상 살아있지 않음

**해결**:
```bash
# 터미널 #2에서 다시 띄우기
cloudflared tunnel --url http://localhost:11434
```

→ **새 URL이 발급됨!** 학생들에게 즉시 재공지:
```
🚨 URL 갱신됨: https://새로운URL.trycloudflare.com
```

학생들은 노트북의 `SERVER_URL` 변수만 바꾸고 셀을 다시 실행하면 됨.

### 증상 B — Ollama 측 문제로 응답 안 옴

**확인**:
```bash
# 새 터미널이나 #1에서
curl http://localhost:11434/v1/models
```

→ 응답 없으면 Ollama 죽은 것.

**해결**:
```bash
sudo systemctl restart ollama
sleep 5
curl http://localhost:11434/v1/models
```

(cloudflared 터널은 그대로 유지 — 같은 URL로 다시 연결됨)

### 증상 C — VRAM 부족

**증상**: 학생 응답이 점점 느려지거나 일부 timeout

**확인**:
```bash
nvidia-smi
# VRAM 사용량 32GB 근접
```

**해결 옵션**:
1. **NUM_PARALLEL 줄이기** (임시 대응)
   ```bash
   sudo systemctl edit ollama
   # OLLAMA_NUM_PARALLEL=4 로 변경
   sudo systemctl restart ollama
   ```
2. **더 가벼운 모델로 교체**
   ```bash
   ollama pull qwen2.5:3b-instruct
   # 학생들에게 MODEL_NAME 변경 공지
   ```

### 증상 D — Quick tunnel이 너무 자주 죽음

trycloudflare의 무상 quick tunnel은 SLA가 없어 종종 끊김.

**일시 대응**: 매번 새로 띄우고 URL 재공지

**근본 대응** (다음 수업까지 여유 있으면): named tunnel 셋업 — Cloudflare 무료 계정으로 고정 URL 발급 가능 (별도 가이드 필요시 요청)

### 증상 E — 한 학생만 안 됨

학생 개별 문제일 가능성 큼:
- URL 오타 / 끝 슬래시
- 노트북 셀 실행 순서
- Colab 런타임 끊김

학생 본인의 노트북 Part 0 셀 결과를 함께 보세요.

---

## 📋 빠른 참조 — 명령어 모음

```bash
# Ollama 서비스 제어
sudo systemctl status ollama    # 상태
sudo systemctl start ollama     # 시작
sudo systemctl stop ollama      # 정지
sudo systemctl restart ollama   # 재시작

# Ollama 로그
journalctl -u ollama -f         # 실시간

# 모델 관리
ollama list                     # 설치된 모델
ollama pull qwen2.5:7b-instruct # 다운로드
ollama rm <모델명>              # 삭제

# 동작 확인
curl http://localhost:11434/v1/models                  # 모델 목록
curl http://localhost:11434/api/tags                    # Ollama 네이티브 API

# cloudflared
cloudflared tunnel --url http://localhost:11434         # 터널 시작
cloudflared --version                                   # 버전

# 모니터링
nvidia-smi                      # GPU 상태
watch -n 2 nvidia-smi          # 실시간
```

---

## 🎯 수업 30분 전 최종 체크리스트

수업 직전 한 번 보면서 다 ✅:

- [ ] 5090 PC 켜짐, 인터넷 연결 OK
- [ ] 화면 절전 비활성화
- [ ] **터미널 #1**에서 `sudo systemctl status ollama` → active (running)
- [ ] **터미널 #1**에서 워밍업 curl → JSON 응답 받음
- [ ] **터미널 #2**에서 `cloudflared tunnel --url http://localhost:11434` 실행 중
- [ ] cloudflared 발급한 URL을 폰 브라우저로 외부 접근 테스트 → JSON
- [ ] URL을 슬라이드/단톡방/칠판에 입력 준비
- [ ] (선택) 본인이 학생 노트북 1개를 처음부터 끝까지 한번 돌려봄
- [ ] 비상시 새 URL 공유할 채널 (단톡방 등) 준비

여기까지 다 ✅면 수업 시작 준비 끝.

---

## 💡 운영 노하우

1. **터미널 #2는 수업 끝까지 같은 창을 유지**. 새 창 열고 거기서 또 cloudflared 실행하면 URL이 두 개 되어 혼란.

2. **노트북 PC인 경우** 충전기 꽂은 채로 진행. 배터리만 쓰면 절전 모드로 갈 수 있음.

3. **수업 중간 휴식 시간에 절대 cloudflared 끄지 말 것**. 끄면 새 URL이라 휴식 후 학생들이 다시 입력해야 함.

4. **학생 30명 넘으면 시연 전에 NUM_PARALLEL 점검**:
   ```bash
   grep PARALLEL /etc/systemd/system/ollama.service.d/override.conf
   # 8 이상이면 OK
   ```

5. **수업 끝나고 본인 5090 보호** — Ollama 서비스 정지하면 idle 부하 0:
   ```bash
   sudo systemctl stop ollama
   ```

---

수업 잘 되시길 바랍니다 🍀
