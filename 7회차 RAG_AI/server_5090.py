"""
============================================================
 7회차 실습용 LLM 서버 (RTX 5090, 32GB VRAM)
============================================================

수강생들이 Colab에서 OpenAI-호환 API로 접속해 동시에 사용합니다.
vLLM의 continuous batching이 동시 요청을 자동으로 효율 처리합니다.

----------- 사전 설치 (5090 PC, 1회만) -----------
    conda create -n vllm-server python=3.11 -y
    conda activate vllm-server
    pip install vllm
    # cloudflared 설치 (Ubuntu):
    #   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    #   sudo dpkg -i cloudflared-linux-amd64.deb

----------- 모델 사전 다운로드 (권장) -----------
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct

----------- 실행 방법 -----------
  터미널 1) python server_5090.py
  터미널 2) cloudflared tunnel --url http://localhost:8000
            -> 콘솔에 뜨는 https://*.trycloudflare.com 주소를 학생들에게 공유

----------- 학생 접속 확인 (Colab) -----------
    from openai import OpenAI
    client = OpenAI(base_url="https://<발급URL>/v1", api_key="not-needed")
    print(client.models.list())
"""

import subprocess
import sys

# ============================================================
#  설정
# ============================================================
# Qwen2.5-7B-Instruct: 한/영 둘 다 강함, tool-use 지원, 32GB VRAM에 여유롭게 들어감
# 다국어 성능 중시 + 동시접속 헤드룸 확보 측면에서 7B가 베스트 밸런스
MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 한국어 응답 품질을 더 끌어올리고 싶을 때 대안:
# MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"  # 한국어 강함, 라이선스 확인 필요
# MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"         # 14B AWQ 양자화도 5090에 들어감

HOST = "0.0.0.0"
PORT = 8000

# 컨텍스트 길이: RAG 환경에서 검색된 문서 + 대화 이력을 담아야 하므로 넉넉히
MAX_MODEL_LEN = 8192

# 동시 요청 처리량: vLLM이 자동 배칭하므로 보통 기본값으로 충분
GPU_MEMORY_UTIL = 0.85

# ============================================================
#  서버 실행
# ============================================================
cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL,
    "--host", HOST,
    "--port", str(PORT),
    "--max-model-len", str(MAX_MODEL_LEN),
    "--gpu-memory-utilization", str(GPU_MEMORY_UTIL),
    "--dtype", "bfloat16",
    "--served-model-name", MODEL,        # 학생이 client.chat.completions.create(model=...)에 쓰는 이름
    # "--enable-auto-tool-choice",       # Qwen2.5의 네이티브 tool-use를 켜고 싶을 때
    # "--tool-call-parser", "hermes",    # ↑ 같이 활성화
]

print("=" * 60)
print(f" Launching vLLM OpenAI-compatible server")
print(f" Model : {MODEL}")
print(f" URL   : http://{HOST}:{PORT}/v1")
print(f" Max len: {MAX_MODEL_LEN}")
print("=" * 60)
print()
print("[NEXT] 다른 터미널에서 다음을 실행해 공개 URL을 만드세요:")
print("       cloudflared tunnel --url http://localhost:8000")
print()

subprocess.run(cmd)
