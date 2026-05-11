"""
자율주행 AI 시뮬레이터 — 웹 서버
================================
실행: python server.py
→ 자동으로 외부 접속 가능한 공개 URL이 생성됩니다!

사전 설치:
  pip install flask pyngrok

ngrok 인증 (최초 1회, 무료):
  1. https://ngrok.com 가입
  2. https://dashboard.ngrok.com/get-started/your-authtoken 에서 토큰 복사
  3. 터미널에서: ngrok config add-authtoken YOUR_TOKEN

  또는 아래 NGROK_AUTH_TOKEN 변수에 직접 입력해도 됩니다.
"""
from flask import Flask, send_from_directory, jsonify
import os, socket

app = Flask(__name__, static_folder='static', template_folder='templates')

# ── ngrok 인증 토큰 (여기에 직접 입력해도 됩니다) ──
NGROK_AUTH_TOKEN = ""  # 예: "2abc123def456..."

# 트랙 이미지 경로
TRACK_DIR = os.path.join(os.path.dirname(__file__), 'static', 'tracks')
os.makedirs(TRACK_DIR, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/tracks')
def list_tracks():
    tracks = []
    for f in sorted(os.listdir(TRACK_DIR)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            tracks.append({"name": f, "url": f"/static/tracks/{f}"})
    return jsonify(tracks)

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def start_ngrok(port):
    """ngrok 터널을 자동으로 시작하고 공개 URL을 반환"""
    try:
        from pyngrok import ngrok

        if NGROK_AUTH_TOKEN:
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)

        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url

        if public_url.startswith("http://"):
            public_url = public_url.replace("http://", "https://", 1)

        return public_url

    except ImportError:
        print()
        print("  ⚠️  pyngrok이 설치되지 않았습니다.")
        print("  외부 접속을 원하면: pip install pyngrok")
        print()
        return None

    except Exception as e:
        error_msg = str(e)
        if "authtoken" in error_msg.lower() or "ERR_NGROK" in error_msg:
            print()
            print("  ⚠️  ngrok 인증이 필요합니다 (무료, 최초 1회)")
            print()
            print("  방법 1: 터미널에서 실행")
            print("    ngrok config add-authtoken YOUR_TOKEN")
            print()
            print("  방법 2: server.py의 NGROK_AUTH_TOKEN 변수에 직접 입력")
            print()
            print("  토큰 발급: https://dashboard.ngrok.com/get-started/your-authtoken")
            print()
        else:
            print(f"  ⚠️  ngrok 오류: {e}")
        return None


if __name__ == '__main__':
    port = 5000
    ip = get_local_ip()

    print()
    print("  ╔════════════════════════════════════════════════╗")
    print("  ║   🚗 자율주행 AI 시뮬레이터 — 웹 버전         ║")
    print("  ╚════════════════════════════════════════════════╝")
    print()

    # ngrok 시작
    public_url = start_ngrok(port)

    print(f"  📍 로컬 접속:    http://localhost:{port}")
    print(f"  📍 내부 네트워크: http://{ip}:{port}")

    if public_url:
        print()
        print(f"  🌐 외부 접속 URL (학생들에게 공유!):")
        print(f"  ┌─────────────────────────────────────────┐")
        print(f"  │  {public_url:<40s}│")
        print(f"  └─────────────────────────────────────────┘")
        print()
        print(f"  이 URL은 어디서든 접속 가능합니다!")
        print(f"  (서버를 종료하면 URL도 비활성화됩니다)")
    else:
        print()
        print(f"  💡 같은 Wi-Fi라면 http://{ip}:{port} 로 접속 가능")

    print()
    print(f"  📁 트랙 이미지: static/tracks/ 폴더에 넣으세요")
    print(f"  🛑 종료: Ctrl+C")
    print()

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
