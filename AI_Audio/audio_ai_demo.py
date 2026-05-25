#!/usr/bin/env python3
"""
음향 AI 인터랙티브 데모 — 강의용 Gradio 웹 서비스

탭 구성:
  1. 🔬 소리 분석       — Waveform / Spectrogram / Mel-Spectrogram / MFCC 시각화
  2. 🎙 음성 인식       — OpenAI Whisper (한국어·영어, 마이크 녹음 지원)
  3. 🔊 음성 합성 (TTS) — Edge TTS, 목소리 비교, Mel-Spectrogram 시각화
  4. 🎵 음원 분리       — HPSS(빠름) 또는 Demucs(딥러닝)
  5. 📊 소리 비교       — MFCC 코사인 유사도 + 차이 히트맵

실행 방법:
  pip install -r requirements_demo.txt
  python audio_ai_demo.py              # 로컬 (http://localhost:7860)
  python audio_ai_demo.py --share      # 공개 URL 자동 생성 (Gradio 터널)
  python audio_ai_demo.py --port 8080  # 포트 변경
"""

import argparse
import asyncio
import os
import platform
import subprocess
import tempfile
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import librosa
import librosa.display
import numpy as np
import soundfile as sf
import gradio as gr

# ─────────────────────────────────────────────────────────
# 한글 폰트 설정 (서버 최초 실행 시 자동 설치 시도)
# ─────────────────────────────────────────────────────────

def _setup_korean_font():
    """
    한글 폰트를 탐색하고 없으면 자동 설치합니다.
    Windows: Malgun Gothic / macOS: AppleGothic / Linux: NanumGothic
    """
    CANDIDATES = [
        "Malgun Gothic",   # Windows
        "AppleGothic",     # macOS
        "NanumGothic",     # Linux (fonts-nanum)
        "NanumBarunGothic",
        "UnDotum",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]

    available = {f.name for f in fm.fontManager.ttflist}

    for name in CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[Font] 한글 폰트 적용: {name}")
            return

    # Linux에서 NanumGothic 자동 설치
    if platform.system() == "Linux":
        print("[Font] 한글 폰트 없음 → fonts-nanum 설치 시도 중...")
        try:
            subprocess.run(
                ["apt-get", "install", "-y", "-q", "fonts-nanum"],
                check=True, capture_output=True, timeout=120,
            )
            # 폰트 캐시 재구성
            fm.fontManager.__init__()
            # 재탐색
            available = {f.name for f in fm.fontManager.ttflist}
            for name in ["NanumGothic", "NanumBarunGothic"]:
                if name in available:
                    plt.rcParams["font.family"] = name
                    plt.rcParams["axes.unicode_minus"] = False
                    print(f"[Font] 설치 완료 후 적용: {name}")
                    return
        except Exception as e:
            print(f"[Font] 자동 설치 실패: {e}")

    # 폰트를 못 찾으면 DejaVu Sans 유지 (한글 깨짐 대신 □ 표시)
    plt.rcParams["axes.unicode_minus"] = False
    print("[Font] 한글 폰트를 찾지 못했습니다. matplotlib 제목의 한글이 깨질 수 있습니다.")
    print("       수동 설치: sudo apt-get install fonts-nanum  (Linux)")


_setup_korean_font()

# ─────────────────────────────────────────────────────────
# 마이크 녹음 JavaScript
# gr.Button.click(fn=None, js=...) 로 Gradio 이벤트 시스템을 통해 실행
# → DOMPurify 우회, onclick 속성 불필요
# ─────────────────────────────────────────────────────────

# 녹음 시작/중지 토글 (Python 호출 없음, 순수 클라이언트)
_REC_TOGGLE_JS = """
async () => {
    const statusEl = document.getElementById('mic-status');
    const audioEl  = document.getElementById('mic-audio');
    const btn      = document.querySelector('#mic-toggle-btn button');

    if (!window._isRec) {
        window._recChunks = []; window._recB64 = null;
        if (audioEl) audioEl.style.display = 'none';

        try {
            /* 오디오 처리 OFF — Windows에서 자동 게인/노이즈 억제가
               소리를 0으로 만드는 경우 방지 */
            const stream = await navigator.mediaDevices.getUserMedia({audio: {
                echoCancellation:  false,
                noiseSuppression:  false,
                autoGainControl:   false,
                channelCount: 1
            }});

            /* 실시간 볼륨 미터 (브라우저가 실제로 소리를 잡는지 확인용) */
            const actx    = new AudioContext();
            const src     = actx.createMediaStreamSource(stream);
            const analyser= actx.createAnalyser();
            analyser.fftSize = 256;
            src.connect(analyser);
            const freqData = new Uint8Array(analyser.frequencyBinCount);

            /* 지원 코덱 우선순위 탐색 */
            const mime = ['audio/webm;codecs=opus','audio/webm',
                          'audio/ogg;codecs=opus','audio/mp4','']
                         .find(m => m === '' || MediaRecorder.isTypeSupported(m));
            window._mr = new MediaRecorder(stream, mime ? {mimeType: mime} : {});
            window._recChunks = [];
            window._mr.ondataavailable = e => { if (e.data.size > 0) window._recChunks.push(e.data); };
            window._mr.onstop = function() {
                const usedMime = window._mr.mimeType || 'audio/webm';
                const blob = new Blob(window._recChunks, {type: usedMime});
                clearInterval(window._recTimer);
                clearInterval(window._levelTimer);
                actx.close();
                window._isRec = false;
                if (btn) btn.textContent = '🔴  녹음 시작';
                if (statusEl) { statusEl.textContent = '⏳ 처리 중...'; statusEl.style.color='#555'; }

                const reader = new FileReader();
                reader.onloadend = () => {
                    const dataUrl = reader.result;
                    window._recB64 = usedMime + '||' + dataUrl.split(',')[1];
                    if (audioEl) { audioEl.src = dataUrl; audioEl.style.display = 'block'; }
                    if (statusEl) statusEl.textContent =
                        '✅ 녹음 완료! 재생으로 확인 후 초록 버튼을 눌러 인식하세요.';
                };
                reader.readAsDataURL(blob);
            };

            window._mr.start(100);
            window._isRec = true;
            if (btn) btn.textContent = '⏹  녹음 중지';

            /* 볼륨 바 표시 — 여기서 레벨이 0이면 브라우저가 소리를 못 잡는 것 */
            let s = 0;
            window._levelTimer = setInterval(() => {
                analyser.getByteFrequencyData(freqData);
                const level = Math.round(freqData.reduce((a,b)=>a+b,0) / freqData.length);
                const bar   = '▮'.repeat(Math.min(20, Math.floor(level / 5)));
                if (statusEl) statusEl.textContent =
                    '🔴 ' + s + '초  [' + bar.padEnd(20,'·') + '] ' + level;
            }, 100);
            window._recTimer = setInterval(() => { s++; }, 1000);

        } catch(e) {
            if (statusEl) { statusEl.textContent = '❌ ' + e.message; statusEl.style.color = '#c62828'; }
        }
    } else {
        window._mr.stop();
        window._mr.stream.getTracks().forEach(t => t.stop());
        clearInterval(window._recTimer);
        clearInterval(window._levelTimer);
    }
}
"""

# 녹음된 base64 데이터를 Gradio outputs 으로 반환
_REC_SEND_JS = "() => { return window._recB64 || ''; }"

# ─────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────

def _fig_to_pil(fig):
    """matplotlib Figure → PIL Image (서버 렌더링용)"""
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def _load(path, sr=22050, max_sec=30):
    """mp3 / wav / ogg / flac / m4a / webm 등 librosa+ffmpeg 지원 포맷 모두 로드"""
    try:
        y, sr_out = librosa.load(path, sr=sr, duration=max_sec)
    except Exception:
        # ffmpeg 백엔드로 재시도 (m4a, webm, aac 등)
        import audioread
        with audioread.audio_open(path) as f:
            sr_out = f.samplerate
            raw = b"".join(f)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if f.channels > 1:
            arr = arr.reshape(-1, f.channels).mean(axis=1)
        y = librosa.resample(arr, orig_sr=sr_out, target_sr=sr)
        sr_out = sr
    return y.astype(np.float32), sr_out


# ─────────────────────────────────────────────────────────
# Tab 1 — 소리 분석
# ─────────────────────────────────────────────────────────

def analyze_audio(audio_path):
    if audio_path is None:
        return None, "⚠️ 오디오 파일을 업로드해주세요."

    y, sr = _load(audio_path)
    duration = len(y) / sr

    # Feature extraction
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    try:
        tempo = float(librosa.beat.beat_track(y=y, sr=sr)[0])
    except Exception:
        tempo = 0.0

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle("소리 분석 결과", fontsize=16, fontweight="bold", y=1.01)

    axes[0].set_title("① Waveform — 시간에 따른 소리의 세기 (사람 귀에 가장 가까운 표현)", fontweight="bold")
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#2196F3")
    axes[0].set_xlabel("Time (s)")

    axes[1].set_title("② Spectrogram — 소리의 X-ray (시간 × 주파수 × 세기)", fontweight="bold")
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                             x_axis="time", y_axis="hz",
                             ax=axes[1], cmap="magma")
    axes[1].set_ylim(0, 8000)
    axes[1].set_ylabel("Frequency (Hz)")

    axes[2].set_title("③ Mel-Spectrogram — AI가 보는 소리 사진 (사람 귀 인식 방식 반영)", fontweight="bold")
    img = librosa.display.specshow(S_mel_db, sr=sr, hop_length=512,
                                   x_axis="time", y_axis="mel",
                                   ax=axes[2], cmap="magma", fmax=8000)
    axes[2].set_ylabel("Mel Frequency")
    fig.colorbar(img, ax=axes[2], format="%+2.0f dB")

    axes[3].set_title("④ MFCC — 소리의 DNA (음성인식·화자인식 AI에서 가장 많이 사용)", fontweight="bold")
    librosa.display.specshow(mfcc, sr=sr, hop_length=512,
                             x_axis="time", ax=axes[3], cmap="coolwarm")
    axes[3].set_ylabel("MFCC 계수")
    axes[3].set_xlabel("Time (s)")

    plt.tight_layout()
    vis = _fig_to_pil(fig)

    info = (
        f"📊 오디오 분석 결과\n\n"
        f"⏱  재생 시간      : {duration:.2f}초\n"
        f"🎚  샘플링 레이트  : {sr:,} Hz\n"
        f"🎵  추정 BPM      : {tempo:.1f}\n"
        f"🌟  평균 밝기(SC)  : {np.mean(sc):.0f} Hz\n"
        f"📈  영교차율(ZCR)  : {np.mean(zcr):.4f}\n\n"
        f"💡 그래프 해설\n"
        f"   ① Waveform     : 언제 크고 작은지만 보임\n"
        f"   ② Spectrogram  : 어떤 음이 언제 나는지 보임\n"
        f"   ③ Mel-Spec     : 사람 귀 기준으로 보정 → AI 입력으로 사용\n"
        f"   ④ MFCC         : 소리의 특징 벡터 → AI 학습 데이터"
    )
    return vis, info


# ─────────────────────────────────────────────────────────
# Tab 2 — 음성 인식 (Whisper)
# ─────────────────────────────────────────────────────────

_whisper_model = None

def _get_whisper(size="small"):
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"[Whisper] 모델 로딩 중: {size} …")
        _whisper_model = whisper.load_model(size)
        print("[Whisper] 로드 완료!")
    return _whisper_model


def transcribe_audio(audio_path, language):
    if audio_path is None:
        return "⚠️ 오디오 파일을 업로드하거나 마이크로 녹음해주세요.", None

    try:
        model = _get_whisper()
        lang_map = {"자동 감지": None, "한국어": "ko", "영어": "en"}
        result = model.transcribe(audio_path, language=lang_map.get(language))
        text = result["text"].strip()
        detected = result.get("language", "?")
        segments = result.get("segments", [])

        seg_lines = "\n".join(
            f"  [{s['start']:.1f}s~{s['end']:.1f}s] {s['text'].strip()}"
            for s in segments[:10]
        )

        output = (
            f"✅ 인식 완료!\n\n"
            f"🗣  인식된 텍스트:\n{text}\n\n"
            f"🌐  감지된 언어: {detected}\n\n"
            f"📝 세그먼트별:\n{seg_lines}\n\n"
            f"💡 Whisper 소개\n"
            f"   • OpenAI 개발, 680,000시간 오디오로 학습\n"
            f"   • 99개 언어 지원 (한국어 포함)\n"
            f"   • 동작 방식: 오디오 → Mel-Spectrogram → 텍스트 변환\n"
            f"   ※ 마이크는 '녹음 완료 후 인식' 방식 (실시간 스트리밍 아님)"
        )

        # Visualization
        y, sr = _load(audio_path, sr=16000)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, axes = plt.subplots(2, 1, figsize=(14, 7))
        axes[0].set_title("Waveform (음성 파형)", fontweight="bold")
        librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#4CAF50")
        axes[0].set_xlabel("Time (s)")

        axes[1].set_title("Mel-Spectrogram — Whisper 가 이 이미지를 보고 문자를 읽습니다", fontweight="bold")
        librosa.display.specshow(S_db, sr=sr, hop_length=160,
                                 x_axis="time", y_axis="mel",
                                 ax=axes[1], cmap="magma", fmax=8000)
        for seg in segments[:8]:
            axes[1].axvline(x=seg["start"], color="white", alpha=0.6, linewidth=1)
        axes[1].set_ylabel("Mel Frequency")
        axes[1].set_xlabel("Time (s)")
        plt.tight_layout()
        vis = _fig_to_pil(fig)

        return output, vis

    except ModuleNotFoundError:
        return "⚠️ Whisper 미설치\n  pip install openai-whisper", None
    except Exception as e:
        return f"오류: {e}", None


# ─────────────────────────────────────────────────────────
# Tab 3 — 음성 합성 (TTS)
# ─────────────────────────────────────────────────────────

VOICE_MAP = {
    "한국어 여성 (SunHi)":  "ko-KR-SunHiNeural",
    "한국어 남성 (InJoon)": "ko-KR-InJoonNeural",
    "영어 여성 (Aria)":     "en-US-AriaNeural",
    "영어 남성 (Guy)":      "en-US-GuyNeural",
    "영어 여성 (Jenny)":    "en-US-JennyNeural",
}

async def _do_tts(text, voice):
    import edge_tts
    f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    out = f.name
    f.close()
    await edge_tts.Communicate(text, voice).save(out)
    return out


def generate_tts(text, voice_label):
    if not text.strip():
        return None, None, "⚠️ 텍스트를 입력해주세요."

    voice = VOICE_MAP.get(voice_label, "ko-KR-SunHiNeural")
    try:
        out_path = asyncio.run(_do_tts(text, voice))
    except ModuleNotFoundError:
        return None, None, "⚠️ edge-tts 미설치\n  pip install edge-tts"
    except Exception as e:
        return None, None, f"오류: {e}"

    y, sr = _load(out_path)
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].set_title(f"생성된 음성 — Waveform  [{voice_label}]", fontweight="bold")
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#9C27B0")
    axes[0].set_xlabel("Time (s)")

    axes[1].set_title("Mel-Spectrogram — TTS 보코더의 최종 출력", fontweight="bold")
    img = librosa.display.specshow(S_mel_db, sr=sr, hop_length=512,
                                   x_axis="time", y_axis="mel",
                                   ax=axes[1], cmap="magma", fmax=8000)
    axes[1].set_ylabel("Mel Frequency")
    axes[1].set_xlabel("Time (s)")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    plt.tight_layout()
    vis = _fig_to_pil(fig)

    info = (
        f"✅ 음성 합성 완료!\n\n"
        f"📝 입력 텍스트 : {text[:80]}{'…' if len(text)>80 else ''}\n"
        f"🗣  목소리      : {voice_label}\n"
        f"⏱  재생 시간   : {len(y)/sr:.2f}초\n\n"
        f"💡 TTS 파이프라인\n"
        f"   텍스트 → [TTS 모델] → Mel-Spectrogram → [보코더] → 음성\n\n"
        f"📊 Mel-Spectrogram 관찰 포인트\n"
        f"   • 밝은 가로줄 = 각 음소의 주요 주파수\n"
        f"   • 가로 너비   = 해당 음절의 발화 시간\n"
        f"   • 에너지 분포 = 목소리 음색(높낮이) 차이"
    )
    return out_path, vis, info


def compare_tts(text):
    """같은 문장을 남성/여성으로 생성 후 Mel-Spectrogram 비교"""
    if not text.strip():
        return None, None, None, "⚠️ 텍스트를 입력해주세요."

    results = {}
    for label, voice in [("한국어 여성 (SunHi)", "ko-KR-SunHiNeural"),
                          ("한국어 남성 (InJoon)", "ko-KR-InJoonNeural")]:
        try:
            path = asyncio.run(_do_tts(text, voice))
            results[label] = path
        except Exception:
            return None, None, None, "⚠️ edge-tts 오류"

    ys, srs, labels = [], [], []
    for lbl, path in results.items():
        y, sr = _load(path)
        ys.append(y)
        srs.append(sr)
        labels.append(lbl)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    colors = ["#E91E63", "#2196F3"]
    for i, (y, sr, lbl, col) in enumerate(zip(ys, srs, labels, colors)):
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)

        axes[i][0].set_title(f"{lbl} — Waveform", fontweight="bold")
        librosa.display.waveshow(y, sr=sr, ax=axes[i][0], color=col)

        axes[i][1].set_title(f"{lbl} — Mel-Spectrogram", fontweight="bold")
        librosa.display.specshow(S_db, sr=sr, hop_length=512,
                                 x_axis="time", y_axis="mel",
                                 ax=axes[i][1], cmap="magma", fmax=8000)

    plt.suptitle(f'"{text[:50]}" — 남성 vs 여성 목소리 비교', fontsize=13, fontweight="bold")
    plt.tight_layout()
    vis = _fig_to_pil(fig)

    paths = list(results.values())
    note = (
        "💡 같은 문장인데 Mel-Spectrogram 구조가 다릅니다!\n"
        "   • 여성 목소리: 에너지가 더 높은 주파수 대역까지 분포\n"
        "   • 남성 목소리: 에너지가 낮은 주파수 대역에 집중\n\n"
        "   AI 커버 = 이 '에너지 분포(음색)'만 바꾸고 내용 패턴은 유지하는 기술!"
    )
    return paths[0], paths[1], vis, note


# ─────────────────────────────────────────────────────────
# Tab 4 — 음원 분리
# ─────────────────────────────────────────────────────────

def separate_audio(audio_path, method):
    if audio_path is None:
        return None, None, None, "⚠️ 오디오 파일을 업로드해주세요."

    y, sr = _load(audio_path, max_sec=30)

    if "HPSS" in method:
        return _hpss(y, sr)
    else:
        try:
            return _demucs(audio_path, y, sr)
        except Exception as e:
            note = f"⚠️ Demucs 오류: {e}\nHPSS 방법으로 대체합니다."
            a1, a2, vis, info = _hpss(y, sr)
            return a1, a2, vis, note + "\n\n" + info


def _hpss(y, sr):
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    H, P = librosa.decompose.hpss(D, margin=3.0)
    y_h = librosa.istft(H, hop_length=512).astype(np.float32)
    y_p = librosa.istft(P, hop_length=512).astype(np.float32)

    h_path = tempfile.mktemp(suffix="_harmonic.wav")
    p_path = tempfile.mktemp(suffix="_percussive.wav")
    sf.write(h_path, y_h, sr)
    sf.write(p_path, y_p, sr)

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    _row(fig, axes, 0, "원본", y, librosa.amplitude_to_db(np.abs(D), ref=np.max),
         sr, "#607D8B", is_mel=False)

    S_h = librosa.amplitude_to_db(np.abs(H), ref=np.max)
    _row(fig, axes, 1, "화성음 (멜로디·화음)", y_h, S_h, sr, "#2196F3", is_mel=False)

    S_p = librosa.amplitude_to_db(np.abs(P), ref=np.max)
    _row(fig, axes, 2, "타악기 (드럼·타건음)", y_p, S_p, sr, "#F44336", is_mel=False)

    plt.suptitle("HPSS 음원 분리 결과", fontsize=14, fontweight="bold")
    plt.tight_layout()
    vis = _fig_to_pil(fig)

    info = (
        "✅ HPSS 분리 완료!\n\n"
        "📌 원리\n"
        "   • 화성음 (Harmonic): 주파수가 지속됨\n"
        "     → Spectrogram에서 가로 줄무늬\n"
        "   • 타악기 (Percussive): 순간적·충격적인 소리\n"
        "     → Spectrogram에서 세로 줄무늬\n\n"
        "💡 AI가 Spectrogram의 패턴 방향을 보고 소리를 분리합니다!"
    )
    return h_path, p_path, vis, info


def _demucs(audio_path, y_fallback, sr_fallback):
    import subprocess
    tmp_wav = tempfile.mktemp(suffix=".wav")
    sf.write(tmp_wav, y_fallback, sr_fallback)
    out_dir = tempfile.mkdtemp()

    res = subprocess.run(
        ["python", "-m", "demucs", "--two-stems=vocals", "-o", out_dir, tmp_wav],
        capture_output=True, text=True, timeout=180
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr[:500])

    base = os.path.splitext(os.path.basename(tmp_wav))[0]
    v_path = os.path.join(out_dir, "htdemucs", base, "vocals.wav")
    m_path = os.path.join(out_dir, "htdemucs", base, "no_vocals.wav")

    if not os.path.exists(v_path):
        raise FileNotFoundError("Demucs 출력 파일 없음")

    y_v, sr = _load(v_path)
    y_m, _  = _load(m_path, sr=sr)

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    S_v = librosa.feature.melspectrogram(y=y_v, sr=sr, n_mels=80, fmax=8000)
    S_m = librosa.feature.melspectrogram(y=y_m, sr=sr, n_mels=80, fmax=8000)
    S_orig = librosa.feature.melspectrogram(y=y_fallback, sr=sr_fallback, n_mels=80, fmax=8000)

    for i, (y_i, S_i, lbl, col) in enumerate([
        (y_fallback, S_orig, "원본",     "#607D8B"),
        (y_v,        S_v,   "보컬",     "#E91E63"),
        (y_m,        S_m,   "반주 (MR)", "#00BCD4"),
    ]):
        _row(fig, axes, i, lbl, y_i,
             librosa.power_to_db(S_i, ref=np.max),
             sr, col, is_mel=True)

    plt.suptitle("Demucs 음원 분리 결과 (보컬 / 반주)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    vis = _fig_to_pil(fig)

    info = (
        "✅ Demucs 분리 완료!\n\n"
        "📌 Demucs (Meta/Facebook AI)\n"
        "   • Transformer + U-Net 아키텍처\n"
        "   • 딥러닝 기반 고품질 분리\n\n"
        "💡 AI 커버 제작 파이프라인\n"
        "   1. Demucs로 원곡에서 MR(반주) 추출\n"
        "   2. Voice Conversion으로 보컬 목소리 교체\n"
        "   3. 교체된 보컬 + MR 합치면 AI 커버 완성!"
    )
    return v_path, m_path, vis, info


def _row(fig, axes, i, label, y, S_db, sr, color, is_mel=True):
    axes[i][0].set_title(f"{label} — Waveform", fontweight="bold")
    librosa.display.waveshow(y, sr=sr, ax=axes[i][0], color=color)

    axes[i][1].set_title(f"{label} — {'Mel-' if is_mel else ''}Spectrogram", fontweight="bold")
    y_axis = "mel" if is_mel else "hz"
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                             x_axis="time", y_axis=y_axis,
                             ax=axes[i][1], cmap="magma",
                             fmax=8000 if is_mel else None)
    if not is_mel:
        axes[i][1].set_ylim(0, 8000)


# ─────────────────────────────────────────────────────────
# Gradio UI 구성
# ─────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="🎵 음향 AI 인터랙티브 데모",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.HTML("""
        <div style='text-align:center;padding:24px 0 8px'>
            <h1 style='font-size:2rem;margin:0'>🎵 음향 AI 인터랙티브 데모</h1>
            <p style='color:#666;font-size:1rem;margin-top:8px'>
                소리를 AI가 어떻게 이해하고 처리하는지 직접 체험해보세요!
            </p>
        </div>
        """)

        # ── Tab 1 ──────────────────────────────────────
        with gr.Tabs():
            with gr.TabItem("🔬 소리 분석"):
                gr.Markdown("""
### 오디오를 업로드하면 AI가 보는 방식으로 4가지 시각화를 동시에 보여줍니다
- **Waveform** : 소리의 심전도
- **Spectrogram** : 소리의 X-ray
- **Mel-Spectrogram** : AI용 소리 사진
- **MFCC** : 소리의 DNA (AI 모델 학습에 사용)
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        a1_in = gr.Audio(label="오디오 업로드 (mp3·wav·ogg 등)", type="filepath")
                        a1_btn = gr.Button("🔍 분석하기", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        a1_img = gr.Image(label="시각화 결과", type="pil")
                        a1_txt = gr.Textbox(label="분석 정보", lines=12)
                a1_btn.click(analyze_audio, [a1_in], [a1_img, a1_txt])

            # ── Tab 2 ──────────────────────────────────
            with gr.TabItem("🎙 음성 인식 (Whisper)"):
                gr.Markdown("### OpenAI Whisper : 음성 → 텍스트")
                with gr.Row():
                    # ── 왼쪽 ─────────────────────────────
                    with gr.Column(scale=1):

                        # 마이크 녹음 섹션
                        gr.HTML("""
<div style="border:2px solid #c5cae9;border-radius:12px;padding:18px;
            background:#f8f9ff;margin-bottom:14px">
  <div style="font-size:15px;font-weight:bold;margin-bottom:10px;color:#3949ab">
    🎤 방법 A — 마이크 녹음 (HTTPS 필요)
  </div>
  <div id="mic-status" style="font-size:13px;color:#555;margin-bottom:8px;min-height:18px">
    아래 버튼을 눌러 녹음하세요
  </div>
  <audio id="mic-audio" controls
    style="width:100%;margin-bottom:8px;border-radius:8px;display:none"></audio>
</div>
                        """)

                        # Gradio 네이티브 버튼 — JS는 .click(fn=None, js=) 로 실행
                        rec_toggle_btn = gr.Button(
                            "🔴  녹음 시작",
                            variant="stop", size="lg",
                            elem_id="mic-toggle-btn",
                        )
                        rec_b64 = gr.Textbox(visible=False, label="rec_b64")
                        rec_send_btn = gr.Button(
                            "✅ 이 녹음으로 음성 인식하기",
                            variant="secondary", size="lg",
                        )

                        gr.HTML("""
<div style="border:2px solid #c8e6c9;border-radius:12px;padding:14px;
            background:#f1f8f1;margin:14px 0">
  <div style="font-size:15px;font-weight:bold;margin-bottom:8px;color:#2e7d32">
    📁 방법 B — 파일 업로드 (mp3 · wav · ogg · m4a)
  </div>
</div>
                        """)
                        a2_in = gr.Audio(
                            label="파일을 끌어다 놓거나 클릭해서 선택",
                            type="filepath",
                            sources=["upload"],
                        )

                        a2_lang = gr.Radio(
                            ["자동 감지", "한국어", "영어"],
                            value="자동 감지", label="언어 선택",
                        )
                        a2_file_btn = gr.Button(
                            "📁 파일로 음성 인식하기 →",
                            variant="primary", size="lg",
                        )

                    # ── 오른쪽: 결과 ─────────────────────
                    with gr.Column(scale=2):
                        a2_txt = gr.Textbox(label="인식 결과", lines=14)
                        a2_img = gr.Image(label="Mel-Spectrogram 시각화", type="pil")

                # 녹음 토글 — 순수 JS, Python 호출 없음
                rec_toggle_btn.click(fn=None, inputs=[], outputs=[], js=_REC_TOGGLE_JS)

                # 녹음 데이터 → rec_b64 textbox 업데이트 (JS return → Gradio output)
                rec_send_btn.click(fn=None, inputs=[], outputs=[rec_b64], js=_REC_SEND_JS)

                # rec_b64 변경 → Python 전사
                def _transcribe_from_b64(b64_payload, language):
                    if not b64_payload or "||" not in b64_payload:
                        return "⚠️ 먼저 마이크로 녹음하세요.", None
                    import base64 as _b64
                    try:
                        mime, data = b64_payload.split("||", 1)
                        ext = ".webm" if "webm" in mime else ".wav"
                        raw = _b64.b64decode(data)
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                            f.write(raw)
                            tmp = f.name
                        return transcribe_audio(tmp, language)
                    except Exception as e:
                        return f"오류: {e}", None

                rec_b64.change(_transcribe_from_b64, [rec_b64, a2_lang], [a2_txt, a2_img])

                # 파일 업로드 경로
                a2_file_btn.click(transcribe_audio, [a2_in, a2_lang], [a2_txt, a2_img])

            # ── Tab 3 ──────────────────────────────────
            with gr.TabItem("🔊 음성 합성 (TTS)"):
                gr.Markdown("""
### Edge TTS (Microsoft) : 텍스트 → 음성
- 텍스트 → Mel-Spectrogram → 보코더 → 음성 파이프라인
- 한국어 / 영어, 남성 / 여성 목소리 선택 가능
                """)
                with gr.Tabs():
                    with gr.TabItem("단일 생성"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                t3_text = gr.Textbox(
                                    label="음성으로 변환할 텍스트",
                                    lines=4,
                                    value="안녕하세요! AI가 이 텍스트를 음성으로 변환합니다.",
                                )
                                t3_voice = gr.Dropdown(
                                    list(VOICE_MAP.keys()),
                                    value="한국어 여성 (SunHi)",
                                    label="목소리 선택",
                                )
                                t3_btn = gr.Button("🔊 음성 생성", variant="primary", size="lg")
                            with gr.Column(scale=2):
                                t3_audio = gr.Audio(label="생성된 음성", type="filepath")
                                t3_img   = gr.Image(label="Mel-Spectrogram", type="pil")
                                t3_info  = gr.Textbox(label="분석 정보", lines=10)
                        t3_btn.click(generate_tts, [t3_text, t3_voice], [t3_audio, t3_img, t3_info])

                    with gr.TabItem("남성 vs 여성 비교"):
                        gr.Markdown("같은 문장을 한국어 남성/여성 목소리로 생성해서 Mel-Spectrogram을 비교합니다.")
                        with gr.Row():
                            with gr.Column(scale=1):
                                t3c_text = gr.Textbox(
                                    label="비교할 텍스트",
                                    value="인공지능이 소리를 분석하는 방법을 배워봅시다.",
                                    lines=3,
                                )
                                t3c_btn = gr.Button("🔄 비교 생성", variant="primary", size="lg")
                            with gr.Column(scale=2):
                                t3c_f   = gr.Audio(label="여성 목소리", type="filepath")
                                t3c_m   = gr.Audio(label="남성 목소리", type="filepath")
                                t3c_img = gr.Image(label="비교 시각화", type="pil")
                                t3c_note = gr.Textbox(label="관찰 포인트", lines=6)
                        t3c_btn.click(compare_tts, [t3c_text], [t3c_f, t3c_m, t3c_img, t3c_note])

            # ── Tab 4 ──────────────────────────────────
            with gr.TabItem("🎵 음원 분리"):
                gr.Markdown("""
### Source Separation : 소리를 성분별로 분리
- **HPSS** (빠름) : 화성음(멜로디·화음) vs 타악기(드럼) 분리
- **Demucs** (딥러닝) : 보컬 vs 반주(MR) 분리 — 별도 설치 필요
- 💡 AI 커버의 핵심 기술! 원곡에서 MR 추출에 바로 사용됩니다
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        a3_in  = gr.Audio(label="오디오 업로드 (최대 30초)", type="filepath")
                        a3_met = gr.Radio(
                            ["화성/타악기 분리 (HPSS, 빠름)", "보컬/반주 분리 (Demucs, 딥러닝)"],
                            value="화성/타악기 분리 (HPSS, 빠름)",
                            label="분리 방법",
                        )
                        a3_btn = gr.Button("🎵 음원 분리 시작", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        a3_img  = gr.Image(label="분리 결과 시각화", type="pil")
                        a3_info = gr.Textbox(label="분석 정보", lines=10)
                with gr.Row():
                    a3_o1 = gr.Audio(label="분리 결과 A (화성음 / 보컬)", type="filepath")
                    a3_o2 = gr.Audio(label="분리 결과 B (타악기 / 반주)", type="filepath")
                a3_btn.click(separate_audio, [a3_in, a3_met], [a3_o1, a3_o2, a3_img, a3_info])

        gr.HTML("""
        <div style='text-align:center;padding:16px;color:#aaa;font-size:13px'>
            음향 AI 인터랙티브 데모 | 강의 실습용
        </div>
        """)

    return demo


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="음향 AI 인터랙티브 데모")
    parser.add_argument("--host",  default="0.0.0.0", help="서버 호스트 (기본: 0.0.0.0)")
    parser.add_argument("--port",  type=int, default=7860, help="서버 포트 (기본: 7860)")
    parser.add_argument("--share", action="store_true", help="Gradio 공개 URL 생성")
    args = parser.parse_args()

    print("=" * 58)
    print("  🎵 음향 AI 인터랙티브 데모 시작")
    print("=" * 58)
    print(f"  로컬 접속  : http://localhost:{args.port}")
    print(f"  네트워크   : http://<서버IP>:{args.port}")
    if args.share:
        print("  공개 URL   : 잠시 후 아래에 표시됩니다...")
    else:
        print("  외부 공유  : python audio_ai_demo.py --share")
    print("=" * 58)

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        favicon_path=None,
    )
