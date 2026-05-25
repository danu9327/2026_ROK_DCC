#!/usr/bin/env python3
"""
AI 커버 파이프라인 — Gradio 웹 서비스

1단계: 🎤 목소리 등록  — 파일 업로드 or 마이크 녹음 + 음성 분석
2단계: 🎵 노래 분리    — Demucs(보컬/MR) + Whisper(가사 추출)
3단계: 🎙 AI 커버 생성 — FreeVC 보컬 변환 + MR 믹싱

실행:
  pip install -r requirements_cover.txt
  python ai_cover_pipeline.py --share   # HTTPS URL (마이크 녹음 시 필요)
"""

import argparse
import asyncio
import base64
import os
import platform
import subprocess
import tempfile
import warnings
warnings.filterwarnings("ignore")

# 1. Matplotlib 백엔드 설정을 가장 먼저 실행 (가장 중요)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 2. 그 다음 오디오 및 기타 라이브러리 임포트
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import gradio as gr

# ─────────────────────────────────────────────────────────
# 한글 폰트
# ─────────────────────────────────────────────────────────

def _setup_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                  "NanumBarunGothic", "UnDotum", "Noto Sans CJK KR"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    if platform.system() == "Linux":
        try:
            subprocess.run(["apt-get", "install", "-y", "-q", "fonts-nanum"],
                           check=True, capture_output=True, timeout=120)
            fm.fontManager.__init__()
            plt.rcParams["font.family"] = "NanumGothic"
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False

_setup_korean_font()

# ─────────────────────────────────────────────────────────
# 마이크 녹음 JS  (gr.Button.click(fn=None, js=...) 로 실행)
# ─────────────────────────────────────────────────────────

_REC_TOGGLE_JS = """
async () => {
    const statusEl = document.getElementById('mic-status');
    const audioEl  = document.getElementById('mic-audio');
    const btn      = document.querySelector('#mic-toggle-btn button');
    const dlBtn    = document.getElementById('mic-download'); // 다운로드 버튼

    if (!window._isRec) {
        window._recChunks = []; window._recB64 = null;
        if (audioEl) audioEl.style.display = 'none';
        if (dlBtn) dlBtn.style.display = 'none'; // 녹음 시작 시 숨김

        try {
            const stream = await navigator.mediaDevices.getUserMedia({audio: {
                echoCancellation: false, noiseSuppression: false,
                autoGainControl: false, channelCount: 1
            }});
            const actx = new AudioContext();
            const src  = actx.createMediaStreamSource(stream);
            const an   = actx.createAnalyser(); an.fftSize = 256;
            src.connect(an);
            const fd = new Uint8Array(an.frequencyBinCount);
            const mime = ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','']
                         .find(m => m==='' || MediaRecorder.isTypeSupported(m));
            window._mr2 = new MediaRecorder(stream, mime ? {mimeType: mime} : {});
            window._recChunks = [];
            window._mr2.ondataavailable = e => { if (e.data.size>0) window._recChunks.push(e.data); };
            window._mr2.onstop = function() {
                const usedMime = window._mr2.mimeType || 'audio/webm';
                const blob = new Blob(window._recChunks, {type: usedMime});
                clearInterval(window._recTimer2); clearInterval(window._levelTimer2);
                actx.close(); window._isRec = false;
                if (btn) btn.textContent = '🔴  녹음 시작';
                if (statusEl) statusEl.textContent = '⏳ 처리 중...';
                
                const reader = new FileReader();
                reader.onloadend = () => {
                    const dataUrl = reader.result;
                    window._recB64 = usedMime + '||' + dataUrl.split(',')[1];
                    
                    if (audioEl) { audioEl.src = dataUrl; audioEl.style.display = 'block'; }
                    
                    // 녹음 완료 시 다운로드 버튼 활성화
                    if (dlBtn) {
                        dlBtn.href = dataUrl;
                        dlBtn.download = usedMime.includes('webm') ? 'my_recorded_voice.webm' : 'my_recorded_voice.ogg';
                        dlBtn.style.display = 'inline-block';
                    }

                    if (statusEl) statusEl.textContent = '✅ 녹음 완료! 아래 초록 버튼을 눌러 등록하세요.';
                };
                reader.readAsDataURL(blob);
            };
            window._mr2.start(100); window._isRec = true;
            if (btn) btn.textContent = '⏹  녹음 중지';
            let s=0;
            window._levelTimer2 = setInterval(() => {
                an.getByteFrequencyData(fd);
                const lv = Math.round(fd.reduce((a,b)=>a+b,0)/fd.length);
                const bar = '▮'.repeat(Math.min(20, Math.floor(lv/5)));
                if (statusEl) statusEl.textContent = '🔴 '+s+'초  ['+bar.padEnd(20,'·')+'] '+lv;
            }, 100);
            window._recTimer2 = setInterval(() => { s++; }, 1000);
        } catch(e) {
            if (statusEl) { statusEl.textContent='❌ '+e.message; statusEl.style.color='#c62828'; }
        }
    } else {
        window._mr2.stop();
        window._mr2.stream.getTracks().forEach(t=>t.stop());
        clearInterval(window._recTimer2); clearInterval(window._levelTimer2);
    }
}
"""

_REC_SEND_JS = "() => { return window._recB64 || ''; }"

# ─────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────

def _fig_to_pil(fig):
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def _load(path, sr=22050, duration=None):
    try:
        y, r = librosa.load(path, sr=sr, duration=duration)
    except Exception:
        import audioread
        with audioread.audio_open(path) as f:
            r = f.samplerate
            raw = b"".join(f)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if f.channels > 1:
            arr = arr.reshape(-1, f.channels).mean(axis=1)
        y = librosa.resample(arr, orig_sr=r, target_sr=sr)
        r = sr
    return y.astype(np.float32), r


def _b64_to_file(payload: str) -> str | None:
    """base64 payload → 임시 파일 경로"""
    if not payload or "||" not in payload:
        return None
    mime, data = payload.split("||", 1)
    ext = ".webm" if "webm" in mime else ".wav"
    raw = base64.b64decode(data)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(raw)
        return f.name

# ─────────────────────────────────────────────────────────
# 1단계: 목소리 분석
# ─────────────────────────────────────────────────────────

def analyze_voice(audio_path):
    if audio_path is None:
        return None, "⚠️ 목소리를 업로드하거나 녹음하세요."

    y, sr = _load(audio_path, duration=60)

    # F0 (pitch)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=700, sr=sr)
    f0_voiced = f0[voiced & ~np.isnan(f0)]

    # Features
    S_mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000), ref=np.max)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    sc    = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    axes[0].set_title("Waveform", fontweight="bold")
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#2196F3")

    times = librosa.times_like(f0, sr=sr)
    axes[1].set_title("F0 (음높이 궤적) — 목소리 음역대", fontweight="bold")
    if len(f0_voiced) > 0:
        axes[1].plot(times[voiced & ~np.isnan(f0)], f0_voiced,
                     color="#E91E63", linewidth=1.5)
        axes[1].set_ylim(max(0, f0_voiced.min()-50), f0_voiced.max()+50)
    axes[1].set_ylabel("Hz"); axes[1].set_xlabel("Time (s)")

    img2 = librosa.display.specshow(S_mel, sr=sr, hop_length=512,
                                    x_axis="time", y_axis="mel",
                                    ax=axes[2], cmap="magma", fmax=8000)
    axes[2].set_title("Mel-Spectrogram — AI가 보는 목소리", fontweight="bold")
    fig.colorbar(img2, ax=axes[2], format="%+2.0f dB")

    librosa.display.specshow(mfcc, sr=sr, hop_length=512,
                             x_axis="time", ax=axes[3], cmap="coolwarm")
    axes[3].set_title("MFCC — 음색 지문 (Voice Conversion의 변환 대상)", fontweight="bold")

    plt.tight_layout()
    vis = _fig_to_pil(fig)

    f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0
    f0_min  = float(np.min(f0_voiced))  if len(f0_voiced) > 0 else 0
    f0_max  = float(np.max(f0_voiced))  if len(f0_voiced) > 0 else 0

    if   f0_mean < 150: vtype = "베이스 (저음 남성)"
    elif f0_mean < 185: vtype = "바리톤 (중저음 남성)"
    elif f0_mean < 255: vtype = "테너 (고음 남성)"
    elif f0_mean < 300: vtype = "알토 (저음 여성)"
    elif f0_mean < 370: vtype = "메조소프라노 (여성)"
    else:               vtype = "소프라노 (고음 여성)"

    info = (
        f"🎤 목소리 분석 결과\n\n"
        f"⏱  길이        : {len(y)/sr:.1f}초\n"
        f"🎵 평균 음높이  : {f0_mean:.0f} Hz\n"
        f"📊 음역대       : {f0_min:.0f} ~ {f0_max:.0f} Hz\n"
        f"🎭 추정 성부    : {vtype}\n"
        f"🌟 음색 밝기    : {np.mean(sc):.0f} Hz\n\n"
        f"💡 AI 커버 원리\n"
        f"   F0  = 음정 (원곡 멜로디 유지)\n"
        f"   MFCC = 음색 지문 (이 목소리로 교체)\n"
        f"   → F0는 유지하고 MFCC만 교체 = Voice Conversion"
    )
    return vis, info


def register_voice_from_b64(b64_payload):
    path = _b64_to_file(b64_payload)
    if path is None:
        return None, None, "⚠️ 먼저 마이크로 녹음하세요."
    vis, info = analyze_voice(path)
    return path, vis, info

# ─────────────────────────────────────────────────────────
# 2단계: 노래 분리 + 가사 추출
# ─────────────────────────────────────────────────────────

def process_song(song_path):
    if song_path is None:
        return None, None, None, "⚠️ 노래 파일을 업로드하세요.", "대기 중"

    y_orig, sr = _load(song_path, duration=120)
    vocals_path, mr_path = _demucs_separate(song_path, y_orig, sr)

    # Whisper 가사
    lyrics = _extract_lyrics(vocals_path)

    # 시각화
    y_v, sr_v = _load(vocals_path)
    y_m, sr_m = _load(mr_path)

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    for i, (y_i, sr_i, lbl, col) in enumerate([
        (y_orig[:sr*60], sr,   "원본",     "#607D8B"),
        (y_v,            sr_v, "보컬",     "#E91E63"),
        (y_m,            sr_m, "MR (반주)", "#2196F3"),
    ]):
        axes[i][0].set_title(f"{lbl} — Waveform", fontweight="bold")
        librosa.display.waveshow(y_i, sr=sr_i, ax=axes[i][0], color=col)
        S = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y_i, sr=sr_i, n_mels=80, fmax=8000), ref=np.max)
        axes[i][1].set_title(f"{lbl} — Mel-Spectrogram", fontweight="bold")
        librosa.display.specshow(S, sr=sr_i, hop_length=512, x_axis="time",
                                 y_axis="mel", ax=axes[i][1], cmap="magma", fmax=8000)
    plt.suptitle("노래 분리 결과", fontsize=14, fontweight="bold")
    plt.tight_layout()
    vis = _fig_to_pil(fig)

    status = "✅ 분리 완료!"
    return vocals_path, mr_path, vis, lyrics, status


def _demucs_separate(song_path, y_fallback, sr):
    """Demucs → (vocals_path, mr_path). 실패 시 HPSS 폴백."""
    try:
        tmp_wav = tempfile.mktemp(suffix=".wav")
        sf.write(tmp_wav, y_fallback, sr)
        out_dir = tempfile.mkdtemp()
        res = subprocess.run(
            ["python", "-m", "demucs", "--two-stems=vocals", "-o", out_dir, tmp_wav],
            capture_output=True, text=True, timeout=300)
        if res.returncode == 0:
            base = os.path.splitext(os.path.basename(tmp_wav))[0]
            v = os.path.join(out_dir, "htdemucs", base, "vocals.wav")
            m = os.path.join(out_dir, "htdemucs", base, "no_vocals.wav")
            if os.path.exists(v):
                return v, m
    except Exception as e:
        print(f"[Demucs] 오류: {e} → HPSS 폴백")

    # HPSS 폴백
    D = librosa.stft(y_fallback, n_fft=2048, hop_length=512)
    H, P = librosa.decompose.hpss(D, margin=3.0)
    y_v = librosa.istft(H, hop_length=512).astype(np.float32)
    y_m = librosa.istft(P, hop_length=512).astype(np.float32)
    vp = tempfile.mktemp(suffix="_vocals.wav")
    mp = tempfile.mktemp(suffix="_mr.wav")
    sf.write(vp, y_v, sr)
    sf.write(mp, y_m, sr)
    return vp, mp


def _extract_lyrics(vocals_path):
    try:
        import whisper
        model = whisper.load_model("small")
        result = model.transcribe(vocals_path)
        return result["text"].strip()
    except Exception as e:
        return f"(가사 추출 실패: {e})"

# ─────────────────────────────────────────────────────────
# 3단계: Voice Conversion + 믹싱
# ─────────────────────────────────────────────────────────

def generate_cover(voice_path, vocals_path, mr_path, vocal_vol, mr_vol):
    if not voice_path:
        return None, None, "⚠️ 1단계에서 목소리를 먼저 등록하세요."
    if not vocals_path or not mr_path:
        return None, None, "⚠️ 2단계에서 먼저 노래를 분리하세요."

    converted_path, method = _voice_convert(voice_path, vocals_path)
    if converted_path is None:
        return None, None, "⚠️ Voice Conversion 실패"

    cover_path, mix_info = _mix(converted_path, mr_path, vocal_vol, mr_vol)

    # 시각화
    y_orig, sr = _load(vocals_path)
    y_conv, _  = _load(converted_path, sr=sr)
    n = min(len(y_orig), len(y_conv))

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    S_orig = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y_orig[:n], sr=sr, n_mels=80, fmax=8000), ref=np.max)
    S_conv = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y_conv[:n], sr=sr, n_mels=80, fmax=8000), ref=np.max)

    axes[0][0].set_title("원본 보컬 — Waveform", fontweight="bold")
    librosa.display.waveshow(y_orig[:n], sr=sr, ax=axes[0][0], color="#E91E63")
    axes[0][1].set_title("원본 보컬 — Mel-Spectrogram", fontweight="bold")
    librosa.display.specshow(S_orig, sr=sr, hop_length=512, x_axis="time",
                             y_axis="mel", ax=axes[0][1], cmap="magma", fmax=8000)

    axes[1][0].set_title("변환된 보컬 — Waveform (등록한 목소리)", fontweight="bold")
    librosa.display.waveshow(y_conv[:n], sr=sr, ax=axes[1][0], color="#9C27B0")
    axes[1][1].set_title("변환된 보컬 — Mel-Spectrogram", fontweight="bold")
    librosa.display.specshow(S_conv, sr=sr, hop_length=512, x_axis="time",
                             y_axis="mel", ax=axes[1][1], cmap="magma", fmax=8000)

    plt.suptitle("Voice Conversion 결과 비교", fontsize=14, fontweight="bold")
    plt.tight_layout()
    vis = _fig_to_pil(fig)

    info = (
        f"✅ AI 커버 생성 완료!\n\n"
        f"🎙 변환 방법 : {method}\n"
        f"🎚 보컬 볼륨 : {vocal_vol:+d} dB\n"
        f"🎚 MR 볼륨   : {mr_vol:+d} dB\n"
        f"{mix_info}\n\n"
        f"💡 Voice Conversion 원리\n"
        f"   원본 보컬의 F0(음정)은 유지하고\n"
        f"   스펙트럼 포락(음색)을 목표 목소리로 교체\n"
        f"   → Mel-Spectrogram 패턴은 비슷하지만\n"
        f"     세부 구조(음색)가 달라진 것을 확인하세요!"
    )
    return converted_path, cover_path, vis, info


def _voice_convert(target_voice_path, source_vocals_path):
    """FreeVC (Coqui TTS) → 실패 시 스펙트럼 포락 전이 폴백"""
    # 1) FreeVC
    try:
        from TTS.api import TTS
        print("[VC] FreeVC 모델 로딩 중 (첫 실행 시 다운로드)...")
        tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24",
                  progress_bar=False, gpu=False)
        out = tempfile.mktemp(suffix="_freevc.wav")
        tts.voice_conversion_to_file(
            source_wav=source_vocals_path,
            target_wav=target_voice_path,
            file_path=out)
        print("[VC] FreeVC 변환 완료!")
        return out, "FreeVC (Coqui TTS)"
    except Exception as e:
        print(f"[VC] FreeVC 실패: {e}")

    # 2) 스펙트럼 포락 전이 폴백
    try:
        out = _spectral_envelope_vc(source_vocals_path, target_voice_path)
        return out, "스펙트럼 포락 전이 (폴백, 음질 낮음)"
    except Exception as e:
        print(f"[VC] 폴백도 실패: {e}")
        return None, ""


def _spectral_envelope_vc(source, target):
    """간단한 스펙트럼 포락 전이 (음색만 바꾸는 단순 방법)"""
    y_src, sr = _load(source)
    y_tgt, _  = _load(target, sr=sr)
    n_fft, hop = 2048, 512

    D_src = librosa.stft(y_src, n_fft=n_fft, hop_length=hop)
    D_tgt = librosa.stft(y_tgt, n_fft=n_fft, hop_length=hop)

    env_src = np.mean(np.abs(D_src), axis=1, keepdims=True) + 1e-8
    env_tgt = np.mean(np.abs(D_tgt), axis=1, keepdims=True) + 1e-8
    D_out = D_src * (env_tgt / env_src)

    y_out = librosa.istft(D_out, hop_length=hop).astype(np.float32)
    out = tempfile.mktemp(suffix="_spec_vc.wav")
    sf.write(out, y_out, sr)
    return out


def _mix(vocals_path, mr_path, vocal_db, mr_db):
    try:
        from pydub import AudioSegment
        v = AudioSegment.from_file(vocals_path) + vocal_db
        m = AudioSegment.from_file(mr_path)     + mr_db
        n = min(len(v), len(m))
        mixed = m[:n].overlay(v[:n])
        out = tempfile.mktemp(suffix="_cover.mp3")
        mixed.export(out, format="mp3", bitrate="192k")
        return out, f"믹싱 완료 ({n/1000:.1f}초)"
    except Exception as e:
        # pydub 없으면 soundfile로 단순 합산
        y_v, sr = _load(vocals_path)
        y_m, _  = _load(mr_path, sr=sr)
        n = min(len(y_v), len(y_m))
        mixed = (y_v[:n] * (10 ** (vocal_db/20)) +
                 y_m[:n] * (10 ** (mr_db/20)))
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-8) * 0.95
        out = tempfile.mktemp(suffix="_cover.wav")
        sf.write(out, mixed, sr)
        return out, f"믹싱 완료 (pydub 없음 → WAV)"

# ─────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="🎙 AI 커버 파이프라인", theme=gr.themes.Soft()) as demo:

        gr.HTML("""
        <div style='text-align:center;padding:24px 0 8px'>
            <h1 style='font-size:2rem;margin:0'>🎙 AI 커버 파이프라인</h1>
            <p style='color:#666;font-size:1rem;margin-top:8px'>
                목소리를 등록하고 → 노래를 분리하고 → 그 목소리로 AI 커버를 만듭니다
            </p>
        </div>
        """)

        # 탭 간 데이터 공유용 State
        voice_state   = gr.State(None)   # 등록된 목소리 파일 경로
        vocals_state  = gr.State(None)   # 분리된 보컬 경로
        mr_state      = gr.State(None)   # 분리된 MR 경로

        with gr.Tabs():

            # ── 1단계: 목소리 등록 ──────────────────────
            with gr.TabItem("🎤 1단계: 목소리 등록"):
                gr.HTML("""
<div style='background:#e8f5e9;border-left:4px solid #43a047;
            padding:12px 16px;border-radius:6px;margin-bottom:12px;font-size:14px'>
    <b>목소리 샘플을 등록합니다.</b><br>
    10~30초 분량의 깨끗한 목소리를 녹음하거나 파일로 업로드하세요.<br>
    말하는 목소리로도 되고, 노래 목소리도 가능합니다.
</div>
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        # 마이크 녹음
                        gr.HTML("""
<div style='border:2px solid #c5cae9;border-radius:12px;padding:16px;
            background:#f8f9ff;margin-bottom:12px'>
  <b style='color:#3949ab'>🎤 방법 A — 마이크 녹음 (HTTPS 필요)</b>
  <div id='mic-status' style='font-size:13px;color:#555;margin:8px 0;min-height:18px'>
    아래 버튼을 눌러 녹음하세요
  </div>
  <audio id='mic-audio' controls
    style='width:100%;margin-bottom:8px;border-radius:8px;display:none'></audio>
    
  <a id='mic-download' 
     style='display:none; text-decoration:none; background:#e8eaf6; color:#3949ab; padding:8px 12px; border-radius:6px; font-size:13px; font-weight:bold; cursor:pointer;'>
     ⬇️ 방금 녹음한 원본 파일 다운로드
  </a>
</div>
                        """)
                        rec_toggle_btn = gr.Button(
                            "🔴  녹음 시작", variant="stop", size="lg",
                            elem_id="mic-toggle-btn")
                        rec_b64 = gr.Textbox(visible=False, label="rec_b64")
                        rec_register_btn = gr.Button(
                            "✅ 이 녹음으로 목소리 등록하기",
                            variant="secondary", size="lg")

                        gr.HTML("<hr style='margin:12px 0'>")

                        # 파일 업로드
                        gr.HTML("""
<div style='border:2px solid #c8e6c9;border-radius:12px;padding:14px;
            background:#f1f8f1;margin-bottom:12px'>
  <b style='color:#2e7d32'>📁 방법 B — 파일 업로드 (mp3·wav·m4a)</b>
</div>
                        """)
                        voice_file = gr.Audio(
                            label="목소리 파일 선택",
                            type="filepath", sources=["upload"])
                        voice_file_btn = gr.Button(
                            "📁 이 파일로 목소리 등록하기",
                            variant="primary", size="lg")

                    with gr.Column(scale=2):
                        voice_vis  = gr.Image(label="목소리 분석", type="pil")
                        voice_info = gr.Textbox(label="분석 결과", lines=14)

                # JS: 녹음 토글
                rec_toggle_btn.click(fn=None, inputs=[], outputs=[], js=_REC_TOGGLE_JS)
                # JS: base64 → rec_b64 textbox
                rec_register_btn.click(fn=None, inputs=[], outputs=[rec_b64], js=_REC_SEND_JS)
                # Python: 녹음 등록
                def _register_mic(b64, cur_voice):
                    path = _b64_to_file(b64)
                    if path is None:
                        return cur_voice, None, "⚠️ 먼저 마이크로 녹음하세요."
                    vis, info = analyze_voice(path)
                    return path, vis, info
                rec_b64.change(
                    _register_mic, [rec_b64, voice_state],
                    [voice_state, voice_vis, voice_info])
                # Python: 파일 등록
                def _register_file(path, cur_voice):
                    if path is None:
                        return cur_voice, None, "⚠️ 파일을 선택하세요."
                    vis, info = analyze_voice(path)
                    return path, vis, info
                voice_file_btn.click(
                    _register_file, [voice_file, voice_state],
                    [voice_state, voice_vis, voice_info])

            # ── 2단계: 노래 분리 ────────────────────────
            with gr.TabItem("🎵 2단계: 노래 분리 & 가사 추출"):
                gr.HTML("""
<div style='background:#e3f2fd;border-left:4px solid #1976d2;
            padding:12px 16px;border-radius:6px;margin-bottom:12px;font-size:14px'>
    <b>AI 커버로 만들고 싶은 노래를 업로드합니다.</b><br>
    Demucs가 <b>보컬</b>과 <b>MR(반주)</b>를 분리하고, Whisper가 <b>가사</b>를 추출합니다.<br>
    ⏱ Demucs는 1~3분 소요됩니다.
</div>
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        song_file = gr.Audio(
                            label="노래 파일 업로드 (mp3·wav·m4a)",
                            type="filepath", sources=["upload"])
                        song_btn = gr.Button(
                            "🎵 노래 분리 시작", variant="primary", size="lg")
                        song_status = gr.Textbox(label="처리 상태", lines=1)
                        gr.HTML("<hr style='margin:12px 0'>")
                        gr.Markdown("**분리 결과 미리듣기**")
                        vocals_out = gr.Audio(label="분리된 보컬", type="filepath")
                        mr_out     = gr.Audio(label="분리된 MR (반주)", type="filepath")

                    with gr.Column(scale=2):
                        song_vis   = gr.Image(label="분리 결과 시각화", type="pil")
                        lyrics_out = gr.Textbox(label="추출된 가사 (Whisper)", lines=12)

                def process_song_wrap(path, cv, cm):
                    if path is None:
                        return cv, cm, None, "⚠️ 노래 파일을 업로드하세요.", "대기 중", None, None
                    vp, mp, vis, lyr, status = process_song(path)
                    return vp, mp, vis, lyr, status, vp, mp

                song_btn.click(
                    process_song_wrap,
                    [song_file, vocals_state, mr_state],
                    [vocals_state, mr_state, song_vis, lyrics_out, song_status,
                     vocals_out, mr_out])

            # ── 3단계: AI 커버 생성 ─────────────────────
            with gr.TabItem("🎙 3단계: AI 커버 생성"):
                gr.HTML("""
<div style='background:#fce4ec;border-left:4px solid #e91e63;
            padding:12px 16px;border-radius:6px;margin-bottom:12px;font-size:14px'>
    <b>1·2단계를 먼저 완료하세요.</b><br>
    FreeVC가 원곡 보컬의 <b>음정은 유지</b>하면서 <b>음색만 등록한 목소리로 교체</b>합니다.<br>
    ⏱ FreeVC 첫 실행 시 모델 다운로드(~500MB)가 필요합니다.
</div>
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**볼륨 조절**")
                        vocal_vol = gr.Slider(-12, 12, value=0, step=1,
                                              label="보컬 볼륨 (dB)")
                        mr_vol    = gr.Slider(-12, 12, value=-3, step=1,
                                              label="MR 볼륨 (dB)")
                        cover_btn = gr.Button(
                            "🎙 AI 커버 생성!", variant="primary", size="lg")
                        gr.HTML("<hr style='margin:12px 0'>")
                        gr.Markdown("**결과 미리듣기**")
                        conv_vocals_out = gr.Audio(
                            label="변환된 보컬 (등록한 목소리)", type="filepath")
                        cover_out = gr.Audio(
                            label="🎵 최종 AI 커버 (보컬 + MR)", type="filepath")

                    with gr.Column(scale=2):
                        cover_vis  = gr.Image(label="변환 결과 시각화", type="pil")
                        cover_info = gr.Textbox(label="생성 정보", lines=14)

                cover_btn.click(
                    generate_cover,
                    [voice_state, vocals_state, mr_state, vocal_vol, mr_vol],
                    [conv_vocals_out, cover_out, cover_vis, cover_info])

        gr.HTML("""
        <div style='text-align:center;padding:16px;color:#aaa;font-size:13px'>
            AI 커버 파이프라인 | 강의 실습용
        </div>
        """)
    return demo

# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI 커버 파이프라인")
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=7861)
    parser.add_argument("--share", action="store_true",
                        help="Gradio 공개 URL 생성 (마이크 사용 시 필요)")
    args = parser.parse_args()

    print("=" * 58)
    print("  🎙 AI 커버 파이프라인 시작")
    print("=" * 58)
    print(f"  로컬  : http://localhost:{args.port}")
    print(f"  네트워크: http://<서버IP>:{args.port}")
    if args.share:
        print("  공개 URL: 아래에 표시됩니다...")
    else:
        print("  마이크 사용하려면: python ai_cover_pipeline.py --share")
    print("=" * 58)

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, show_error=True)
