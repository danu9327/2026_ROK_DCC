"""
🔍 01_explore_data.py - 한국어 대화 데이터셋 구경하기

실행: python 01_explore_data.py

HuggingFace에서 바로 다운로드되는 한국어 데이터셋 2종을 탐색합니다.
별도 회원가입이나 승인 없이 바로 사용 가능!

사용 데이터셋:
  1. jojo0217/korean_safe_conversation (26K, 일상대화 Q&A)
  2. heegyu/open-korean-instructions (한국어 챗봇 통합 데이터)
"""
import random
from datasets import load_dataset


def print_box(title, content=""):
    """예쁜 박스 출력"""
    width = 60
    print(f"\n{'━' * width}")
    print(f"  {title}")
    print(f"{'━' * width}")
    if content:
        print(content)


# =========================================================================
# 1) 한국어 안전 대화 데이터 (약 26,000개)
#    성균관대-VAIV 산학협력으로 구축한 일상대화 챗봇용 데이터
# =========================================================================
print_box("📦 데이터셋 1: korean_safe_conversation 다운로드 중...")

safe_conv = load_dataset("jojo0217/korean_safe_conversation", split="train")

print(f"  총 {len(safe_conv):,}개 대화 쌍")
print(f"  컬럼: {safe_conv.column_names}")
print(f"  예시 구조: {safe_conv[0]}")

print_box("💬 랜덤 대화 5개 구경하기")
cols = safe_conv.column_names
for i in random.sample(range(len(safe_conv)), 5):
    item = safe_conv[i]
    # 컬럼명 자동 감지
    if "instruction" in cols:
        q = item.get("instruction", "")
        a = item.get("output", "")
    elif "Q" in cols:
        q = item.get("Q", "")
        a = item.get("A", "")
    else:
        q = str(item[cols[0]])
        a = str(item[cols[1]]) if len(cols) > 1 else ""

    print(f"  Q: {str(q)[:120]}")
    print(f"  A: {str(a)[:120]}")
    print(f"  {'─' * 50}")


# =========================================================================
# 2) Open Korean Instructions (4가지 한국어 챗봇 데이터 통합)
#    KoAlpaca, ShareGPT-ko, KorQuAD-Chat 등 포함
# =========================================================================
print_box("📦 데이터셋 2: open-korean-instructions 다운로드 중...")

oki = load_dataset("heegyu/open-korean-instructions", split="train")

print(f"  총 {len(oki):,}개 대화")
print(f"  컬럼: {oki.column_names}")

# source별 분포
if "source" in oki.column_names:
    from collections import Counter
    sources = Counter(item["source"] for item in oki)
    print(f"\n  📊 데이터 출처별 분포:")
    for src, cnt in sources.most_common():
        print(f"    {src}: {cnt:,}개")

print_box("🎭 Open Korean Instructions 대화 3개 구경하기")
for i in random.sample(range(len(oki)), 3):
    item = oki[i]
    text = item.get("text", "")

    print(f"  📌 출처: {item.get('source', '?')}")

    # <usr>, <bot> 토큰을 읽기 좋게 변환
    display = text[:300]
    display = display.replace("<usr>", "\n    🧑 사용자: ")
    display = display.replace("<bot>", "\n    🤖 봇: ")
    display = display.replace("<sys>", "\n    📋 시스템: ")
    print(f"  {display}")

    if len(text) > 300:
        print(f"    ... (이하 {len(text) - 300}자 생략)")
    print(f"  {'═' * 50}")


# =========================================================================
# 3) 간단한 통계
# =========================================================================
print_box("📊 데이터 통계")

# 안전 대화 데이터
ans_col = next((c for c in ["output", "A", "answer"] if c in cols), cols[-1])
a_lens = [len(str(item[ans_col])) for item in safe_conv]
print(f"  [한국어 안전 대화 - {len(safe_conv):,}개]")
print(f"    응답 평균 길이: {sum(a_lens)/len(a_lens):.0f}자")
print(f"    최단: {min(a_lens)}자 / 최장: {max(a_lens)}자")

# OKI
text_lens = [len(item.get("text", "")) for item in oki]
print(f"\n  [Open Korean Instructions - {len(oki):,}개]")
print(f"    텍스트 평균 길이: {sum(text_lens)/len(text_lens):.0f}자")
print(f"    최단: {min(text_lens)}자 / 최장: {max(text_lens)}자")

print_box(
    "✅ 탐색 완료!",
    "  다음 단계: python 02_finetune.py  (파인튜닝)\n"
    "  또는 바로: python 03_story_game.py (게임 플레이)\n"
)
