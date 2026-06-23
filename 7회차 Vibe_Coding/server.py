"""
미니 데이콘 챌린지 서버 (5090 PC에서 실행) — 제출 무제한 버전

기능:
- 데이터 배포 (/data/<task>/train.csv, test.csv)
- 제출 엔드포인트 (/api/submit) — 점수 자동 채점
- 리더보드 (HTML + JSON API, 자동 새로고침)
- 🔥 제출 횟수 무제한
- SQLite에 모든 제출 기록 저장

실행:
    pip install fastapi uvicorn pandas scikit-learn python-multipart
    python server.py

cloudflared로 공개:
    cloudflared tunnel --url http://localhost:9000
"""

import sqlite3
import os
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import mean_squared_error, accuracy_score

# ============================================================
# 설정
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "leaderboard.db"
SUBMISSION_LIMIT = None    # None = 무제한 (숫자로 바꾸면 제한 부활)

TASKS = {
    "regression": {
        "title": "🎬 K-드라마 시청률 예측",
        "metric": "RMSE",
        "metric_lower_is_better": True,
        "target_col": "avg_rating",
        "id_col": "id",
    },
    "classification": {
        "title": "🎯 영화 흥행 등급 분류",
        "metric": "Accuracy",
        "metric_lower_is_better": False,
        "target_col": "box_office_grade",
        "id_col": "id",
    },
}

# 정답 로드 (메모리 캐시)
SOLUTIONS = {}
for task in TASKS:
    sol_path = DATA_DIR / task / "solution.csv"
    if sol_path.exists():
        SOLUTIONS[task] = pd.read_csv(sol_path)


# ============================================================
# DB 초기화
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            nickname TEXT NOT NULL,
            score REAL NOT NULL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_task_nick ON submissions(task, nickname)")
    conn.commit()
    conn.close()


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


init_db()

# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="Mini DACON")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ============================================================
# 데이터 배포
# ============================================================
@app.get("/data/{task}/{filename}")
def serve_data(task: str, filename: str):
    if task not in TASKS:
        raise HTTPException(404, f"Unknown task: {task}")
    if filename not in ("train.csv", "test.csv"):
        raise HTTPException(404, "Only train.csv / test.csv allowed")
    path = DATA_DIR / task / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="text/csv", filename=filename)


# ============================================================
# 제출
# ============================================================
class Submission(BaseModel):
    task: str = Field(..., description="regression 또는 classification")
    nickname: str = Field(..., min_length=1, max_length=20)
    predictions: list = Field(..., description="test.csv 순서대로의 예측 리스트")


def grade(task: str, predictions: list) -> float:
    """제출된 예측에 대해 점수 계산"""
    if task not in SOLUTIONS:
        raise HTTPException(500, f"No solution loaded for {task}")

    sol = SOLUTIONS[task]
    target_col = TASKS[task]["target_col"]

    if len(predictions) != len(sol):
        raise HTTPException(
            400,
            f"제출 길이({len(predictions)}) ≠ test 길이({len(sol)}). "
            f"test.csv를 순서대로 모두 예측했는지 확인하세요.",
        )

    y_true = sol[target_col].values
    y_pred = np.asarray(predictions)

    try:
        if task == "regression":
            y_pred = y_pred.astype(float)
            score = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        else:  # classification
            y_pred = y_pred.astype(int)
            score = float(accuracy_score(y_true, y_pred))
    except (ValueError, TypeError) as e:
        raise HTTPException(400, f"예측값 타입 오류: {e}")

    return score


@app.post("/api/submit")
def submit(s: Submission):
    if s.task not in TASKS:
        raise HTTPException(400, f"Unknown task: {s.task}")

    nick = s.nickname.strip()
    if not nick:
        raise HTTPException(400, "닉네임은 비어있을 수 없습니다")

    # 본인 누적 제출 수 카운트 (응답용)
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) AS c FROM submissions WHERE task=? AND nickname=?",
        (s.task, nick),
    )
    used = cur.fetchone()["c"]

    # 제출 제한 (SUBMISSION_LIMIT이 None이면 통과)
    if SUBMISSION_LIMIT is not None and used >= SUBMISSION_LIMIT:
        conn.close()
        raise HTTPException(
            429,
            f"제출 횟수 초과 — {nick}님은 [{s.task}]에 이미 {used}/{SUBMISSION_LIMIT}회 제출",
        )

    # 채점
    score = grade(s.task, s.predictions)

    # 기록
    cur.execute(
        "INSERT INTO submissions(task, nickname, score) VALUES (?, ?, ?)",
        (s.task, nick, score),
    )
    conn.commit()

    # 본인 최고 점수
    lower_better = TASKS[s.task]["metric_lower_is_better"]
    op = "MIN" if lower_better else "MAX"
    cur.execute(
        f"SELECT {op}(score) AS best FROM submissions WHERE task=? AND nickname=?",
        (s.task, nick),
    )
    best = cur.fetchone()["best"]

    # 본인 등수
    cmp = "<" if lower_better else ">"
    cur.execute(
        f"""SELECT COUNT(DISTINCT nickname) + 1 AS rank
            FROM submissions
            WHERE task=? AND nickname != ?
              AND nickname IN (
                  SELECT nickname FROM submissions WHERE task=?
                  GROUP BY nickname
                  HAVING {op}(score) {cmp} ?
              )""",
        (s.task, nick, s.task, best),
    )
    rank = cur.fetchone()["rank"]
    conn.close()

    return {
        "ok": True,
        "task": s.task,
        "nickname": nick,
        "score_this": round(score, 5),
        "score_best": round(best, 5),
        "metric": TASKS[s.task]["metric"],
        "submissions_used": used + 1,
        "submissions_left": (SUBMISSION_LIMIT - (used + 1)) if SUBMISSION_LIMIT is not None else None,
        "rank": rank,
    }


# ============================================================
# 리더보드
# ============================================================
def get_leaderboard(task: str):
    lower_better = TASKS[task]["metric_lower_is_better"]
    op = "MIN" if lower_better else "MAX"
    order = "ASC" if lower_better else "DESC"

    conn = db()
    cur = conn.cursor()
    cur.execute(
        f"""SELECT nickname,
                   {op}(score) AS best_score,
                   COUNT(*) AS submissions,
                   MAX(submitted_at) AS last_submit
            FROM submissions
            WHERE task=?
            GROUP BY nickname
            ORDER BY best_score {order}
            LIMIT 100""",
        (task,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


@app.get("/api/leaderboard/{task}")
def leaderboard_api(task: str):
    if task not in TASKS:
        raise HTTPException(404, f"Unknown task: {task}")
    return JSONResponse(get_leaderboard(task))


@app.get("/api/quota/{task}/{nickname}")
def quota(task: str, nickname: str):
    if task not in TASKS:
        raise HTTPException(404, f"Unknown task: {task}")
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) AS used FROM submissions WHERE task=? AND nickname=?",
        (task, nickname),
    )
    used = cur.fetchone()["used"]
    conn.close()
    return {
        "task": task,
        "nickname": nickname,
        "used": used,
        "left": (SUBMISSION_LIMIT - used) if SUBMISSION_LIMIT is not None else None,
        "limit": SUBMISSION_LIMIT,
    }


# ============================================================
# 메인 HTML 리더보드 (실시간 새로고침)
# ============================================================
INDEX_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>🏆 미니 데이콘 챌린지</title>
<meta http-equiv="refresh" content="5">
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Noto Sans KR', sans-serif;
         background:#0f172a; color:#e2e8f0; margin:0; padding:20px; }
  .container { max-width: 1400px; margin: 0 auto; }
  h1 { text-align:center; font-size:2.5em; margin-bottom:8px;
       background: linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
       -webkit-background-clip:text; -webkit-text-fill-color: transparent; }
  .subtitle { text-align:center; color:#94a3b8; margin-bottom:30px; font-size:1.1em; }
  .boards { display:grid; grid-template-columns: 1fr 1fr; gap:24px; }
  @media (max-width: 900px) { .boards { grid-template-columns: 1fr; } }
  .board { background:#1e293b; border-radius:12px; padding:20px;
           box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
  .board h2 { margin-top:0; font-size:1.5em; border-bottom:2px solid #334155;
              padding-bottom:10px; }
  .metric { color:#94a3b8; font-size:0.9em; }
  table { width:100%; border-collapse: collapse; margin-top:12px; }
  th { text-align:left; padding:10px 8px; color:#94a3b8; font-weight:600;
       font-size:0.85em; border-bottom:1px solid #334155; text-transform:uppercase;
       letter-spacing:0.5px; }
  td { padding:12px 8px; border-bottom:1px solid #1e293b; }
  tr:hover { background:#334155; }
  .rank { font-weight:700; font-size:1.1em; width:50px; }
  .rank-1 { color:#fbbf24; font-size:1.5em; }
  .rank-2 { color:#cbd5e1; font-size:1.3em; }
  .rank-3 { color:#d97706; font-size:1.2em; }
  .nick { font-weight:600; }
  .score { font-family: 'JetBrains Mono', monospace; font-size:1.05em; color:#34d399; }
  .meta { color:#64748b; font-size:0.85em; }
  .empty { text-align:center; color:#64748b; padding:40px; font-style:italic; }
  .footer { text-align:center; margin-top:30px; color:#64748b; font-size:0.85em; }
  .live { display:inline-block; width:10px; height:10px; background:#10b981;
          border-radius:50%; margin-right:6px; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
</style>
</head>
<body>
  <div class="container">
    <h1>🏆 미니 데이콘 챌린지</h1>
    <p class="subtitle"><span class="live"></span>실시간 리더보드 · 5초마다 자동 갱신</p>
    <div class="boards">
      __BOARD_REGRESSION__
      __BOARD_CLASSIFICATION__
    </div>
    <div class="footer">
      __LIMIT_TEXT__ · 갱신: __NOW__
    </div>
  </div>
</body>
</html>"""


def render_board(task: str) -> str:
    info = TASKS[task]
    rows = get_leaderboard(task)
    metric = info["metric"]

    if not rows:
        body = '<div class="empty">아직 제출이 없습니다. 첫 제출자가 되어보세요!</div>'
    else:
        body = '<table><thead><tr><th>순위</th><th>닉네임</th>'
        body += f'<th>{metric}</th><th>제출수</th><th>마지막 제출</th></tr></thead><tbody>'
        for i, r in enumerate(rows, 1):
            rank_cls = f"rank-{i}" if i <= 3 else ""
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            last = r["last_submit"][:19] if r["last_submit"] else "-"
            body += (
                f'<tr><td class="rank {rank_cls}">{medal}</td>'
                f'<td class="nick">{r["nickname"]}</td>'
                f'<td class="score">{r["best_score"]:.4f}</td>'
                f'<td class="meta">{r["submissions"]}회</td>'
                f'<td class="meta">{last}</td></tr>'
            )
        body += "</tbody></table>"

    return f"""<div class="board">
      <h2>{info["title"]}</h2>
      <div class="metric">평가 지표: {metric}
        ({"낮을수록 좋음" if info["metric_lower_is_better"] else "높을수록 좋음"})</div>
      {body}
    </div>"""


@app.get("/", response_class=HTMLResponse)
def index():
    if SUBMISSION_LIMIT is None:
        limit_text = "🔥 제출 무제한"
    else:
        limit_text = f"제출 제한: 1인당 task별 {SUBMISSION_LIMIT}회"

    html = INDEX_HTML \
        .replace("__BOARD_REGRESSION__", render_board("regression")) \
        .replace("__BOARD_CLASSIFICATION__", render_board("classification")) \
        .replace("__LIMIT_TEXT__", limit_text) \
        .replace("__NOW__", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return html


@app.get("/health")
def health():
    return {"ok": True, "tasks": list(TASKS.keys()), "limit": SUBMISSION_LIMIT}


# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print(" 🏆 Mini DACON Server")
    print("=" * 60)
    print(f" Tasks       : {list(TASKS.keys())}")
    print(f" Submit limit: {'무제한 🔥' if SUBMISSION_LIMIT is None else SUBMISSION_LIMIT}")
    print(f" DB          : {DB_PATH}")
    print(f" URL         : http://0.0.0.0:9000")
    print("=" * 60)
    print()
    print("[NEXT] 공개 URL은 cloudflared로:")
    print("       cloudflared tunnel --url http://localhost:9000")
    print()
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
