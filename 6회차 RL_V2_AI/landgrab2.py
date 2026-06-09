"""
땅따먹기 강화학습 배틀 (single-file FastAPI + WebSocket)

실행:
    pip install fastapi uvicorn numpy
    python landgrab.py

학생 접속: http://<서버IP>:8000/  (닉네임 입력)
관리자 접속: http://<서버IP>:8000/  (닉네임 칸에 ADMIN_PASSWORD 입력)

핵심:
 - 솔로: 학생마다 numpy 정책망(REINFORCE)이 10Hz로 학습. 에피소드마다 베스트 가중치 저장.
 - 관리자 CCTV: 모든 학생의 실시간 그리드 상태를 격자 뷰로 시청.
 - 단체전: 학생들의 베스트 모델을 큰 맵에 랜덤 배치 후 60초 경쟁.
"""

import asyncio
import base64
import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# ========================= 설정 =========================
ADMIN_PASSWORD = "관리자비번"   # 이 문자열을 닉네임으로 입력하면 관리자
HOST = "0.0.0.0"
PORT = 8000

SOLO_GRID = 36
TEAM_GRID = 64
TICK_HZ = 10
SOLO_EP_TICKS = 200          # 솔로 에피소드 1회 = 20초
TEAM_TICKS = 60 * TICK_HZ    # 단체전 60초

COLORS = [
    "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#ec4899",
    "#14b8a6", "#eab308", "#f97316", "#06b6d4", "#84cc16", "#d946ef",
    "#f43f5e", "#0ea5e9", "#10b981", "#facc15", "#8b5cf6", "#fb7185",
    "#34d399", "#fbbf24",
]
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]   # up, right, down, left
MAPS = ("square", "circle", "diamond", "star", "cross", "ring")

# 보상함수 컴포넌트 (학생이 슬라이더로 가중치 조절)
REWARD_KEYS = (
    "new_paint",       # 빈 칸을 새로 칠함
    "steal",           # 적 칸을 덮어씀
    "revisit",         # 내 칸을 또 밟음
    "wall_hit",        # 이동 실패(벽/경계)
    "near_enemy",      # 5x5 시야에 적 있음
    "frontier",        # 인접 4칸에 빈 칸 있음(탐험 여지)
    "territory_gain",  # 매 틱 내 영토 증가량(/10 정규화)
    "survive",         # 매 틱 +1
)
DEFAULT_WEIGHTS = {
    "new_paint": 1.0, "steal": 0.5, "revisit": -0.1, "wall_hit": -0.3,
    "near_enemy": 0.0, "frontier": 0.1, "territory_gain": 0.0, "survive": 0.0,
}
PC_FIXED = (("random", "PC1(random)"), ("straight", "PC2(straight)"))


# ========================= 맵 생성 =========================
def make_map(shape: str, size: int) -> np.ndarray:
    """-1 = wall, 0 = empty playable cell"""
    g = np.full((size, size), -1, dtype=np.int8)
    cy = cx = (size - 1) / 2.0
    if shape == "square":
        g[2:-2, 2:-2] = 0
    elif shape == "circle":
        r = size / 2 - 2
        Y, X = np.ogrid[:size, :size]
        g[((Y - cy) ** 2 + (X - cx) ** 2) < r * r] = 0
    elif shape == "diamond":
        r = size / 2 - 1
        Y, X = np.ogrid[:size, :size]
        g[np.abs(Y - cy) + np.abs(X - cx) < r] = 0
    elif shape == "star":
        R = size / 2 - 1
        ri = R * 0.45
        Y, X = np.indices((size, size))
        dx = X - cx; dy = Y - cy
        ang = np.arctan2(dy, dx)
        dist = np.sqrt(dx * dx + dy * dy)
        k = 5
        a = ang % (2 * math.pi / k)
        t = np.abs(a - math.pi / k) / (math.pi / k)
        rad_at = ri + (R - ri) * (1 - t)
        g[dist < rad_at] = 0
    elif shape == "cross":
        th = max(4, size // 4)
        c = size // 2
        g[2:-2, c - th // 2:c + th // 2] = 0
        g[c - th // 2:c + th // 2, 2:-2] = 0
    elif shape == "ring":
        Ro = size / 2 - 2
        Ri = size / 4
        Y, X = np.ogrid[:size, :size]
        d2 = (Y - cy) ** 2 + (X - cx) ** 2
        g[(d2 < Ro * Ro) & (d2 > Ri * Ri)] = 0
    else:
        g[2:-2, 2:-2] = 0
    return g


def random_empty(grid: np.ndarray):
    empties = np.argwhere(grid == 0)
    if len(empties) == 0:
        return None
    y, x = empties[random.randint(0, len(empties) - 1)]
    return int(x), int(y)


# ========================= 정책망 (REINFORCE) =========================
class PolicyNet:
    IN, HID, OUT = 31, 32, 4

    def __init__(self, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((self.IN, self.HID)) * 0.15).astype(np.float32)
        self.b1 = np.zeros(self.HID, dtype=np.float32)
        self.W2 = (rng.standard_normal((self.HID, self.OUT)) * 0.15).astype(np.float32)
        self.b2 = np.zeros(self.OUT, dtype=np.float32)
        self.lr = 0.02

    def _fwd(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(logits)
        p = e / e.sum(axis=-1, keepdims=True)
        return p, h

    def act(self, x: np.ndarray) -> int:
        p, _ = self._fwd(x[None].astype(np.float32))
        return int(np.random.choice(self.OUT, p=p[0]))

    def update(self, states, actions, returns):
        if len(states) < 4:
            return
        S = np.asarray(states, dtype=np.float32)
        A = np.asarray(actions)
        R = np.asarray(returns, dtype=np.float32)
        R = (R - R.mean()) / (R.std() + 1e-6)
        p, h = self._fwd(S)
        N = len(A)
        oh = np.zeros_like(p)
        oh[np.arange(N), A] = 1.0
        d = (p - oh) * R[:, None] / N
        dW2 = h.T @ d
        db2 = d.sum(0)
        dh = (d @ self.W2.T) * (h > 0)
        dW1 = S.T @ dh
        db1 = dh.sum(0)
        for g in (dW1, db1, dW2, db2):
            np.clip(g, -1, 1, out=g)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def snapshot(self) -> dict:
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def load(self, s: dict):
        self.W1 = s["W1"].copy(); self.b1 = s["b1"].copy()
        self.W2 = s["W2"].copy(); self.b2 = s["b2"].copy()


# ========================= Agent & Game =========================
@dataclass
class Agent:
    pid: int
    x: int
    y: int
    dir: int
    algo: str = "model"
    policy: Optional[PolicyNet] = None
    nickname: str = ""
    color: str = "#ffffff"
    zig_cnt: int = 0
    weights: Optional[dict] = None        # 모델 에이전트의 보상 가중치
    prev_tiles: int = 0                   # territory_gain 계산용
    traj_s: list = field(default_factory=list)
    traj_a: list = field(default_factory=list)
    traj_r: list = field(default_factory=list)


class GameInstance:
    def __init__(self, grid: np.ndarray, agents: List[Agent], max_ticks: int):
        self.grid = grid.copy().astype(np.int8)
        self.size = self.grid.shape[0]
        self.agents = agents
        self.tick = 0
        self.max_ticks = max_ticks
        self.done = False
        for a in agents:
            self.grid[a.y, a.x] = a.pid

    def _features(self, a: Agent) -> np.ndarray:
        sz = self.size
        half = 2  # 5x5 view
        f = np.zeros(31, dtype=np.float32)
        idx = 0
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                yy = a.y + dy; xx = a.x + dx
                if 0 <= yy < sz and 0 <= xx < sz:
                    v = int(self.grid[yy, xx])
                    if v == -1:
                        f[idx] = -2.0
                    elif v == 0:
                        f[idx] = 0.0
                    elif v == a.pid:
                        f[idx] = 1.0
                    else:
                        f[idx] = -1.0
                else:
                    f[idx] = -2.0
                idx += 1
        f[25] = a.x / sz
        f[26] = a.y / sz
        f[27 + a.dir] = 1.0
        return f

    def _choose(self, a: Agent) -> int:
        if a.algo == "model" and a.policy is not None:
            feat = self._features(a)
            act = a.policy.act(feat)
            a.traj_s.append(feat); a.traj_a.append(act)
            return act
        if a.algo == "straight":
            d = a.dir
            dx, dy = DIRS[d]
            nx, ny = a.x + dx, a.y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size) or self.grid[ny, nx] == -1:
                return random.randint(0, 3)
            return d
        if a.algo == "center":
            if random.random() < 0.15:
                return random.randint(0, 3)
            cx, cy = self.size // 2, self.size // 2
            dxc = cx - a.x; dyc = cy - a.y
            if abs(dxc) > abs(dyc):
                return 1 if dxc > 0 else 3
            if dyc != 0:
                return 2 if dyc > 0 else 0
            return random.randint(0, 3)
        if a.algo == "zigzag":
            a.zig_cnt += 1
            if a.zig_cnt % 5 == 0:
                return (a.dir + 1) % 4
            if a.zig_cnt % 9 == 0:
                return (a.dir + 3) % 4
            d = a.dir
            dx, dy = DIRS[d]
            nx, ny = a.x + dx, a.y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size) or self.grid[ny, nx] == -1:
                return random.randint(0, 3)
            return d
        # random
        return random.randint(0, 3)

    def _reward(self, a: Agent, prev_cell: int, moved: bool, tile_delta: int) -> float:
        """학생이 설정한 가중치로 매 틱 보상을 계산."""
        w = a.weights or DEFAULT_WEIGHTS
        r = 0.0
        if prev_cell == 0:
            r += w.get("new_paint", 0.0)
        elif prev_cell > 0 and prev_cell != a.pid:
            r += w.get("steal", 0.0)
        elif prev_cell == a.pid:
            r += w.get("revisit", 0.0)
        if not moved:
            r += w.get("wall_hit", 0.0)
        sz = self.size
        y0, y1 = max(0, a.y - 2), min(sz, a.y + 3)
        x0, x1 = max(0, a.x - 2), min(sz, a.x + 3)
        view = self.grid[y0:y1, x0:x1]
        if np.any((view > 0) & (view != a.pid)):
            r += w.get("near_enemy", 0.0)
        for dx, dy in DIRS:
            nx, ny = a.x + dx, a.y + dy
            if 0 <= nx < sz and 0 <= ny < sz and self.grid[ny, nx] == 0:
                r += w.get("frontier", 0.0)
                break
        r += w.get("territory_gain", 0.0) * (tile_delta / 10.0)
        r += w.get("survive", 0.0)
        return r

    def step(self):
        if self.done:
            return
        # territory_gain 정확히 계산하려고 이번 틱 시작 시 영토 카운트 저장
        for a in self.agents:
            if a.algo == "model":
                a.prev_tiles = int((self.grid == a.pid).sum())
        for a in self.agents:
            act = self._choose(a)
            a.dir = act
            dx, dy = DIRS[act]
            nx, ny = a.x + dx, a.y + dy
            moved = False
            if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny, nx] != -1:
                a.x, a.y = nx, ny
                moved = True
            prev = int(self.grid[a.y, a.x])
            self.grid[a.y, a.x] = a.pid
            if a.algo == "model":
                now_tiles = int((self.grid == a.pid).sum())
                a.traj_r.append(self._reward(a, prev, moved, now_tiles - a.prev_tiles))
        self.tick += 1
        if self.tick >= self.max_ticks:
            self.done = True

    def scores(self) -> Dict[int, int]:
        out = {}
        for a in self.agents:
            out[a.pid] = int((self.grid == a.pid).sum())
        return out

    def serialize(self) -> dict:
        return {
            "size": self.size,
            "grid": base64.b64encode(self.grid.tobytes()).decode(),
            "tick": self.tick,
            "max_ticks": self.max_ticks,
            "agents": [
                {"pid": a.pid, "x": a.x, "y": a.y, "color": a.color,
                 "nick": a.nickname, "algo": a.algo}
                for a in self.agents
            ],
            "scores": {str(k): v for k, v in self.scores().items()},
        }


def compute_returns(rewards: list, gamma: float = 0.95) -> list:
    R = 0.0; out = []
    for r in reversed(rewards):
        R = r + gamma * R
        out.append(R)
    out.reverse()
    return out


# ========================= Server State =========================
@dataclass
class StudentSession:
    nickname: str
    ws: Optional[WebSocket] = None
    policy: Optional[PolicyNet] = None
    best_policy: Optional[dict] = None
    best_score: int = 0
    episode_count: int = 0
    config: dict = field(default_factory=lambda: {
        "map": "square",
        "pc_count": 2,
        "weights": dict(DEFAULT_WEIGHTS),
    })
    current_game: Optional[GameInstance] = None
    running: bool = False


class S:
    phase: str = "lobby"     # lobby | solo | team_ready | team
    students: Dict[str, StudentSession] = {}
    admin_ws: set = set()
    team_game: Optional[GameInstance] = None
    team_task: Optional[asyncio.Task] = None
    team_map: str = "square"
    team_speed: int = 1      # 단체전 배속 (1/2/4/8)


# ========================= Broadcast helpers =========================
async def _send(ws: WebSocket, msg: str):
    try:
        await ws.send_text(msg)
    except Exception:
        pass


async def broadcast_phase():
    msg = json.dumps({
        "type": "phase",
        "phase": S.phase,
        "students": list(S.students.keys()),
    })
    for ws in list(S.admin_ws):
        await _send(ws, msg)
    for s in list(S.students.values()):
        if s.ws:
            await _send(s.ws, msg)


async def broadcast_student_state(s: StudentSession):
    if s.current_game is None:
        return
    payload = json.dumps({
        "type": "game_state",
        "context": "solo",
        "nickname": s.nickname,
        "state": s.current_game.serialize(),
        "episode": s.episode_count,
        "best_score": s.best_score,
    })
    if s.ws:
        await _send(s.ws, payload)
    for ws in list(S.admin_ws):
        await _send(ws, payload)


async def broadcast_team_state(final: bool = False):
    if S.team_game is None:
        return
    payload = json.dumps({
        "type": "game_state",
        "context": "team",
        "state": S.team_game.serialize(),
        "final": final,
        "map": S.team_map,
    })
    for ws in list(S.admin_ws):
        await _send(ws, payload)
    for s in list(S.students.values()):
        if s.ws:
            await _send(s.ws, payload)


# ========================= Game loops =========================
async def student_solo_loop(s: StudentSession):
    s.running = True
    try:
        while S.phase == "solo":
            cfg = dict(s.config)
            grid = make_map(cfg["map"], SOLO_GRID)
            pos = random_empty(grid)
            if pos is None:
                await asyncio.sleep(0.5)
                continue
            student_agent = Agent(
                pid=1, x=pos[0], y=pos[1], dir=random.randint(0, 3),
                algo="model", policy=s.policy, nickname=s.nickname,
                color=COLORS[0], weights=dict(cfg.get("weights", DEFAULT_WEIGHTS)),
            )
            agents = [student_agent]
            grid[pos[1], pos[0]] = -2  # mark to avoid reuse
            pc_count = int(cfg.get("pc_count", 2))
            for i in range(pc_count):
                p = random_empty(grid)
                if p is None:
                    break
                grid[p[1], p[0]] = -2
                algo = "random" if i % 2 == 0 else "straight"
                agents.append(Agent(
                    pid=2 + i, x=p[0], y=p[1], dir=random.randint(0, 3),
                    algo=algo, nickname=f"PC{i+1}({algo})",
                    color=COLORS[(i + 1) % len(COLORS)],
                ))
            grid[grid == -2] = 0

            game = GameInstance(grid, agents, SOLO_EP_TICKS)
            s.current_game = game
            SOLO_SPEED = 2  # 솔로는 2배속 고정
            tick_i = 0
            while not game.done and S.phase == "solo":
                game.step()
                if tick_i % SOLO_SPEED == 0:
                    await broadcast_student_state(s)
                tick_i += 1
                await asyncio.sleep(1.0 / (TICK_HZ * SOLO_SPEED))
            # 마지막 상태 한 번 더 송출
            await broadcast_student_state(s)
            if S.phase != "solo":
                break

            my_score = game.scores().get(1, 0)
            if student_agent.traj_r:
                returns = compute_returns(student_agent.traj_r, 0.95)
                s.policy.update(student_agent.traj_s, student_agent.traj_a, returns)
            s.episode_count += 1
            if my_score > s.best_score:
                s.best_score = my_score
                s.best_policy = s.policy.snapshot()
    finally:
        s.running = False


async def team_loop(map_shape: str):
    grid = make_map(map_shape, TEAM_GRID)
    items = list(S.students.items())
    random.shuffle(items)
    agents: List[Agent] = []
    pid = 1
    for i, (nick, s) in enumerate(items):
        snap = s.best_policy or (s.policy.snapshot() if s.policy else None)
        if snap is None:
            continue
        pol = PolicyNet()
        pol.load(snap)
        p = random_empty(grid)
        if p is None:
            break
        grid[p[1], p[0]] = -2
        agents.append(Agent(
            pid=pid, x=p[0], y=p[1], dir=random.randint(0, 3),
            algo="model", policy=pol, nickname=nick,
            color=COLORS[(pid - 1) % len(COLORS)],
        ))
        pid += 1
    grid[grid == -2] = 0
    if not agents:
        return
    game = GameInstance(grid, agents, TEAM_TICKS)
    S.team_game = game
    tick_i = 0
    while not game.done and S.team_game is game and S.phase == "team":
        game.step()
        speed = max(1, min(8, int(S.team_speed)))
        if tick_i % speed == 0:
            await broadcast_team_state()
        tick_i += 1
        await asyncio.sleep(1.0 / (TICK_HZ * speed))
    await broadcast_team_state(final=True)


# ========================= Admin commands =========================
async def cmd_start_solo():
    if S.phase != "lobby":
        return
    S.phase = "solo"
    await broadcast_phase()
    for s in list(S.students.values()):
        if s.policy is None:
            s.policy = PolicyNet()
        asyncio.create_task(student_solo_loop(s))


async def cmd_end_solo():
    if S.phase != "solo":
        return
    S.phase = "team_ready"
    await asyncio.sleep(0.3)
    for s in S.students.values():
        if s.best_policy is None and s.policy is not None:
            s.best_policy = s.policy.snapshot()
    await broadcast_phase()


async def cmd_start_team(map_shape: str):
    if S.phase not in ("team_ready", "team"):
        return
    if map_shape not in MAPS:
        map_shape = "square"
    S.team_map = map_shape
    if S.team_game is not None:
        S.team_game.done = True
        S.team_game = None
        await asyncio.sleep(0.25)
    S.phase = "team"
    await broadcast_phase()
    S.team_task = asyncio.create_task(team_loop(map_shape))


async def cmd_reset():
    """관리자가 처음으로 되돌리고 싶을 때."""
    S.phase = "lobby"
    if S.team_game is not None:
        S.team_game.done = True
        S.team_game = None
    S.students.clear()
    await broadcast_phase()


# ========================= Web =========================
HTML_PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>땅따먹기 강화학습 배틀</title>
<style>
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#0f172a;color:#e2e8f0}
.container{max-width:1500px;margin:0 auto;padding:16px}
button{padding:9px 16px;background:#3b82f6;color:#fff;border:0;border-radius:6px;cursor:pointer;font-size:14px;font-weight:500}
button:hover{background:#2563eb}
button.danger{background:#ef4444}
button.danger:hover{background:#dc2626}
button.success{background:#22c55e}
button.success:hover{background:#16a34a}
input,select{padding:8px 12px;background:#1e293b;color:#fff;border:1px solid #334155;border-radius:6px;font-size:14px}
.card{background:#1e293b;padding:16px;border-radius:10px;margin:10px 0}
canvas{background:#000;image-rendering:pixelated;border-radius:4px;display:block}
.grid-students{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}
.student-tile{background:#0b1220;padding:10px;border-radius:8px;border:1px solid #1e293b}
h1{font-size:22px;margin:8px 0}
h2{font-size:17px;margin:8px 0}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.stat{font-size:12px;color:#94a3b8;margin-top:4px}
.tag{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;margin-right:4px;vertical-align:middle;font-weight:600;color:#000}
.scoreboard{background:#0b1220;padding:12px;border-radius:8px;margin-top:10px;min-width:260px}
.scoreboard>div{padding:4px 0;font-size:14px;display:flex;align-items:center;gap:6px}
.phase-pill{padding:4px 12px;background:#1e293b;border-radius:999px;font-size:12px;color:#94a3b8}
label.algo{padding:6px 10px;background:#0b1220;border-radius:6px;cursor:pointer;display:inline-flex;align-items:center;gap:6px;font-size:13px}
.game-flex{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
.kbd{background:#0b1220;padding:2px 6px;border-radius:4px;font-family:monospace;font-size:12px}
</style></head><body><div class="container" id="app">Loading…</div>
<script>
let ws=null, role=null, nickname=null;
let state={phase:'lobby', students:[]};
let lastSolo=null, lastTeam=null, soloByNick={}, adminTiles={};

function $(s){return document.querySelector(s);}
function el(tag, attrs, ...kids){
  const e=document.createElement(tag);
  for(const k in (attrs||{})){
    if(k==='onclick') e.onclick=attrs[k];
    else if(k==='style') e.style.cssText=attrs[k];
    else if(k==='class') e.className=attrs[k];
    else e.setAttribute(k, attrs[k]);
  }
  for(const c of kids){
    if(c==null) continue;
    e.append(c.nodeType?c:document.createTextNode(c));
  }
  return e;
}

function connect(){
  const proto = location.protocol==='https:'?'wss:':'ws:';
  ws = new WebSocket(proto+'//'+location.host+'/ws');
  ws.onmessage = ev => { try{ handleMsg(JSON.parse(ev.data)); }catch(e){} };
  ws.onclose = () => setTimeout(connect, 1500);
}
function send(m){ if(ws && ws.readyState===1) ws.send(JSON.stringify(m)); }

function handleMsg(m){
  if(m.type==='joined'){
    role = m.role; nickname = m.nickname;
    render();
  } else if(m.type==='phase'){
    const prevPhase = state.phase;
    const prevStudents = state.students.join(',');
    state.phase = m.phase;
    state.students = m.students || [];
    if(prevPhase !== state.phase){
      lastSolo = null; lastTeam = null; soloByNick = {};
      render();
    } else if(role==='admin' && prevStudents !== state.students.join(',')){
      render();
    }
  } else if(m.type==='game_state'){
    if(m.context==='solo'){
      soloByNick[m.nickname] = m;
      if(role==='student' && m.nickname===nickname){
        lastSolo = m; renderStudentSolo();
      }
      if(role==='admin') updateAdminTile(m.nickname);
    } else if(m.context==='team'){
      lastTeam = m; renderTeam();
    }
  } else if(m.type==='error'){
    alert(m.message);
  }
}

function colorMap(st){
  const cm = {0:'#1a1a2e', '-1':'#050510'};
  for(const a of st.agents) cm[a.pid] = a.color;
  return cm;
}

function drawGrid(st, canvas){
  const ctx = canvas.getContext('2d');
  const sz = st.size, W = canvas.width, H = canvas.height;
  const cell = W/sz;
  const bin = atob(st.grid);
  const arr = new Int8Array(bin.length);
  for(let i=0;i<bin.length;i++) arr[i] = (bin.charCodeAt(i)<<24)>>24;
  const cm = colorMap(st);
  ctx.fillStyle = '#000'; ctx.fillRect(0,0,W,H);
  for(let y=0;y<sz;y++){
    for(let x=0;x<sz;x++){
      ctx.fillStyle = cm[arr[y*sz+x]] || '#222';
      ctx.fillRect(x*cell, y*cell, cell+0.7, cell+0.7);
    }
  }
  // agent heads
  for(const a of st.agents){
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(a.x*cell, a.y*cell, cell, cell);
    ctx.strokeStyle = a.color;
    ctx.lineWidth = Math.max(1, cell*0.2);
    ctx.strokeRect(a.x*cell+0.5, a.y*cell+0.5, cell-1, cell-1);
  }
}

function scoreboardEl(st, sort=true){
  const sb = el('div', {class:'scoreboard'});
  const agents = sort ? [...st.agents].sort((a,b)=> (st.scores[b.pid]||0)-(st.scores[a.pid]||0)) : st.agents;
  const total = st.agents.reduce((acc,a)=> acc + (st.scores[a.pid]||0), 0) || 1;
  let rank = 1;
  for(const a of agents){
    const s = st.scores[a.pid] || 0;
    const pct = (s*100/total).toFixed(1);
    sb.append(el('div', {},
      el('span', {class:'tag', style:'background:'+a.color}, '#'+rank),
      ' '+a.nick+(a.algo!=='model'?' ['+a.algo+']':''),
      el('span', {style:'margin-left:auto;color:#94a3b8'}, s+'  ('+pct+'%)')
    ));
    rank++;
  }
  return sb;
}

// ============== Join screen ==============
function renderJoin(){
  const app = $('#app'); app.innerHTML='';
  app.append(
    el('h1', {}, '🎨 땅따먹기 강화학습 배틀'),
    el('p', {style:'color:#94a3b8'},
      '닉네임을 입력하세요. (관리자: 닉네임 칸에 관리자 비번 입력)'),
    (()=>{
      const inp = el('input', {placeholder:'닉네임', style:'width:240px'});
      inp.addEventListener('keydown', e=>{ if(e.key==='Enter') btn.click(); });
      const btn = el('button', {onclick:()=>{
        const v = inp.value.trim(); if(!v) return;
        send({type:'join', nickname:v});
      }}, '참가');
      const row = el('div', {class:'row'}, inp, btn);
      setTimeout(()=>inp.focus(), 30);
      return row;
    })(),
    el('div', {class:'card', style:'margin-top:20px;color:#94a3b8;font-size:13px;line-height:1.7'},
      el('div', {}, '· 스플래툰처럼 지나가는 칸이 내 색으로 칠해집니다.'),
      el('div', {}, '· 솔로 학습 시간 동안 본인의 정책 신경망이 강화학습으로 발전합니다.'),
      el('div', {}, '· 학습이 끝나면 각자의 베스트 모델이 단체전 맵에 배치되어 대결합니다.'),
    )
  );
}

// ============== Student views ==============
function renderStudent(app){
  app.append(phasePill());
  if(state.phase==='lobby'){
    app.append(el('div', {class:'card'},
      el('h2', {}, '대기실'),
      '관리자가 시작 버튼을 누를 때까지 기다려주세요.',
      el('div', {class:'stat'}, '현재 참가자: '+state.students.length+'명')
    ));
  } else if(state.phase==='solo'){
    app.append(buildConfigCard());
    app.append(el('div', {id:'student-solo-area'}));
    renderStudentSolo();
  } else if(state.phase==='team_ready'){
    app.append(el('div', {class:'card'},
      el('h2', {}, '학습 완료'),
      '관리자가 단체전을 시작할 때까지 기다려주세요.'
    ));
  } else if(state.phase==='team'){
    app.append(el('div', {id:'team-area'}));
    renderTeam();
  }
}

function buildConfigCard(){
  const card = el('div', {class:'card'});
  card.append(el('h2', {}, '⚙️ 학습 설정 (적용 누른 다음 에피소드부터 반영)'));

  const REWARD_INFO = [
    ['new_paint',      '빈 칸을 칠함 (영토 확장)',          1.0],
    ['steal',          '적 칸을 덮음 (공격)',               0.5],
    ['revisit',        '내 칸 재방문 (보통 음수)',          -0.1],
    ['wall_hit',       '벽에 막힘 (보통 음수)',             -0.3],
    ['near_enemy',     '근처에 적이 있을 때',                0.0],
    ['frontier',       '인접에 빈 칸 있음 (탐험 여지)',       0.1],
    ['territory_gain', '내 영토 증가량 가중치',              0.0],
    ['survive',        '매 틱 +1 (생존 보너스)',             0.0],
  ];
  const PRESETS = {
    '균형형':  {new_paint:1.0, steal:0.5, revisit:-0.1, wall_hit:-0.3, near_enemy:0.0, frontier:0.1, territory_gain:0.0, survive:0.0},
    '공격형':  {new_paint:0.5, steal:1.5, revisit:-0.2, wall_hit:-0.2, near_enemy:0.5, frontier:0.0, territory_gain:0.0, survive:0.0},
    '방어형':  {new_paint:0.8, steal:0.2, revisit:-0.2, wall_hit:-0.5, near_enemy:-0.5, frontier:0.0, territory_gain:1.0, survive:0.0},
    '탐험형':  {new_paint:0.8, steal:0.3, revisit:-0.3, wall_hit:-0.5, near_enemy:0.0, frontier:0.5, territory_gain:0.0, survive:0.0},
  };

  const cfg = (window._myCfg ||= {map:'square', pc_count:2, weights:{...PRESETS['균형형']}});

  // 맵 선택
  const mapSel = el('select');
  for(const m of ['square','circle','diamond','star','cross','ring']){
    const o = el('option', {value:m}, m);
    if(m===cfg.map) o.selected = true;
    mapSel.append(o);
  }

  // PC 수 선택 (알고리즘은 random/straight 번갈아 고정)
  const pcSel = el('select');
  for(let i=0;i<=6;i++){
    const o = el('option', {value:String(i)}, i+'마리');
    if(i===cfg.pc_count) o.selected = true;
    pcSel.append(o);
  }

  // 슬라이더들
  const sliders = {};
  const sliderBox = el('div', {style:'display:grid;grid-template-columns:1fr;gap:8px;margin-top:10px'});
  function renderSliders(){
    sliderBox.innerHTML = '';
    for(const [key, label, _] of REWARD_INFO){
      const val = (cfg.weights[key] ?? 0).toFixed(2);
      const numLabel = el('span', {style:'min-width:48px;text-align:right;font-family:monospace;color:#22c55e'}, val);
      const slider = el('input', {type:'range', min:'-2', max:'2', step:'0.05', style:'flex:1'});
      slider.value = String(cfg.weights[key] ?? 0);
      slider.oninput = ()=>{
        cfg.weights[key] = parseFloat(slider.value);
        numLabel.textContent = parseFloat(slider.value).toFixed(2);
        numLabel.style.color = parseFloat(slider.value) >= 0 ? '#22c55e' : '#ef4444';
      };
      sliders[key] = slider;
      const row = el('div', {class:'row', style:'gap:10px'},
        el('div', {style:'min-width:200px;font-size:13px'},
          el('div', {style:'font-weight:600'}, key),
          el('div', {class:'stat', style:'margin:0'}, label)
        ),
        slider, numLabel
      );
      sliderBox.append(row);
    }
  }
  renderSliders();

  // 프리셋 버튼
  const presetBox = el('div', {class:'row', style:'margin-top:6px'});
  for(const name of Object.keys(PRESETS)){
    presetBox.append(el('button', {style:'background:#475569', onclick:()=>{
      cfg.weights = {...PRESETS[name]};
      renderSliders();
    }}, '🎯 '+name));
  }
  presetBox.append(el('button', {style:'background:#475569', onclick:()=>{
    for(const k in cfg.weights) cfg.weights[k] = 0;
    renderSliders();
  }}, '0으로 리셋'));

  const apply = el('button', {class:'success', onclick:()=>{
    cfg.map = mapSel.value;
    cfg.pc_count = parseInt(pcSel.value);
    send({type:'config', map: cfg.map, pc_count: cfg.pc_count, weights: cfg.weights});
  }}, '✅ 적용 (다음 에피소드부터)');

  card.append(
    el('div', {class:'row'},
      el('span', {style:'font-weight:600'}, '맵:'), mapSel,
      el('span', {style:'font-weight:600;margin-left:14px'}, 'PC 수:'), pcSel,
      el('span', {class:'stat', style:'margin-left:auto'},
        'PC 알고리즘: random + straight 번갈아 배치 (고정)')
    ),
    el('div', {style:'margin-top:14px;font-weight:600'}, '🎨 보상함수 가중치 조합'),
    el('div', {class:'stat', style:'margin-top:4px'},
      '슬라이더로 각 보상의 가중치를 조절하세요. 매 틱 모든 항목을 합산해서 학습 신호로 씁니다.'),
    presetBox,
    sliderBox,
    el('div', {style:'margin-top:12px'}, apply)
  );
  return card;
}

function renderStudentSolo(){
  const area = $('#student-solo-area');
  if(!area) return;
  area.innerHTML = '';
  if(!lastSolo){
    area.append(el('div', {class:'card'}, '에피소드 시작 중…'));
    return;
  }
  const st = lastSolo.state;
  const canvas = el('canvas', {width:'480', height:'480'});
  const flex = el('div', {class:'game-flex'},
    el('div', {class:'card'},
      el('h2', {}, '솔로 학습 #'+lastSolo.episode),
      el('div', {class:'stat'}, '진행 '+st.tick+'/'+st.max_ticks+
        '  ·  내 최고 점수 '+lastSolo.best_score),
      canvas
    ),
    scoreboardEl(st)
  );
  area.append(flex);
  drawGrid(st, canvas);
}

function renderTeam(){
  const area = $('#team-area');
  if(!area) return;
  area.innerHTML = '';
  if(!lastTeam){
    area.append(el('div', {class:'card'}, '단체전 준비 중…'));
    return;
  }
  const st = lastTeam.state;
  const canvas = el('canvas', {width:'640', height:'640'});
  const flex = el('div', {class:'game-flex'},
    el('div', {class:'card'},
      el('h2', {}, '🏆 단체전 — 맵: '+(lastTeam.map||'?')+(lastTeam.final?' (종료)':'')),
      el('div', {class:'stat'}, '진행 '+st.tick+'/'+st.max_ticks),
      canvas
    ),
    scoreboardEl(st)
  );
  area.append(flex);
  drawGrid(st, canvas);
}

// ============== Admin views ==============
function renderAdmin(app){
  app.append(phasePill());

  const ctl = el('div', {class:'card row'});
  if(state.phase==='lobby'){
    ctl.append(el('button', {class:'success', onclick:()=>send({type:'admin_cmd', cmd:'start_solo'})},
      '▶ 시작 (솔로 학습)'));
    ctl.append(el('span', {class:'stat'}, '학생들이 모두 입장한 뒤 누르세요.'));
  } else if(state.phase==='solo'){
    ctl.append(el('button', {class:'danger', onclick:()=>send({type:'admin_cmd', cmd:'end_solo'})},
      '⏹ 학습 종료'));
  } else if(state.phase==='team_ready' || state.phase==='team'){
    const mapSel = el('select');
    for(const m of ['square','circle','diamond','star','cross','ring']){
      const o = el('option', {value:m}, m);
      if(m===(lastTeam && lastTeam.map)) o.selected = true;
      mapSel.append(o);
    }
    const spdSel = el('select');
    for(const sp of [1,2,4]){
      const o = el('option', {value:String(sp)}, sp+'x');
      spdSel.append(o);
    }
    spdSel.onchange = ()=>{
      // 단체전 진행 중이면 실시간 반영
      send({type:'admin_cmd', cmd:'set_team_speed', speed: parseInt(spdSel.value)});
    };
    ctl.append(
      el('span', {}, '단체전 맵:'), mapSel,
      el('span', {style:'margin-left:8px'}, '배속:'), spdSel,
      el('button', {class:'success', onclick:()=>send({
        type:'admin_cmd', cmd:'start_team',
        map: mapSel.value, speed: parseInt(spdSel.value)
      })},
        state.phase==='team_ready' ? '▶ 단체전 시작' : '🔄 이 맵으로 다시 시작'),
    );
  }
  ctl.append(el('button', {class:'danger', onclick:()=>{
    if(confirm('전체 세션을 초기화할까요? (모든 학생 제거)')) send({type:'admin_cmd', cmd:'reset'});
  }, style:'margin-left:auto'}, '↺ 리셋'));
  app.append(ctl);

  app.append(el('div', {class:'card'},
    el('h2', {}, '참가자 ('+state.students.length+'명)'),
    el('div', {style:'color:#94a3b8;font-size:13px'},
      state.students.length ? state.students.join(' · ') : '아직 없음')
  ));

  if(state.phase==='solo'){
    app.append(el('h2', {}, '👁 학생 CCTV'));
    app.append(el('div', {class:'grid-students', id:'admin-cctv'}));
    buildAdminGrid();
  } else if(state.phase==='team' || state.phase==='team_ready'){
    app.append(el('div', {id:'team-area'}));
    renderTeam();
  }
}

function buildAdminGrid(){
  const area = $('#admin-cctv');
  if(!area) return;
  area.innerHTML = '';
  adminTiles = {};
  for(const nick of state.students){
    const canvas = el('canvas', {width:'210', height:'210'});
    const info = el('div', {class:'stat'}, '대기 중…');
    const tile = el('div', {class:'student-tile'},
      el('div', {style:'font-weight:600;margin-bottom:6px'}, nick),
      canvas, info);
    adminTiles[nick] = {canvas, info};
    area.append(tile);
    if(soloByNick[nick]) updateAdminTile(nick);
  }
}

function updateAdminTile(nick){
  const t = adminTiles[nick]; const data = soloByNick[nick];
  if(!t || !data) return;
  drawGrid(data.state, t.canvas);
  const my = data.state.scores[1] || 0;
  t.info.textContent = 'Ep '+data.episode+' · 최고 '+data.best_score+' · 지금 '+my;
}

function phasePill(){
  const names = {lobby:'대기실', solo:'솔로 학습', team_ready:'학습 종료', team:'단체전'};
  return el('div', {class:'row', style:'margin-bottom:8px'},
    el('span', {class:'phase-pill'}, '단계: '+(names[state.phase]||state.phase)),
    el('span', {class:'phase-pill'}, '참가: '+state.students.length+'명'),
  );
}

function render(){
  if(!role){ renderJoin(); return; }
  const app = $('#app'); app.innerHTML = '';
  app.append(el('h1', {},
    (role==='admin' ? '🛠 [관리자] ' : '🎨 ') + '땅따먹기 배틀'
      + (role==='student' ? ' · ' + nickname : '')));
  if(role==='admin') renderAdmin(app); else renderStudent(app);
}

render();
connect();
</script></body></html>"""


app = FastAPI()


@app.get("/")
async def index():
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    role: Optional[str] = None
    student: Optional[StudentSession] = None
    try:
        while True:
            data = await ws.receive_text()
            try:
                m = json.loads(data)
            except Exception:
                continue
            t = m.get("type")

            if t == "join" and role is None:
                nick = (m.get("nickname") or "").strip()
                if not nick:
                    await _send(ws, json.dumps({"type": "error", "message": "닉네임을 입력하세요"}))
                    continue
                if nick == ADMIN_PASSWORD:
                    role = "admin"
                    S.admin_ws.add(ws)
                    await _send(ws, json.dumps({"type": "joined", "role": "admin", "nickname": "관리자"}))
                else:
                    if nick in S.students:
                        student = S.students[nick]
                        student.ws = ws
                    else:
                        if S.phase != "lobby":
                            await _send(ws, json.dumps({"type": "error",
                                "message": "이미 게임이 시작되어 신규 입장이 불가합니다."}))
                            continue
                        student = StudentSession(nickname=nick, ws=ws)
                        S.students[nick] = student
                    role = "student"
                    await _send(ws, json.dumps({"type": "joined", "role": "student", "nickname": nick}))
                await broadcast_phase()

            elif t == "config" and role == "student" and student is not None:
                shape = m.get("map", "square")
                if shape not in MAPS:
                    shape = "square"
                try:
                    pc = int(m.get("pc_count", 2))
                except Exception:
                    pc = 2
                pc = max(0, min(6, pc))
                in_w = m.get("weights") or {}
                weights = {}
                for k in REWARD_KEYS:
                    try:
                        v = float(in_w.get(k, DEFAULT_WEIGHTS[k]))
                    except Exception:
                        v = DEFAULT_WEIGHTS[k]
                    weights[k] = max(-2.0, min(2.0, v))
                student.config = {"map": shape, "pc_count": pc, "weights": weights}

            elif t == "admin_cmd" and role == "admin":
                cmd = m.get("cmd")
                if cmd == "start_solo":
                    await cmd_start_solo()
                elif cmd == "end_solo":
                    await cmd_end_solo()
                elif cmd == "start_team":
                    try:
                        sp = int(m.get("speed", S.team_speed))
                    except Exception:
                        sp = S.team_speed
                    S.team_speed = sp if sp in (1, 2, 4) else 1
                    await cmd_start_team(m.get("map", "square"))
                elif cmd == "set_team_speed":
                    try:
                        sp = int(m.get("speed", 1))
                    except Exception:
                        sp = 1
                    S.team_speed = sp if sp in (1, 2, 4) else 1
                elif cmd == "reset":
                    await cmd_reset()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("ws error:", repr(e))
    finally:
        if role == "admin":
            S.admin_ws.discard(ws)
        elif role == "student" and student is not None and student.ws is ws:
            student.ws = None


if __name__ == "__main__":
    print("=" * 60)
    print(" 땅따먹기 강화학습 배틀 서버")
    print("=" * 60)
    print(f"  주소         : http://{HOST}:{PORT}/")
    print(f"  관리자 비번  : '{ADMIN_PASSWORD}'  (이 문자열을 닉네임으로 입력)")
    print(f"  솔로 그리드  : {SOLO_GRID}x{SOLO_GRID}, 에피소드={SOLO_EP_TICKS/TICK_HZ:.0f}초")
    print(f"  단체전 그리드: {TEAM_GRID}x{TEAM_GRID}, 시간={TEAM_TICKS/TICK_HZ:.0f}초")
    print("=" * 60)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
