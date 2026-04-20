"""
================================================================
  자율주행 AI 시뮬레이터 v4 — 교육용 (2시간 실습)
  ================================================
  [v3 대비 주요 개선]
  1. 실시간 세대별 적합도 그래프 (우측 패널)
  2. GA 연산 시각화 (선택/교차/변이 과정 표시)
  3. 학습 로그 자동 저장 (CSV)
  4. 인구 다양성 지표 표시
  5. 단계별 한글 튜토리얼 툴팁
  6. 실험 비교 모드 (시드 고정 → 보상 함수 효과 비교)
  7. 미니맵 + 최적 경로 표시
  8. 정체 감지 개선 (각도 변화량 추가)
  9. 세대별 통계 히스토리 (완주율, 평균 적합도 등)
  10. 결과 요약 화면 강화
================================================================
"""
import pygame
import numpy as np
import math
import os
import platform
import csv
import time

# ====================================================================
# 0. 상수 및 색상
# ====================================================================
WIDTH = 1200
HEIGHT = 800
CONFIG_W = 1200
CONFIG_H = 800
MAX_STEPS = 1500

STATE_CONFIG1 = -3
STATE_CONFIG2 = -2
STATE_SETUP = -1
STATE_SIMULATING = 0
STATE_FINISHED = 1

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
LIGHT_GRAY = (240, 240, 240)
BLUE = (0, 100, 255)
LIGHT_BLUE = (180, 210, 255)
RED = (255, 50, 50)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)
GOLD = (255, 215, 0)
ORANGE = (255, 165, 0)
PANEL_BG = (245, 245, 248)
ACCENT = (60, 60, 180)
PURPLE = (140, 80, 200)
CYAN = (0, 180, 200)
PINK = (255, 120, 160)

SENSOR_LEN = 150
MAX_SPEED = 3.0
STAGNATION_CHECK_INTERVAL = 60
STAGNATION_MIN_DISPLACEMENT = 30
STAGNATION_MIN_ANGLE_CHANGE = 0.3  # 새로 추가: 회전만 하는 정체도 감지

# 우측 정보 패널 너비
INFO_PANEL_W = 320

# ====================================================================
# 보상 함수 정의
# ====================================================================
REWARD_METHODS = [
    {
        "id": "max_dist",
        "name": "최대 도달 거리",
        "desc": "시작점에서 가장 멀리 간 직선거리를 보상",
        "detail": "fitness += 시작점~현재위치 직선거리",
        "icon": "▶",
        "color": (70, 130, 230),
    },
    {
        "id": "cumulative",
        "name": "누적 이동 거리",
        "desc": "총 이동한 경로 길이를 합산하여 보상",
        "detail": "fitness += 매 스텝 이동거리 합산 × 0.5",
        "icon": "~",
        "color": (50, 180, 100),
    },
    {
        "id": "survival",
        "name": "생존 시간",
        "desc": "충돌 없이 오래 살아남을수록 보상",
        "detail": "fitness += 생존 스텝 수 × 0.3",
        "icon": "♥",
        "color": (220, 80, 80),
    },
    {
        "id": "wall_avoid",
        "name": "벽 회피 보상",
        "desc": "센서 평균 거리가 클수록 보상 (벽에서 멀리)",
        "detail": "fitness += 센서평균거리 누적 × 0.5",
        "icon": "◇",
        "color": (200, 150, 30),
    },
    {
        "id": "straight",
        "name": "직진 보상",
        "desc": "조향이 적을수록 보상 (효율적 주행)",
        "detail": "fitness += max(0, 500 - 조향합 × 0.5)",
        "icon": "→",
        "color": (140, 80, 200),
    },
]


# ====================================================================
# 한글 폰트
# ====================================================================
def find_korean_font():
    system = platform.system()
    candidates = []
    if system == "Windows":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        candidates = [
            os.path.join(windir, "Fonts", "malgun.ttf"),
            os.path.join(windir, "Fonts", "malgunbd.ttf"),
            os.path.join(windir, "Fonts", "gulim.ttc"),
        ]
    elif system == "Darwin":
        candidates = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/AppleGothic.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def make_font(size, bold=False):
    fp = find_korean_font()
    if fp:
        try:
            f = pygame.font.Font(fp, size)
            f.set_bold(bold)
            return f
        except Exception:
            pass
    for name in ["malgungothic", "malgun gothic", "applegothic", "nanumgothic"]:
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f.render("가", True, BLACK).get_width() > 2:
                return f
        except Exception:
            continue
    return pygame.font.SysFont("Arial", size, bold=bold)


# ====================================================================
# 트랙 함수
# ====================================================================
track_img = None
TRACK_W = 0
TRACK_H = 0


def load_track(track_path=None):
    global track_img, TRACK_W, TRACK_H, WIDTH, HEIGHT
    if track_path and os.path.exists(track_path):
        path = track_path
    else:
        path = "./assets/track01.png"
        if not os.path.exists(path):
            path = "track.png"
        if not os.path.exists(path):
            s = pygame.Surface((CONFIG_W, CONFIG_H))
            s.fill(WHITE)
            pygame.draw.rect(s, BLACK, (0, 0, CONFIG_W, CONFIG_H), 20)
            pygame.image.save(s, path)
    track_img = pygame.image.load(path).convert()
    TRACK_W = track_img.get_width()
    TRACK_H = track_img.get_height()
    # 우측 패널 포함 크기
    WIDTH = TRACK_W + INFO_PANEL_W
    HEIGHT = max(TRACK_H, 700)  # 최소 높이 보장
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("자율주행 AI 시뮬레이터 v4 — 교육용")
    return screen


def is_wall(x, y):
    if x < 0 or x >= TRACK_W or y < 0 or y >= TRACK_H:
        return True
    c = track_img.get_at((int(x), int(y)))
    return c[0] < 50 and c[1] < 50 and c[2] < 50


def cast_ray(sx, sy, angle):
    dx, dy = math.cos(angle), math.sin(angle)
    x, y = sx, sy
    for d in range(SENSOR_LEN):
        if is_wall(x, y):
            return d
        x += dx
        y += dy
    return SENSOR_LEN


def check_collision(x, y, angle, verts):
    pts = []
    for v in verts:
        nx = v[0] * math.cos(angle) - v[1] * math.sin(angle) + x
        ny = v[0] * math.sin(angle) + v[1] * math.cos(angle) + y
        pts.append((nx, ny))
    for i in range(len(pts)):
        p1, p2 = pts[i], pts[(i + 1) % len(pts)]
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(1, int(dist))
        for j in range(steps + 1):
            t = j / steps
            if is_wall(p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t):
                return True
    return False


def check_finish(x, y, angle, verts):
    pts = []
    for v in verts:
        nx = v[0] * math.cos(angle) - v[1] * math.sin(angle) + x
        ny = v[0] * math.sin(angle) + v[1] * math.cos(angle) + y
        pts.append((nx, ny))
    for i in range(len(pts)):
        p1, p2 = pts[i], pts[(i + 1) % len(pts)]
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(1, int(dist))
        for j in range(steps + 1):
            t = j / steps
            cx = p1[0] + (p2[0] - p1[0]) * t
            cy = p1[1] + (p2[1] - p1[1]) * t
            if 0 <= cx < TRACK_W and 0 <= cy < TRACK_H:
                c = track_img.get_at((int(cx), int(cy)))
                if c[0] > 200 and c[1] < 100 and c[2] < 100:
                    return True
    return False


# ====================================================================
# CarBrain — 신경망 (교육용 주석 추가)
# ====================================================================
class CarBrain:
    """
    2층 신경망 (Multi-Layer Perceptron)
    ───────────────────────────────────
    구조: 입력(센서) → 은닉층(ReLU) → 출력(softmax → 좌/우 확률)

    [학습 포인트]
    - 이 신경망은 역전파(backprop)로 학습하지 않음
    - 대신 유전 알고리즘이 가중치를 '진화'시킴
    - 좋은 가중치 = 높은 적합도 → 다음 세대에 전달
    """

    def __init__(self, n_sensors, w1=None, b1=None, w2=None, b2=None):
        self.n_sensors = n_sensors
        self.hidden = max(8, n_sensors * 2)
        if w1 is None:
            # He 초기화: 깊은 네트워크 학습에 유리한 초기화 방법
            self.w1 = np.random.randn(n_sensors, self.hidden) * np.sqrt(2.0 / n_sensors)
            self.b1 = np.zeros(self.hidden)
            self.w2 = np.random.randn(self.hidden, 2) * np.sqrt(2.0 / self.hidden)
            self.b2 = np.zeros(2)
        else:
            self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2

    def copy(self):
        return CarBrain(self.n_sensors, self.w1.copy(), self.b1.copy(),
                        self.w2.copy(), self.b2.copy())

    def forward(self, data):
        """순전파: 센서 데이터 → 조향 확률"""
        h = np.maximum(0, np.dot(data, self.w1) + self.b1)  # ReLU
        o = np.dot(h, self.w2) + self.b2
        e = np.exp(o - np.max(o))  # softmax (수치 안정)
        return e / e.sum()

    def mutate(self, rate=0.15, strength=0.2):
        """
        변이 (Mutation)
        ─────────────
        - rate: 각 가중치가 변이될 확률 (0~1)
        - strength: 변이 크기 (표준편차)
        - 생물학의 유전자 돌연변이에 해당
        """
        def m(mat):
            mask = np.random.rand(*mat.shape) < rate
            return mat + np.random.randn(*mat.shape) * strength * mask
        return CarBrain(self.n_sensors, m(self.w1), m(self.b1), m(self.w2), m(self.b2))

    def crossover(self, other):
        """
        교차 (Crossover)
        ──────────────
        - 두 부모의 가중치를 50:50 확률로 섞음
        - 생물학의 유성생식에 해당
        """
        def c(a, b):
            mask = np.random.rand(*a.shape) < 0.5
            return np.where(mask, a, b)
        return CarBrain(self.n_sensors, c(self.w1, other.w1), c(self.b1, other.b1),
                        c(self.w2, other.w2), c(self.b2, other.b2))

    def get_weight_stats(self):
        """가중치 통계 (다양성 측정용)"""
        all_w = np.concatenate([self.w1.flatten(), self.b1, self.w2.flatten(), self.b2])
        return {"mean": float(np.mean(all_w)), "std": float(np.std(all_w)),
                "min": float(np.min(all_w)), "max": float(np.max(all_w))}


# ====================================================================
# Car
# ====================================================================
class Car:
    def __init__(self, car_id, sx, sy, n_sensors, brain=None):
        self.id = car_id
        self.x, self.y = sx, sy
        self.start_x, self.start_y = sx, sy
        self.angle = -math.pi / 2
        self.speed = MAX_SPEED
        self.n_sensors = n_sensors
        self.brain = brain if brain else CarBrain(n_sensors)
        self.sensor_data = np.zeros(n_sensors)
        self.vertices = [(25, 0), (-15, 15), (-15, -15)]
        self.color = BLUE
        self.crashed = False
        self.finished = False
        self.stagnated = False
        self.sensor_angles = []
        self.sensor_dists = []
        self.max_dist = 0.0
        self.cumul_dist = 0.0
        self.prev_x, self.prev_y = sx, sy
        self.step_count = 0
        self.cp_x, self.cp_y = sx, sy
        self.cp_angle = self.angle  # 정체 감지용 각도 체크포인트
        self.alive_steps = 0
        self.avg_sensor_sum = 0.0
        self.steer_penalty_sum = 0.0
        self.trajectory = [(sx, sy)]  # 경로 기록
        self.birth_method = "random"  # elite / crossover / mutant / random

    @property
    def is_stopped(self):
        return self.crashed or self.finished or self.stagnated

    def compute_fitness(self, reward_ids):
        fit = 0.0
        if "max_dist" in reward_ids:
            fit += self.max_dist
        if "cumulative" in reward_ids:
            fit += self.cumul_dist * 0.5
        if "survival" in reward_ids:
            fit += self.alive_steps * 0.3
        if "wall_avoid" in reward_ids:
            fit += self.avg_sensor_sum * 0.5
        if "straight" in reward_ids:
            fit += max(0, 500 - self.steer_penalty_sum * 0.5)
        if self.finished:
            fit += 5000
        if self.stagnated:
            fit -= 200
        return fit

    def update(self):
        if self.is_stopped:
            return
        self.step_count += 1
        self.alive_steps += 1

        # 개선된 정체 감지: 위치 + 각도 변화 모두 확인
        if self.step_count % STAGNATION_CHECK_INTERVAL == 0:
            disp = math.hypot(self.x - self.cp_x, self.y - self.cp_y)
            angle_diff = abs(self.angle - self.cp_angle)
            if disp < STAGNATION_MIN_DISPLACEMENT and angle_diff < STAGNATION_MIN_ANGLE_CHANGE:
                self.stagnated = True
                self.color = ORANGE
                return
            self.cp_x, self.cp_y = self.x, self.y
            self.cp_angle = self.angle

        spread = math.pi / (self.n_sensors - 1) if self.n_sensors > 1 else 0
        half = (self.n_sensors - 1) / 2
        self.sensor_angles = []
        self.sensor_dists = []
        for i in range(self.n_sensors):
            a = self.angle + (i - half) * spread * 0.8
            self.sensor_angles.append(a)
            d = cast_ray(self.x, self.y, a)
            self.sensor_data[i] = d / SENSOR_LEN
            self.sensor_dists.append(d)

        self.avg_sensor_sum += sum(self.sensor_dists) / self.n_sensors

        nx = self.x + self.speed * math.cos(self.angle)
        ny = self.y + self.speed * math.sin(self.angle)

        if check_finish(nx, ny, self.angle, self.vertices):
            self.finished = True
            self.color = GREEN
            return
        if check_collision(nx, ny, self.angle, self.vertices):
            self.crashed = True
            self.color = RED
            return

        prob = self.brain.forward(self.sensor_data)
        steer = prob[1] - prob[0]
        self.angle += steer * 0.15
        self.steer_penalty_sum += abs(steer)
        cur_speed = self.speed * (1.0 - abs(steer) * 0.6)
        self.x += cur_speed * math.cos(self.angle)
        self.y += cur_speed * math.sin(self.angle)

        step_d = math.hypot(self.x - self.prev_x, self.y - self.prev_y)
        self.cumul_dist += step_d
        self.prev_x, self.prev_y = self.x, self.y

        d_start = math.hypot(self.x - self.start_x, self.y - self.start_y)
        if d_start > self.max_dist:
            self.max_dist = d_start

        # 경로 기록 (10스텝마다)
        if self.step_count % 10 == 0:
            self.trajectory.append((self.x, self.y))

    def draw(self, surface, font_obj, show_sensors=True, show_id=True):
        if not self.is_stopped and self.sensor_angles and show_sensors:
            for i in range(self.n_sensors):
                a = self.sensor_angles[i]
                d = self.sensor_dists[i]
                ex = self.x + d * math.cos(a)
                ey = self.y + d * math.sin(a)
                t = d / SENSOR_LEN
                lr = int(255 * (1 - t))
                lg = int(60 * (1 - t) + 220 * t)
                lb = int(60 * (1 - t) + 80 * t)
                pygame.draw.line(surface, (lr, lg, lb), (self.x, self.y), (ex, ey), 1)
                pygame.draw.circle(surface, (255, 60, 60), (int(ex), int(ey)), 3)

        pts = [
            (v[0] * math.cos(self.angle) - v[1] * math.sin(self.angle) + self.x,
             v[0] * math.sin(self.angle) + v[1] * math.cos(self.angle) + self.y)
            for v in self.vertices
        ]
        pygame.draw.polygon(surface, self.color, pts)
        pygame.draw.polygon(surface, BLACK, pts, 2)

        if show_id:
            lc = WHITE if self.color in (BLUE, GREEN, DARK_GREEN) else BLACK
            txt = font_obj.render(str(self.id), True, lc)
            surface.blit(txt, (self.x - txt.get_width() // 2, self.y - txt.get_height() // 2))


# ====================================================================
# 세대 통계 기록
# ====================================================================
class GenerationStats:
    """매 세대의 통계를 기록하여 그래프와 분석에 활용"""

    def __init__(self):
        self.history = []

    def record(self, generation, cars, reward_ids):
        fitnesses = [c.compute_fitness(reward_ids) for c in cars]
        finished = sum(1 for c in cars if c.finished)
        crashed = sum(1 for c in cars if c.crashed)
        stagnated = sum(1 for c in cars if c.stagnated)

        # 인구 다양성: 가중치 표준편차의 평균
        weight_stds = [c.brain.get_weight_stats()["std"] for c in cars]

        entry = {
            "gen": generation,
            "best_fit": max(fitnesses),
            "avg_fit": sum(fitnesses) / len(fitnesses),
            "worst_fit": min(fitnesses),
            "finished": finished,
            "crashed": crashed,
            "stagnated": stagnated,
            "total": len(cars),
            "diversity": sum(weight_stds) / len(weight_stds),
            "best_dist": max(c.max_dist for c in cars),
        }
        self.history.append(entry)
        return entry

    def save_csv(self, filename="ga_log.csv"):
        """학습 로그를 CSV로 저장"""
        if not self.history:
            return
        keys = self.history[0].keys()
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)


# ====================================================================
# 설정 화면 1: 자동차 수 / 센서 수
# ====================================================================
class ConfigPage1:
    def __init__(self, screen, fonts):
        self.screen = screen
        self.fonts = fonts
        self.num_cars = 10
        self.num_sensors = 5
        self.seed = 42
        self.done = False
        self.go_back = False

        cx = CONFIG_W // 2
        self.car_minus = pygame.Rect(cx + 30, 215, 50, 50)
        self.car_plus = pygame.Rect(cx + 170, 215, 50, 50)
        self.sen_minus = pygame.Rect(cx + 30, 335, 50, 50)
        self.sen_plus = pygame.Rect(cx + 170, 335, 50, 50)
        self.seed_minus = pygame.Rect(cx + 30, 455, 50, 50)
        self.seed_plus = pygame.Rect(cx + 170, 455, 50, 50)
        self.seed_random_btn = pygame.Rect(cx + 225, 463, 50, 35)
        self.btn_next = pygame.Rect(cx - 120, CONFIG_H - 100, 240, 55)

        # 교육용: 파라미터 설명 툴팁
        self.tooltips = {
            "cars": [
                "💡 자동차 수가 많을수록:",
                "  • 해를 더 넓게 탐색 (탐험↑)",
                "  • 계산량 증가 → 속도 감소",
                "  • 권장: 10~20대로 시작",
            ],
            "sensors": [
                "💡 센서가 많을수록:",
                "  • 환경 인식 능력 향상",
                "  • 신경망 입력 차원 증가",
                "  • 학습 난이도도 증가",
                "  • 권장: 5개로 시작",
            ],
            "seed": [
                "💡 랜덤 시드란?",
                "  • 같은 시드 = 같은 초기 가중치",
                "  • 보상 함수만 바꿔 비교 가능",
                "  • 실험의 재현성 보장",
            ],
        }
        self.active_tooltip = None

    def handle_click(self, pos):
        if self.car_minus.collidepoint(pos):
            self.num_cars = max(2, self.num_cars - 1)
        elif self.car_plus.collidepoint(pos):
            self.num_cars = min(30, self.num_cars + 1)
        elif self.sen_minus.collidepoint(pos):
            self.num_sensors = max(2, self.num_sensors - 1)
        elif self.sen_plus.collidepoint(pos):
            self.num_sensors = min(12, self.num_sensors + 1)
        elif self.seed_minus.collidepoint(pos):
            self.seed = max(0, self.seed - 1)
        elif self.seed_plus.collidepoint(pos):
            self.seed = min(9999, self.seed + 1)
        elif self.seed_random_btn.collidepoint(pos):
            import random
            self.seed = random.randint(0, 9999)
        elif self.btn_next.collidepoint(pos):
            self.done = True

    def draw(self):
        self.screen.fill(PANEL_BG)
        f_xl, f_lg, f_md, f_sm = self.fonts
        cx = CONFIG_W // 2

        self._draw_steps(1)

        t = f_xl.render("자율주행 AI 시뮬레이터", True, ACCENT)
        self.screen.blit(t, (cx - t.get_width() // 2, 30))

        # 교육용 서브타이틀
        sub_edu = f_sm.render("유전 알고리즘(GA)으로 자동차의 '두뇌(신경망)'를 진화시킵니다", True, DARK_GRAY)
        self.screen.blit(sub_edu, (cx - sub_edu.get_width() // 2, 75))

        pygame.draw.line(self.screen, GRAY, (cx - 280, 105), (cx + 280, 105), 2)

        sub = f_lg.render("STEP 1: 학습 파라미터 설정", True, DARK_GRAY)
        self.screen.blit(sub, (cx - sub.get_width() // 2, 120))

        # 파라미터 카드들
        card_x = cx - 280
        card_w = 560

        self._draw_param_card(card_x, 175, card_w, 110,
                              "자동차 수 (인구 크기)", "동시에 학습하는 자동차 대수 (2~30)",
                              self.num_cars, self.car_minus, self.car_plus)

        self._draw_param_card(card_x, 295, card_w, 110,
                              "센서 수 (입력 차원)", "각 자동차의 벽 감지 센서 개수 (2~12)",
                              self.num_sensors, self.sen_minus, self.sen_plus)

        self._draw_param_card(card_x, 415, card_w, 110,
                              "랜덤 시드 (재현성)", "같은 시드 → 같은 초기 조건 → 비교 실험 가능",
                              self.seed, self.seed_minus, self.seed_plus)

        # 랜덤 버튼
        hover_r = self.seed_random_btn.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.screen, (ACCENT if hover_r else DARK_GRAY), self.seed_random_btn, 0, 8)
        rt = f_sm.render("랜덤", True, WHITE)
        self.screen.blit(rt, (self.seed_random_btn.centerx - rt.get_width() // 2,
                              self.seed_random_btn.centery - rt.get_height() // 2))

        # 툴팁 표시
        mouse = pygame.mouse.get_pos()
        card_regions = [
            (pygame.Rect(card_x, 175, card_w, 110), "cars"),
            (pygame.Rect(card_x, 295, card_w, 110), "sensors"),
            (pygame.Rect(card_x, 415, card_w, 110), "seed"),
        ]
        self.active_tooltip = None
        for rect, key in card_regions:
            if rect.collidepoint(mouse):
                self.active_tooltip = key
                break

        if self.active_tooltip:
            self._draw_tooltip(mouse, self.tooltips[self.active_tooltip])

        # 미리보기
        preview = f_sm.render(
            f"자동차 {self.num_cars}대 × 센서 {self.num_sensors}개  |  시드: {self.seed}  |  "
            f"신경망 파라미터: {self.num_sensors * max(8, self.num_sensors * 2) + max(8, self.num_sensors * 2) + max(8, self.num_sensors * 2) * 2 + 2}개",
            True, DARK_GRAY,
        )
        self.screen.blit(preview, (cx - preview.get_width() // 2, 545))

        # 다음 버튼
        hover = self.btn_next.collidepoint(pygame.mouse.get_pos())
        bg = ACCENT if hover else BLUE
        pygame.draw.rect(self.screen, bg, self.btn_next, 0, 12)
        nt = f_lg.render("다음 →", True, WHITE)
        self.screen.blit(nt, (self.btn_next.centerx - nt.get_width() // 2,
                              self.btn_next.centery - nt.get_height() // 2))

    def _draw_tooltip(self, mouse_pos, lines):
        f_xl, f_lg, f_md, f_sm = self.fonts
        w = 300
        h = len(lines) * 22 + 16
        x = min(mouse_pos[0] + 15, CONFIG_W - w - 10)
        y = min(mouse_pos[1] - 10, CONFIG_H - h - 10)

        # 배경
        tip_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(tip_surf, (40, 40, 60, 230), (0, 0, w, h), 0, 8)
        self.screen.blit(tip_surf, (x, y))

        for i, line in enumerate(lines):
            t = f_sm.render(line, True, (230, 230, 240))
            self.screen.blit(t, (x + 10, y + 8 + i * 22))

    def _draw_param_card(self, x, y, w, h, title, desc, value, btn_minus, btn_plus):
        f_xl, f_lg, f_md, f_sm = self.fonts
        card = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, WHITE, card, 0, 12)
        pygame.draw.rect(self.screen, GRAY, card, 2, 12)

        tt = f_lg.render(title, True, BLACK)
        self.screen.blit(tt, (x + 25, y + 15))
        dt = f_sm.render(desc, True, DARK_GRAY)
        self.screen.blit(dt, (x + 25, y + 48))

        hover_m = btn_minus.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.screen, (DARK_GRAY if hover_m else GRAY), btn_minus, 0, 8)
        mt = f_lg.render("-", True, WHITE)
        self.screen.blit(mt, (btn_minus.centerx - mt.get_width() // 2,
                              btn_minus.centery - mt.get_height() // 2))

        vt = f_xl.render(str(value), True, ACCENT)
        val_x = (btn_minus.right + btn_plus.left) // 2 - vt.get_width() // 2
        val_y = btn_minus.centery - vt.get_height() // 2
        self.screen.blit(vt, (val_x, val_y))

        hover_p = btn_plus.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.screen, (DARK_GRAY if hover_p else GRAY), btn_plus, 0, 8)
        pt = f_lg.render("+", True, WHITE)
        self.screen.blit(pt, (btn_plus.centerx - pt.get_width() // 2,
                              btn_plus.centery - pt.get_height() // 2))

    def _draw_steps(self, current):
        f_xl, f_lg, f_md, f_sm = self.fonts
        labels = ["1. 파라미터", "2. 보상 함수", "3. 맵 선택", "4. 시뮬레이션"]
        total_w = 650
        start_x = (CONFIG_W - total_w) // 2
        step_w = total_w // len(labels)
        y = CONFIG_H - 40
        for i, lbl in enumerate(labels):
            x = start_x + i * step_w
            color = ACCENT if i + 1 == current else (GRAY if i + 1 > current else DARK_GREEN)
            t = f_sm.render(lbl, True, color)
            self.screen.blit(t, (x + step_w // 2 - t.get_width() // 2, y))
            if i < len(labels) - 1:
                lx = x + step_w
                pygame.draw.line(self.screen, GRAY, (lx - 30, y + 8), (lx + 10, y + 8), 1)


# ====================================================================
# 설정 화면 2: 보상 함수 선택
# ====================================================================
class ConfigPage2:
    def __init__(self, screen, fonts):
        self.screen = screen
        self.fonts = fonts
        self.selected = {"max_dist"}
        self.done = False
        self.go_back = False

        cx = CONFIG_W // 2
        card_w, card_h = 340, 120  # 높이 증가 (수식 표시)
        gap = 20
        row1_x = cx - (card_w * 3 + gap * 2) // 2
        row2_x = cx - (card_w * 2 + gap) // 2

        self.card_rects = []
        for i in range(3):
            self.card_rects.append(pygame.Rect(row1_x + i * (card_w + gap), 230, card_w, card_h))
        for i in range(2):
            self.card_rects.append(pygame.Rect(row2_x + i * (card_w + gap), 230 + card_h + gap, card_w, card_h))

        self.btn_start = pygame.Rect(cx - 120, CONFIG_H - 100, 240, 55)
        self.btn_back = pygame.Rect(cx - 320, CONFIG_H - 100, 140, 55)

    def handle_click(self, pos):
        for i, r in enumerate(self.card_rects):
            if r.collidepoint(pos):
                rid = REWARD_METHODS[i]["id"]
                if rid in self.selected:
                    if len(self.selected) > 1:
                        self.selected.discard(rid)
                else:
                    self.selected.add(rid)
                return
        if self.btn_start.collidepoint(pos) and self.selected:
            self.done = True
        elif self.btn_back.collidepoint(pos):
            self.go_back = True

    def draw(self):
        self.screen.fill(PANEL_BG)
        f_xl, f_lg, f_md, f_sm = self.fonts
        cx = CONFIG_W // 2

        self._draw_steps(2)

        t = f_xl.render("보상 함수 선택", True, ACCENT)
        self.screen.blit(t, (cx - t.get_width() // 2, 30))

        # 교육용 설명
        edu_lines = [
            "보상 함수 = AI가 '잘 했다/못 했다'를 판단하는 기준",
            "같은 환경이라도 보상 함수에 따라 완전히 다른 행동을 학습합니다",
        ]
        for i, line in enumerate(edu_lines):
            lt = f_sm.render(line, True, DARK_GRAY)
            self.screen.blit(lt, (cx - lt.get_width() // 2, 80 + i * 22))

        pygame.draw.line(self.screen, GRAY, (cx - 280, 135), (cx + 280, 135), 2)

        sub = f_md.render("학습 기준을 선택하세요 (다중 선택 시 합산 적용)", True, DARK_GRAY)
        self.screen.blit(sub, (cx - sub.get_width() // 2, 150))

        cnt_t = f_md.render(f"{len(self.selected)}개 선택됨", True, ACCENT)
        self.screen.blit(cnt_t, (cx - cnt_t.get_width() // 2, 180))

        # 실험 제안
        suggest = f_sm.render("🔬 실험 제안: 같은 시드로 보상 함수만 바꿔 결과를 비교해 보세요!", True, PURPLE)
        self.screen.blit(suggest, (cx - suggest.get_width() // 2, 205))

        for i, rm in enumerate(REWARD_METHODS):
            rect = self.card_rects[i]
            selected = rm["id"] in self.selected
            hover = rect.collidepoint(pygame.mouse.get_pos())
            self._draw_reward_card(rect, rm, selected, hover)

        # 뒤로
        hover_b = self.btn_back.collidepoint(pygame.mouse.get_pos())
        bg_b = DARK_GRAY if hover_b else GRAY
        pygame.draw.rect(self.screen, bg_b, self.btn_back, 0, 12)
        bt = f_md.render("← 이전", True, WHITE)
        self.screen.blit(bt, (self.btn_back.centerx - bt.get_width() // 2,
                              self.btn_back.centery - bt.get_height() // 2))

        # 시작
        can_start = len(self.selected) > 0
        hover_s = self.btn_start.collidepoint(pygame.mouse.get_pos())
        bg_s = ACCENT if (hover_s and can_start) else (BLUE if can_start else GRAY)
        pygame.draw.rect(self.screen, bg_s, self.btn_start, 0, 12)
        st = f_lg.render("다음 →", True, WHITE)
        self.screen.blit(st, (self.btn_start.centerx - st.get_width() // 2,
                              self.btn_start.centery - st.get_height() // 2))

    def _draw_reward_card(self, rect, rm, selected, hover):
        f_xl, f_lg, f_md, f_sm = self.fonts
        theme = rm["color"]

        if selected:
            bg = (*theme,)
            border = BLACK
            name_color = WHITE
            desc_color = (220, 220, 230)
            detail_color = (200, 230, 255)
            icon_color = WHITE
        elif hover:
            bg = (theme[0] // 3 + 170, theme[1] // 3 + 170, theme[2] // 3 + 170)
            border = theme
            name_color = BLACK
            desc_color = DARK_GRAY
            detail_color = ACCENT
            icon_color = theme
        else:
            bg = WHITE
            border = GRAY
            name_color = BLACK
            desc_color = DARK_GRAY
            detail_color = DARK_GRAY
            icon_color = theme

        pygame.draw.rect(self.screen, bg, rect, 0, 12)
        pygame.draw.rect(self.screen, border, rect, 2, 12)

        indicator_x = rect.right - 25
        indicator_y = rect.top + 15
        if selected:
            pygame.draw.circle(self.screen, WHITE, (indicator_x, indicator_y), 10)
            pygame.draw.circle(self.screen, theme, (indicator_x, indicator_y), 7)
        else:
            pygame.draw.circle(self.screen, GRAY, (indicator_x, indicator_y), 10, 2)

        icon_t = f_lg.render(rm["icon"], True, icon_color)
        self.screen.blit(icon_t, (rect.x + 18, rect.y + 12))

        name_t = f_md.render(rm["name"], True, name_color)
        self.screen.blit(name_t, (rect.x + 55, rect.y + 15))

        desc_t = f_sm.render(rm["desc"], True, desc_color)
        self.screen.blit(desc_t, (rect.x + 20, rect.y + 55))

        # 수식 표시 (교육용)
        detail_t = f_sm.render(rm["detail"], True, detail_color)
        self.screen.blit(detail_t, (rect.x + 20, rect.y + 82))

    def _draw_steps(self, current):
        f_xl, f_lg, f_md, f_sm = self.fonts
        labels = ["1. 파라미터", "2. 보상 함수", "3. 맵 선택", "4. 시뮬레이션"]
        total_w = 650
        start_x = (CONFIG_W - total_w) // 2
        step_w = total_w // len(labels)
        y = CONFIG_H - 40
        for i, lbl in enumerate(labels):
            x = start_x + i * step_w
            color = ACCENT if i + 1 == current else (GRAY if i + 1 > current else DARK_GREEN)
            t = f_sm.render(lbl, True, color)
            self.screen.blit(t, (x + step_w // 2 - t.get_width() // 2, y))
            if i < len(labels) - 1:
                lx = x + step_w
                pygame.draw.line(self.screen, GRAY, (lx - 30, y + 8), (lx + 10, y + 8), 1)


# ====================================================================
# 설정 화면 3: 맵 선택
# ====================================================================
MAP_PATHS = [
    "./assets/track01.png",
    "./assets/track02.png",
    "./assets/track03.png",
]
MAP_NAMES = ["Track 01", "Track 02", "Track 03"]


class ConfigPage3:
    def __init__(self, screen, fonts):
        self.screen = screen
        self.fonts = fonts
        self.selected_idx = 0
        self.done = False
        self.go_back = False

        self.thumbnails = []
        self.available = []
        self.map_sizes = []
        thumb_w, thumb_h = 320, 210
        for p in MAP_PATHS:
            if os.path.exists(p):
                img = pygame.image.load(p).convert()
                self.map_sizes.append((img.get_width(), img.get_height()))
                thumb = pygame.transform.smoothscale(img, (thumb_w, thumb_h))
                self.thumbnails.append(thumb)
                self.available.append(True)
            else:
                placeholder = pygame.Surface((thumb_w, thumb_h))
                placeholder.fill(DARK_GRAY)
                self.thumbnails.append(placeholder)
                self.available.append(False)
                self.map_sizes.append((0, 0))

        cx = CONFIG_W // 2
        gap = 30
        total = thumb_w * 3 + gap * 2
        start_x = cx - total // 2

        self.card_rects = []
        card_h = thumb_h + 75
        for i in range(3):
            x = start_x + i * (thumb_w + gap)
            self.card_rects.append(pygame.Rect(x, 250, thumb_w, card_h))

        self.btn_start = pygame.Rect(cx - 120, CONFIG_H - 100, 240, 55)
        self.btn_back = pygame.Rect(cx - 320, CONFIG_H - 100, 140, 55)

    def handle_click(self, pos):
        for i, r in enumerate(self.card_rects):
            if r.collidepoint(pos) and self.available[i]:
                self.selected_idx = i
                return
        if self.btn_start.collidepoint(pos):
            self.done = True
        elif self.btn_back.collidepoint(pos):
            self.go_back = True

    def get_selected_path(self):
        return MAP_PATHS[self.selected_idx]

    def draw(self):
        self.screen.fill(PANEL_BG)
        f_xl, f_lg, f_md, f_sm = self.fonts
        cx = CONFIG_W // 2

        self._draw_steps(3)

        t = f_xl.render("맵 선택", True, ACCENT)
        self.screen.blit(t, (cx - t.get_width() // 2, 50))

        pygame.draw.line(self.screen, GRAY, (cx - 280, 120), (cx + 280, 120), 2)

        sub = f_md.render("트랙을 선택하세요 (화면이 맵 크기에 맞게 조정됩니다)", True, DARK_GRAY)
        self.screen.blit(sub, (cx - sub.get_width() // 2, 145))

        for i in range(3):
            rect = self.card_rects[i]
            selected = (i == self.selected_idx)
            hover = rect.collidepoint(pygame.mouse.get_pos())
            available = self.available[i]

            if selected:
                bg = LIGHT_BLUE
                border_color = ACCENT
                border_w = 4
            elif hover and available:
                bg = (230, 230, 240)
                border_color = DARK_GRAY
                border_w = 2
            else:
                bg = WHITE
                border_color = GRAY
                border_w = 2

            pygame.draw.rect(self.screen, bg, rect, 0, 12)
            pygame.draw.rect(self.screen, border_color, rect, border_w, 12)

            thumb_x = rect.x + (rect.width - self.thumbnails[i].get_width()) // 2
            thumb_y = rect.y + 8
            self.screen.blit(self.thumbnails[i], (thumb_x, thumb_y))
            thumb_rect = pygame.Rect(thumb_x, thumb_y,
                                     self.thumbnails[i].get_width(),
                                     self.thumbnails[i].get_height())
            pygame.draw.rect(self.screen, border_color, thumb_rect, 1)

            name = MAP_NAMES[i]
            if not available:
                name += " (없음)"
            name_color = ACCENT if selected else (DARK_GRAY if available else GRAY)
            nt = f_md.render(name, True, name_color)
            self.screen.blit(nt, (rect.centerx - nt.get_width() // 2, rect.bottom - 50))

            if available:
                mw, mh = self.map_sizes[i]
                size_t = f_sm.render(f"{mw} x {mh} px", True, DARK_GRAY if not selected else ACCENT)
                self.screen.blit(size_t, (rect.centerx - size_t.get_width() // 2, rect.bottom - 28))

            if selected:
                ind_x = rect.right - 20
                ind_y = rect.top + 20
                pygame.draw.circle(self.screen, ACCENT, (ind_x, ind_y), 12)
                check = f_sm.render("V", True, WHITE)
                self.screen.blit(check, (ind_x - check.get_width() // 2,
                                         ind_y - check.get_height() // 2))

        hover_b = self.btn_back.collidepoint(pygame.mouse.get_pos())
        bg_b = DARK_GRAY if hover_b else GRAY
        pygame.draw.rect(self.screen, bg_b, self.btn_back, 0, 12)
        bt = f_md.render("← 이전", True, WHITE)
        self.screen.blit(bt, (self.btn_back.centerx - bt.get_width() // 2,
                              self.btn_back.centery - bt.get_height() // 2))

        hover_s = self.btn_start.collidepoint(pygame.mouse.get_pos())
        bg_s = ACCENT if hover_s else BLUE
        pygame.draw.rect(self.screen, bg_s, self.btn_start, 0, 12)
        st = f_lg.render("START →", True, WHITE)
        self.screen.blit(st, (self.btn_start.centerx - st.get_width() // 2,
                              self.btn_start.centery - st.get_height() // 2))

    def _draw_steps(self, current):
        f_xl, f_lg, f_md, f_sm = self.fonts
        labels = ["1. 파라미터", "2. 보상 함수", "3. 맵 선택", "4. 시뮬레이션"]
        total_w = 650
        start_x = (CONFIG_W - total_w) // 2
        step_w = total_w // len(labels)
        y = CONFIG_H - 40
        for i, lbl in enumerate(labels):
            x = start_x + i * step_w
            color = ACCENT if i + 1 == current else (GRAY if i + 1 > current else DARK_GREEN)
            t = f_sm.render(lbl, True, color)
            self.screen.blit(t, (x + step_w // 2 - t.get_width() // 2, y))
            if i < len(labels) - 1:
                lx = x + step_w
                pygame.draw.line(self.screen, GRAY, (lx - 30, y + 8), (lx + 10, y + 8), 1)


# ====================================================================
# 시뮬레이션 세션 (대폭 개선)
# ====================================================================
class Session:
    def __init__(self, screen, fonts, num_cars, num_sensors, reward_ids, seed=42):
        self.screen = screen
        self.fonts = fonts
        self.num_cars = num_cars
        self.num_sensors = num_sensors
        self.reward_ids = reward_ids
        self.seed = seed
        self.generation = 1
        self.state = STATE_SETUP
        self.steps = 0
        self.cars = []
        self.start_pos = (0, 0)
        self.best_brain = None
        self.first_finish_gen = None
        self.paused = False
        self.speed_options = [1, 2, 5, 10]
        self.speed_idx = 0
        self.speed_mult = 1
        self.finish_history = []
        self.start_time = None

        # 교육용 추가
        self.gen_stats = GenerationStats()
        self.show_best_trail = True
        self.show_ga_info = True
        self.best_trajectory = []  # 역대 최고 경로

        # GA 연산 기록 (마지막 세대 전환 정보)
        self.last_ga_info = {
            "parent_a_fit": 0,
            "parent_b_fit": 0,
            "n_elite": 0,
            "n_weak_mutant": 0,
            "n_crossover": 0,
            "n_strong_mutant": 0,
        }

        # 컨트롤 버튼 (트랙 영역 상단)
        self.btn_pause = pygame.Rect(TRACK_W - 400, 15, 85, 35)
        self.btn_speed = pygame.Rect(TRACK_W - 305, 15, 85, 35)
        self.btn_reset = pygame.Rect(TRACK_W - 210, 15, 85, 35)
        self.btn_back = pygame.Rect(TRACK_W - 115, 15, 100, 35)

        # 토글 버튼 (우측 패널 하단)
        self.btn_trail = pygame.Rect(TRACK_W + 10, HEIGHT - 75, 145, 30)
        self.btn_ga_toggle = pygame.Rect(TRACK_W + 165, HEIGHT - 75, 145, 30)
        self.btn_save_log = pygame.Rect(TRACK_W + 10, HEIGHT - 38, 300, 30)

        # 완료 화면 버튼
        cx = TRACK_W // 2
        self.btn_restart = pygame.Rect(cx - 130, HEIGHT // 2 + 80, 260, 55)
        self.btn_quit = pygame.Rect(cx - 130, HEIGHT // 2 + 150, 260, 55)

    def _make_cars(self, sx, sy, brains=None):
        cars = []
        for i in range(self.num_cars):
            b = brains[i] if brains and i < len(brains) else None
            cars.append(Car(i + 1, sx, sy, self.num_sensors, b))
        return cars

    def _fitness(self, car):
        return car.compute_fitness(self.reward_ids)

    def next_generation(self):
        # 통계 기록
        entry = self.gen_stats.record(self.generation, self.cars, self.reward_ids)

        # 최고 경로 갱신
        best_car = max(self.cars, key=lambda c: self._fitness(c))
        if not self.best_trajectory or self._fitness(best_car) > (self.gen_stats.history[-2]["best_fit"] if len(self.gen_stats.history) > 1 else 0):
            self.best_trajectory = best_car.trajectory.copy()

        sx, sy = self.start_pos
        sorted_cars = sorted(self.cars, key=lambda c: self._fitness(c), reverse=True)

        p_a = sorted_cars[0].brain
        p_b = sorted_cars[1].brain if len(sorted_cars) > 1 else p_a

        brains = []
        birth_methods = []
        n = self.num_cars

        # 엘리트 (상위 복제)
        n_elite = max(2, int(n * 0.4))
        for i in range(n_elite):
            parent = p_a if i % 2 == 0 else p_b
            brains.append(parent.copy())
            birth_methods.append("elite")

        # 약한 변이
        n_weak = max(1, int(n * 0.2))
        for i in range(n_weak):
            parent = p_a if i % 2 == 0 else p_b
            brains.append(parent.mutate(rate=0.05, strength=0.1))
            birth_methods.append("weak_mutant")

        # 교차 + 변이
        n_cross = max(1, int(n * 0.2))
        for _ in range(n_cross):
            brains.append(p_a.crossover(p_b).mutate(rate=0.15, strength=0.2))
            birth_methods.append("crossover")

        # 강한 변이 (탐험)
        n_strong = 0
        while len(brains) < n:
            parent = p_a if len(brains) % 2 == 0 else p_b
            brains.append(parent.mutate(rate=0.3, strength=0.4))
            birth_methods.append("strong_mutant")
            n_strong += 1

        # GA 정보 기록
        self.last_ga_info = {
            "parent_a_fit": int(self._fitness(sorted_cars[0])),
            "parent_b_fit": int(self._fitness(sorted_cars[1])) if len(sorted_cars) > 1 else 0,
            "n_elite": n_elite,
            "n_weak_mutant": n_weak,
            "n_crossover": n_cross,
            "n_strong_mutant": n_strong,
        }

        self.cars = self._make_cars(sx, sy, brains)
        for i, car in enumerate(self.cars):
            if i < len(birth_methods):
                car.birth_method = birth_methods[i]

        self.generation += 1
        self.steps = 0

    def handle_click(self, pos):
        if self.state == STATE_SETUP:
            # 트랙 영역 안에서만 시작점 설정
            if pos[0] < TRACK_W and pos[1] < TRACK_H:
                self.start_pos = pos
                np.random.seed(self.seed)
                self.cars = self._make_cars(pos[0], pos[1])
                self.state = STATE_SIMULATING
                self.start_time = time.time()
            return

        if self.state == STATE_SIMULATING:
            if self.btn_pause.collidepoint(pos):
                self.paused = not self.paused
            elif self.btn_speed.collidepoint(pos):
                self.speed_idx = (self.speed_idx + 1) % len(self.speed_options)
                self.speed_mult = self.speed_options[self.speed_idx]
            elif self.btn_reset.collidepoint(pos):
                self._reset()
            elif self.btn_back.collidepoint(pos):
                self.state = None
            elif self.btn_trail.collidepoint(pos):
                self.show_best_trail = not self.show_best_trail
            elif self.btn_ga_toggle.collidepoint(pos):
                self.show_ga_info = not self.show_ga_info
            elif self.btn_save_log.collidepoint(pos):
                self._save_log()
            return

        if self.state == STATE_FINISHED:
            if self.btn_restart.collidepoint(pos):
                self._reset()
            elif self.btn_quit.collidepoint(pos):
                self.state = None
            return

    def _save_log(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ga_log_{timestamp}.csv"
        self.gen_stats.save_csv(filename)
        self.log_saved_msg = f"저장됨: {filename}"
        self.log_saved_time = time.time()

    def _reset(self):
        self.generation = 1
        self.steps = 0
        self.cars = []
        self.best_brain = None
        self.first_finish_gen = None
        self.paused = False
        self.finish_history = []
        self.gen_stats = GenerationStats()
        self.best_trajectory = []
        self.state = STATE_SETUP
        self.start_time = None

    def update(self):
        if self.state != STATE_SIMULATING or self.paused:
            return

        for _ in range(self.speed_mult):
            if self.state != STATE_SIMULATING:
                break
            self.steps += 1
            for car in self.cars:
                car.update()
            all_stopped = all(c.is_stopped for c in self.cars)
            finished_count = sum(1 for c in self.cars if c.finished)

            if self.steps >= MAX_STEPS or all_stopped:
                self.finish_history.append(finished_count)
                if finished_count > 0 and self.first_finish_gen is None:
                    self.first_finish_gen = self.generation
                if finished_count >= self.num_cars:
                    self.gen_stats.record(self.generation, self.cars, self.reward_ids)
                    self.best_brain = max(self.cars, key=lambda c: self._fitness(c)).brain
                    self.state = STATE_FINISHED
                    return
                self.next_generation()

    def draw(self):
        f_xl, f_lg, f_md, f_sm = self.fonts

        # 트랙 영역
        self.screen.blit(track_img, (0, 0))

        # 트랙 아래 여백 채우기
        if TRACK_H < HEIGHT:
            pygame.draw.rect(self.screen, PANEL_BG, (0, TRACK_H, TRACK_W, HEIGHT - TRACK_H))

        if self.state == STATE_FINISHED:
            self._draw_finish()
            self._draw_info_panel()
            return

        if self.state == STATE_SETUP:
            # 트랙 위 안내
            msg = f_lg.render("트랙 위 시작점을 클릭하세요", True, RED)
            self.screen.blit(msg, (TRACK_W // 2 - msg.get_width() // 2, HEIGHT // 2))
            hint = f_sm.render("(검은 벽이 아닌 흰색 도로 위를 클릭)", True, DARK_GRAY)
            self.screen.blit(hint, (TRACK_W // 2 - hint.get_width() // 2, HEIGHT // 2 + 35))
            self._draw_info_panel()
            return

        # 최적 경로 표시
        if self.show_best_trail and len(self.best_trajectory) > 1:
            for i in range(len(self.best_trajectory) - 1):
                p1 = self.best_trajectory[i]
                p2 = self.best_trajectory[i + 1]
                pygame.draw.line(self.screen, (100, 255, 100, 80),
                                 (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])), 2)

        # 자동차 그리기
        for car in self.cars:
            car.draw(self.screen, f_sm)

        # HUD
        hud = pygame.Surface((TRACK_W, 65))
        hud.set_alpha(210)
        hud.fill(WHITE)
        self.screen.blit(hud, (0, 0))

        finished_count = sum(1 for c in self.cars if c.finished)
        alive = sum(1 for c in self.cars if not c.is_stopped)
        stuck = sum(1 for c in self.cars if c.stagnated)
        crashed = sum(1 for c in self.cars if c.crashed)
        best_fit = max(self._fitness(c) for c in self.cars) if self.cars else 0

        line1 = f_md.render(
            f"Gen {self.generation}  |  Step {self.steps}/{MAX_STEPS}  |  "
            f"Cars {self.num_cars}  |  Sensors {self.num_sensors}  |  Seed {self.seed}",
            True, BLACK,
        )
        self.screen.blit(line1, (15, 5))

        line2 = f_sm.render(
            f"Best: {int(best_fit)}  Alive: {alive}  "
            f"Finish: {finished_count}/{self.num_cars}  Crash: {crashed}  Stuck: {stuck}",
            True, BLUE,
        )
        self.screen.blit(line2, (15, 32))

        # 완주율 바
        if finished_count > 0:
            bar_x, bar_w, bar_h = 15, 250, 10
            bar_y = 52
            ratio = finished_count / self.num_cars
            pygame.draw.rect(self.screen, GRAY, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, GREEN, (bar_x, bar_y, int(bar_w * ratio), bar_h))
            pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_w, bar_h), 1)

        # 컨트롤 버튼
        pause_label = "일시정지" if not self.paused else "재개"
        self._draw_ctrl(self.btn_pause, pause_label, ORANGE if self.paused else BLUE, f_sm)
        self._draw_ctrl(self.btn_speed, f"x{self.speed_mult} 배속", DARK_GREEN, f_sm)
        self._draw_ctrl(self.btn_reset, "리셋", RED, f_sm)
        self._draw_ctrl(self.btn_back, "설정으로", DARK_GRAY, f_sm)

        # 우측 정보 패널
        self._draw_info_panel()

        # 일시정지 오버레이
        if self.paused:
            overlay = pygame.Surface((TRACK_W, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 80))
            self.screen.blit(overlay, (0, 0))
            pt = f_xl.render("PAUSED", True, WHITE)
            self.screen.blit(pt, (TRACK_W // 2 - pt.get_width() // 2,
                                  HEIGHT // 2 - pt.get_height() // 2))
            hint = f_md.render("SPACE로 재개", True, (200, 200, 200))
            self.screen.blit(hint, (TRACK_W // 2 - hint.get_width() // 2,
                                    HEIGHT // 2 + 45))

    def _draw_info_panel(self):
        """우측 정보 패널: 그래프 + GA 설명 + 통계"""
        f_xl, f_lg, f_md, f_sm = self.fonts
        px = TRACK_W  # 패널 시작 x
        pw = INFO_PANEL_W
        ph = HEIGHT

        # 배경
        pygame.draw.rect(self.screen, PANEL_BG, (px, 0, pw, ph))
        pygame.draw.line(self.screen, DARK_GRAY, (px, 0), (px, ph), 2)

        y = 10
        # 타이틀
        title = f_md.render("📊 학습 대시보드", True, ACCENT)
        self.screen.blit(title, (px + 10, y))
        y += 30

        if self.state == STATE_SETUP:
            # 설정 안내
            self._draw_panel_text(px, y, [
                ("GA(유전 알고리즘) 학습 과정", ACCENT, f_md),
                ("", BLACK, f_sm),
                ("1. 랜덤 신경망으로 시작", BLACK, f_sm),
                ("2. 자동차가 주행 시도", BLACK, f_sm),
                ("3. 적합도(보상) 계산", BLACK, f_sm),
                ("4. 상위 개체 선택 (선택)", BLUE, f_sm),
                ("5. 가중치 섞기 (교차)", GREEN, f_sm),
                ("6. 가중치 흔들기 (변이)", RED, f_sm),
                ("7. 다음 세대 생성", BLACK, f_sm),
                ("8. 2~7 반복 → 진화!", ACCENT, f_sm),
                ("", BLACK, f_sm),
                ("트랙 위 흰색 도로를 클릭하여", DARK_GRAY, f_sm),
                ("시작점을 지정하세요.", DARK_GRAY, f_sm),
            ])
            return

        # === 적합도 그래프 ===
        graph_x = px + 15
        graph_y = y
        graph_w = pw - 30
        graph_h = 140

        self._draw_fitness_graph(graph_x, graph_y, graph_w, graph_h)
        y += graph_h + 15

        # === 완주율 그래프 ===
        self._draw_finish_graph(graph_x, y, graph_w, 80)
        y += 95

        # === GA 연산 정보 ===
        if self.show_ga_info and self.generation > 1:
            pygame.draw.line(self.screen, GRAY, (px + 10, y), (px + pw - 10, y), 1)
            y += 8

            ga_title = f_sm.render("🧬 세대 구성 (GA 연산)", True, ACCENT)
            self.screen.blit(ga_title, (px + 10, y))
            y += 22

            info = self.last_ga_info
            ga_lines = [
                (f"부모 A 적합도: {info['parent_a_fit']}", BLUE),
                (f"부모 B 적합도: {info['parent_b_fit']}", BLUE),
                (f"엘리트 복제: {info['n_elite']}대 (상위 그대로)", DARK_GREEN),
                (f"약한 변이:   {info['n_weak_mutant']}대 (미세 조정)", ORANGE),
                (f"교차+변이:   {info['n_crossover']}대 (유전자 섞기)", PURPLE),
                (f"강한 변이:   {info['n_strong_mutant']}대 (새로운 탐험)", RED),
            ]
            for text, color in ga_lines:
                t = f_sm.render(text, True, color)
                self.screen.blit(t, (px + 15, y))
                y += 18
            y += 5

        # === 현재 세대 통계 ===
        pygame.draw.line(self.screen, GRAY, (px + 10, y), (px + pw - 10, y), 1)
        y += 8

        stat_title = f_sm.render("📈 현재 통계", True, ACCENT)
        self.screen.blit(stat_title, (px + 10, y))
        y += 22

        if self.gen_stats.history:
            latest = self.gen_stats.history[-1]
            stat_lines = [
                (f"최고 적합도: {int(latest['best_fit'])}", DARK_GREEN),
                (f"평균 적합도: {int(latest['avg_fit'])}", BLUE),
                (f"최고 도달거리: {int(latest['best_dist'])}px", BLACK),
                (f"인구 다양성: {latest['diversity']:.3f}", PURPLE),
            ]
            for text, color in stat_lines:
                t = f_sm.render(text, True, color)
                self.screen.blit(t, (px + 15, y))
                y += 18
            y += 5

        # 경과 시간
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            time_t = f_sm.render(f"경과 시간: {mins}분 {secs}초", True, DARK_GRAY)
            self.screen.blit(time_t, (px + 15, y))
            y += 22

        # 선택된 보상 함수 표시
        pygame.draw.line(self.screen, GRAY, (px + 10, y), (px + pw - 10, y), 1)
        y += 8
        rf_title = f_sm.render("🎯 보상 함수", True, ACCENT)
        self.screen.blit(rf_title, (px + 10, y))
        y += 20
        for rm in REWARD_METHODS:
            if rm["id"] in self.reward_ids:
                t = f_sm.render(f"  {rm['icon']} {rm['name']}", True, rm["color"])
                self.screen.blit(t, (px + 10, y))
                y += 18

        # === 하단 토글/저장 버튼 ===
        trail_color = DARK_GREEN if self.show_best_trail else GRAY
        self._draw_ctrl(self.btn_trail, "경로 표시" if self.show_best_trail else "경로 숨김",
                        trail_color, f_sm)

        ga_color = ACCENT if self.show_ga_info else GRAY
        self._draw_ctrl(self.btn_ga_toggle, "GA정보 표시" if self.show_ga_info else "GA정보 숨김",
                        ga_color, f_sm)

        self._draw_ctrl(self.btn_save_log, "📁 학습 로그 CSV 저장", DARK_GRAY, f_sm)

        # 저장 완료 메시지
        if hasattr(self, 'log_saved_time') and time.time() - self.log_saved_time < 3:
            msg = f_sm.render(self.log_saved_msg, True, DARK_GREEN)
            self.screen.blit(msg, (px + 10, HEIGHT - 95))

    def _draw_panel_text(self, px, y, lines):
        """패널에 여러 줄 텍스트 표시"""
        for text, color, font in lines:
            if text:
                t = font.render(text, True, color)
                self.screen.blit(t, (px + 15, y))
            y += 22

    def _draw_fitness_graph(self, x, y, w, h):
        """실시간 적합도 그래프"""
        f_xl, f_lg, f_md, f_sm = self.fonts

        # 배경
        pygame.draw.rect(self.screen, WHITE, (x, y, w, h), 0, 4)
        pygame.draw.rect(self.screen, GRAY, (x, y, w, h), 1, 4)

        title = f_sm.render("세대별 적합도", True, BLACK)
        self.screen.blit(title, (x + 5, y + 3))

        hist = self.gen_stats.history
        if len(hist) < 2:
            no_data = f_sm.render("데이터 수집 중...", True, DARK_GRAY)
            self.screen.blit(no_data, (x + w // 2 - no_data.get_width() // 2, y + h // 2))
            return

        pad_top = 22
        pad_bottom = 18
        pad_left = 5
        pad_right = 5
        gx = x + pad_left
        gy = y + pad_top
        gw = w - pad_left - pad_right
        gh = h - pad_top - pad_bottom

        best_vals = [e["best_fit"] for e in hist]
        avg_vals = [e["avg_fit"] for e in hist]
        all_vals = best_vals + avg_vals
        max_val = max(all_vals) if all_vals else 1
        min_val = min(min(all_vals), 0)
        val_range = max(max_val - min_val, 1)

        n = len(hist)
        dx = gw / max(n - 1, 1)

        def to_screen(i, v):
            sx = gx + i * dx
            sy = gy + gh - (v - min_val) / val_range * gh
            return (int(sx), int(sy))

        # 그리드
        for i in range(5):
            gy_line = gy + gh * i // 4
            pygame.draw.line(self.screen, LIGHT_GRAY, (gx, gy_line), (gx + gw, gy_line), 1)

        # 평균 적합도 선
        if n >= 2:
            avg_pts = [to_screen(i, v) for i, v in enumerate(avg_vals)]
            pygame.draw.lines(self.screen, CYAN, False, avg_pts, 1)

        # 최고 적합도 선
        if n >= 2:
            best_pts = [to_screen(i, v) for i, v in enumerate(best_vals)]
            pygame.draw.lines(self.screen, BLUE, False, best_pts, 2)

        # 범례
        pygame.draw.line(self.screen, BLUE, (x + w - 120, y + 6), (x + w - 100, y + 6), 2)
        lt1 = f_sm.render("최고", True, BLUE)
        self.screen.blit(lt1, (x + w - 95, y + 1))

        pygame.draw.line(self.screen, CYAN, (x + w - 60, y + 6), (x + w - 40, y + 6), 1)
        lt2 = f_sm.render("평균", True, CYAN)
        self.screen.blit(lt2, (x + w - 35, y + 1))

        # 최고값 표시
        val_t = f_sm.render(f"{int(max_val)}", True, DARK_GRAY)
        self.screen.blit(val_t, (x + 5, y + h - 15))

    def _draw_finish_graph(self, x, y, w, h):
        """세대별 완주 수 그래프"""
        f_xl, f_lg, f_md, f_sm = self.fonts

        pygame.draw.rect(self.screen, WHITE, (x, y, w, h), 0, 4)
        pygame.draw.rect(self.screen, GRAY, (x, y, w, h), 1, 4)

        title = f_sm.render("세대별 완주 수", True, BLACK)
        self.screen.blit(title, (x + 5, y + 3))

        hist = self.gen_stats.history
        if not hist:
            return

        pad_top = 20
        pad_bottom = 5
        gx = x + 5
        gy = y + pad_top
        gw = w - 10
        gh = h - pad_top - pad_bottom

        n = len(hist)
        max_display = 50
        display_hist = hist[-max_display:]
        dn = len(display_hist)
        bar_w = max(2, gw // max(dn, 1) - 1)

        for i, entry in enumerate(display_hist):
            ratio = entry["finished"] / entry["total"]
            bh = int(gh * ratio)
            bx = gx + i * (bar_w + 1)
            by = gy + gh - bh

            color = GREEN if ratio >= 1.0 else (BLUE if ratio > 0 else GRAY)
            pygame.draw.rect(self.screen, color, (bx, by, bar_w, bh))

    def _draw_ctrl(self, rect, label, color, font_obj):
        hover = rect.collidepoint(pygame.mouse.get_pos())
        bg = color if hover else LIGHT_GRAY
        tc = WHITE if hover else BLACK
        pygame.draw.rect(self.screen, bg, rect, 0, 8)
        pygame.draw.rect(self.screen, BLACK, rect, 1, 8)
        t = font_obj.render(label, True, tc)
        self.screen.blit(t, (rect.centerx - t.get_width() // 2,
                             rect.centery - t.get_height() // 2))

    def _draw_finish(self):
        f_xl, f_lg, f_md, f_sm = self.fonts

        overlay = pygame.Surface((TRACK_W, HEIGHT))
        overlay.set_alpha(220)
        overlay.fill(WHITE)
        self.screen.blit(overlay, (0, 0))

        t = f_xl.render("ALL CARS FINISHED!", True, GOLD)
        self.screen.blit(t, (TRACK_W // 2 - t.get_width() // 2, HEIGHT // 2 - 180))

        lines = [
            f"총 {self.generation} 세대 만에 전원 완주!",
            f"자동차 {self.num_cars}대  |  센서 {self.num_sensors}개  |  시드 {self.seed}",
        ]
        if self.first_finish_gen:
            lines.append(f"첫 완주 세대: {self.first_finish_gen}")
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            lines.append(f"소요 시간: {mins}분 {secs}초")

        rnames = [rm["name"] for rm in REWARD_METHODS if rm["id"] in self.reward_ids]
        lines.append(f"보상 함수: {', '.join(rnames)}")

        if self.gen_stats.history:
            last = self.gen_stats.history[-1]
            lines.append(f"최종 최고 적합도: {int(last['best_fit'])}")

        for i, line in enumerate(lines):
            lt = f_md.render(line, True, BLACK)
            self.screen.blit(lt, (TRACK_W // 2 - lt.get_width() // 2, HEIGHT // 2 - 100 + i * 30))

        # CSV 자동 저장 안내
        save_hint = f_sm.render("💡 학습 로그는 우측 패널에서 CSV로 저장할 수 있습니다", True, DARK_GRAY)
        self.screen.blit(save_hint, (TRACK_W // 2 - save_hint.get_width() // 2, HEIGHT // 2 + 55))

        for rect, label, color in [
            (self.btn_restart, "다시 시작", BLUE),
            (self.btn_quit, "설정으로 돌아가기", DARK_GRAY),
        ]:
            hover = rect.collidepoint(pygame.mouse.get_pos())
            bg = color if hover else LIGHT_GRAY
            tc = WHITE if hover else BLACK
            pygame.draw.rect(self.screen, bg, rect, 0, 12)
            pygame.draw.rect(self.screen, BLACK, rect, 2, 12)
            bt = f_lg.render(label, True, tc)
            self.screen.blit(bt, (rect.centerx - bt.get_width() // 2,
                                  rect.centery - bt.get_height() // 2))


# ====================================================================
# 메인 루프
# ====================================================================
def main():
    global WIDTH, HEIGHT
    pygame.init()
    screen = pygame.display.set_mode((CONFIG_W, CONFIG_H))
    pygame.display.set_caption("자율주행 AI 시뮬레이터 v4 — 교육용")
    clock = pygame.time.Clock()

    fonts = (
        make_font(38, bold=True),
        make_font(26, bold=True),
        make_font(18),
        make_font(14),
    )

    while True:
        WIDTH, HEIGHT = CONFIG_W, CONFIG_H
        screen = pygame.display.set_mode((CONFIG_W, CONFIG_H))

        current_page = 1
        page1 = ConfigPage1(screen, fonts)
        page2 = ConfigPage2(screen, fonts)
        page3 = ConfigPage3(screen, fonts)

        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    return
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    if current_page == 1:
                        page1.handle_click(ev.pos)
                    elif current_page == 2:
                        page2.handle_click(ev.pos)
                    elif current_page == 3:
                        page3.handle_click(ev.pos)

            if current_page == 1:
                if page1.done:
                    current_page = 2
                page1.draw()
            elif current_page == 2:
                if page2.go_back:
                    page2.go_back = False
                    page2.done = False
                    current_page = 1
                    page1.done = False
                elif page2.done:
                    current_page = 3
                page2.draw()
            elif current_page == 3:
                if page3.go_back:
                    page3.go_back = False
                    page3.done = False
                    current_page = 2
                    page2.done = False
                elif page3.done:
                    break
                page3.draw()

            pygame.display.flip()
            clock.tick(60)

        # 맵 로드
        screen = load_track(page3.get_selected_path())

        # 시뮬레이션
        session = Session(screen, fonts, page1.num_cars,
                          page1.num_sensors, page2.selected, page1.seed)
        running_session = True
        while running_session:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    return
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    session.handle_click(ev.pos)
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_SPACE:
                        session.paused = not session.paused
                    elif ev.key == pygame.K_r:
                        session._reset()
                    elif ev.key == pygame.K_t:
                        session.show_best_trail = not session.show_best_trail
                    elif ev.key == pygame.K_g:
                        session.show_ga_info = not session.show_ga_info
                    elif ev.key == pygame.K_s:
                        session._save_log()

            if session.state is None:
                running_session = False
                continue

            session.update()
            session.draw()
            pygame.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    main()
