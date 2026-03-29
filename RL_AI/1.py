import pygame
import numpy as np
import math
import os
import argparse
import platform

# ====================================================================
# 0. 초기 설정 및 하이퍼파라미터
# ====================================================================
WIDTH = 1700
HEIGHT = 800
MAX_STEPS = 1200

STATE_SETUP = -1
STATE_SIMULATING = 0
STATE_SELECTING = 1
STATE_FINISHED = 2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 100, 255)
RED = (255, 50, 50)
GREEN = (0, 200, 0)
GOLD = (255, 215, 0)
DARK_GREEN = (0, 150, 0)
ORANGE = (255, 165, 0)
UI_BACK_COLOR = (230, 230, 230)

NUM_CARS = 10
NUM_SENSORS = 5
SENSOR_LEN = 150
MAX_SPEED = 3.0
MUTATION_RATE = 0.15

# 정체 감지 파라미터
STAGNATION_CHECK_INTERVAL = 60   # 60스텝마다 체크
STAGNATION_MIN_DISPLACEMENT = 30  # 60스텝 동안 최소 30px은 이동해야 함


# ====================================================================
# 한글 폰트 자동 탐색
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
            "/Library/Fonts/NanumGothic.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def make_font(size, bold=False):
    font_path = find_korean_font()
    if font_path:
        try:
            f = pygame.font.Font(font_path, size)
            f.set_bold(bold)
            return f
        except Exception:
            pass
    for name in ["malgungothic", "malgun gothic", "applegothic", "nanumgothic", "gulim"]:
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            test = f.render("가", True, (0, 0, 0))
            if test.get_width() > 2:
                return f
        except Exception:
            continue
    return pygame.font.SysFont("Arial", size, bold=bold)


# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("자율주행 AI 시뮬레이터")
clock = pygame.time.Clock()
font = make_font(20)
large_font = make_font(30, bold=True)
title_font = make_font(50, bold=True)

# 맵 이미지 로드
TRACK_PATH = "./assets/track01.png"
if not os.path.exists(TRACK_PATH):
    TRACK_PATH = "track.png"
if not os.path.exists(TRACK_PATH):
    temp_surface = pygame.Surface((WIDTH, HEIGHT))
    temp_surface.fill(WHITE)
    pygame.draw.rect(temp_surface, BLACK, (0, 0, WIDTH, HEIGHT), 20)
    pygame.image.save(temp_surface, TRACK_PATH)

track_img = pygame.image.load(TRACK_PATH).convert()


# ====================================================================
# 1. 트랙 감지 유틸리티 함수
# ====================================================================
def is_wall(x, y):
    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
        return True
    color = track_img.get_at((int(x), int(y)))
    return color[0] < 50 and color[1] < 50 and color[2] < 50


def cast_ray_image(start_x, start_y, angle):
    dx, dy = math.cos(angle), math.sin(angle)
    x, y = start_x, start_y
    for dist in range(SENSOR_LEN):
        if is_wall(x, y):
            return dist
        x += dx
        y += dy
    return SENSOR_LEN


def check_car_collision_image(x, y, angle, vertices):
    points = []
    for v in vertices:
        nx = v[0] * math.cos(angle) - v[1] * math.sin(angle) + x
        ny = v[0] * math.sin(angle) + v[1] * math.cos(angle) + y
        points.append((nx, ny))
    edges = [(points[0], points[1]), (points[1], points[2]), (points[2], points[0])]
    for p1, p2 in edges:
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(1, int(dist))
        for i in range(steps + 1):
            t = i / steps
            if is_wall(p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t):
                return True
    return False


def check_car_finish_image(x, y, angle, vertices):
    points = []
    for v in vertices:
        nx = v[0] * math.cos(angle) - v[1] * math.sin(angle) + x
        ny = v[0] * math.sin(angle) + v[1] * math.cos(angle) + y
        points.append((nx, ny))
    edges = [(points[0], points[1]), (points[1], points[2]), (points[2], points[0])]
    for p1, p2 in edges:
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(1, int(dist))
        for i in range(steps + 1):
            t = i / steps
            cx = p1[0] + (p2[0] - p1[0]) * t
            cy = p1[1] + (p2[1] - p1[1]) * t
            if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                color = track_img.get_at((int(cx), int(cy)))
                if color[0] > 200 and color[1] < 100 and color[2] < 100:
                    return True
    return False


# ====================================================================
# 2. 신경망 및 차량 클래스
# ====================================================================
class CarBrain:
    def __init__(self, w1=None, b1=None, w2=None, b2=None):
        self.hidden_size = 8
        if w1 is None:
            self.w1 = np.random.randn(NUM_SENSORS, self.hidden_size) * np.sqrt(2.0 / NUM_SENSORS)
            self.b1 = np.zeros(self.hidden_size)
            self.w2 = np.random.randn(self.hidden_size, 2) * np.sqrt(2.0 / self.hidden_size)
            self.b2 = np.zeros(2)
        else:
            self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2

    def copy(self):
        return CarBrain(
            self.w1.copy(), self.b1.copy(),
            self.w2.copy(), self.b2.copy()
        )

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, sensor_data):
        hidden = self.relu(np.dot(sensor_data, self.w1) + self.b1)
        raw_output = np.dot(hidden, self.w2) + self.b2
        exp_out = np.exp(raw_output - np.max(raw_output))
        return exp_out / exp_out.sum()

    def mutate(self, rate=None, strength=0.2):
        if rate is None:
            rate = MUTATION_RATE
        def apply_mutation(matrix):
            mask = np.random.rand(*matrix.shape) < rate
            return matrix + (np.random.randn(*matrix.shape) * strength * mask)
        return CarBrain(
            apply_mutation(self.w1), apply_mutation(self.b1),
            apply_mutation(self.w2), apply_mutation(self.b2)
        )

    def crossover(self, other):
        def cross(a, b):
            mask = np.random.rand(*a.shape) < 0.5
            return np.where(mask, a, b)
        return CarBrain(
            cross(self.w1, other.w1), cross(self.b1, other.b1),
            cross(self.w2, other.w2), cross(self.b2, other.b2)
        )

    def repulse(self, worst_brain, alpha=0.1):
        """
        하위 개체(worst_brain)의 가중치와 반대 방향으로 미세하게 이동합니다.
        alpha: 회피 강도 (너무 크면 가중치가 발산할 수 있으니 주의)
        """
        def avoid(w_parent, w_worst):
            # (부모 - 실패) 방향으로 벡터 이동: 실패한 녀석과 반대로 행동하게 함
            return w_parent + alpha * (w_parent - w_worst)

        return CarBrain(
            avoid(self.w1, worst_brain.w1), avoid(self.b1, worst_brain.b1),
            avoid(self.w2, worst_brain.w2), avoid(self.b2, worst_brain.b2)
        )

class Car:
    def __init__(self, car_id, start_x, start_y, brain=None):
        self.id = car_id
        self.x, self.y = start_x, start_y
        self.start_x, self.start_y = start_x, start_y
        self.angle = -math.pi / 2
        self.speed = MAX_SPEED
        self.brain = brain if brain else CarBrain()
        self.sensor_data = np.zeros(NUM_SENSORS)
        self.vertices = [(25, 0), (-15, 15), (-15, -15)]
        self.color = BLUE
        self.crashed = False
        self.finished = False
        self.stagnated = False  # 정체 판정

        # === 센서 시각화용 ===
        self.sensor_angles = []   # 센서 방향각
        self.sensor_dists = []    # 센서 감지 거리 (px)

        # === FITNESS 관련 ===
        # 시작점으로부터 도달한 최대 직선거리 (제자리 회전 방지)
        self.max_distance_from_start = 0.0
        # 누적 이동 거리 (보조 지표)
        self.cumulative_distance = 0.0
        self.prev_x, self.prev_y = start_x, start_y

        # === 정체 감지용 ===
        self.step_count = 0
        self.checkpoint_x = start_x  # 마지막 체크 시점의 위치
        self.checkpoint_y = start_y

    @property
    def fitness(self):
        """
        주 지표: 시작점에서 가장 멀리 도달한 거리 (max_distance_from_start)
        → 제자리 회전은 이 값이 안 올라감
        보조 지표: 누적 이동 거리의 10% (같은 거리 도달 시 더 효율적인 경로 우대)
        완주 보너스: +5000
        """
        bonus = 5000 if self.finished else 0
        # 정체로 죽은 차는 페널티
        stag_penalty = 200 if self.stagnated else 0
        return self.max_distance_from_start + self.cumulative_distance * 0.1 + bonus - stag_penalty

    @property
    def is_stopped(self):
        return self.crashed or self.finished or self.stagnated

    def update(self):
        if self.is_stopped:
            return

        self.step_count += 1

        # === 정체 감지 ===
        # 매 STAGNATION_CHECK_INTERVAL 스텝마다 체크포인트 대비 이동량 확인
        if self.step_count % STAGNATION_CHECK_INTERVAL == 0:
            displacement = math.hypot(
                self.x - self.checkpoint_x,
                self.y - self.checkpoint_y
            )
            if displacement < STAGNATION_MIN_DISPLACEMENT:
                # 60스텝 동안 30px도 못 벗어남 → 정체(빙글빙글 or 벽 앞 진동)
                self.stagnated = True
                self.color = ORANGE
                return
            # 체크포인트 갱신
            self.checkpoint_x = self.x
            self.checkpoint_y = self.y

        # 센서 데이터
        self.sensor_angles = [
            self.angle,
            self.angle - math.pi / 4,
            self.angle + math.pi / 4,
            self.angle - math.pi / 2,
            self.angle + math.pi / 2,
        ]
        for i, a in enumerate(self.sensor_angles[:NUM_SENSORS]):
            raw_dist = cast_ray_image(self.x, self.y, a)
            self.sensor_data[i] = raw_dist / SENSOR_LEN
        self.sensor_dists = [self.sensor_data[i] * SENSOR_LEN for i in range(NUM_SENSORS)]

        nx = self.x + self.speed * math.cos(self.angle)
        ny = self.y + self.speed * math.sin(self.angle)

        if check_car_finish_image(nx, ny, self.angle, self.vertices):
            self.finished = True
            self.color = GREEN
            return
        if check_car_collision_image(nx, ny, self.angle, self.vertices):
            self.crashed = True
            self.color = RED
            return

        prob = self.brain.forward(self.sensor_data)
        steer = prob[1] - prob[0]
        self.angle += steer * 0.15
        cur_speed = self.speed * (1.0 - abs(steer) * 0.6)
        self.x += cur_speed * math.cos(self.angle)
        self.y += cur_speed * math.sin(self.angle)

        # 누적 이동 거리
        step_dist = math.hypot(self.x - self.prev_x, self.y - self.prev_y)
        self.cumulative_distance += step_dist
        self.prev_x, self.prev_y = self.x, self.y

        # 시작점으로부터 최대 도달 거리 갱신
        dist_from_start = math.hypot(self.x - self.start_x, self.y - self.start_y)
        if dist_from_start > self.max_distance_from_start:
            self.max_distance_from_start = dist_from_start

    def draw(self, surface):
        # 센서 선 그리기 (살아있는 차만)
        if not self.is_stopped and self.sensor_angles:
            SENSOR_GREEN = (0, 220, 80)
            SENSOR_RED_TIP = (255, 60, 60)
            for i in range(min(NUM_SENSORS, len(self.sensor_angles))):
                a = self.sensor_angles[i]
                dist = self.sensor_dists[i] if i < len(self.sensor_dists) else SENSOR_LEN
                end_x = self.x + dist * math.cos(a)
                end_y = self.y + dist * math.sin(a)
                # 거리에 따라 색상 변화: 가까우면 빨강, 멀면 초록
                t = dist / SENSOR_LEN  # 0(벽 근접) ~ 1(멀리)
                line_r = int(255 * (1 - t) + 0 * t)
                line_g = int(60 * (1 - t) + 220 * t)
                line_b = int(60 * (1 - t) + 80 * t)
                line_color = (line_r, line_g, line_b)
                pygame.draw.line(surface, line_color, (self.x, self.y), (end_x, end_y), 1)
                # 끝점에 작은 점
                pygame.draw.circle(surface, SENSOR_RED_TIP, (int(end_x), int(end_y)), 3)

        # 차체 그리기
        pts = [
            (
                v[0] * math.cos(self.angle) - v[1] * math.sin(self.angle) + self.x,
                v[0] * math.sin(self.angle) + v[1] * math.cos(self.angle) + self.y,
            )
            for v in self.vertices
        ]
        pygame.draw.polygon(surface, self.color, pts)
        pygame.draw.polygon(surface, BLACK, pts, 2)
        label_color = WHITE if self.color == BLUE or self.color == GREEN else BLACK
        txt = font.render(str(self.id), True, label_color)
        surface.blit(txt, (self.x - txt.get_width() // 2, self.y - txt.get_height() // 2))


# ====================================================================
# 3. 인터랙티브 UI 루프
# ====================================================================
class UISession:
    def __init__(self, mode="manual"):
        self.generation = 1
        self.state = STATE_SETUP
        self.mode = mode
        self.steps = 0
        self.cars = []
        self.start_pos = (0, 0)
        self.best_brain = None
        self.first_finish_gen = None
        self.finish_history = []
        # 동적 과반수: ceil(NUM_CARS / 2), Continue 누르면 NUM_CARS로 변경
        self.finish_target = math.ceil(NUM_CARS / 2)

        btn_w = 150
        self.buttons = [
            pygame.Rect(
                (WIDTH - (NUM_CARS * btn_w + (NUM_CARS - 1) * 20)) / 2 + i * (btn_w + 20),
                HEIGHT - 120, btn_w, 60,
            )
            for i in range(NUM_CARS)
        ]

        btn_width, btn_height = 250, 60
        center_x = WIDTH // 2
        self.btn_continue = pygame.Rect(
            center_x - btn_width // 2, HEIGHT // 2 + 10, btn_width, btn_height
        )
        self.btn_restart_manual = pygame.Rect(
            center_x - btn_width - 20, HEIGHT // 2 + 90, btn_width, btn_height
        )
        self.btn_restart_auto = pygame.Rect(
            center_x + 20, HEIGHT // 2 + 90, btn_width, btn_height
        )
        self.btn_quit = pygame.Rect(
            center_x - btn_width // 2, HEIGHT // 2 + 170, btn_width, btn_height
        )

    def next_generation(self, selected_brain):
        sx, sy = self.start_pos
        # 1. 성적순 정렬 (Fitness 기준)
        sorted_cars = sorted(self.cars, key=lambda c: c.fitness, reverse=True)
        
        # 2. 상위 20% (Elite)와 하위 20% (Worst) 추출
        num_elite = max(1, int(NUM_CARS * 0.2))
        num_worst = max(1, int(NUM_CARS * 0.2))
        
        elites = [c.brain for c in sorted_cars[:num_elite]]
        worsts = [c.brain for c in sorted_cars[-num_worst:]]
        
        # 하위 개체들의 '평균 실패 가중치' 계산 (간단하게 하위 1등만 써도 됨)
        avg_worst_brain = worsts[0] # 가장 못난 녀석

        next_gen = []
        
        # [그룹 A] 인간이 선택한 뇌 (보존 2대)
        next_gen.append(Car(1, sx, sy, selected_brain.copy()))
        next_gen.append(Car(2, sx, sy, selected_brain.copy()))
        
        # [그룹 B] 하위 20%를 회피하도록 개조된 유전자 (2대) - 핵심!
        # "실패한 놈이랑 반대로 해라"라는 명령을 가중치에 주입
        repulsed_brain = selected_brain.repulse(avg_worst_brain, alpha=0.15)
        next_gen.append(Car(3, sx, sy, repulsed_brain.mutate(rate=0.05)))
        next_gen.append(Car(4, sx, sy, repulsed_brain.mutate(rate=0.05)))

        # [그룹 C] 상위 엘리트와 하위 회피 뇌의 교차 (2대)
        crossed_brain = elites[0].crossover(repulsed_brain)
        next_gen.append(Car(5, sx, sy, crossed_brain.mutate()))
        next_gen.append(Car(6, sx, sy, crossed_brain.mutate()))

        # [그룹 D] 돌연변이 및 탐험 (4대)
        for i in range(7, NUM_CARS + 1):
            parent = selected_brain if i % 2 == 0 else elites[0]
            next_gen.append(Car(i, sx, sy, parent.mutate(rate=0.25, strength=0.3)))

        self.cars = next_gen
        self.generation += 1
        self.steps = 0
        self.state = STATE_SIMULATING

    def update(self):
        if self.state == STATE_SIMULATING:
            self.steps += 1
            all_stopped = all(c.is_stopped for c in self.cars)
            for car in self.cars:
                car.update()

            finished_count = sum(1 for c in self.cars if c.finished)

            if self.steps >= MAX_STEPS or all_stopped:
                self.finish_history.append(finished_count)

                if finished_count > 0 and self.first_finish_gen is None:
                    self.first_finish_gen = self.generation

                if finished_count >= self.finish_target:
                    self.best_brain = max(self.cars, key=lambda c: c.fitness).brain
                    self.state = STATE_FINISHED
                    return

                if self.mode == "auto":
                    best_car = max(self.cars, key=lambda c: c.fitness)
                    self.next_generation(best_car.brain)
                else:
                    self.state = STATE_SELECTING

    def draw(self):
        screen.blit(track_img, (0, 0))

        if self.state == STATE_FINISHED:
            self._draw_finish_screen()
            return

        for car in self.cars:
            car.draw(screen)

        # 상단 HUD
        overlay = pygame.Surface((WIDTH, 100))
        overlay.set_alpha(200)
        overlay.fill(WHITE)
        screen.blit(overlay, (0, 0))

        mode_str = "AUTO" if self.mode == "auto" else "MANUAL"
        info_txt = font.render(
            f"Generation {self.generation}  |  Mode: {mode_str}  |  Step: {self.steps}/{MAX_STEPS}",
            True, BLACK,
        )
        screen.blit(info_txt, (20, 10))

        if self.cars:
            best_fit = max(c.fitness for c in self.cars)
            finished_count = sum(1 for c in self.cars if c.finished)
            alive_count = sum(1 for c in self.cars if not c.is_stopped)
            stag_count = sum(1 for c in self.cars if c.stagnated)
            crashed_count = sum(1 for c in self.cars if c.crashed)

            stat_txt = font.render(
                f"Best: {int(best_fit)}  |  Alive: {alive_count}  |  Finish: {finished_count}  |  Crash: {crashed_count}  |  Stuck: {stag_count}",
                True, BLUE,
            )
            screen.blit(stat_txt, (20, 40))

            if finished_count > 0:
                ratio = finished_count / NUM_CARS
                bar_x, bar_y, bar_w, bar_h = 20, 68, 300, 14
                pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(screen, GREEN, (bar_x, bar_y, int(bar_w * ratio), bar_h))
                pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_w, bar_h), 1)
                pct_txt = font.render(f"완주율: {finished_count}/{NUM_CARS} (목표: {self.finish_target})", True, DARK_GREEN)
                screen.blit(pct_txt, (bar_x + bar_w + 10, bar_y - 3))

        if self.state == STATE_SETUP:
            msg = large_font.render("Click to set START point", True, RED)
            screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2))

        elif self.state == STATE_SIMULATING:
            bar_x, bar_y, bar_w, bar_h = 600, 15, 350, 20
            progress = 1 - self.steps / MAX_STEPS
            pygame.draw.rect(screen, BLUE, (bar_x, bar_y, int(bar_w * progress), bar_h))
            pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_w, bar_h), 2)

        elif self.state == STATE_SELECTING:
            msg = large_font.render("Select the best car", True, GREEN)
            screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 200))
            for i, rect in enumerate(self.buttons):
                hovering = rect.collidepoint(pygame.mouse.get_pos())
                car = self.cars[i]
                if car.finished:
                    bg = GREEN if hovering else (180, 240, 180)
                elif car.stagnated:
                    bg = ORANGE if hovering else (255, 220, 180)
                elif car.crashed:
                    bg = RED if hovering else (240, 180, 180)
                else:
                    bg = GRAY if hovering else UI_BACK_COLOR
                pygame.draw.rect(screen, bg, rect, 0, 10)
                pygame.draw.rect(screen, BLACK, rect, 2, 10)
                if car.finished:
                    status = "F"
                elif car.stagnated:
                    status = "S"
                elif car.crashed:
                    status = "X"
                else:
                    status = "~"
                btn_txt = font.render(
                    f"Car {i+1} [{status}]: {int(car.fitness)}", True, BLACK
                )
                screen.blit(
                    btn_txt,
                    (rect.centerx - btn_txt.get_width() // 2,
                     rect.centery - btn_txt.get_height() // 2),
                )

    def _draw_finish_screen(self):
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(220)
        overlay.fill(WHITE)
        screen.blit(overlay, (0, 0))

        finished_count = sum(1 for c in self.cars if c.finished)

        finish_txt = title_font.render("FINISH!", True, GOLD)
        screen.blit(finish_txt, (WIDTH // 2 - finish_txt.get_width() // 2, HEIGHT // 2 - 170))

        sub_txt = large_font.render(
            f"{finished_count}/{NUM_CARS} completed  |  Generation {self.generation}  |  Target: {self.finish_target}",
            True, DARK_GREEN,
        )
        screen.blit(sub_txt, (WIDTH // 2 - sub_txt.get_width() // 2, HEIGHT // 2 - 100))

        best_car = max(self.cars, key=lambda c: c.fitness)
        lines = [
            f"Best Car: #{best_car.id}  |  Fitness: {int(best_car.fitness)}",
        ]
        if self.first_finish_gen:
            lines.append(f"First finish at Generation {self.first_finish_gen}")

        for i, line in enumerate(lines):
            txt = font.render(line, True, BLACK)
            screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, HEIGHT // 2 - 55 + i * 28))

        # Continue 버튼 라벨: 아직 전원 완주 목표가 아니면 "전원 완주까지"
        continue_label = f"Train until {NUM_CARS}/{NUM_CARS}" if self.finish_target < NUM_CARS else "Continue Training"

        for rect, label, color in [
            (self.btn_continue, continue_label, BLUE),
            (self.btn_restart_manual, "Restart (Manual)", GRAY),
            (self.btn_restart_auto, "Restart (Auto)", DARK_GREEN),
            (self.btn_quit, "Quit", RED),
        ]:
            hovering = rect.collidepoint(pygame.mouse.get_pos())
            bg = color if hovering else UI_BACK_COLOR
            txt_color = WHITE if hovering else BLACK
            pygame.draw.rect(screen, bg, rect, 0, 12)
            pygame.draw.rect(screen, BLACK, rect, 2, 12)
            btn_txt = large_font.render(label, True, txt_color)
            screen.blit(
                btn_txt,
                (rect.centerx - btn_txt.get_width() // 2,
                 rect.centery - btn_txt.get_height() // 2),
            )

    def handle_click(self, pos):
        if self.state == STATE_SETUP:
            self.start_pos = pos
            self.cars = [Car(i + 1, pos[0], pos[1]) for i in range(NUM_CARS)]
            self.state = STATE_SIMULATING

        elif self.state == STATE_SELECTING and self.mode == "manual":
            for i, rect in enumerate(self.buttons):
                if rect.collidepoint(pos):
                    self.next_generation(self.cars[i].brain)
                    break

        elif self.state == STATE_FINISHED:
            if self.btn_continue.collidepoint(pos):
                # 전원 완주까지 자동 반복
                self.finish_target = NUM_CARS
                self.mode = "auto"
                self.next_generation(self.best_brain)
            elif self.btn_restart_manual.collidepoint(pos):
                self._restart("manual")
            elif self.btn_restart_auto.collidepoint(pos):
                self._restart("auto")
            elif self.btn_quit.collidepoint(pos):
                pygame.quit()
                exit()

    def _restart(self, mode):
        self.mode = mode
        self.generation = 1
        self.steps = 0
        self.cars = []
        self.best_brain = None
        self.first_finish_gen = None
        self.finish_history = []
        self.finish_target = math.ceil(NUM_CARS / 2)
        self.state = STATE_SETUP


# ====================================================================
# 4. 메인 실행 루프
# ====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Car Simulation")
    parser.add_argument(
        "--mode", type=str, default="manual", choices=["manual", "auto"],
        help="manual: best car 선택, auto: fitness 기반 자동 선택",
    )
    args = parser.parse_args()

    session = UISession(mode=args.mode)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                session.handle_click(event.pos)
        session.update()
        session.draw()
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()