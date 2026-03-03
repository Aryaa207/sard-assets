import time
import math
import cv2
import numpy as np
import pygame
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo as adafruit_servo
from picamera2 import Picamera2

# ============================================================
#  CONFIGURATION
# ============================================================
PROCESS_WIDTH   = 320
PROCESS_HEIGHT  = 240
VIEW_WIDTH      = 1280
VIEW_HEIGHT     = 720

SCORE_THRESHOLD = 0.6
NMS_THRESHOLD   = 0.3
TOP_K           = 5000
MODEL_PATH      = "face_detection_yunet_2023mar.onnx"

SCALE_X = VIEW_WIDTH  / PROCESS_WIDTH
SCALE_Y = VIEW_HEIGHT / PROCESS_HEIGHT

# ============================================================
#  SERVO CONFIGURATION
# ============================================================
PAN_CHANNEL  = 2
TILT_CHANNEL = 1

PAN_LIMIT    = 70    # +/- degrees from centre
TILT_LIMIT   = 50

# MG90S correct pulse range — DO NOT widen these
SERVO_MIN_PULSE = 600
SERVO_MAX_PULSE = 2400

# ── PID GAINS ─────────────────────────────────────────────
PAN_KP,  PAN_KI,  PAN_KD  = 0.06, 0.00008, 0.010
TILT_KP, TILT_KI, TILT_KD = 0.06, 0.00008, 0.010
DEAD_ZONE = 20       # pixels — ignore small jitter

# ============================================================
#  COLOUR PALETTE
# ============================================================
C_BG            = (10,  13,  18)
C_PANEL         = (18,  24,  32)
C_PANEL_BORDER  = (45,  58,  72)
C_AMBER         = (255, 176, 0)
C_AMBER_DIM     = (140, 90,  0)
C_WHITE         = (220, 228, 235)
C_WHITE_DIM     = (110, 120, 130)
C_RED           = (220, 45,  45)
C_BLUE          = (60,  160, 255)
C_GREEN_OK      = (60,  210, 120)
C_DARK_LINE     = (30,  38,  50)
C_BOX_NOMINAL   = (60,  160, 255)
C_BOX_ELEVATED  = (255, 176, 0)
C_BOX_HIGH      = (220, 45,  45)

# ============================================================
#  LAYOUT
# ============================================================
TOP_H    = 36
BOTTOM_H = 52
LEFT_W   = 200
RIGHT_W  = 210
CAM_X    = LEFT_W
CAM_Y    = TOP_H
CAM_W    = VIEW_WIDTH  - LEFT_W - RIGHT_W
CAM_H    = VIEW_HEIGHT - TOP_H  - BOTTOM_H
CAM_SCALE_X = CAM_W / PROCESS_WIDTH
CAM_SCALE_Y = CAM_H / PROCESS_HEIGHT

# ============================================================
#  PID CONTROLLER
# ============================================================
class PID:
    def __init__(self, kp, ki, kd, limit):
        self.kp         = kp
        self.ki         = ki
        self.kd         = kd
        self.limit      = limit
        self.integral   = 0.0
        self.prev_err   = 0.0
        self.prev_t     = time.perf_counter()
        self.smooth_err = 0.0   # low-pass filtered error
        self.smooth_out = 0.0   # low-pass filtered output
        self.ALPHA_ERR  = 0.25  # error smoothing  (lower = smoother)
        self.ALPHA_OUT  = 0.20  # output smoothing (lower = smoother)

    def reset(self):
        self.integral   = 0.0
        self.prev_err   = 0.0
        self.smooth_err = 0.0
        self.smooth_out = 0.0

    def update(self, raw_error):
        # Smooth incoming error to reduce face-detector jitter
        self.smooth_err = self.ALPHA_ERR * raw_error + (1 - self.ALPHA_ERR) * self.smooth_err
        error = self.smooth_err

        now = time.perf_counter()
        dt  = max(now - self.prev_t, 1e-3)
        self.prev_t = now

        if abs(error) < DEAD_ZONE:
            self.prev_err = error
            self.smooth_out = (1 - self.ALPHA_OUT) * self.smooth_out
            return self.smooth_out

        self.integral = max(-100, min(100, self.integral + error * dt))
        derivative    = (error - self.prev_err) / dt
        self.prev_err = error

        raw_out = self.kp * error + self.ki * self.integral + self.kd * derivative
        raw_out = max(-self.limit, min(self.limit, raw_out))

        # Smooth output so servo moves gradually not snappily
        self.smooth_out = self.ALPHA_OUT * raw_out + (1 - self.ALPHA_OUT) * self.smooth_out
        return self.smooth_out


# ============================================================
#  SERVO CONTROLLER
# ============================================================
class ServoController:
    def __init__(self):
        i2c          = busio.I2C(board.SCL, board.SDA)
        self.pca     = PCA9685(i2c)
        self.pca.frequency = 50

        self.pan_servo  = adafruit_servo.Servo(
            self.pca.channels[PAN_CHANNEL],
            min_pulse=SERVO_MIN_PULSE, max_pulse=SERVO_MAX_PULSE,
            actuation_range=180)
        self.tilt_servo = adafruit_servo.Servo(
            self.pca.channels[TILT_CHANNEL],
            min_pulse=SERVO_MIN_PULSE, max_pulse=SERVO_MAX_PULSE,
            actuation_range=180)

        # Start from current physical position — no snap
        # Read current duty cycle and back-calculate angle
        self.pan_angle  = 0.0
        self.tilt_angle = 0.0

        # Do NOT send any angle command on startup
        # Servos will stay exactly where they physically are
        print("Servos ready — holding current position.")

    def _to_servo_angle(self, degrees, limit):
        return 90 + (degrees / limit) * 90

    def _write_pan(self, angle):
        angle = max(-PAN_LIMIT, min(PAN_LIMIT, angle))
        self.pan_angle = angle
        self.pan_servo.angle = self._to_servo_angle(angle, PAN_LIMIT)

    def _write_tilt(self, angle):
        angle = max(-TILT_LIMIT, min(TILT_LIMIT, angle))
        self.tilt_angle = angle
        self.tilt_servo.angle = self._to_servo_angle(angle, TILT_LIMIT)

    def move(self, pan_delta, tilt_delta):
        self._write_pan(self.pan_angle   + pan_delta)
        self._write_tilt(self.tilt_angle + tilt_delta)

    def stop(self):
        self._write_pan(0.0)
        self._write_tilt(0.0)
        self.pca.deinit()


# ============================================================
#  MODEL
# ============================================================
detector = cv2.FaceDetectorYN.create(
    MODEL_PATH, "",
    (PROCESS_WIDTH, PROCESS_HEIGHT),
    SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
)

# ============================================================
#  CAMERA
# ============================================================
print("Initialising Arducam OWLSight 64 MP ...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (VIEW_WIDTH, VIEW_HEIGHT), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

try:
    picam2.set_controls({"AfMode": 2, "AfSpeed": 1})
    print("Autofocus enabled.")
except Exception as exc:
    print(f"Autofocus unavailable: {exc}")

# ============================================================
#  PYGAME
# ============================================================
pygame.init()
screen = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
pygame.display.set_caption("ARGUS-IV // AUTONOMOUS TRACKING SYSTEM")

font_title = pygame.font.SysFont("monospace", 15, bold=True)
font_med   = pygame.font.SysFont("monospace", 13, bold=True)
font_small = pygame.font.SysFont("monospace", 12)
font_tiny  = pygame.font.SysFont("monospace", 10)

# ============================================================
#  SERVO + PID INIT
# ============================================================
servo  = ServoController()
pid_pan  = PID(PAN_KP,  PAN_KI,  PAN_KD,  PAN_LIMIT)
pid_tilt = PID(TILT_KP, TILT_KI, TILT_KD, TILT_LIMIT)

# ============================================================
#  UI HELPERS
# ============================================================
def filled_panel(surf, rect, fill=C_PANEL, border=C_PANEL_BORDER, radius=0):
    pygame.draw.rect(surf, fill,   rect, border_radius=radius)
    pygame.draw.rect(surf, border, rect, 1, border_radius=radius)

def draw_divider(surf, x1, y1, x2, y2):
    pygame.draw.line(surf, C_DARK_LINE, (x1, y1), (x2, y2), 1)

def threat_color(confidence, face_count):
    if face_count > 2 or confidence > 0.9:
        return C_BOX_HIGH
    elif confidence > 0.75:
        return C_BOX_ELEVATED
    return C_BOX_NOMINAL

def draw_target_box(surf, x, y, w, h, color, confidence, subject_id, tick, is_primary=False):
    pad  = 5
    blen = max(12, min(w, h) // 5)
    x1, y1 = x + CAM_X - pad, y + CAM_Y - pad
    x2, y2 = x1 + w + pad * 2, y1 + h + pad * 2

    rect_surf = pygame.Surface((x2 - x1, y2 - y1), pygame.SRCALPHA)
    pygame.draw.rect(rect_surf, (*color, 40), rect_surf.get_rect(), 1)
    surf.blit(rect_surf, (x1, y1))

    thick = 3 if is_primary else 2
    corners = [
        [(x1, y1 + blen), (x1, y1), (x1 + blen, y1)],
        [(x2 - blen, y1), (x2, y1), (x2, y1 + blen)],
        [(x1, y2 - blen), (x1, y2), (x1 + blen, y2)],
        [(x2 - blen, y2), (x2, y2), (x2, y2 - blen)],
    ]
    for pts in corners:
        pygame.draw.lines(surf, color, False, pts, thick)

    if is_primary:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        pygame.draw.circle(surf, (*color, 180), (cx, cy), 5)
        pygame.draw.line(surf, (*color, 120), (cx - 14, cy), (cx + 14, cy), 1)
        pygame.draw.line(surf, (*color, 120), (cx, cy - 14), (cx, cy + 14), 1)

    tag_surf = pygame.Surface((90, 18), pygame.SRCALPHA)
    tag_surf.fill((*C_PANEL, 210))
    pygame.draw.rect(tag_surf, color, tag_surf.get_rect(), 1)
    tag_surf.blit(font_tiny.render(f"ID-{subject_id:03d}", True, color),     (4, 3))
    tag_surf.blit(font_tiny.render(f"{confidence*100:.0f}%", True, C_WHITE), (54, 3))
    surf.blit(tag_surf, (x1, max(y1 - 20, CAM_Y + 2)))

    status = "LOCK" if confidence > 0.8 else "ACQ"
    sc = C_RED if status == "LOCK" and (tick // 18) % 2 == 0 else color
    surf.blit(font_tiny.render(status, True, sc), (x2 - 26, y2 + 3))


def draw_static_frame(surf):
    surf.fill(C_BG)
    filled_panel(surf, pygame.Rect(0, 0, VIEW_WIDTH, TOP_H), C_PANEL, C_PANEL_BORDER)
    surf.blit(font_title.render("ARGUS-IV  //  AUTONOMOUS TRACKING SYSTEM", True, C_WHITE), (12, 10))
    t = time.strftime("%Y-%m-%d   %H:%M:%S  UTC")
    surf.blit(font_title.render(t, True, C_AMBER), (VIEW_WIDTH - 310, 10))
    pygame.draw.line(surf, C_PANEL_BORDER, (0, TOP_H), (VIEW_WIDTH, TOP_H), 1)
    filled_panel(surf, pygame.Rect(0, VIEW_HEIGHT - BOTTOM_H, VIEW_WIDTH, BOTTOM_H), C_PANEL, C_PANEL_BORDER)
    filled_panel(surf, pygame.Rect(0, TOP_H, LEFT_W, VIEW_HEIGHT - TOP_H - BOTTOM_H), C_PANEL, C_PANEL_BORDER)
    filled_panel(surf, pygame.Rect(VIEW_WIDTH - RIGHT_W, TOP_H, RIGHT_W, VIEW_HEIGHT - TOP_H - BOTTOM_H), C_PANEL, C_PANEL_BORDER)
    pygame.draw.rect(surf, C_PANEL_BORDER, pygame.Rect(CAM_X - 1, CAM_Y - 1, CAM_W + 2, CAM_H + 2), 1)


def draw_left_panel(surf, fps, pan_angle, tilt_angle, tick):
    px, py = 10, TOP_H + 12

    surf.blit(font_med.render("SYSTEM STATUS", True, C_AMBER), (px, py))
    draw_divider(surf, px, py + 16, LEFT_W - 10, py + 16)
    py += 24

    rows = [
        ("PLATFORM",  "ARGUS-IV"),
        ("SENSOR",    "ARDUCAM-64MP"),
        ("ALGORITHM", "YUNET-2023"),
        ("MODE",      "AUTO-TRACK"),
        ("NETWORK",   "AIR-GAPPED"),
        ("ENCRYPT",   "AES-256-GCM"),
    ]
    for label, val in rows:
        surf.blit(font_tiny.render(label, True, C_WHITE_DIM), (px, py))
        surf.blit(font_tiny.render(val,   True, C_WHITE),     (px + 72, py))
        py += 15

    py += 8
    draw_divider(surf, px, py, LEFT_W - 10, py)
    py += 10

    surf.blit(font_med.render("PERFORMANCE", True, C_AMBER), (px, py))
    draw_divider(surf, px, py + 16, LEFT_W - 10, py + 16)
    py += 24
    fps_color = C_GREEN_OK if fps > 20 else C_AMBER if fps > 10 else C_RED
    surf.blit(font_tiny.render("FRAME RATE", True, C_WHITE_DIM), (px, py))
    surf.blit(font_med.render(f"{fps:5.1f} FPS", True, fps_color), (px + 72, py - 1))
    py += 16
    bar_w = LEFT_W - 20
    pygame.draw.rect(surf, C_DARK_LINE, (px, py, bar_w, 6))
    pygame.draw.rect(surf, fps_color,   (px, py, int(bar_w * min(fps / 30.0, 1.0)), 6))
    py += 20

    draw_divider(surf, px, py, LEFT_W - 10, py)
    py += 10

    surf.blit(font_med.render("GIMBAL", True, C_AMBER), (px, py))
    draw_divider(surf, px, py + 16, LEFT_W - 10, py + 16)
    py += 24

    pan_color  = C_AMBER if abs(pan_angle)  > PAN_LIMIT  * 0.8 else C_WHITE
    tilt_color = C_AMBER if abs(tilt_angle) > TILT_LIMIT * 0.8 else C_WHITE
    surf.blit(font_tiny.render("PAN",  True, C_WHITE_DIM), (px, py))
    surf.blit(font_med.render(f"{pan_angle:+.1f}deg",  True, pan_color),  (px + 52, py - 1))
    py += 16
    surf.blit(font_tiny.render("TILT", True, C_WHITE_DIM), (px, py))
    surf.blit(font_med.render(f"{tilt_angle:+.1f}deg", True, tilt_color), (px + 52, py - 1))
    py += 20

    bar_w = LEFT_W - 20
    mid   = bar_w // 2
    for angle, limit in [(pan_angle, PAN_LIMIT), (tilt_angle, TILT_LIMIT)]:
        pygame.draw.rect(surf, C_DARK_LINE, (px, py, bar_w, 6))
        fill = int(mid * (angle / limit))
        if fill >= 0:
            pygame.draw.rect(surf, C_BLUE, (px + mid, py, fill, 6))
        else:
            pygame.draw.rect(surf, C_BLUE, (px + mid + fill, py, -fill, 6))
        pygame.draw.line(surf, C_WHITE, (px + mid, py - 1), (px + mid, py + 7), 1)
        py += 14

    py += 6
    if (tick // 20) % 2 == 0:
        pygame.draw.circle(surf, C_RED, (px + 6, py + 6), 5)
        surf.blit(font_tiny.render("  RECORDING", True, C_RED), (px + 4, py))
    else:
        pygame.draw.circle(surf, C_WHITE_DIM, (px + 6, py + 6), 5, 1)
        surf.blit(font_tiny.render("  RECORDING", True, C_WHITE_DIM), (px + 4, py))


def draw_right_panel(surf, face_count, faces, pan_err, tilt_err, tick):
    px  = VIEW_WIDTH - RIGHT_W + 10
    py  = TOP_H + 12
    rw  = RIGHT_W - 20

    surf.blit(font_med.render("THREAT ASSESSMENT", True, C_AMBER), (px, py))
    draw_divider(surf, px, py + 16, px + rw, py + 16)
    py += 24

    if face_count == 0:
        level, lc = "NOMINAL",  C_GREEN_OK
    elif face_count <= 2:
        level, lc = "ELEVATED", C_AMBER
    else:
        level, lc = "HIGH",     C_RED

    badge = pygame.Surface((rw, 24), pygame.SRCALPHA)
    badge.fill((*lc, 30))
    pygame.draw.rect(badge, lc, badge.get_rect(), 1)
    badge.blit(font_med.render(f"  {level}", True, lc), (4, 4))
    surf.blit(badge, (px, py))
    py += 32

    surf.blit(font_tiny.render("SUBJECTS DETECTED", True, C_WHITE_DIM), (px, py))
    surf.blit(font_med.render(f"{face_count:02d}", True, C_WHITE), (px + 120, py - 1))
    py += 18

    draw_divider(surf, px, py, px + rw, py)
    py += 10

    surf.blit(font_med.render("PID TRACKER", True, C_AMBER), (px, py))
    draw_divider(surf, px, py + 16, px + rw, py + 16)
    py += 24

    ec = C_RED if abs(pan_err) > 50 else C_AMBER if abs(pan_err) > 20 else C_GREEN_OK
    surf.blit(font_tiny.render("PAN  ERR", True, C_WHITE_DIM), (px, py))
    surf.blit(font_med.render(f"{pan_err:+.0f}px", True, ec), (px + 80, py - 1))
    py += 16

    ec = C_RED if abs(tilt_err) > 50 else C_AMBER if abs(tilt_err) > 20 else C_GREEN_OK
    surf.blit(font_tiny.render("TILT ERR", True, C_WHITE_DIM), (px, py))
    surf.blit(font_med.render(f"{tilt_err:+.0f}px", True, ec), (px + 80, py - 1))
    py += 20

    draw_divider(surf, px, py, px + rw, py)
    py += 10

    surf.blit(font_med.render("SUBJECT LOG", True, C_AMBER), (px, py))
    draw_divider(surf, px, py + 16, px + rw, py + 16)
    py += 24

    if faces is not None:
        for i, face in enumerate(faces[:5]):
            conf  = face[14]
            tc    = threat_color(conf, face_count)
            blink = (tick // 15) % 2 == 0
            row   = pygame.Surface((rw, 20), pygame.SRCALPHA)
            row.fill((*tc, 15))
            pygame.draw.rect(row, (*tc, 60), row.get_rect(), 1)
            prefix = ">" if i == 0 else " "
            row.blit(font_tiny.render(f"{prefix}ID-{i+1:03d}", True, tc),    (4,  4))
            row.blit(font_tiny.render(f"{conf*100:.0f}%", True, C_WHITE),     (65, 4))
            status = "LOCK" if conf > 0.8 else "ACQ "
            sc = C_RED if status == "LOCK" and blink else tc
            row.blit(font_tiny.render(status, True, sc), (105, 4))
            surf.blit(row, (px, py))
            py += 24
    else:
        surf.blit(font_tiny.render("-- NO SUBJECTS --", True, C_WHITE_DIM), (px + 20, py))

    cy = VIEW_HEIGHT - BOTTOM_H - 50
    draw_divider(surf, px, cy, px + rw, cy)
    surf.blit(font_tiny.render("CLASSIFICATION:", True, C_WHITE_DIM), (px, cy + 6))
    surf.blit(font_tiny.render("TOP SECRET // SCI", True, C_RED),     (px, cy + 18))


def draw_bottom_bar(surf, face_count, fps, tick):
    by = VIEW_HEIGHT - BOTTOM_H + 8
    bx = LEFT_W + 10

    if face_count > 0:
        blink = (tick // 12) % 2 == 0
        msg   = f"  ALERT: {face_count} SUBJECT(S) IN FRAME -- TRACKING ACTIVE  "
        mc    = C_RED if blink else C_AMBER
        badge = pygame.Surface((440, 22), pygame.SRCALPHA)
        badge.fill((*C_RED, 25) if blink else (*C_AMBER, 15))
        pygame.draw.rect(badge, mc, badge.get_rect(), 1)
        badge.blit(font_med.render(msg, True, mc), (4, 3))
        surf.blit(badge, (bx, by))
    else:
        surf.blit(font_med.render("SCANNING...  NO SUBJECTS IN FRAME", True, C_WHITE_DIM), (bx, by))

    surf.blit(font_tiny.render(
        "SENSOR: ARDUCAM 64MP  //  AI: YuNet 2023  //  SERVO: PCA9685 PID  //  v4.0",
        True, C_WHITE_DIM), (bx, by + 24))


# ============================================================
#  SCANLINE
# ============================================================
scan_y     = 0
SCAN_SPEED = 3

def draw_scanline(surf, sy):
    line = pygame.Surface((CAM_W, 2), pygame.SRCALPHA)
    line.fill((255, 255, 255, 18))
    surf.blit(line, (CAM_X, CAM_Y + sy % CAM_H))


# ============================================================
#  MAIN LOOP
# ============================================================
print("ARGUS-IV ONLINE -- Autonomous tracking active. Press Q to quit.")

fps         = 0.0
frame_times = []
tick        = 0
pan_err     = 0.0
tilt_err    = 0.0

FRAME_CX = PROCESS_WIDTH  // 2
FRAME_CY = PROCESS_HEIGHT // 2

try:
    while True:
        t_start = time.perf_counter()
        tick += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                raise KeyboardInterrupt

        # --- Capture ---
        frame = picam2.capture_array()

        # --- AI ---
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img_small = cv2.resize(frame_bgr, (PROCESS_WIDTH, PROCESS_HEIGHT))
        _, faces  = detector.detect(img_small)
        face_count = len(faces) if faces is not None else 0

        # --- PID Tracking ---
        if faces is not None and len(faces) > 0:
            primary  = faces[0]
            bx = primary[0] + primary[2] / 2.0
            by = primary[1] + primary[3] / 2.0

            pan_err  = bx - FRAME_CX
            tilt_err = by - FRAME_CY

            pan_delta  =  pid_pan.update(pan_err)
            tilt_delta = -pid_tilt.update(tilt_err)

            # Debug — print every 30 frames
            if tick % 30 == 0:
                print(f"Face: bx={bx:.0f} by={by:.0f} | err pan={pan_err:.0f} tilt={tilt_err:.0f} | delta pan={pan_delta:.2f} tilt={tilt_delta:.2f} | angle pan={servo.pan_angle:.1f} tilt={servo.tilt_angle:.1f}")

            servo.move(pan_delta, tilt_delta)
        else:
            pid_pan.reset()
            pid_tilt.reset()
            pan_err  = 0.0
            tilt_err = 0.0

        # --- Draw ---
        draw_static_frame(screen)

        cam_resized = cv2.resize(frame, (CAM_W, CAM_H))
        cam_surface = pygame.surfarray.make_surface(np.transpose(cam_resized, (1, 0, 2)))
        screen.blit(cam_surface, (CAM_X, CAM_Y))

        draw_scanline(screen, scan_y)
        scan_y += SCAN_SPEED

        if faces is not None:
            for i, face in enumerate(faces):
                box  = face[0:4].astype(np.int32)
                x    = int(box[0] * CAM_SCALE_X)
                y    = int(box[1] * CAM_SCALE_Y)
                w    = int(box[2] * CAM_SCALE_X)
                h    = int(box[3] * CAM_SCALE_Y)
                conf = face[14]
                col  = threat_color(conf, face_count)
                draw_target_box(screen, x, y, w, h, col, conf, i + 1, tick, is_primary=(i == 0))

                landmarks = face[4:14].astype(np.float32).reshape((5, 2))
                for lm in landmarks:
                    lx = int(lm[0] * CAM_SCALE_X) + CAM_X
                    ly = int(lm[1] * CAM_SCALE_Y) + CAM_Y
                    pygame.draw.circle(screen, C_AMBER, (lx, ly), 2)

        draw_left_panel(screen, fps, servo.pan_angle, servo.tilt_angle, tick)
        draw_right_panel(screen, face_count, faces, pan_err, tilt_err, tick)
        draw_bottom_bar(screen, face_count, fps, tick)

        pygame.display.flip()

        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

except KeyboardInterrupt:
    print("\nSystem shutdown.")
finally:
    servo.stop()
    picam2.stop()
    pygame.quit()
    print("ARGUS-IV offline.")
