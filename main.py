"""
Vizuálne navádzanie masážneho pohybu pomocou hand trackingu
============================================================
Knižnice: OpenCV (video), MediaPipe (hand tracking), NumPy (výpočty)

Inštalácia:
    pip install opencv-python mediapipe numpy

Spustenie:
    python massage_guidance.py

Ovládanie:
    q / ESC  - koniec
    r        - reset šípky z klávesnice
    d        - otočí šípku o 45 stupňov

Ako to funguje:
    * Webkamera -> MediaPipe Hands -> 21 bodov ruky (články prstov + dlaň)
    * V strede obrazu je šípka. Špičkou ukazováka (bod 8) ju "prejdeš".
    * Rýchlosť pohybu v smere šípky sa meria v dĺžkach šípky za sekundu.
    * Každý úsek šípky sa vyfarbí podľa rýchlosti, akou ho prešiel ukazovák:
          červená = príliš rýchlo
          zelená  = správna rýchlosť (a správny smer)
          žltá    = príliš pomaly
    * Tlačidlo RESET vpravo hore: podrž na ňom ukazovák ~0.6 s -> šípka sa vynuluje.
"""

import time
import math

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------
# KONFIGURÁCIA
# ----------------------------------------------------------------------
CAM_INDEX = 0                 # index webkamery
FRAME_W, FRAME_H = 1280, 720  # požadované rozlíšenie

ARROW_LENGTH_RATIO = 0.55     # dĺžka šípky ako podiel šírky obrazu
ARROW_HALF_WIDTH = 28         # polovica hrúbky driekov šípky (px)
ARROW_HEAD_RATIO = 0.20       # podiel dĺžky, ktorý tvorí hrot
ARROW_ANGLE_DEG = 0           # 0 = doprava (menej: 'd' otáča o 45 st.)
N_SEGMENTS = 50               # počet úsekov pre postupný prechod farby
LATERAL_TOLERANCE = 110       # max. kolmá vzdialenosť prsta od osi šípky (px)
MAX_JUMP = 0.12               # max. skok postupu medzi snímkami (kontinuita)

TARGET_SPEED = 0.45           # cieľová rýchlosť [dĺžka šípky / s]
SPEED_OK_LOW = 0.70           # pomer k cieľu: pod tým je to "pomaly"
SPEED_OK_HIGH = 1.30          # pomer k cieľu: nad tým je to "rýchlo"

BUTTON_SIZE = (200, 70)       # šírka, výška tlačidla
BUTTON_MARGIN = 20
BUTTON_DWELL_S = 0.6          # ako dlho treba držať prst na tlačidle

SMOOTH_POS = 0.5              # EMA vyhladenie polohy (0..1, vyššie = rýchlejšie)
SMOOTH_SPEED = 0.25           # EMA vyhladenie rýchlosti

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

# Spojenia medzi 21 bodmi ruky (zápästie, dlaň, články prstov)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # palec
    (0, 5), (5, 6), (6, 7), (7, 8),            # ukazovák
    (5, 9), (9, 10), (10, 11), (11, 12),       # prostredník
    (9, 13), (13, 14), (14, 15), (15, 16),     # prstenník
    (13, 17), (17, 18), (18, 19), (19, 20),    # malíček
    (0, 17),                                   # okraj dlane
]


def ensure_model():
    """Stiahne model hand_landmarker.task, ak ešte nie je pri skripte."""
    # znovu stiahni, ak súbor chýba alebo je poškodený (príliš malý)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        print("Sťahujem model ruky (jednorazovo)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Hotovo:", MODEL_PATH)
    return MODEL_PATH


# ----------------------------------------------------------------------
# POMOCNÉ FUNKCIE
# ----------------------------------------------------------------------
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def speed_ratio_to_bgr(ratio):
    """Prevod pomeru rýchlosti (skutočná/cieľová) na farbu BGR.

    OpenCV HSV: hue 0 = červená, 30 = žltá, 60 = zelená.
    Plynulý prechod: pomaly (žltá) -> správne (zelená) -> rýchlo (červená).
    """
    if ratio < SPEED_OK_LOW:                       # príliš pomaly
        k = clamp((SPEED_OK_LOW - ratio) / SPEED_OK_LOW)
        hue = 60 - 30 * k
    elif ratio <= SPEED_OK_HIGH:                   # správna rýchlosť
        hue = 60
    else:                                          # príliš rýchlo
        k = clamp((ratio - SPEED_OK_HIGH) / 0.7)
        hue = 60 * (1 - k)
    hsv = np.uint8([[[int(hue), 230, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


# ----------------------------------------------------------------------
# ŠÍPKA S VIZUÁLNYM NAVÁDZANÍM
# ----------------------------------------------------------------------
class GuidanceArrow:
    def __init__(self, frame_w, frame_h):
        self.center = np.array([frame_w / 2, frame_h / 2], dtype=np.float32)
        self.length = frame_w * ARROW_LENGTH_RATIO
        self.angle = math.radians(ARROW_ANGLE_DEG)
        self._update_axes()
        self.reset()

    def _update_axes(self):
        self.d = np.array([math.cos(self.angle), math.sin(self.angle)], np.float32)
        self.n = np.array([-self.d[1], self.d[0]], np.float32)

    def rotate(self, deg):
        self.angle += math.radians(deg)
        self._update_axes()
        self.reset()

    def reset(self):
        self.progress = 0.0                       # 0..1 po osi šípky
        self.colors = [None] * N_SEGMENTS         # farba každého úseku
        self.tip_t = None                         # vyhladená poloha prsta na osi
        self.speed = 0.0                          # vyhladená rýchlosť [dĺžky/s]
        self.speed_samples = []
        self.active = False                       # je prst na šípke?

    # --- geometria -----------------------------------------------------
    def point(self, t, side=0.0):
        """Bod na šípke: t = 0..1 pozdĺž osi, side = kolmý posun v px."""
        base = self.center + self.d * (t - 0.5) * self.length
        return base + self.n * side

    def half_width(self, t):
        head_start = 1.0 - ARROW_HEAD_RATIO
        if t <= head_start:
            return ARROW_HALF_WIDTH
        k = (t - head_start) / ARROW_HEAD_RATIO
        return ARROW_HALF_WIDTH * 1.9 * (1 - k)

    def segment_polygon(self, i):
        t0, t1 = i / N_SEGMENTS, (i + 1) / N_SEGMENTS
        w0, w1 = self.half_width(t0 + 1e-6), self.half_width(t1 - 1e-6)
        if t0 >= 1.0 - ARROW_HEAD_RATIO - 1e-6:   # v hrote
            w0, w1 = self.half_width(t0), self.half_width(t1)
        pts = [self.point(t0, -w0), self.point(t1, -w1),
               self.point(t1, +w1), self.point(t0, +w0)]
        return np.array(pts, np.int32)

    # --- logika --------------------------------------------------------
    def update(self, finger_xy, dt):
        """Spracuje polohu špičky ukazováka (px) alebo None, ak ruka nie je vidieť."""
        if finger_xy is None:
            self.active = False
            self.tip_t = None
            self.speed *= 0.9
            return

        rel = np.array(finger_xy, np.float32) - self.center
        t = float(np.dot(rel, self.d)) / self.length + 0.5   # poloha pozdĺž šípky
        lateral = abs(float(np.dot(rel, self.n)))            # kolmá odchýlka

        self.active = lateral < LATERAL_TOLERANCE and -0.08 < t < 1.08
        if not self.active:
            self.tip_t = None
            self.speed *= 0.9
            return

        if self.tip_t is None:                               # prst práve vošiel
            self.tip_t = t
            return

        prev = self.tip_t
        self.tip_t += SMOOTH_POS * (t - self.tip_t)
        inst_speed = (self.tip_t - prev) / max(dt, 1e-3)     # dĺžky šípky / s
        self.speed += SMOOTH_SPEED * (inst_speed - self.speed)

        # Vyfarbuj len pri pohybe správnym smerom a nadväzne na doterajší postup
        if self.speed > 0 and self.progress < 1.0:
            if 0 <= self.tip_t - self.progress < MAX_JUMP or (
                    self.progress == 0 and self.tip_t < MAX_JUMP):
                ratio = self.speed / TARGET_SPEED
                color = speed_ratio_to_bgr(ratio)
                new_progress = clamp(max(self.progress, self.tip_t))
                a = int(self.progress * N_SEGMENTS)
                b = int(math.ceil(new_progress * N_SEGMENTS))
                for i in range(a, min(b, N_SEGMENTS)):
                    self.colors[i] = color
                self.speed_samples.append(self.speed)
                self.progress = new_progress

    # --- vykresľovanie -------------------------------------------------
    def draw(self, frame):
        overlay = frame.copy()
        for i in range(N_SEGMENTS):
            color = self.colors[i] if self.colors[i] is not None else (110, 110, 110)
            cv2.fillPoly(overlay, [self.segment_polygon(i)], color)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # obrys šípky
        outline = [self.point(0, -ARROW_HALF_WIDTH),
                   self.point(1 - ARROW_HEAD_RATIO, -ARROW_HALF_WIDTH),
                   self.point(1 - ARROW_HEAD_RATIO, -ARROW_HALF_WIDTH * 1.9),
                   self.point(1, 0),
                   self.point(1 - ARROW_HEAD_RATIO, ARROW_HALF_WIDTH * 1.9),
                   self.point(1 - ARROW_HEAD_RATIO, ARROW_HALF_WIDTH),
                   self.point(0, ARROW_HALF_WIDTH)]
        cv2.polylines(frame, [np.array(outline, np.int32)], True,
                      (255, 255, 255), 2, cv2.LINE_AA)

        # značka aktuálnej polohy prsta na šípke
        if self.active and self.tip_t is not None:
            p = self.point(clamp(self.tip_t))
            cv2.circle(frame, tuple(int(v) for v in p), 6, (255, 255, 255), -1, cv2.LINE_AA)


# ----------------------------------------------------------------------
# RESET TLAČIDLO (aktivuje sa podržaním ukazováka)
# ----------------------------------------------------------------------
class ResetButton:
    def __init__(self, frame_w):
        w, h = BUTTON_SIZE
        self.x1 = frame_w - BUTTON_MARGIN - w
        self.y1 = BUTTON_MARGIN
        self.x2, self.y2 = self.x1 + w, self.y1 + h
        self.hover_time = 0.0

    def inside(self, xy):
        return (xy is not None and
                self.x1 <= xy[0] <= self.x2 and self.y1 <= xy[1] <= self.y2)

    def update(self, finger_xy, dt):
        """Vráti True, keď sa tlačidlo aktivovalo."""
        if self.inside(finger_xy):
            self.hover_time += dt
            if self.hover_time >= BUTTON_DWELL_S:
                self.hover_time = 0.0
                return True
        else:
            self.hover_time = max(0.0, self.hover_time - dt * 2)
        return False

    def draw(self, frame):
        k = clamp(self.hover_time / BUTTON_DWELL_S)
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (50, 50, 50), -1)
        # ukazovateľ držania
        cv2.rectangle(frame, (self.x1, self.y1),
                      (self.x1 + int((self.x2 - self.x1) * k), self.y2), (0, 170, 255), -1)
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (255, 255, 255), 2)
        cv2.putText(frame, "RESET", (self.x1 + 42, self.y1 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


# ----------------------------------------------------------------------
# VYKRESLENIE RUKY (všetky články prstov + dlaň)
# ----------------------------------------------------------------------
def draw_hand(frame, landmarks):
    """landmarks = zoznam 21 bodov (normalizované x, y)."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # dlaň - vyplnený polygon z bodov: zápästie + základy prstov
    palm = np.array([pts[i] for i in (0, 1, 5, 9, 13, 17)], np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [palm], (255, 180, 80))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # kostra ruky: spojenia medzi článkami a 21 kĺbov
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(frame, p, 5, (0, 0, 255) if i in (4, 8, 12, 16, 20) else (0, 200, 0),
                   -1, cv2.LINE_AA)

    # zvýraznená špička ukazováka (bod 8)
    cv2.circle(frame, pts[8], 11, (0, 255, 255), 3, cv2.LINE_AA)


# ----------------------------------------------------------------------
# HLAVNÁ SLUČKA
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError("Nepodarilo sa otvoriť webkameru.")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Nepodarilo sa prečítať snímok z kamery.")
    h, w = frame.shape[:2]

    arrow = GuidanceArrow(w, h)
    button = ResetButton(w)

    # Model načítame cez Python do pamäte, lebo MediaPipe nevie otvoriť
    # cesty s diakritikou (napr. "Škola", "Výberový") na Windowse.
    with open(ensure_model(), "rb") as f:
        model_bytes = f.read()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    start_time = time.perf_counter()
    prev_time = start_time
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                       # zrkadlový obraz

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-3))

        # --- hand tracking ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((now - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        finger_xy = None
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]               # zoznam 21 bodov
            draw_hand(frame, hand)
            tip = hand[8]                                 # špička ukazováka
            finger_xy = (tip.x * w, tip.y * h)

        # --- logika ---
        if button.update(finger_xy, dt):
            arrow.reset()
        arrow.update(finger_xy, dt)

        # --- vykreslenie ---
        arrow.draw(frame)
        button.draw(frame)

        ratio = arrow.speed / TARGET_SPEED if TARGET_SPEED else 0
        if arrow.active:
            if ratio > SPEED_OK_HIGH:
                msg, col = "PRILIS RYCHLO", (0, 0, 255)
            elif ratio < SPEED_OK_LOW and arrow.speed > 0.02:
                msg, col = "PRILIS POMALY", (0, 200, 255)
            elif arrow.speed > 0.02:
                msg, col = "SPRAVNE", (0, 220, 0)
            else:
                msg, col = "ZACNI POHYB V SMERE SIPKY", (255, 255, 255)
        else:
            msg, col = "Daj ukazovak na zaciatok sipky", (255, 255, 255)
        if arrow.progress >= 0.99:
            avg = np.mean(arrow.speed_samples) / TARGET_SPEED if arrow.speed_samples else 0
            msg, col = f"HOTOVO (priemer {avg*100:.0f}% cielovej rychlosti)", (0, 220, 0)

        cv2.putText(frame, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Rychlost: {arrow.speed:5.2f} (ciel {TARGET_SPEED:.2f}) dlzky/s",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:4.1f}  |  q = koniec, r = reset, d = otocit sipku",
                    (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

        cv2.imshow("Masazne navadzanie - hand tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            arrow.reset()
        elif key == ord('d'):
            arrow.rotate(45)

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
