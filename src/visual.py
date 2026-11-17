"""Optional OpenCV skeleton view for record.py and live_tflite.py (--visual).

Two panels, in Leap millimetres:
  FRONT  x (left/right) vs y (height)   - the plane your swipes move in
  TOP    x (left/right) vs z (depth)    - how far toward/away from the screen

Also shows a fading palm trail (last ~1.5 s), the palm speed, pinch/grab and
any status lines the caller passes in (gate state, last prediction, REC ...).

OpenCV windows must be drawn from the main thread (macOS), which is where
both scripts call show().
"""

import collections
import math

try:
    import cv2
    import numpy as np
except ImportError as e:  # only needed with --visual
    raise SystemExit(f"--visual needs OpenCV: pip install opencv-python ({e})")

W, H = 980, 520          # window size
PANEL_W = W // 2
X_RANGE = (-250, 250)    # mm, left/right
Y_RANGE = (0, 500)       # mm, height above the sensor
Z_RANGE = (-250, 250)    # mm, depth (+z toward you)
FINGER_COLORS = [(80, 180, 255), (90, 230, 120), (240, 200, 70), (230, 120, 220), (120, 120, 255)]
BG = (24, 24, 28)
GRID = (55, 55, 62)
TEXT = (225, 225, 230)
ACCENT = (60, 200, 255)


def _map(v, lo, hi, size, flip=False):
    t = (v - lo) / (hi - lo)
    t = min(max(t, 0.0), 1.0)
    return int((1 - t) * (size - 1)) if flip else int(t * (size - 1))


class SkeletonView:
    def __init__(self, title="Leap skeleton", trail_frames=75):
        self.title = title
        self.trail = collections.deque(maxlen=trail_frames)
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)

    # --- projections ------------------------------------------------------
    def _front(self, p):
        return _map(p[0], *X_RANGE, PANEL_W), _map(p[1], *Y_RANGE, H, flip=True)

    def _top(self, p):
        return PANEL_W + _map(p[0], *X_RANGE, PANEL_W), _map(p[2], *Z_RANGE, H)

    # --- drawing ----------------------------------------------------------
    def _panel(self, img, x0, label, ylabel):
        for i in range(1, 5):
            x = x0 + i * PANEL_W // 5
            cv2.line(img, (x, 0), (x, H), GRID, 1)
            y = i * H // 5
            cv2.line(img, (x0, y), (x0 + PANEL_W, y), GRID, 1)
        cv2.putText(img, label, (x0 + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1, cv2.LINE_AA)
        cv2.putText(img, ylabel, (x0 + 10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRID, 1, cv2.LINE_AA)

    def _hand(self, img, skel, proj):
        for f, joints in enumerate(skel["digits"]):
            pts = [proj(j) for j in joints]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, FINGER_COLORS[f % 5], 3, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, 4, (240, 240, 240), -1, cv2.LINE_AA)
        # knuckle line across the metacarpal tops
        knuckles = [proj(j[1]) for j in skel["digits"]]
        for a, b in zip(knuckles, knuckles[1:]):
            cv2.line(img, a, b, (150, 150, 160), 2, cv2.LINE_AA)
        cv2.circle(img, proj(skel["palm"]), 8, ACCENT, -1, cv2.LINE_AA)

    def _trail(self, img, proj):
        pts = list(self.trail)
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            fade = i / len(pts)
            c = tuple(int(v * fade) for v in ACCENT)
            cv2.line(img, proj(pts[i - 1]), proj(pts[i]), c, 2, cv2.LINE_AA)

    def show(self, skel, row=None, lines=(), progress=None, recording=False):
        """Draw one frame. Returns the key pressed (or -1)."""
        img = np.full((H, W, 3), BG, np.uint8)
        self._panel(img, 0, "FRONT  (x: left/right, y: height)", "swipe plane")
        self._panel(img, PANEL_W, "TOP  (x: left/right, z: depth)", "toward you is down")
        cv2.line(img, (PANEL_W, 0), (PANEL_W, H), (90, 90, 100), 2)

        self.trail.append(skel["palm"] if skel else None)
        self._trail(img, self._front)
        self._trail(img, self._top)
        if skel:
            self._hand(img, skel, self._front)
            self._hand(img, skel, self._top)

        info = []
        if row is not None and any(row):
            speed = math.sqrt(row[3] ** 2 + row[4] ** 2 + row[5] ** 2)
            info.append(f"speed {speed:5.0f} mm/s   pinch {row[9]:.2f}   grab {row[10]:.2f}")
            info.append(f"palm  x {row[0]:6.0f}  y {row[1]:6.0f}  z {row[2]:6.0f} mm"
                        + (f"   ({skel['side']} hand)" if skel else ""))
        else:
            info.append("no hand")
        info.extend(lines)
        for i, t in enumerate(info):
            cv2.putText(img, t, (12, 48 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT, 1, cv2.LINE_AA)

        if recording:
            cv2.circle(img, (W - 30, 26), 10, (60, 60, 230), -1, cv2.LINE_AA)
            cv2.putText(img, "REC", (W - 80, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 230), 2, cv2.LINE_AA)
        if progress is not None:
            cv2.rectangle(img, (0, H - 6), (int(W * progress), H), (60, 60, 230), -1)

        cv2.imshow(self.title, img)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyWindow(self.title)
