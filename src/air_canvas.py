"""Air writing canvas: write in the air above the Leap, see it on screen.

  python src/air_canvas.py --pen key      # hold SPACE or the mouse button to write
  python src/air_canvas.py                 # pinch thumb + index to write
  python src/air_canvas.py --pen push      # push the index finger forward to write
  python src/air_canvas.py --fullscreen    # e.g. on the TV / monitor
  python src/air_canvas.py --pen key --debug   # log + plot the fingertip path

How it works
  - The pen is the point between your thumb and index fingertips (pinch mode)
    or your index fingertip (push mode), in the FRONT plane: left/right and
    height above the sensor, like writing on an invisible whiteboard.
  - Pinch (or push past the plane) = pen down. Open (or pull back) = pen up.
  - Key mode: the index fingertip only moves; hold SPACE (or the mouse /
    trackpad button inside the window) with your other hand to put ink down.
    This separates "is the tracking messy?" from "is the pen gesture messy?".
  - Lift the pen for --word-pause seconds and the word is "done": it is saved
    to out/words/ as a clean black-on-white PNG (what TrOCR will read later)
    plus the raw strokes as JSON, and shown in the corner preview.

Keys: C = clear   U / Backspace = undo stroke   S = save now   F = fullscreen
      Q / Esc = quit.  Gesture: hold a fist for ~0.8 s to clear.
"""

import argparse
import json
import math
import pathlib
import time

import cv2
import numpy as np

import leap
from leap_features import open_leap

ROOT = pathlib.Path(__file__).resolve().parent.parent

INK = (20, 20, 20)
PAPER = (245, 243, 238)
GUIDE = (215, 212, 205)
CURSOR_UP = (40, 140, 230)
CURSOR_DOWN = (40, 40, 220)
TEXT = (90, 90, 90)


# --- smoothing ---------------------------------------------------------------
class OneEuro:
    """One Euro filter (Casiez et al.): smooth when slow, responsive when fast."""

    def __init__(self, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(1e-3, t - self.t)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x

    def reset(self):
        self.x = self.dx = self.t = None


# --- pen ---------------------------------------------------------------------
class Pen:
    """Decides pen up/down with hysteresis and gives the pen position in mm."""

    def __init__(self, mode, pinch_down=0.80, pinch_up=0.55, push_z=-10.0, push_gap=15.0):
        self.mode = mode
        self.pinch_down, self.pinch_up = pinch_down, pinch_up
        self.push_z, self.push_gap = push_z, push_gap
        self.down = False
        self.held = False       # key mode: set from SPACE / mouse button

    def update(self, row, skel):
        """Returns (raw pen position (x, y, z) in mm or None, pen down)."""
        if not skel:
            self.down = False
            return None, False
        thumb, index = skel["digits"][0][-1], skel["digits"][1][-1]
        if self.mode == "key":
            self.down = self.held
            return (index[0], index[1], index[2]), self.down
        if self.mode == "pinch":
            pinch = row[9]
            if self.down and pinch < self.pinch_up:
                self.down = False
            elif not self.down and pinch > self.pinch_down:
                self.down = True
            pos = ((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2, (thumb[2] + index[2]) / 2)
        else:  # push: index tip moves toward the screen (-z) past a plane
            z = index[2]
            if self.down and z > self.push_z + self.push_gap:
                self.down = False
            elif not self.down and z < self.push_z:
                self.down = True
            pos = (index[0], index[1], index[2])
        return pos, self.down


# --- strokes -> image ----------------------------------------------------------
def render_word(strokes, height=128, pad=0.25, thickness=None):
    """Render strokes (lists of (x_mm, y_mm)) as black ink on white, cropped to
    the writing with padding, keeping the aspect ratio. This is the image
    TrOCR will get: it was trained on scanned handwriting, so it should look
    like ink on paper."""
    pts = [p for s in strokes for p in s]
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    w_mm = max(xs) - min(xs)
    h_mm = max(ys) - min(ys)
    size_mm = max(h_mm, w_mm / 4, 20.0)           # don't blow up tiny scribbles
    scale = height * (1 - 2 * pad) / size_mm
    W = int(w_mm * scale + 2 * pad * height) + 1
    H = height
    img = np.full((H, max(W, H), 3), 255, np.uint8)
    off_x = (img.shape[1] - w_mm * scale) / 2
    off_y = (H - h_mm * scale) / 2
    th = thickness or max(2, height // 22)

    def to_px(p):  # y up in mm -> y down in pixels
        return (int(off_x + (p[0] - min(xs)) * scale),
                int(off_y + (max(ys) - p[1]) * scale))

    for s in strokes:
        px = [to_px(p) for p in s]
        if len(px) == 1:
            cv2.circle(img, px[0], th // 2, (0, 0, 0), -1, cv2.LINE_AA)
        for a, b in zip(px, px[1:]):
            cv2.line(img, a, b, (0, 0, 0), th, cv2.LINE_AA)
    return img


# --- app ---------------------------------------------------------------------
class Canvas:
    def __init__(self, args):
        self.a = args
        self.W, self.H = args.width, args.height
        self.strokes = []        # finished strokes, mm
        self.current = None      # stroke in progress, mm
        self.current_t = []      # timestamps of the points in `current`
        self.last_pen_up = None
        self.word_saved = True
        self.preview = None
        self.status = ""
        self.fist_since = None
        self.trail = []
        self.out_dir = pathlib.Path(args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fx, self.fy = OneEuro(args.min_cutoff, args.beta), OneEuro(args.min_cutoff, args.beta)
        self.key_hint = "hold SPACE or mouse button to write"
        self.debug = None
        if args.debug:
            from air_debug import DebugLog
            self.debug = DebugLog(self.out_dir, args.pen)

    def _trim(self, pts, ts):
        """Drop the first/last --trim-ms of a stroke: closing and opening the
        pinch moves the fingers, which leaves little hooks at the ends."""
        if not pts or self.a.trim_ms <= 0:
            return pts
        t0, t1, tr = ts[0], ts[-1], self.a.trim_ms / 1000.0
        if t1 - t0 < 3 * tr:          # short stroke or a dot: keep it as is
            return pts
        return [p for p, t in zip(pts, ts) if t0 + tr <= t <= t1 - tr] or pts

    # mm -> canvas pixels (front plane, y up)
    def to_px(self, p):
        x0, x1 = -self.a.area_w / 2, self.a.area_w / 2
        y0, y1 = self.a.area_y, self.a.area_y + self.a.area_h
        u = (p[0] - x0) / (x1 - x0)
        v = (p[1] - y0) / (y1 - y0)
        return int(u * (self.W - 1)), int((1 - v) * (self.H - 1))

    def clear(self, why):
        if self.strokes and self.a.save_on_clear and not self.word_saved:
            self.save_word(auto=False)
        self.strokes, self.current = [], None
        self.word_saved, self.status = True, f"cleared ({why})"

    def undo(self):
        if self.current:
            self.current = None
        elif self.strokes:
            self.strokes.pop()
        self.word_saved = not self.strokes
        self.status = "undo"

    def save_word(self, auto):
        if not self.strokes:
            self.status = "nothing to save"
            return
        img = render_word(self.strokes, height=self.a.export_height)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        png = self.out_dir / f"word-{stamp}.png"
        cv2.imwrite(str(png), img)
        (self.out_dir / f"word-{stamp}.json").write_text(json.dumps(
            {"strokes_mm": self.strokes, "pen": self.a.pen, "time": stamp}))
        cv2.imwrite(str(self.out_dir / "last_word.png"), img)
        self.preview, self.word_saved = img, True
        self.status = f"{'word done' if auto else 'saved'}: {png.name}"
        print(self.status)
        if self.debug:
            summ = self.debug.word_done(stamp)
            if summ:
                for i, st in enumerate(summ["strokes"]):
                    print(f"  stroke {i + 1}: {st['duration_s']:.2f} s, jitter x {st['jitter_x_mm']:.2f} "
                          f"y {st['jitter_y_mm']:.2f} mm, edge jump start {st['start_jump_mm']:.1f} "
                          f"end {st['end_jump_mm']:.1f} mm")
                if summ["hand_lost_while_down"]:
                    print(f"  hand lost while pen down: {summ['hand_lost_while_down']} frames")
                print(f"  debug plot: {self.debug.dir / ('word-' + stamp + '-debug.png')}")

    def step(self, row, skel, now, leap_frame=0):
        # pen
        raw, down = self.pen.update(row, skel)
        if raw is not None:
            pos = (self.fx(raw[0], now), self.fy(raw[1], now))
        else:
            pos = None
            self.fx.reset()
            self.fy.reset()
        if self.debug:
            speed = math.sqrt(row[3] ** 2 + row[4] ** 2 + row[5] ** 2)
            self.debug.log(now, leap_frame, raw, pos, down, row, speed)

        if down and pos is not None:
            if self.current is None:
                self.current, self.current_t = [], []
            if not self.current or math.dist(self.current[-1], pos) >= self.a.min_step:
                self.current.append((round(pos[0], 2), round(pos[1], 2)))
                self.current_t.append(now)
            self.word_saved = False
        elif self.current is not None:
            stroke = self._trim(self.current, self.current_t)
            if len(stroke) >= 2 or (stroke and self.a.keep_dots):
                self.strokes.append(stroke)
            self.current = None
            self.last_pen_up = now

        # word done after a pause with the pen up
        if (not down and not self.word_saved and self.strokes and self.last_pen_up
                and now - self.last_pen_up > self.a.word_pause):
            self.save_word(auto=True)
            if self.a.auto_clear:
                self.strokes = []

        # fist held = clear
        if skel and row[10] > 0.9 and not down:
            self.fist_since = self.fist_since or now
            if now - self.fist_since > self.a.fist_clear:
                self.clear("fist")
                self.fist_since = None
        else:
            self.fist_since = None
        return pos, down

    def draw(self, pos, down, row, now):
        img = np.full((self.H, self.W, 3), PAPER, np.uint8)
        # ruled guide lines, like a notebook
        for i in range(1, 4):
            y = i * self.H // 4
            cv2.line(img, (0, y), (self.W, y), GUIDE, 1, cv2.LINE_AA)
        th = self.a.ink
        for s in self.strokes + ([self.current] if self.current else []):
            px = [self.to_px(p) for p in s]
            if len(px) == 1:
                cv2.circle(img, px[0], th // 2, INK, -1, cv2.LINE_AA)
            for a, b in zip(px, px[1:]):
                cv2.line(img, a, b, INK, th, cv2.LINE_AA)

        # cursor
        if pos is not None:
            c = self.to_px(pos)
            if down:
                cv2.circle(img, c, th // 2 + 4, CURSOR_DOWN, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, c, 12, CURSOR_UP, 2, cv2.LINE_AA)
                if self.a.pen == "pinch":  # ring fills as you pinch
                    r = int(12 * min(1.0, row[9] / self.pen.pinch_down))
                    if r > 2:
                        cv2.circle(img, c, r, CURSOR_UP, -1, cv2.LINE_AA)

        # fist-clear progress
        if self.fist_since:
            p = min(1.0, (now - self.fist_since) / self.a.fist_clear)
            cv2.ellipse(img, (self.W - 40, 40), (22, 22), -90, 0, int(360 * p), (60, 60, 220), 4)
            cv2.putText(img, "clear", (self.W - 62, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)

        # word-done countdown
        if not down and not self.word_saved and self.strokes and self.last_pen_up:
            p = min(1.0, (now - self.last_pen_up) / self.a.word_pause)
            cv2.rectangle(img, (0, self.H - 5), (int(self.W * p), self.H), (60, 160, 60), -1)

        # preview of what TrOCR will see
        if self.preview is not None:
            ph = 90
            pw = min(int(self.preview.shape[1] * ph / self.preview.shape[0]), self.W // 3)
            thumb = cv2.resize(self.preview, (pw, ph), interpolation=cv2.INTER_AREA)
            x0, y0 = self.W - pw - 14, self.H - ph - 30
            img[y0:y0 + ph, x0:x0 + pw] = thumb
            cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph), (150, 150, 150), 1)
            label = "last word (TrOCR input)"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(img, label, (self.W - tw - 14, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)

        hand = "pen DOWN" if down else ("pen up" if pos is not None else "no hand")
        extra = f"  pinch {row[9]:.2f}" if self.a.pen == "pinch" and pos is not None else ""
        if self.a.pen == "key":
            extra = f"  ({self.key_hint})"
        lines = [f"{hand}{extra}   strokes {len(self.strokes)}",
                 "C clear  U undo  S save  F fullscreen  Q quit  |  fist = clear",
                 self.status]
        if self.debug:
            lines.append(self.debug.live_text())
        for i, t in enumerate(lines):
            cv2.putText(img, t, (14, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT, 1, cv2.LINE_AA)
        return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pen", choices=["pinch", "push", "key"], default="pinch",
                    help="key = index fingertip moves, hold SPACE / mouse button for ink")
    ap.add_argument("--debug", action="store_true",
                    help="log every frame and save a fingertip-path plot per word (out/words/debug)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--area-w", type=float, default=300, help="writing area width, mm")
    ap.add_argument("--area-h", type=float, default=170, help="writing area height, mm")
    ap.add_argument("--area-y", type=float, default=130, help="bottom of the writing area above the sensor, mm")
    ap.add_argument("--ink", type=int, default=9, help="on-screen ink thickness, px")
    ap.add_argument("--min-step", type=float, default=0.8, help="min mm between stored points")
    ap.add_argument("--trim-ms", type=int, default=None,
                    help="drop this much at each stroke end (default 60 for pinch, 0 otherwise)")
    ap.add_argument("--min-cutoff", type=float, default=1.2, help="One Euro smoothing: lower = smoother")
    ap.add_argument("--beta", type=float, default=0.02, help="One Euro: higher = less lag when fast")
    ap.add_argument("--word-pause", type=float, default=1.5, help="pen-up seconds that end a word")
    ap.add_argument("--auto-clear", action="store_true", help="clear the canvas after each word")
    ap.add_argument("--save-on-clear", action="store_true", help="save unsaved writing when clearing")
    ap.add_argument("--keep-dots", action="store_true", help="keep single-point strokes (dots on i/j)")
    ap.add_argument("--fist-clear", type=float, default=0.8, help="seconds of fist to clear")
    ap.add_argument("--export-height", type=int, default=128, help="height of saved word images, px")
    ap.add_argument("--out", default=str(ROOT / "out" / "words"))
    ap.add_argument("--fps", type=int, default=90)
    args = ap.parse_args()
    if args.trim_ms is None:
        args.trim_ms = 60 if args.pen == "pinch" else 0

    canvas = Canvas(args)
    canvas.pen = Pen(args.pen)
    win = "Air writing"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)

    # Key mode: real press/release events. OpenCV only reports key *presses*
    # (with auto-repeat), so holding SPACE needs pynput; the mouse button works
    # through OpenCV directly.
    held = {"mouse": False, "space": False, "toggle": False}
    kb = None
    if args.pen == "key":
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                held["mouse"] = True
            elif event == cv2.EVENT_LBUTTONUP:
                held["mouse"] = False
        try:
            cv2.setMouseCallback(win, on_mouse)
        except cv2.error as e:
            print(f"(mouse button pen not available: {e.__class__.__name__})")
        try:
            from pynput import keyboard

            def on_press(k):
                if k == keyboard.Key.space:
                    held["space"] = True

            def on_release(k):
                if k == keyboard.Key.space:
                    held["space"] = False

            kb = keyboard.Listener(on_press=on_press, on_release=on_release)
            kb.start()
            print("Key mode: hold SPACE (or the mouse button in the window) to write.")
            print("  If SPACE does nothing on macOS, allow your terminal under System Settings >"
                  " Privacy & Security > Input Monitoring, or use the mouse button.")
        except Exception as e:  # pynput missing or no permission
            canvas.key_hint = "SPACE toggles the pen, or hold the mouse button"
            print(f"Key mode: pynput not available ({e.__class__.__name__}); SPACE toggles the pen,"
                  " or hold the mouse button in the window. (pip install pynput for hold-to-write)")
    fullscreen = args.fullscreen
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"Pen mode: {args.pen}. " + {"pinch": "Pinch thumb and index to write.",
                                        "push": "Push your index finger forward to write.",
                                        "key": "Your index fingertip is the pen."}[args.pen])
    conn, listener = open_leap()
    with conn.open():
        conn.set_tracking_mode(leap.TrackingMode.Desktop)
        dt = 1.0 / args.fps
        t_next = time.monotonic()
        while True:
            now = time.monotonic()
            row, skel = listener.latest(), listener.skeleton()
            canvas.pen.held = held["mouse"] or held["space"] or held["toggle"]
            pos, down = canvas.step(row, skel, now, listener.frames)
            cv2.imshow(win, canvas.draw(pos, down, row, now))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and args.pen == "key" and kb is None:
                held["toggle"] = not held["toggle"]
            if key == ord("c"):
                canvas.clear("key")
            elif key in (ord("u"), 8, 127):
                canvas.undo()
            elif key == ord("s"):
                canvas.save_word(auto=False)
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            t_next += dt
            time.sleep(max(0.0, t_next - time.monotonic()))
    if kb is not None:
        kb.stop()
    if canvas.debug:
        canvas.debug.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
