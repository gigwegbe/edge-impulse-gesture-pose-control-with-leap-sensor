# src/live_pose.py
# Live pose classification with the Edge Impulse float32 TFLite model.
# Optional Spotify control (macOS): --media
# Shows pose confidence bars, Spotify now playing + volume, and command history.
#
# Mapping:
#   fist       -> play / pause (toggle)
#   open_palm  -> next track
#   thumb      -> volume up    (repeats while held)
#   pinky      -> volume down  (repeats while held)
#   peace      -> no action (kept in the model)
#
# Usage: python src/live_pose.py --media
import time, argparse, threading, collections, subprocess
import numpy as np
import cv2
import leap
import pose_detect
from pose_features import pose_features
from pose_detect import draw_view, to_front, to_top, distance_state, extended

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

# Edge Impulse orders classes alphabetically.
DEFAULT_LABELS = sorted(["fist", "none", "open_palm", "peace", "pinky", "thumb"])

VOL_STEP = 8   # Spotify volume change per step (0-100)

# pose -> (Spotify AppleScript command, label shown on screen, repeats while held)
ACTIONS = {
    "fist": ("playpause", "PLAY/PAUSE", False),
    "open_palm": ("next track", "NEXT", False),
    "thumb": (f"set sound volume to (sound volume + {VOL_STEP})", "VOL +", True),
    "pinky": (f"set sound volume to (sound volume - {VOL_STEP})", "VOL -", True),
}
GUIDE = "fist PLAY/PAUSE   palm NEXT   thumb VOL+   pinky VOL-"

SIDE_W = 340
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------- Spotify ----------------
def spotify(cmd):
    try:
        subprocess.run(["osascript", "-e", f'tell application "Spotify" to {cmd}'],
                       check=True, capture_output=True, timeout=3)
        return True
    except Exception as e:
        print(f"Spotify command failed: {e}")
        return False

NOW_PLAYING_SCRIPT = '''
if application "Spotify" is running then
    tell application "Spotify"
        set s to (player state as string)
        set v to (sound volume as string)
        try
            set t to name of current track
            set a to artist of current track
            set p to player position
            set d to duration of current track
            return s & "|" & t & "|" & a & "|" & (p as string) & "|" & (d as string) & "|" & v
        on error
            return s & "||||" & "|" & v
        end try
    end tell
else
    return "off|||||"
end if
'''

def _num(s, default=0.0):
    try:
        return float(s.replace(",", "."))
    except (ValueError, AttributeError):
        return default

def _parse_now_playing(out):
    state, track, artist, pos, dur, vol = (out.split("|") + [""] * 6)[:6]
    return {"state": state or "unknown", "track": track, "artist": artist,
            "pos": _num(pos), "dur": _num(dur) / 1000.0,   # ms -> s
            "vol": _num(vol, -1)}

class SpotifyWatcher(threading.Thread):
    """Polls Spotify once a second in the background."""
    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.lock = threading.Lock()
        self.info = {"state": "unknown", "track": "", "artist": "", "pos": 0.0, "dur": 0.0, "vol": -1}

    def _poll(self):
        try:
            out = subprocess.run(["osascript", "-e", NOW_PLAYING_SCRIPT],
                                 capture_output=True, text=True, timeout=3).stdout.strip()
            info = _parse_now_playing(out)
            with self.lock:
                self.info = info
        except Exception:
            pass

    def run(self):
        while True:
            self._poll()
            time.sleep(self.interval)

    def latest(self):
        with self.lock:
            return dict(self.info)

    def nudge(self):
        # refresh soon after a command
        threading.Thread(target=lambda: (time.sleep(0.3), self._poll()), daemon=True).start()

# ---------------- trigger ----------------
class Trigger:
    """Turns confirmed poses into commands.

    - One-shot poses (fist, open_palm) fire once, then need a re-arm: the hand
      goes to "none" or leaves view. A global cooldown stops quick double fires.
    - Repeating poses (thumb, pinky) fire once, then again every `repeat` seconds
      for as long as the same pose stays confirmed.
    """
    def __init__(self, cooldown, repeat):
        self.cooldown = cooldown
        self.repeat = repeat
        self.armed = True
        self.last_fire = -1e9
        self.held = None        # pose currently repeating

    def update(self, confirmed, now, repeating=()):
        if confirmed in (None, "none"):
            self.armed, self.held = True, None
            return None
        if self.held is not None:
            if confirmed == self.held and now - self.last_fire >= self.repeat:
                self.last_fire = now
                return confirmed
            if confirmed != self.held:
                self.held = None   # switched pose without relaxing: wait for re-arm
            return None
        if not self.armed or now - self.last_fire < self.cooldown:
            return None
        self.armed = False
        self.last_fire = now
        if confirmed in repeating:
            self.held = confirmed
        return confirmed

    def state(self):
        if self.held:
            return f"REPEATING {self.held}"
        return "ARMED" if self.armed else "waiting for none / hand away"

# ---------------- pose confirmation ----------------
class PoseConfirmer:
    """Decides when a pose is "confirmed" (held long enough).

    - A pose must be the top label for `hold` seconds.
    - Brief flickers to another label shorter than `grace` seconds are ignored,
      so one shaky frame no longer restarts the hold.
    - "none" (relaxed hand) confirms after only `none_hold` seconds, so a quick
      relax between two identical poses re-arms the trigger.
    """
    def __init__(self, hold, none_hold, grace):
        self.hold, self.none_hold, self.grace = hold, none_hold, grace
        self.reset()

    def reset(self):
        self.candidate, self.since = None, 0.0
        self.other, self.other_since = None, 0.0
        self.confirmed = None

    def required(self):
        return self.none_hold if self.candidate == "none" else self.hold

    def update(self, label, now):
        if self.candidate is None:
            self.candidate, self.since = label, now
        if label == self.candidate:
            self.other = None
        else:
            if label != self.other:
                self.other, self.other_since = label, now
            if now - self.other_since >= self.grace:
                # the new label has lasted past the grace period: switch to it,
                # counting its hold from when it actually started
                self.candidate, self.since = label, self.other_since
                self.other = None
        if self.candidate != self.confirmed and now - self.since >= self.required():
            self.confirmed = self.candidate
            return self.confirmed
        return None

    def progress(self, now):
        if self.candidate in (None, self.confirmed):
            return 0.0
        return min((now - self.since) / max(self.required(), 1e-6), 1.0)

# ---------------- Leap ----------------
class Listener(leap.Listener):
    def __init__(self):
        self.lock = threading.Lock()
        self.snap = None

    def on_tracking_event(self, event):
        if not event.hands:
            with self.lock:
                self.snap = None
            return
        h = event.hands[0]
        bones = [[((b.prev_joint.x, b.prev_joint.y, b.prev_joint.z),
                   (b.next_joint.x, b.next_joint.y, b.next_joint.z))
                  for b in (d.metacarpal, d.proximal, d.intermediate, d.distal)]
                 for d in h.digits]
        p = h.palm.position
        with self.lock:
            self.snap = {"bones": bones, "ext": extended(h), "palm": (p.x, p.y, p.z),
                         "dist": distance_state(p.y), "feats": pose_features(h)}

    def latest(self):
        with self.lock:
            return self.snap

# ---------------- drawing helpers ----------------
def fit_text(text, max_px, scale, thick=1):
    if not text:
        return ""
    while cv2.getTextSize(text, FONT, scale, thick)[0][0] > max_px and len(text) > 3:
        text = text[:-4] + "..."
    return text

def mmss(sec):
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"

def draw_spotify(img, x0, y0, w, info, media_on):
    cv2.putText(img, "SPOTIFY", (x0 + 10, y0 + 25), FONT, 0.6, (30, 215, 96), 2)
    if not media_on:
        cv2.putText(img, "control off (run with --media)", (x0 + 10, y0 + 55),
                    FONT, 0.45, (150, 150, 150), 1)
        return
    state = info["state"]
    if state == "off":
        cv2.putText(img, "Spotify is not running", (x0 + 10, y0 + 55),
                    FONT, 0.5, (150, 150, 150), 1)
        return
    col = {"playing": (30, 215, 96), "paused": (0, 170, 255)}.get(state, (180, 180, 180))
    cv2.putText(img, state.upper(), (x0 + w - 110, y0 + 25), FONT, 0.55, col, 2)
    cv2.putText(img, fit_text(info["track"], w - 20, 0.6, 2), (x0 + 10, y0 + 58),
                FONT, 0.6, (255, 255, 255), 2)
    cv2.putText(img, fit_text(info["artist"], w - 20, 0.5), (x0 + 10, y0 + 82),
                FONT, 0.5, (190, 190, 190), 1)
    # track progress
    by = y0 + 100
    cv2.rectangle(img, (x0 + 10, by), (x0 + w - 10, by + 6), (70, 70, 70), -1)
    if info["dur"] > 0:
        frac = min(info["pos"] / info["dur"], 1.0)
        cv2.rectangle(img, (x0 + 10, by), (x0 + 10 + int((w - 20) * frac), by + 6), col, -1)
        cv2.putText(img, mmss(info["pos"]), (x0 + 10, by + 24), FONT, 0.4, (160, 160, 160), 1)
        cv2.putText(img, mmss(info["dur"]), (x0 + w - 45, by + 24), FONT, 0.4, (160, 160, 160), 1)
    # volume
    vy = y0 + 140
    vol = info.get("vol", -1)
    cv2.putText(img, "VOL", (x0 + 10, vy + 7), FONT, 0.45, (160, 160, 160), 1)
    cv2.rectangle(img, (x0 + 50, vy), (x0 + w - 60, vy + 8), (70, 70, 70), -1)
    if vol >= 0:
        cv2.rectangle(img, (x0 + 50, vy), (x0 + 50 + int((w - 110) * min(vol, 100) / 100), vy + 8),
                      (0, 200, 255), -1)
        cv2.putText(img, f"{int(vol)}", (x0 + w - 50, vy + 8), FONT, 0.5, (220, 220, 220), 1)

def draw_history(img, x0, y0, w, h, history):
    cv2.putText(img, "COMMAND HISTORY", (x0 + 10, y0 + 25), FONT, 0.6, (0, 220, 255), 2)
    cv2.putText(img, f"{len(history)} shown", (x0 + w - 90, y0 + 25), FONT, 0.4, (130, 130, 130), 1)
    if not history:
        cv2.putText(img, "no commands yet", (x0 + 10, y0 + 55), FONT, 0.5, (130, 130, 130), 1)
        return
    row_h = 30
    for i, item in enumerate(reversed(history)):
        y = y0 + 55 + i * row_h
        if y > y0 + h - 10:
            break
        t, pose, name, status, count = item
        newest = i == 0
        col = {"ok": (80, 220, 80), "failed": (60, 60, 255), "dry": (180, 180, 180)}[status]
        if newest:
            cv2.rectangle(img, (x0 + 5, y - 20), (x0 + w - 5, y + 8), (40, 40, 40), -1)
        cv2.putText(img, time.strftime("%H:%M:%S", time.localtime(t)), (x0 + 10, y),
                    FONT, 0.45, (150, 150, 150), 1)
        label = name + (f" x{count}" if count > 1 else "")
        cv2.putText(img, label, (x0 + 95, y), FONT, 0.55, col, 2 if newest else 1)
        cv2.putText(img, pose, (x0 + 215, y), FONT, 0.45, (170, 170, 170), 1)
        tag = {"ok": "", "failed": "failed", "dry": "dry"}[status]
        if tag:
            cv2.putText(img, tag, (x0 + w - 45, y), FONT, 0.4, col, 1)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pose_float32.tflite")
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS),
                    help="model classes in Edge Impulse order (alphabetical)")
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="min confidence for a pose (lower = more sensitive)")
    ap.add_argument("--hold", type=float, default=None,
                    help="seconds a pose must be held (default 0.4, or 0.6 with --media)")
    ap.add_argument("--none-hold", type=float, default=0.2,
                    help="seconds of relaxed hand needed to re-arm")
    ap.add_argument("--grace", type=float, default=0.15,
                    help="ignore flickers shorter than this while holding a pose")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--near", type=float, default=120)
    ap.add_argument("--far", type=float, default=450)
    ap.add_argument("--media", action="store_true", help="control Spotify")
    ap.add_argument("--cooldown", type=float, default=1.0, help="seconds between different commands")
    ap.add_argument("--repeat", type=float, default=0.5, help="seconds between volume steps while held")
    ap.add_argument("--history", type=int, default=12, help="commands kept in the list")
    args = ap.parse_args()
    if args.hold is None:
        args.hold = 0.6 if args.media else 0.4
    pose_detect.NEAR_MM, pose_detect.FAR_MM = args.near, args.far
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]

    interp = Interpreter(model_path=args.model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    n_out = int(np.prod(out["shape"]))
    if n_out != len(labels):
        raise SystemExit(
            f"The model has {n_out} classes but {len(labels)} labels were given ({', '.join(labels)}).\n"
            "Retrain in Edge Impulse with the new poses and download the new model, or pass the\n"
            "old model's classes, e.g. --labels fist,none,open_palm,peace")
    missing = [p for p in ACTIONS if p not in labels]
    if missing:
        print(f"Note: no {', '.join(missing)} class in this model, so those commands are off.")
    repeating = {p for p, (_, _, rep) in ACTIONS.items() if rep}

    smooth = collections.deque(maxlen=args.smooth)
    history = collections.deque(maxlen=args.history)
    confirmer = PoseConfirmer(args.hold, args.none_hold, args.grace)
    trigger = Trigger(args.cooldown, args.repeat)
    why = ""
    last_action, last_action_t = None, 0.0

    watcher = SpotifyWatcher()
    if args.media:
        watcher.start()

    listener = Listener()
    conn = leap.Connection()
    conn.add_listener(listener)
    with conn.open():
        conn.set_tracking_mode(leap.TrackingMode.Desktop)
        mode = "Spotify control ON" if args.media else "display only"
        print(f"Show a pose ({mode}, range {args.near:.0f}-{args.far:.0f} mm). Press Q to quit.")
        print(GUIDE)
        while True:
            now = time.time()
            snap = listener.latest()
            probs = None

            top_name, top_p = None, 0.0
            if snap and snap["dist"] == "good":
                x = np.array([snap["feats"]], dtype=np.float32)
                interp.set_tensor(inp["index"], x)
                interp.invoke()
                smooth.append(interp.get_tensor(out["index"])[0])
                probs = np.mean(smooth, axis=0)
                top = int(np.argmax(probs))
                top_name, top_p = labels[top], float(probs[top])
                label = top_name if top_p >= args.threshold else "none"
                newly = confirmer.update(label, now)
                if newly and newly != "none":
                    print(f"POSE: {newly}  ({top_p:.2f})")
            else:
                # no hand, or out of range: start fresh, and count it as a re-arm
                smooth.clear()
                if confirmer.confirmed not in (None, "none"):
                    print("-- hand lost" if snap is None else "-- out of range")
                confirmer.reset()
            confirmed = confirmer.confirmed

            fired = trigger.update(confirmed, now, repeating)

            # why is (or isn't) something happening? shown on screen
            if snap is None:
                why = "no hand"
            elif snap["dist"] != "good":
                why = "TOO CLOSE: move your hand away" if snap["dist"] == "too close" \
                    else "TOO FAR: move your hand closer"
            elif top_name and top_name != "none" and top_p < args.threshold:
                why = f"unsure: {top_name} {top_p:.2f} < {args.threshold:.2f}, make the pose clearer"
            elif confirmer.candidate not in (None, "none", confirmer.confirmed):
                why = f"holding {confirmer.candidate}..."
            elif trigger.held:
                why = f"repeating {ACTIONS[trigger.held][1]} while held"
            elif confirmed not in (None, "none") and not fired and trigger.held is None \
                    and not trigger.armed:
                why = "done: relax your hand (or move away) to fire again"
            elif confirmed not in (None, "none") and trigger.armed \
                    and now - trigger.last_fire < args.cooldown:
                why = f"cooldown {args.cooldown - (now - trigger.last_fire):.1f} s"
            else:
                why = "ready"
            if fired and fired in ACTIONS:
                cmd, name, _ = ACTIONS[fired]
                if args.media:
                    ok = spotify(cmd)
                    status = "ok" if ok else "failed"
                    watcher.nudge()
                    print(f"ACTION: {name}" + ("" if ok else " (failed)"))
                else:
                    status = "dry"
                    print(f"(would send {name})")
                # merge repeated volume steps into one history row
                if history and history[-1][2] == name and fired in repeating and now - history[-1][0] < 2 * args.repeat + 0.5:
                    t0, p0, n0, s0, c0 = history[-1]
                    history[-1] = (now, p0, n0, status, c0 + 1)
                else:
                    history.append((now, fired, name, status, 1))
                last_action, last_action_t = name, now

            # ----- left: pose panel + skeleton views -----
            panel = np.full((80 + 27 * len(labels) + 70, 840, 3), 15, np.uint8)
            cv2.putText(panel, f"Confirmed: {confirmed or '-'}", (10, 40), FONT, 1.0, (255, 255, 255), 2)
            if snap:
                ok = snap["dist"] == "good"
                msg = f"{snap['palm'][1] / 10:.0f} cm  " + ("in range" if ok else snap["dist"].upper())
                cv2.putText(panel, msg, (460, 40), FONT, 0.8, (80, 220, 80) if ok else (0, 140, 255), 2)
            for i, name in enumerate(labels):
                y = 70 + i * 27
                v = float(probs[i]) if probs is not None else 0.0
                act = ACTIONS.get(name, (None, "", False))[1]
                cv2.putText(panel, name, (10, y + 15), FONT, 0.6, (200, 200, 200), 1)
                cv2.rectangle(panel, (130, y), (130 + int(450 * v), y + 18),
                              (80, 220, 80) if v >= args.threshold else (90, 90, 90), -1)
                cv2.putText(panel, f"{v:.2f}", (595, y + 15), FONT, 0.6, (200, 200, 200), 1)
                if act:
                    cv2.putText(panel, act, (660, y + 15), FONT, 0.5, (130, 130, 130), 1)
            ty = panel.shape[0] - 14
            # hold progress bar + reason line
            py = ty - 42
            prog = confirmer.progress(now)
            cv2.rectangle(panel, (10, py), (330, py + 12), (60, 60, 60), -1)
            if prog > 0:
                cv2.rectangle(panel, (10, py), (10 + int(320 * prog), py + 12), (0, 200, 255), -1)
            why_col = (80, 220, 80) if why in ("ready",) or why.startswith("holding") else (0, 170, 255)
            cv2.putText(panel, why, (345, py + 11), FONT, 0.55, why_col, 1)
            cv2.putText(panel, f"Trigger: {trigger.state()}", (10, ty), FONT, 0.6,
                        (80, 220, 80) if trigger.armed or trigger.held else (0, 170, 255), 1)
            if last_action and now - last_action_t < 1.5:
                cv2.putText(panel, last_action, (560, ty), FONT, 1.0, (0, 220, 255), 3)

            views = np.hstack([draw_view(snap, to_front, "FRONT", front=True),
                               draw_view(snap, to_top, "TOP")])
            left = np.vstack([panel, views])

            # ----- right: Spotify + command history -----
            side = np.full((left.shape[0], SIDE_W, 3), 22, np.uint8)
            cv2.line(side, (0, 0), (0, side.shape[0]), (60, 60, 60), 2)
            draw_spotify(side, 0, 0, SIDE_W, watcher.latest(), args.media)
            cv2.line(side, (10, 170), (SIDE_W - 10, 170), (60, 60, 60), 1)
            draw_history(side, 0, 175, SIDE_W, side.shape[0] - 205, history)
            cv2.putText(side, GUIDE.replace("   ", "  "), (10, side.shape[0] - 12),
                        FONT, 0.38, (140, 140, 140), 1)

            cv2.imshow("Live poses", np.hstack([left, side]))
            if cv2.waitKey(15) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
