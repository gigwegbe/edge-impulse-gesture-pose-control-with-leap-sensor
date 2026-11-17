"""Control media with Leap gestures classified by an Edge Impulse model.

Same idea as the MediaPipe finger-count script, but the gesture comes from
your trained .eim model instead of hand-written rules.

  python src/media_control.py model.eim                 # send key presses
  python src/media_control.py model.eim --dry-run       # print only
  python src/media_control.py model.eim --backend playerctl
  python src/media_control.py model.eim --backend spotify   # macOS, Spotify app

Edit ACTIONS to match your labels. Unknown labels and "idle" do nothing.
"""

import argparse
import collections
import shutil
import subprocess
import time

import leap

from leap_features import N_AXES, open_leap

# label -> (key for pyautogui, playerctl command)
ACTIONS = {
    "swipe_right": ("right", ["playerctl", "next"]),
    "swipe_left": ("left", ["playerctl", "previous"]),
    "swipe_up": ("volumeup", ["playerctl", "volume", "0.1+"]),
    "swipe_down": ("volumedown", ["playerctl", "volume", "0.1-"]),
    "grab": ("space", ["playerctl", "play-pause"]),
}
REPEATABLE = {"swipe_up", "swipe_down"}  # volume can fire again while held

# label -> AppleScript for the Spotify desktop app on macOS. Works with Spotify
# in the background. Note: "previous track" restarts the current song if it has
# played for more than ~3 s; swipe left twice to go back a track.
SPOTIFY = {
    "swipe_right": "next track",
    "swipe_left": "previous track",
    "swipe_up": "set sound volume to (sound volume + 10)",
    "swipe_down": "set sound volume to (sound volume - 10)",
    "grab": "playpause",
}
BACKENDS = ["keys", "playerctl", "spotify"]


def action_name(backend, label):
    if backend == "spotify":
        return SPOTIFY.get(label, "")
    if backend == "playerctl":
        return " ".join(ACTIONS[label][1][1:])
    return ACTIONS[label][0]


class Debouncer:
    """Fire a label only when it is confident for `stable` classifications in
    a row, then wait `cooldown` seconds. Non-repeatable labels must go back to
    idle (or something else) before they can fire again."""

    def __init__(self, threshold, stable, cooldown):
        self.threshold, self.cooldown = threshold, cooldown
        self.recent = collections.deque(maxlen=stable)
        self.last_fire_t = 0.0
        self.last_fired = None

    def update(self, scores):
        best = max(scores, key=scores.get)
        label = best if scores[best] >= self.threshold else None
        self.recent.append(label)

        if label is None or label == "idle" or label not in ACTIONS:
            if label != self.last_fired:
                self.last_fired = None  # re-arm once the gesture is released
            return None
        if len(self.recent) < self.recent.maxlen or len(set(self.recent)) != 1:
            return None
        if time.monotonic() - self.last_fire_t < self.cooldown:
            return None
        if label == self.last_fired and label not in REPEATABLE:
            return None

        self.last_fire_t = time.monotonic()
        self.last_fired = label
        self.recent.clear()
        return label


def make_sender(backend, dry_run):
    if dry_run:
        return lambda label: None
    if backend == "spotify":
        if not shutil.which("osascript"):
            raise SystemExit("The spotify backend needs macOS (osascript).")

        def spotify(label):
            if label not in SPOTIFY:
                return
            script = ('if application "Spotify" is running then\n'
                      f'  tell application "Spotify" to {SPOTIFY[label]}\n'
                      'end if')
            # Popen so the gesture loop never waits for AppleScript.
            subprocess.Popen(["osascript", "-e", script],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return spotify
    if backend == "playerctl":
        if not shutil.which("playerctl"):
            raise SystemExit("playerctl not found: sudo apt install playerctl")
        return lambda label: subprocess.run(ACTIONS[label][1], check=False)
    import pyautogui  # needs an X11 session
    return lambda label: pyautogui.press(ACTIONS[label][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--backend", choices=BACKENDS, default="keys")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--stable", type=int, default=3, help="confident results in a row before firing")
    ap.add_argument("--cooldown", type=float, default=0.6, help="seconds between actions")
    ap.add_argument("--every-ms", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    send = make_sender(args.backend, args.dry_run)
    deb = Debouncer(args.threshold, args.stable, args.cooldown)

    from edge_impulse_linux.runner import ImpulseRunner  # only needed for .eim

    runner = ImpulseRunner(args.model)
    try:
        mp = runner.init()["model_parameters"]
        interval_ms = mp["interval_ms"]
        window = mp["input_features_count"] // N_AXES
        missing = [l for l in mp["labels"] if l != "idle" and l not in ACTIONS]
        if missing:
            print("No action mapped for:", missing)

        buf = collections.deque(maxlen=window)
        conn, listener = open_leap()
        with conn.open():
            conn.set_tracking_mode(leap.TrackingMode.Desktop)
            print(f"Ready ({args.backend}{', dry run' if args.dry_run else ''}). Ctrl+C to quit.")
            dt = interval_ms / 1000.0
            every = max(1, round(args.every_ms / interval_ms))
            step, t_next = 0, time.monotonic()
            while True:
                buf.append(listener.latest())
                step += 1
                if len(buf) == window and step % every == 0:
                    scores = runner.classify([v for row in buf for v in row])["result"]["classification"]
                    label = deb.update(scores)
                    if label:
                        print(f"{label:>12s} -> {action_name(args.backend, label)}")
                        send(label)
                t_next += dt
                time.sleep(max(0.0, t_next - time.monotonic()))
    finally:
        runner.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
