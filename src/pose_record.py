# src/pose_record.py
# Record static pose samples to CSV for Edge Impulse (one row per frame).
# Keys in the window: 1-N pick pose, SPACE start/stop, Q quit.
#
# Record only the two new poses (recommended, keeps your old data):
#   python src/pose_record.py --poses thumb,pinky
# Record everything:
#   python src/pose_record.py
#
# Poses:
#   fist       all fingers curled                      -> play / pause
#   open_palm  all five fingers straight               -> next track
#   thumb      fist with ONLY the thumb out            -> volume up
#   pinky      fist with ONLY the little finger out    -> volume down
#   peace      index + middle up (kept, no action yet)
#   none       anything else (half-curled, pointing, three fingers, relaxed)
import os, time, csv, math, argparse, threading
import numpy as np
import cv2
import leap
import pose_detect
from pose_features import pose_features, N_FEATURES
from pose_detect import draw_view, to_front, to_top, distance_state, extended

ALL_POSES = ["fist", "open_palm", "peace", "none", "thumb", "pinky"]
HINTS = {
    "fist": "all fingers curled",
    "open_palm": "all five fingers straight",
    "peace": "index + middle up",
    "none": "anything else: half-curled, pointing, 3 fingers",
    "thumb": "fist, ONLY thumb out (any direction)",
    "pinky": "fist, ONLY little finger out",
}
RATE_HZ = 20
COUNTDOWN_S = 3
BANDS = ["near", "mid", "far"]

def band_of(y_mm):
    span = pose_detect.FAR_MM - pose_detect.NEAR_MM
    if y_mm < pose_detect.NEAR_MM + span / 3:
        return "near"
    if y_mm < pose_detect.NEAR_MM + 2 * span / 3:
        return "mid"
    return "far"

class RecListener(leap.Listener):
    def __init__(self):
        self.lock = threading.Lock()
        self.snap = None

    def on_tracking_event(self, event):
        if not event.hands:
            with self.lock:
                self.snap = None
            return
        hand = event.hands[0]
        bones = [[((b.prev_joint.x, b.prev_joint.y, b.prev_joint.z),
                   (b.next_joint.x, b.next_joint.y, b.next_joint.z))
                  for b in (d.metacarpal, d.proximal, d.intermediate, d.distal)]
                 for d in hand.digits]
        p = hand.palm.position
        with self.lock:
            self.snap = {"bones": bones, "ext": extended(hand),
                         "palm": (p.x, p.y, p.z), "dist": distance_state(p.y),
                         "feats": pose_features(hand)}

    def latest(self):
        with self.lock:
            return self.snap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default=",".join(ALL_POSES),
                    help="comma list of poses to record, e.g. thumb,pinky")
    ap.add_argument("--near", type=float, default=120, help="min palm distance in mm")
    ap.add_argument("--far", type=float, default=450, help="max palm distance in mm")
    args = ap.parse_args()
    pose_detect.NEAR_MM, pose_detect.FAR_MM = args.near, args.far

    poses = [p.strip() for p in args.poses.split(",") if p.strip()]
    unknown = [p for p in poses if p not in ALL_POSES]
    if unknown or not poses:
        raise SystemExit(f"Unknown pose(s) {unknown}. Choose from: {', '.join(ALL_POSES)}")
    if len(poses) > 9:
        raise SystemExit("At most 9 poses per session (keys 1-9).")

    os.makedirs("data/poses", exist_ok=True)
    path = time.strftime("data/poses/poses_%Y%m%d_%H%M%S.csv")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["label"] + [f"f{i}" for i in range(N_FEATURES)])

    counts = {p: 0 for p in poses}
    bands = {p: {b: 0 for b in BANDS} for p in poses}
    current, recording, start_at, last_save = 0, False, 0.0, 0.0

    listener = RecListener()
    conn = leap.Connection()
    conn.add_listener(listener)
    with conn.open():
        conn.set_tracking_mode(leap.TrackingMode.Desktop)
        print(f"Saving to {path}  (range {args.near:.0f}-{args.far:.0f} mm)")
        print("Keys: " + "  ".join(f"{i + 1}={p}" for i, p in enumerate(poses)) + "  SPACE=start/stop  Q=quit")
        while True:
            now = time.time()
            snap = listener.latest()
            pose = poses[current]

            live = recording and now >= start_at
            if live and snap and snap["dist"] == "good" and now - last_save >= 1.0 / RATE_HZ:
                w.writerow([pose] + [f"{x:.4f}" for x in snap["feats"]])
                counts[pose] += 1
                bands[pose][band_of(snap["palm"][1])] += 1
                last_save = now

            panel = np.full((220, 840, 3), 15, np.uint8)
            if recording and now < start_at:
                status, col = f"GET READY {math.ceil(start_at - now)}", (0, 200, 255)
            elif recording:
                status, col = "RECORDING", (60, 60, 255)
            else:
                status, col = "paused (SPACE to record)", (180, 180, 180)
            cv2.putText(panel, f"Pose: {pose}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(panel, status, (430, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            cv2.putText(panel, HINTS[pose], (10, 72), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (150, 200, 255), 1)

            if snap:
                dist = f"{snap['dist']}  {snap['palm'][1] / 10:.0f} cm  ({band_of(snap['palm'][1])})"
                dcol = (80, 220, 80) if snap["dist"] == "good" else (0, 140, 255)
            else:
                dist, dcol = "no hand", (0, 140, 255)
            cv2.putText(panel, f"Distance: {dist}", (10, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, dcol, 2)

            b = bands[pose]
            cv2.putText(panel, f"{pose} by distance:  near {b['near']}   mid {b['mid']}   far {b['far']}",
                        (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 1)
            cnt = "   ".join(f"{i + 1}:{p} {counts[p]}" for i, p in enumerate(poses))
            cv2.putText(panel, cnt, (10, 190), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)

            views = np.hstack([draw_view(snap, to_front, "FRONT", front=True),
                               draw_view(snap, to_top, "TOP")])
            cv2.imshow("Pose recorder", np.vstack([panel, views]))

            k = cv2.waitKey(15) & 0xFF
            if k in (ord("q"), 27):
                break
            if ord("1") <= k < ord("1") + len(poses):
                current = k - ord("1")
                recording = False
            if k == ord(" "):
                recording = not recording
                if recording:
                    start_at = now + COUNTDOWN_S
                    f.flush()

    f.close()
    cv2.destroyAllWindows()
    print("Saved:", counts, "->", path)
    for p in poses:
        print(f"  {p}: {bands[p]}")

if __name__ == "__main__":
    main()
