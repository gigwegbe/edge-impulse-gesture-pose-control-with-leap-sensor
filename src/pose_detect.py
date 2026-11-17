# src/pose_detect.py
# Static hand pose detection on Leap Motion (Gemini / leapc bindings)
# with a live OpenCV view of the hand skeleton and a distance guide.
import time, math, argparse, threading
import numpy as np
import cv2
import leap

STILL_MM_S = 40.0
HOLD_S = 0.5
NEAR_MM = 150.0       # palm height below this = too close
FAR_MM = 350.0        # palm height above this = too far
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

def speed(v):
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

def extended(hand):
    return [bool(d.is_extended) for d in hand.digits]

def thumb_up_ratio(hand):
    b = hand.digits[0].distal
    dx = b.next_joint.x - b.prev_joint.x
    dy = b.next_joint.y - b.prev_joint.y
    dz = b.next_joint.z - b.prev_joint.z
    length = math.sqrt(dx * dx + dy * dy + dz * dz) or 1
    return dy / length

def distance_state(palm_y):
    if palm_y < NEAR_MM:
        return "too close"
    if palm_y > FAR_MM:
        return "too far"
    return "good"

def classify(hand):
    ext = extended(hand)
    n = sum(ext)
    thumb, index, middle, ring, pinky = ext
    others = index + middle + ring + pinky

    if hand.pinch_strength > 0.6 and (middle + ring + pinky) >= 2:
        return "ok"
    if n == 0 or hand.grab_strength > 0.9:
        return "fist"
    if n == 5:
        return "open_palm"
    if thumb and others <= 1:
        return "thumbs_up" if thumb_up_ratio(hand) > 0.5 else "thumb_side"
    if index and n == 1:
        return "point"
    if index and middle and n == 2:
        return "peace"
    return f"count_{n}"

class PoseListener(leap.Listener):
    def __init__(self):
        self.lock = threading.Lock()
        self.snapshot = None
        self.candidate = None
        self.since = 0.0
        self.reported = None

    def on_tracking_event(self, event):
        now = time.time()
        if not event.hands:
            if self.reported:
                print("-- hand lost")
            self.candidate = self.reported = None
            with self.lock:
                self.snapshot = None
            return
        hand = event.hands[0]
        pose = classify(hand)
        still = speed(hand.palm.velocity) < STILL_MM_S
        p = hand.palm.position
        dist = distance_state(p.y)
        in_range = dist == "good"

        if pose != self.candidate or not still or not in_range:
            self.candidate, self.since = pose, now
        elif now - self.since >= HOLD_S and pose != self.reported:
            self.reported = pose
            print(f"POSE: {pose}")

        bones = []
        for d in hand.digits:
            seg = []
            for b in (d.metacarpal, d.proximal, d.intermediate, d.distal):
                seg.append(((b.prev_joint.x, b.prev_joint.y, b.prev_joint.z),
                            (b.next_joint.x, b.next_joint.y, b.next_joint.z)))
            bones.append(seg)
        with self.lock:
            self.snapshot = {
                "pose": pose, "reported": self.reported, "still": still,
                "ext": extended(hand), "bones": bones,
                "palm": (p.x, p.y, p.z), "dist": dist,
                "pinch": hand.pinch_strength, "grab": hand.grab_strength,
                "thumb_up": thumb_up_ratio(hand),
            }

    def latest(self):
        with self.lock:
            return self.snapshot

# ---------- drawing ----------
W, H = 420, 420
SCALE_Y = H / 500.0   # front view covers 0 to 500 mm height

def to_front(x, y, z):
    return int(W / 2 + x * 1.2), int(H - y * SCALE_Y)

def to_top(x, y, z):
    return int(W / 2 + x * 1.2), int(H / 2 + z * 1.2)

def draw_view(snap, project, title, front=False):
    img = np.full((H, W, 3), 25, np.uint8)
    if front:
        # shade the sweet spot band
        y_far, y_near = int(H - FAR_MM * SCALE_Y), int(H - NEAR_MM * SCALE_Y)
        cv2.rectangle(img, (0, y_far), (W, y_near), (35, 55, 35), -1)
        cv2.putText(img, "sweet spot", (8, y_far + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 160, 90), 1)
        cv2.rectangle(img, (W // 2 - 30, H - 8), (W // 2 + 30, H), (120, 120, 120), -1)
    cv2.putText(img, title, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (160, 160, 160), 1)
    if snap is None:
        cv2.putText(img, "no hand", (W // 2 - 40, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
        return img
    for i, seg in enumerate(snap["bones"]):
        colour = (80, 220, 80) if snap["ext"][i] else (70, 70, 230)
        for a, b in seg:
            cv2.line(img, project(*a), project(*b), colour, 3)
            cv2.circle(img, project(*b), 4, (240, 240, 240), -1)
        tip = project(*seg[-1][1])
        cv2.putText(img, FINGERS[i][0].upper(), (tip[0] + 6, tip[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    cv2.circle(img, project(*snap["palm"]), 8, (0, 200, 255), -1)
    return img

def draw_panel(snap):
    img = np.full((150, W * 2, 3), 15, np.uint8)
    if snap is None:
        cv2.putText(img, "Show your hand 20-30 cm above the sensor", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return img
    colour = (80, 220, 80) if snap["still"] else (0, 170, 255)
    cv2.putText(img, f"Pose: {snap['pose']}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
    cv2.putText(img, f"Confirmed: {snap['reported'] or '-'}", (430, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    d = snap["dist"]
    dcol = (80, 220, 80) if d == "good" else (0, 140, 255)
    height_cm = snap["palm"][1] / 10.0
    msg = {"good": "distance OK", "too close": "TOO CLOSE - move up",
           "too far": "TOO FAR - move down"}[d]
    cv2.putText(img, f"Height {height_cm:.0f} cm   {msg}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, dcol, 2)

    ext = " ".join(f"{f[0].upper()}:{int(e)}" for f, e in zip(FINGERS, snap["ext"]))
    info = (f"{ext}   pinch {snap['pinch']:.2f}   grab {snap['grab']:.2f}   "
            f"thumb_up {snap['thumb_up']:+.2f}   {'STILL' if snap['still'] else 'moving'}")
    cv2.putText(img, info, (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)
    return img

def main():
    argparse.ArgumentParser().parse_args()
    listener = PoseListener()
    conn = leap.Connection()
    conn.add_listener(listener)
    with conn.open():
        conn.set_tracking_mode(leap.TrackingMode.Desktop)
        print("Hold a pose still for half a second. Press Q in the window to quit.")
        while True:
            snap = listener.latest()
            views = np.hstack([draw_view(snap, to_front, "FRONT", front=True),
                               draw_view(snap, to_top, "TOP")])
            cv2.imshow("Leap poses", np.vstack([draw_panel(snap), views]))
            if cv2.waitKey(15) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
