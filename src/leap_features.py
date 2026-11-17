"""Shared Leap (Gemini / LeapC) helpers: a listener that keeps the latest
hand features, plus the feature layout used for Edge Impulse.

Feature vector per frame (11 values, first hand only, zeros if no hand):
  palm_x, palm_y, palm_z          mm, Leap coordinates (y is up from the device)
  vel_x, vel_y, vel_z             mm/s, palm velocity
  index_x, index_y, index_z       mm, index fingertip (end of distal bone)
  pinch, grab                     0..1
"""

import threading
import time

import leap

AXES = [
    ("palm_x", "mm"), ("palm_y", "mm"), ("palm_z", "mm"),
    ("vel_x", "mm/s"), ("vel_y", "mm/s"), ("vel_z", "mm/s"),
    ("index_x", "mm"), ("index_y", "mm"), ("index_z", "mm"),
    ("pinch", "ratio"), ("grab", "ratio"),
]
N_AXES = len(AXES)
EMPTY = [0.0] * N_AXES


def hand_features(hand):
    """Turn one leap Hand into the flat feature list. Must be called inside the
    tracking callback: the underlying LeapC memory is only valid there."""
    p = hand.palm.position
    v = hand.palm.velocity
    tip = hand.index.distal.next_joint
    return [
        p.x, p.y, p.z,
        v.x, v.y, v.z,
        tip.x, tip.y, tip.z,
        hand.pinch_strength, hand.grab_strength,
    ]


def hand_skeleton(hand):
    """Joint positions (mm) for drawing: palm plus 5 digits x 5 joints
    (metacarpal start, then the end of each of the 4 bones)."""
    digits = []
    for d in hand.digits:
        bones = d.bones
        pts = [bones[0].prev_joint] + [b.next_joint for b in bones]
        digits.append([(p.x, p.y, p.z) for p in pts])
    p = hand.palm.position
    side = "left" if "left" in str(hand.type).lower() else "right"
    return {"palm": (p.x, p.y, p.z), "digits": digits, "side": side}


class LatestHandListener(leap.Listener):
    """Keeps the most recent feature vector and some simple stats."""

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._latest = list(EMPTY)
        self._skeleton = None
        self.hand_present = False
        self.frames = 0
        self.device_serial = None
        self.connected = False

    # --- LeapC events -----------------------------------------------------
    def on_connection_event(self, event):
        self.connected = True

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()
        self.device_serial = info.serial

    def on_tracking_event(self, event):
        hands = event.hands
        feats = hand_features(hands[0]) if hands else list(EMPTY)
        skel = None
        if hands:
            try:
                skel = hand_skeleton(hands[0])
            except Exception:  # drawing data is optional; never break tracking
                skel = None
        with self._lock:
            self._latest = feats
            self._skeleton = skel
            self.hand_present = bool(hands)
            self.frames += 1

    # --- reading ----------------------------------------------------------
    def latest(self):
        with self._lock:
            return list(self._latest)

    def skeleton(self):
        with self._lock:
            return self._skeleton


def open_leap(mode=leap.TrackingMode.Desktop):
    """Return (connection, listener). Use as:
        conn, listener = open_leap()
        with conn.open():
            conn.set_tracking_mode(...)
    """
    listener = LatestHandListener()
    conn = leap.Connection()
    conn.add_listener(listener)
    return conn, listener


def sample_window(listener, seconds, interval_ms, on_tick=None):
    """Sample the latest features at a fixed rate for `seconds`.
    Returns a list of rows (one per tick). `on_tick(i, n)` is called each tick
    (used to refresh the --visual window while recording)."""
    n = int(round(seconds * 1000 / interval_ms))
    dt = interval_ms / 1000.0
    rows = []
    t_next = time.monotonic()
    for i in range(n):
        rows.append(listener.latest())
        if on_tick:
            on_tick(i, n)
        t_next += dt
        delay = t_next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
    return rows
