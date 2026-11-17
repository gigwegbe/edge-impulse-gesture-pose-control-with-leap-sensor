"""Speed-based gesture detector (the "detector" in front of the classifier).

Instead of classifying every 100 ms, watch the palm speed:

  waiting  --speed > start for `start_frames` frames-->  moving
  moving   --speed < end for `end_frames` frames------->  settle (wait post_ms)
  settle   --------------------------------------------> CLASSIFY ONCE -> waiting

Slow movements (like the relaxed return after a swipe) never pass the start
threshold, so they are never classified. After a gesture fires, the opposite
gesture is ignored for `return_ms` as a second guard against return strokes.
"""

import collections
import math

OPPOSITE = {
    "swipe_left": "swipe_right", "swipe_right": "swipe_left",
    "swipe_up": "swipe_down", "swipe_down": "swipe_up",
}

VEL = slice(3, 6)  # vel_x, vel_y, vel_z in the leap_features layout


def palm_speed(row):
    vx, vy, vz = row[VEL]
    return math.sqrt(vx * vx + vy * vy + vz * vz)


class SpeedGate:
    def __init__(self, interval_ms, start=300.0, end=150.0, start_frames=2,
                 end_frames=4, post_ms=200, min_ms=80, max_ms=1500):
        self.start, self.end = start, end
        self.start_frames, self.end_frames = start_frames, end_frames
        self.post = max(0, round(post_ms / interval_ms))
        self.min_len = max(1, round(min_ms / interval_ms))
        self.max_len = max(self.min_len + 1, round(max_ms / interval_ms))
        self.state = "waiting"
        self.count = 0      # consecutive frames above/below threshold
        self.length = 0     # frames since the gesture started
        self.wait = 0       # frames left in settle

    def update(self, row):
        """Feed one frame. Returns True on the frame where a gesture should be
        classified (the caller then uses its rolling window)."""
        hand = any(row)
        s = palm_speed(row) if hand else 0.0

        if self.state == "waiting":
            self.count = self.count + 1 if s > self.start else 0
            if self.count >= self.start_frames:
                self.state, self.length, self.count = "moving", self.count, 0
            return False

        if self.state == "moving":
            self.length += 1
            self.count = self.count + 1 if s < self.end else 0
            if self.count >= self.end_frames or self.length >= self.max_len or not hand:
                if self.length - self.count < self.min_len:  # a twitch, not a gesture
                    self._reset()
                    return False
                self.state, self.wait = "settle", self.post
            return False

        # settle: let the end of the gesture land inside the window
        self.wait -= 1
        if self.wait <= 0:
            self._reset()
            return True
        return False

    def _reset(self):
        self.state, self.count, self.length, self.wait = "waiting", 0, 0, 0


# Shared by record.py --segment and live_tflite.py --segment. Change them in
# one place so training samples and live windows are always cut the same way.
SEGMENT_DEFAULTS = {
    "speed_start": 300.0,   # mm/s to start a gesture
    "speed_end": 150.0,     # mm/s to end it
    "pre_ms": 200,          # keep this much from just before the start
    "post_ms": 200,         # and this much after the hand slows down
    "length_ms": 1500,      # fixed sample length (= Edge Impulse window)
    "rearm_ms": 600,        # record.py: hold still this long before the next capture
}


class Segmenter:
    """Cuts gestures out of the frame stream, the same way for recording and
    for live use (the idea from the mmWave repo's Logger):

      start   = speed gate starts (plus `pre_ms` of history)
      end     = speed gate fires (hand slowed down, plus `post_ms`)
      output  = aligned at the start, cropped/padded to `length_ms`
                (padding repeats the last frame: the hand holding still)

    feed(row) returns a list of rows (one fixed-length segment) when a gesture
    has just ended, otherwise None. Twitches the gate rejects are dropped.
    """

    def __init__(self, interval_ms, speed_start=300.0, speed_end=150.0, pre_ms=200,
                 post_ms=200, length_ms=1500, **_ignored):
        self.n = max(1, round(length_ms / interval_ms))
        self.pre = max(0, round(pre_ms / interval_ms))
        post = max(0, round(post_ms / interval_ms))
        # Let the gesture itself use whatever is left of the fixed length.
        max_ms = max(interval_ms * 2, length_ms - pre_ms - post_ms)
        self.gate = SpeedGate(interval_ms, start=speed_start, end=speed_end,
                              post_ms=post_ms, max_ms=max_ms)
        self.history = collections.deque(maxlen=self.pre + self.gate.start_frames)
        self.capture = None

    @property
    def state(self):
        return self.gate.state

    def feed(self, row):
        self.history.append(row)
        was_waiting = self.gate.state == "waiting"
        fired = self.gate.update(row)

        if self.capture is None:
            if was_waiting and self.gate.state == "moving":
                self.capture = list(self.history)  # pre-roll + the start frames
            return None

        self.capture.append(row)
        if fired:
            seg, self.capture = self.capture, None
            return self._fix_length(seg)
        if self.gate.state == "waiting":  # gate rejected it (twitch)
            self.capture = None
        return None

    def _fix_length(self, seg):
        seg = [list(r) for r in seg[: self.n]]
        while len(seg) < self.n:
            seg.append(list(seg[-1]))
        return seg


class ReturnGuard:
    """Blocks the opposite of the last fired gesture for `return_ms`."""

    def __init__(self, return_ms):
        self.return_s = return_ms / 1000.0
        self.last, self.t = None, -1e9

    def allow(self, label, now):
        if self.last and OPPOSITE.get(self.last) == label and now - self.t < self.return_s:
            return False
        return True

    def fired(self, label, now):
        self.last, self.t = label, now
