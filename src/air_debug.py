"""Debug log for air_canvas.py (--debug).

Answers one question: is messy writing caused by the pen switch, or by the
tracking itself? It records every frame the canvas sees and, for each word:

  out/words/debug/word-<stamp>-debug.json   every frame since the last word
  out/words/debug/word-<stamp>-debug.png    plot (needs matplotlib):
      1. path: raw fingertip (grey) vs smoothed ink (blue = pen down)
      2. x / y / z over time, pen-down periods shaded
      3. jitter over time (frame-to-frame noise), per-stroke numbers
  out/words/debug/session-<stamp>.csv       every frame of the session

Jitter is estimated from the second difference of the raw position,
x[t] - 2x[t-1] + x[t-2], which is near zero for smooth motion and grows with
frame-to-frame noise. For white noise its std is sqrt(6) x the noise std, so
we divide by sqrt(6) to get millimetres of noise.

How to read it:
  - jitter < ~0.5 mm and the grey raw path already looks like your letters
      -> tracking is fine; the mess came from the pen switch or smoothing.
  - jitter > ~1-2 mm, or "hand lost" while the pen is down
      -> tracking problem: lighting, hand angle, distance, sensor position.
  - big jumps at the start/end of strokes only -> the pen gesture (pinch/push)
      moves your fingertip; use --pen key.
"""

import csv
import json
import math
import pathlib
import statistics
import time

FIELDS = ["t", "leap_frame", "repeat", "hand", "raw_x", "raw_y", "raw_z", "ink_x", "ink_y",
          "pen_down", "pinch", "grab", "palm_speed"]


def jitter_mm(values):
    """Noise estimate (mm) from the second difference of a 1-D series."""
    if len(values) < 5:
        return float("nan")
    d2 = [values[i] - 2 * values[i - 1] + values[i - 2] for i in range(2, len(values))]
    return statistics.pstdev(d2) / math.sqrt(6)


class DebugLog:
    def __init__(self, out_dir, pen_mode):
        self.dir = pathlib.Path(out_dir) / "debug"
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.csv_path = self.dir / f"session-{stamp}.csv"
        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=FIELDS)
        self._w.writeheader()
        self.pen_mode = pen_mode
        self.word = []          # frames since the last saved word
        self.stroke_jitter = [] # (jx, jy) per finished stroke, whole session
        self.lost_down = 0
        self.frames = 0
        self.leap_first = self.leap_last = None
        self.t_first = self.t_last = None
        self.recent = []        # raw (x, y) for the live jitter readout
        self.dups = 0
        self._last_leap = None

    def log(self, now, leap_frame, raw, ink, down, row, palm_speed):
        # The canvas can redraw before a new Leap frame arrives; a repeated
        # frame would look like noise during motion, so it is marked and left
        # out of the jitter numbers.
        repeat = int(self._last_leap == leap_frame)
        rec = {
            "t": round(now, 4), "leap_frame": leap_frame, "repeat": repeat, "hand": int(raw is not None),
            "raw_x": round(raw[0], 2) if raw else "", "raw_y": round(raw[1], 2) if raw else "",
            "raw_z": round(raw[2], 2) if raw else "",
            "ink_x": round(ink[0], 2) if ink else "", "ink_y": round(ink[1], 2) if ink else "",
            "pen_down": int(bool(down)), "pinch": round(row[9], 3), "grab": round(row[10], 3),
            "palm_speed": round(palm_speed, 1),
        }
        self._w.writerow(rec)
        self.word.append(rec)
        self.frames += 1
        self.dups += repeat
        self._last_leap = leap_frame
        self.leap_first = self.leap_first if self.leap_first is not None else leap_frame
        self.leap_last = leap_frame
        self.t_first = self.t_first or now
        self.t_last = now
        if down and raw is None:
            self.lost_down += 1
        if raw is not None and not repeat:
            self.recent.append((raw[0], raw[1]))
            self.recent = self.recent[-40:]

    def live_text(self):
        if len(self.recent) < 10:
            return ""
        jx = jitter_mm([p[0] for p in self.recent])
        jy = jitter_mm([p[1] for p in self.recent])
        fps = self.leap_fps()
        return f"debug: jitter x {jx:.2f} y {jy:.2f} mm   leap {fps:.0f} fps"

    def leap_fps(self):
        if self.t_last is None or self.t_last == self.t_first:
            return 0.0
        return (self.leap_last - self.leap_first) / (self.t_last - self.t_first)

    # --- per word -------------------------------------------------------------
    def word_done(self, stamp):
        frames, self.word = self.word, []
        if not frames:
            return None
        base = self.dir / f"word-{stamp}-debug"
        strokes = self._strokes(frames)
        summary = {
            "pen": self.pen_mode,
            "frames": len(frames),
            "pen_down_frames": sum(f["pen_down"] for f in frames),
            "hand_lost_while_down": sum(1 for f in frames if f["pen_down"] and not f["hand"]),
            "strokes": [{
                "frames": len(s),
                "duration_s": round(s[-1]["t"] - s[0]["t"], 3),
                "jitter_x_mm": round(jitter_mm([f["raw_x"] for f in s]), 3),
                "jitter_y_mm": round(jitter_mm([f["raw_y"] for f in s]), 3),
                "start_jump_mm": round(self._end_jump(s, start=True), 2),
                "end_jump_mm": round(self._end_jump(s, start=False), 2),
            } for s in strokes],
        }
        self.stroke_jitter += [(st["jitter_x_mm"], st["jitter_y_mm"]) for st in summary["strokes"]]
        base.with_suffix(".json").write_text(json.dumps({"summary": summary, "frames": frames}))
        self._plot(frames, strokes, summary, base.with_suffix(".png"))
        return summary

    @staticmethod
    def _strokes(frames):
        out, cur = [], []
        for f in frames:
            if f["repeat"]:
                continue
            if f["pen_down"] and f["hand"]:
                cur.append(f)
            elif cur:
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        return [s for s in out if len(s) >= 3]

    @staticmethod
    def _end_jump(s, start):
        """How much bigger the largest frame-to-frame move in the first/last
        100 ms of a stroke is than the large moves (95th percentile) in the
        middle of it. ~0 = the stroke ends look like the rest; several mm = the
        pen gesture itself (pinch/push) moved the fingertip."""
        steps = [math.dist((a["raw_x"], a["raw_y"]), (b["raw_x"], b["raw_y"])) for a, b in zip(s, s[1:])]
        ts = [f["t"] for f in s[1:]]
        if len(steps) < 6:
            return 0.0
        lo, hi = s[0]["t"] + 0.1, s[-1]["t"] - 0.1
        edge = [d for d, t in zip(steps, ts) if (t <= lo if start else t >= hi)]
        middle = sorted(d for d, t in zip(steps, ts) if lo < t < hi)
        if not edge or len(middle) < 4:
            return 0.0
        p95 = middle[min(len(middle) - 1, int(0.95 * len(middle)))]
        return max(0.0, max(edge) - p95)

    def _plot(self, frames, strokes, summary, path):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (pip install matplotlib for debug plots)")
            return
        hand = [f for f in frames if f["hand"] and not f["repeat"]]
        if not hand:
            return
        t0 = frames[0]["t"]
        fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.25, 1.4, 1]})

        a = ax[0]
        a.plot([f["raw_x"] for f in hand], [f["raw_y"] for f in hand], color="0.75", lw=0.8, label="raw (all)")
        for i, s in enumerate(strokes):
            a.plot([f["raw_x"] for f in s], [f["raw_y"] for f in s], color="0.35", lw=0.8,
                   label="raw (pen down)" if i == 0 else None)
            a.plot([f["ink_x"] for f in s], [f["ink_y"] for f in s], color="tab:blue", lw=2.2,
                   label="smoothed ink" if i == 0 else None)
        a.set_aspect("equal")
        a.set_title("Fingertip path (front view, mm)")
        a.set_xlabel("x  left/right")
        a.set_ylabel("y  height")
        a.legend(fontsize=8, loc="best")

        a = ax[1]
        ts = [f["t"] - t0 for f in hand]
        for key, c in (("raw_x", "tab:red"), ("raw_y", "tab:green"), ("raw_z", "tab:purple")):
            a.plot(ts, [f[key] for f in hand], color=c, lw=1, label=key[-1])
        for s in strokes:
            a.axvspan(s[0]["t"] - t0, s[-1]["t"] - t0, color="tab:blue", alpha=0.12)
        lost = [f["t"] - t0 for f in frames if not f["hand"]]
        if lost:
            a.plot(lost, [a.get_ylim()[0]] * len(lost), "|", color="black", ms=10, label="hand lost")
        a.set_title("Position over time (shaded = pen down)")
        a.set_xlabel("s")
        a.set_ylabel("mm")
        a.legend(fontsize=8, ncol=4)

        a = ax[2]
        win = 15
        jt, jv = [], []
        xs = [f["raw_x"] for f in hand]
        ys = [f["raw_y"] for f in hand]
        for i in range(win, len(hand)):
            jt.append(ts[i])
            jv.append((jitter_mm(xs[i - win:i]) + jitter_mm(ys[i - win:i])) / 2)
        a.plot(jt, jv, color="tab:orange", lw=1.2)
        for s in strokes:
            a.axvspan(s[0]["t"] - t0, s[-1]["t"] - t0, color="tab:blue", alpha=0.12)
        a.axhline(0.5, color="tab:green", ls="--", lw=0.8, label="good < 0.5")
        a.axhline(1.5, color="tab:red", ls="--", lw=0.8, label="bad > 1.5")
        a.set_ylim(0, max(2.0, max(jv, default=0) * 1.1))
        a.set_title("Jitter, avg of x and y (mm, 0.3 s window)")
        a.set_xlabel("s")
        a.legend(fontsize=8, loc="upper right")
        lines = [f"stroke {i + 1}: jitter {s['jitter_x_mm']:.2f}/{s['jitter_y_mm']:.2f} mm, "
                 f"edge jump {s['start_jump_mm']:.1f}/{s['end_jump_mm']:.1f} mm"
                 for i, s in enumerate(summary["strokes"][:6])]
        lines.append(f"hand lost while pen down: {summary['hand_lost_while_down']} frames")
        fig.suptitle(f"pen = {self.pen_mode}", fontsize=10)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.text(0.01, 0.01, "\n".join(lines), va="bottom", fontsize=8.5, family="monospace")
        fig.savefig(path, dpi=110)
        plt.close(fig)

    # --- end of session ----------------------------------------------------------
    def close(self):
        self._f.close()
        if not self.frames:
            return
        good = [j for j in self.stroke_jitter if not any(math.isnan(v) for v in j)]
        jx = statistics.median(j[0] for j in good) if good else float("nan")
        jy = statistics.median(j[1] for j in good) if good else float("nan")
        print("\n--- debug summary ---")
        print(f"frames logged      {self.frames}  (session log: {self.csv_path})")
        print(f"Leap frame rate    {self.leap_fps():.0f} fps   "
              f"(repeated frames skipped: {self.dups})")
        print(f"jitter, pen down   x {jx:.2f} mm, y {jy:.2f} mm   median of {len(good)} strokes "
              "(< 0.5 good, > 1.5 bad)")
        print(f"hand lost, pen down {self.lost_down} frames")
