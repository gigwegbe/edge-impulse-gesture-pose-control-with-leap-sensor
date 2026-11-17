"""Step 4: live gesture recognition with an Edge Impulse Linux model (.eim).

Download the model on the device with:
  edge-impulse-linux-runner --download model.eim
Then:
  python src/live_classify.py model.eim

A rolling buffer holds the last window of Leap features (same 11 axes and
interval as training) and is classified every --every-ms milliseconds.
"""

import argparse
import collections
import time

import leap
from edge_impulse_linux.runner import ImpulseRunner

from leap_features import N_AXES, open_leap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--every-ms", type=int, default=100, help="how often to classify")
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    runner = ImpulseRunner(args.model)
    try:
        info = runner.init()
        mp = info["model_parameters"]
        labels = mp["labels"]
        interval_ms = mp["interval_ms"]
        n_features = mp["input_features_count"]
        if mp.get("axis_count", N_AXES) != N_AXES:
            raise SystemExit(f"Model expects {mp['axis_count']} axes, recorder produces {N_AXES}")
        window = n_features // N_AXES
        print(f"Model: {info['project']['owner']} / {info['project']['name']}")
        print(f"Labels {labels}, window {window} frames @ {interval_ms} ms")

        buf = collections.deque(maxlen=window)
        conn, listener = open_leap()
        with conn.open():
            conn.set_tracking_mode(leap.TrackingMode.Desktop)
            dt = interval_ms / 1000.0
            steps_per_classify = max(1, round(args.every_ms / interval_ms))
            step = 0
            last = None
            t_next = time.monotonic()
            while True:
                buf.append(listener.latest())
                step += 1
                if len(buf) == window and step % steps_per_classify == 0:
                    features = [v for row in buf for v in row]
                    t0 = time.perf_counter()
                    res = runner.classify(features)
                    ms = (time.perf_counter() - t0) * 1000
                    scores = res["result"]["classification"]
                    best = max(scores, key=scores.get)
                    if scores[best] >= args.threshold and best != "idle" and best != last:
                        print(f"{best:>12s}  {scores[best]:.2f}  ({ms:.1f} ms)")
                    last = best if scores[best] >= args.threshold else None
                t_next += dt
                delay = t_next - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    finally:
        runner.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
