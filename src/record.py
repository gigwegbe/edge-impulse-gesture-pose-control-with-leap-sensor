"""Step 2: record labelled gesture samples from the Leap.

Each sample is saved locally in Edge Impulse's data acquisition JSON format
(data/<label>/<label>.<timestamp>.json) and, if EI_API_KEY is set, uploaded
straight to your project via the ingestion API.

Examples:
  python src/record.py swipe_left --count 30
  python src/record.py idle --count 30 --seconds 2
  EI_API_KEY=ei_xxx python src/record.py circle --count 30 --category testing

Segment mode (recommended): no Enter/Space per sample. Just do the gesture;
the speed gate cuts it out exactly the way live_tflite.py --segment will:
  python src/record.py swipe_left --segment --count 40 --visual
  python src/record.py none --segment --count 60 --visual   # withdrawals, fidgets...
Keys in the --visual window: D = delete last sample, P = pause, Q/Esc = quit.

Already have local files? Upload them later with the Edge Impulse CLI:
  edge-impulse-uploader --category split data/*/*.json
"""

import argparse
import json
import os
import pathlib
import time

import leap
import requests

from leap_features import AXES, open_leap, sample_window
from speed_gate import SEGMENT_DEFAULTS, Segmenter, palm_speed

INGEST = "https://ingestion.edgeimpulse.com/api/{category}/data"


def to_ei_json(rows, interval_ms):
    return {
        "protected": {"ver": "v1", "alg": "none", "iat": int(time.time())},
        "signature": "0" * 64,
        "payload": {
            "device_name": "leap-motion-v1",
            "device_type": "LEAP_MOTION",
            "interval_ms": interval_ms,
            "sensors": [{"name": n, "units": u} for n, u in AXES],
            "values": rows,
        },
    }


def upload(api_key, category, label, filename, body):
    r = requests.post(
        INGEST.format(category=category),
        data=json.dumps(body),
        headers={
            "x-api-key": api_key,
            "x-label": label,
            "x-file-name": filename,
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    return r.ok, (r.text if not r.ok else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label")
    ap.add_argument("--count", type=int, default=10, help="number of samples to record")
    ap.add_argument("--seconds", type=float, default=2.0, help="length of each sample")
    ap.add_argument("--interval-ms", type=int, default=20, help="20 ms = 50 Hz")
    ap.add_argument("--category", choices=["training", "testing"], default="training")
    ap.add_argument("--out", default="data")
    ap.add_argument("--auto", action="store_true", help="no Enter between samples, 1.5 s pause instead")
    ap.add_argument("--visual", action="store_true",
                    help="show the hand skeleton; press SPACE in the window to record, Q/Esc to quit")
    sg = ap.add_argument_group("segment mode")
    sg.add_argument("--segment", action="store_true",
                    help="auto-capture each gesture with the speed gate (same cut as live --segment)")
    sg.add_argument("--speed-start", type=float, default=SEGMENT_DEFAULTS["speed_start"])
    sg.add_argument("--speed-end", type=float, default=SEGMENT_DEFAULTS["speed_end"])
    sg.add_argument("--pre-ms", type=int, default=SEGMENT_DEFAULTS["pre_ms"])
    sg.add_argument("--post-ms", type=int, default=SEGMENT_DEFAULTS["post_ms"])
    sg.add_argument("--length-ms", type=int, default=SEGMENT_DEFAULTS["length_ms"])
    sg.add_argument("--rearm-ms", type=int, default=SEGMENT_DEFAULTS["rearm_ms"],
                    help="hold still this long before the next capture (stops return strokes being saved)")
    args = ap.parse_args()

    api_key = os.environ.get("EI_API_KEY")
    out_dir = pathlib.Path(args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.segment:
        print(f"Segment mode: {args.count} x '{args.label}', {args.length_ms} ms each at {1000 // args.interval_ms} Hz")
    else:
        print(f"Recording {args.count} x {args.seconds}s of '{args.label}' at {1000 // args.interval_ms} Hz")
    print("Upload:", f"on ({args.category})" if api_key else "off (set EI_API_KEY to enable)")

    conn, listener = open_leap()
    with conn.open():
        conn.set_tracking_mode(leap.TrackingMode.Desktop)
        time.sleep(1.0)  # let the service connect
        if not listener.connected:
            print("Warning: no connection event yet. Is the tracking service running?")

        view = None
        status = ""
        if args.visual:
            from visual import SkeletonView
            view = SkeletonView(f"record: {args.label}")

        def draw(lines, progress=None, recording=False):
            return view.show(listener.skeleton(), listener.latest(), lines,
                             progress=progress, recording=recording)

        if args.segment:
            try:
                record_segments(args, listener, view, draw, api_key, out_dir)
            finally:
                if view:
                    view.close()
            return

        saved = 0
        for i in range(1, args.count + 1):
            head = f"'{args.label}'  sample {i}/{args.count}  (saved {saved})"
            if args.auto:
                t_end = time.monotonic() + 1.5
                while time.monotonic() < t_end:
                    if view:
                        if draw([head, "get ready...", status]) in (ord("q"), 27):
                            return
                    time.sleep(0.02)
            elif view:
                print(f"[{i}/{args.count}] press SPACE in the window, then do the gesture")
                while True:
                    key = draw([head, "SPACE = record   Q/Esc = quit", status])
                    if key == ord(" "):
                        break
                    if key in (ord("q"), 27):
                        return
                    time.sleep(0.02)
            else:
                input(f"[{i}/{args.count}] Enter, then do the gesture... ")
            print("  go")
            tick = (lambda k, n: draw([head, "recording..."], progress=(k + 1) / n, recording=True)) \
                if view else None
            rows = sample_window(listener, args.seconds, args.interval_ms, on_tick=tick)

            no_hand = sum(1 for r in rows if not any(r))
            if args.label != "idle" and no_hand > len(rows) * 0.5:
                status = f"last: skipped, no hand in {no_hand}/{len(rows)} frames"
                print(f"  skipped: no hand in {no_hand}/{len(rows)} frames")
                continue
            saved += 1

            body = to_ei_json(rows, args.interval_ms)
            name = f"{args.label}.{int(time.time() * 1000)}.json"
            (out_dir / name).write_text(json.dumps(body))
            msg = f"  saved {out_dir / name}"
            if api_key:
                ok, err = upload(api_key, args.category, args.label, name, body)
                msg += " | uploaded" if ok else f" | upload failed: {err[:200]}"
            print(msg)
            status = "last: saved" + (" + uploaded" if api_key and ok else "")

        if view:
            view.close()


class PendingSample:
    """Holds the newest sample back from upload until the next one arrives (or
    you quit), so the D key can still delete it everywhere."""

    def __init__(self, api_key, category, label):
        self.api_key, self.category, self.label = api_key, category, label
        self.item = None  # (path, name, body)

    def add(self, path, name, body):
        self.flush()
        self.item = (path, name, body)

    def delete(self):
        if not self.item:
            return False
        self.item[0].unlink(missing_ok=True)
        self.item = None
        return True

    def flush(self):
        if not self.item:
            return ""
        path, name, body = self.item
        self.item = None
        if not self.api_key:
            return ""
        ok, err = upload(self.api_key, self.category, self.label, name, body)
        msg = f"uploaded {name}" if ok else f"upload failed for {name}: {err[:200]}"
        print("  " + msg)
        return msg


def record_segments(args, listener, view, draw, api_key, out_dir):
    seg = Segmenter(args.interval_ms, speed_start=args.speed_start, speed_end=args.speed_end,
                    pre_ms=args.pre_ms, post_ms=args.post_ms, length_ms=args.length_ms)
    rearm_frames = max(1, round(args.rearm_ms / args.interval_ms))
    pending = PendingSample(api_key, args.category, args.label)
    gesture_like = args.label not in ("idle", "none")

    saved, still, armed, armed_at_start = 0, 0, False, False
    paused, status = False, ""
    dt = args.interval_ms / 1000.0
    t_next = time.monotonic()
    print("Do the gesture whenever the window says ARMED. "
          + ("D = delete last, P = pause, Q = quit." if view else "Ctrl+C to stop."))
    try:
        while saved < args.count:
            row = listener.latest()
            speed = palm_speed(row) if any(row) else 0.0
            still = still + 1 if speed < args.speed_end else 0
            if not armed and still >= rearm_frames:
                armed = True

            was_idle = seg.capture is None
            out = None if paused else seg.feed(row)
            if was_idle and seg.capture is not None:
                armed_at_start = armed  # decide at the start, not at the end

            if out is not None:
                armed, still = False, 0
                no_hand = sum(1 for r in out if not any(r))
                if not armed_at_start:
                    status = "ignored: started before you were armed (return stroke?)"
                elif gesture_like and no_hand > len(out) * 0.5:
                    status = f"ignored: no hand in {no_hand}/{len(out)} frames"
                else:
                    body = to_ei_json(out, args.interval_ms)
                    name = f"{args.label}.{int(time.time() * 1000)}.json"
                    path = out_dir / name
                    path.write_text(json.dumps(body))
                    pending.add(path, name, body)
                    saved += 1
                    status = f"saved #{saved}  ({name})"
                    print(f"  [{saved}/{args.count}] saved {path}")
                if status.startswith("ignored"):
                    print("  " + status)

            if view:
                if paused:
                    mode = "PAUSED (P to resume)"
                elif seg.capture is not None:
                    mode = "capturing..."
                else:
                    mode = "ARMED: do the gesture" if armed else "hold still..."
                key = view.show(listener.skeleton(), row,
                                [f"'{args.label}'  saved {saved}/{args.count}   [{mode}]",
                                 "D = delete last   P = pause   Q/Esc = quit", status],
                                recording=seg.capture is not None)
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = not paused
                    seg = Segmenter(args.interval_ms, speed_start=args.speed_start,
                                    speed_end=args.speed_end, pre_ms=args.pre_ms,
                                    post_ms=args.post_ms, length_ms=args.length_ms)
                    armed, still = False, 0
                if key == ord("d"):
                    if pending.delete():
                        saved -= 1
                        status = "deleted last sample"
                        print("  deleted last sample")
                    else:
                        status = "nothing to delete (already uploaded or none yet)"

            t_next += dt
            time.sleep(max(0.0, t_next - time.monotonic()))
    finally:
        pending.flush()
        print(f"Done: {saved} '{args.label}' samples in {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
