"""Live gesture recognition with the Edge Impulse TFLite file (no .eim needed).

Download from Edge Impulse > Dashboard > "Download block output":
  Classifier model, TensorFlow Lite (float32)  -> e.g. models/classifier.tflite
(int8 also works; the script handles quantization.)

The .tflite only contains the neural network. The processing block runs here
in Python first, and must match your impulse:

  --dsp spectral   Spectral Analysis block. Uses Edge Impulse's own Python code
                   (third_party/processing-blocks). Put your block settings in
                   a JSON file (see config/spectral.json) and pass --dsp-config.
  --dsp raw        Raw Data block. Just the flattened window (x scale_axes).

Examples:
  python src/live_tflite.py models/classifier.tflite \
      --labels idle,swipe_left,swipe_right --window-ms 1500 \
      --dsp spectral --dsp-config config/spectral.json

  ... add --media (and --dry-run) to drive media_control.py's actions.

Labels must be in the same order as in Studio (alphabetical by default).
"""

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np

import leap
from leap_features import AXES, N_AXES, open_leap

ROOT = pathlib.Path(__file__).resolve().parent.parent
INTERVAL_MS = 20  # must match record.py / Studio frequency (50 Hz)


def load_interpreter(path):
    for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
        try:
            m = __import__(mod, fromlist=["Interpreter"])
            return m.Interpreter(model_path=path)
        except ImportError:
            pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=path)
    except ImportError:
        raise SystemExit("No TFLite runtime. Install one: pip install ai-edge-litert")


class TFLiteClassifier:
    def __init__(self, path):
        self.it = load_interpreter(path)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.n_in = int(np.prod(self.inp["shape"]))

    def __call__(self, feats):
        x = np.asarray(feats, dtype=np.float32).reshape(self.inp["shape"])
        if self.inp["dtype"] != np.float32:
            s, z = self.inp["quantization"]
            x = np.clip(np.round(x / s + z), np.iinfo(self.inp["dtype"]).min,
                        np.iinfo(self.inp["dtype"]).max).astype(self.inp["dtype"])
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"]).reshape(-1)
        if self.out["dtype"] != np.float32:
            s, z = self.out["quantization"]
            y = (y.astype(np.float32) - z) * s
        return y


def make_dsp(kind, config_path):
    cfg = json.loads(pathlib.Path(config_path).read_text()) if config_path else {}
    scale = float(cfg.get("scale_axes", 1))
    if kind == "raw":
        return lambda window: (np.asarray(window, dtype=np.float32) * scale).reshape(-1)

    blocks = ROOT / "third_party" / "processing-blocks"
    sys.path.insert(0, str(blocks))
    sys.path.insert(0, str(blocks / "spectral_analysis"))  # dsp.py imports "common.*"
    try:
        # Load dsp.py directly: the package __init__ imports an "autotune"
        # module that isn't in the public repo.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ei_spectral_dsp", blocks / "spectral_analysis" / "dsp.py")
        if spec is None:
            raise ImportError("spectral_analysis/dsp.py missing")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        generate_features = mod.generate_features
    except (ImportError, FileNotFoundError) as e:
        raise SystemExit(
            "Spectral block code not found. Run:\n"
            "  git clone --depth 1 https://github.com/edgeimpulse/processing-blocks.git "
            "third_party/processing-blocks\n"
            "  pip install scipy PeakUtils PyWavelets\n"
            f"({e})")

    p = {
        "implementation_version": 4, "scale_axes": 1, "input_decimation_ratio": 1,
        "filter_type": "none", "filter_cutoff": 3, "filter_order": 6,
        "analysis_type": "FFT", "fft_length": 16, "do_log": True,
        "do_fft_overlap": True, "extra_low_freq": False,
        "wavelet_level": 1, "wavelet": "db4",
        "spectral_peaks_count": 3, "spectral_peaks_threshold": 0.1,
        "spectral_power_edges": "0.1, 0.5, 1.0, 2.0, 5.0",
    }
    p.update(cfg)
    axes = [n for n, _ in AXES]
    freq = 1000 / INTERVAL_MS

    import warnings

    def spectral(window):
        raw = np.asarray(window, dtype=np.float32).reshape(-1)
        with warnings.catch_warnings():
            # Flat axes (no hand, pinch=0) make skew/kurtosis warn and return NaN.
            warnings.simplefilter("ignore", RuntimeWarning)
            out = _generate(raw)
        return np.nan_to_num(np.asarray(out["features"], dtype=np.float32))

    def _generate(raw):
        return generate_features(
            p["implementation_version"], False, raw, axes, freq, p["scale_axes"],
            p["input_decimation_ratio"], p["filter_type"], p["filter_cutoff"],
            p["filter_order"], p["analysis_type"], p["fft_length"],
            p["spectral_peaks_count"], p["spectral_peaks_threshold"],
            p["spectral_power_edges"], p["do_log"], p["do_fft_overlap"],
            p["wavelet_level"], p["wavelet"], p["extra_low_freq"])

    return spectral


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--labels", required=True, help="comma separated, Studio order")
    ap.add_argument("--window-ms", type=int, default=1500)
    ap.add_argument("--dsp", choices=["spectral", "raw"], default="spectral")
    ap.add_argument("--dsp-config", help="JSON with your block's settings")
    ap.add_argument("--every-ms", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--media", action="store_true", help="send media actions (see media_control.py)")
    ap.add_argument("--backend", choices=["keys", "playerctl", "spotify"], default="keys",
                    help="spotify = control the Spotify app on macOS via AppleScript")
    ap.add_argument("--dry-run", action="store_true")
    g = ap.add_argument_group("speed trigger (default)")
    g.add_argument("--trigger", choices=["speed", "continuous"], default="speed",
                   help="speed: classify once per fast movement; continuous: every --every-ms")
    g.add_argument("--speed-start", type=float, default=300, help="mm/s to start a gesture")
    g.add_argument("--speed-end", type=float, default=150, help="mm/s to end it")
    g.add_argument("--post-ms", type=int, default=200, help="wait after the hand slows before classifying")
    g.add_argument("--segment", action="store_true",
                   help="cut the window like record.py --segment: start-aligned, padded to --window-ms. "
                        "Use with models trained on segment-mode data")
    g.add_argument("--pre-ms", type=int, default=200, help="--segment: history kept before the start")
    g.add_argument("--return-ms", type=int, default=1000, help="ignore the opposite gesture this long")
    g.add_argument("--cooldown", type=float, default=0.4, help="seconds between any two actions")
    g.add_argument("--show-speed", action="store_true", help="print palm speed to tune thresholds")
    g.add_argument("--verbose", action="store_true", help="print all scores for each classification")
    ap.add_argument("--visual", action="store_true", help="show the hand skeleton and live results (Q/Esc quits)")
    args = ap.parse_args()

    if args.segment and args.trigger != "speed":
        raise SystemExit("--segment works with the speed trigger only")
    labels = [l.strip() for l in args.labels.split(",")]
    clf = TFLiteClassifier(args.model)
    dsp = make_dsp(args.dsp, args.dsp_config)
    window = args.window_ms // INTERVAL_MS

    # Check the feature count matches the model before going live.
    n_feat = dsp([[0.0] * N_AXES] * window).size
    if n_feat != clf.n_in:
        raise SystemExit(f"DSP gives {n_feat} features but the model expects {clf.n_in}. "
                         "Check --dsp, --window-ms and the block settings in --dsp-config.")
    n_out = clf(np.zeros(n_feat)).size
    if n_out != len(labels):
        raise SystemExit(f"Model has {n_out} outputs but {len(labels)} labels were given.")

    deb = send = None
    ACTIONS = {}
    describe = lambda label: ""
    if args.media:
        from media_control import ACTIONS, Debouncer, action_name, make_sender
        send = make_sender(args.backend, args.dry_run)
        describe = lambda label: action_name(args.backend, label)
        if args.trigger == "continuous":
            deb = Debouncer(args.threshold, 3, 0.6)

    print(f"Model OK: {n_feat} features -> {labels}. Window {args.window_ms} ms. Trigger: {args.trigger}{' + segment' if args.segment else ''}.")
    view = None
    if args.visual:
        from visual import SkeletonView
        view = SkeletonView("live: " + ", ".join(labels))

    if args.trigger == "speed":
        run_speed(args, labels, clf, dsp, window, ACTIONS, send, describe, view)
        return
    shown = {"scores": "", "fired": ""}

    buf = collections.deque(maxlen=window)
    conn, listener = open_leap()
    with conn.open():
        conn.set_tracking_mode(leap.TrackingMode.Desktop)
        dt = INTERVAL_MS / 1000.0
        every = max(1, round(args.every_ms / INTERVAL_MS))
        step, last, t_next = 0, None, time.monotonic()
        while True:
            buf.append(listener.latest())
            step += 1
            if len(buf) == window and step % every == 0:
                t0 = time.perf_counter()
                probs = clf(dsp(list(buf)))
                ms = (time.perf_counter() - t0) * 1000
                scores = dict(zip(labels, probs.tolist()))
                best = max(scores, key=scores.get)
                shown["scores"] = "  ".join(f"{k} {v:.2f}" for k, v in scores.items())
                if deb is not None:
                    fired = deb.update(scores)
                    if fired:
                        print(f"{fired:>12s} -> {describe(fired)}  ({ms:.1f} ms)")
                        shown["fired"] = f"{fired} -> {describe(fired)}"
                        send(fired)
                elif scores[best] >= args.threshold and best != "idle" and best != last:
                    print(f"{best:>12s}  {scores[best]:.2f}  ({ms:.1f} ms)")
                    shown["fired"] = f"{best} {scores[best]:.2f}"
                last = best if scores[best] >= args.threshold else None
            if view and view.show(listener.skeleton(), buf[-1],
                                  ["trigger: continuous", "scores: " + shown["scores"],
                                   "last: " + shown["fired"]]) in (ord("q"), 27):
                return
            t_next += dt
            time.sleep(max(0.0, t_next - time.monotonic()))


def run_speed(args, labels, clf, dsp, window, actions, send, describe, view=None):
    """Classify once per fast movement, detected from palm speed."""
    from speed_gate import ReturnGuard, Segmenter, SpeedGate, palm_speed

    seg = None
    if args.segment:
        # Same cut as record.py --segment, so live windows match training samples.
        seg = Segmenter(INTERVAL_MS, speed_start=args.speed_start, speed_end=args.speed_end,
                        pre_ms=args.pre_ms, post_ms=args.post_ms, length_ms=args.window_ms)
        gate = seg.gate
    else:
        gate = SpeedGate(INTERVAL_MS, start=args.speed_start, end=args.speed_end, post_ms=args.post_ms)
    guard = ReturnGuard(args.return_ms)
    buf = collections.deque(maxlen=window)
    last_fire = -1e9
    shown = {"scores": "", "fired": "", "ignored": ""}
    conn, listener = open_leap()
    with conn.open():
        conn.set_tracking_mode(leap.TrackingMode.Desktop)
        dt = INTERVAL_MS / 1000.0
        step, t_next = 0, time.monotonic()
        while True:
            row = listener.latest()
            buf.append(row)
            step += 1
            if args.show_speed and step % 5 == 0:
                print(f"speed {palm_speed(row):7.0f} mm/s  [{gate.state}]")

            if seg is not None:
                sample = seg.feed(row)
            else:
                sample = list(buf) if gate.update(row) and len(buf) == window else None

            if sample is not None:
                t0 = time.perf_counter()
                probs = clf(dsp(sample))
                ms = (time.perf_counter() - t0) * 1000
                scores = dict(zip(labels, probs.tolist()))
                best = max(scores, key=scores.get)
                now = time.monotonic()
                if args.verbose:
                    print("  " + "  ".join(f"{k}={v:.2f}" for k, v in scores.items()))
                shown["scores"] = "  ".join(f"{k} {v:.2f}" for k, v in scores.items())

                if best == "idle" or scores[best] < args.threshold:
                    reason = "idle" if best == "idle" else f"low {best} {scores[best]:.2f}"
                elif not guard.allow(best, now):
                    reason = f"return stroke ({best})"
                elif now - last_fire < args.cooldown:
                    reason = "cooldown"
                else:
                    reason = None

                if reason:
                    shown["ignored"] = reason
                    if args.verbose:
                        print(f"  ignored: {reason}")
                else:
                    last_fire = now
                    guard.fired(best, now)
                    action = f" -> {describe(best)}" if best in actions else ""
                    print(f"{best:>12s}  {scores[best]:.2f}{action}  ({ms:.1f} ms)")
                    shown["fired"], shown["ignored"] = f"{best} {scores[best]:.2f}{action}", ""
                    if send and best in actions:
                        send(best)

            if view and view.show(listener.skeleton(), row,
                                  [f"trigger: {'segment' if seg else 'speed'}  [{gate.state}]  start>{args.speed_start:.0f}  end<{args.speed_end:.0f} mm/s",
                                   "scores: " + shown["scores"],
                                   "last: " + shown["fired"],
                                   ("ignored: " + shown["ignored"]) if shown["ignored"] else ""]) in (ord("q"), 27):
                return
            t_next += dt
            time.sleep(max(0.0, t_next - time.monotonic()))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
