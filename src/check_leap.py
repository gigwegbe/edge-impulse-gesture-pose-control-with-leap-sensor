"""Step 1 sanity check: is the Gemini service running, is the Leap detected,
and is it tracking? Prints connection, device serial, frame rate and palm
position once per second. Ctrl+C to quit."""

import time

import leap

from leap_features import open_leap


def main():
    conn, listener = open_leap()
    with conn.open():
        conn.set_tracking_mode(leap.TrackingMode.Desktop)
        print("Waiting for the tracking service... (hold a hand ~20 cm above the sensor)")
        last_frames = 0
        while True:
            time.sleep(1)
            fps = listener.frames - last_frames
            last_frames = listener.frames
            f = listener.latest()
            hand = f"palm=({f[0]:7.1f}, {f[1]:7.1f}, {f[2]:7.1f}) mm pinch={f[9]:.2f} grab={f[10]:.2f}" \
                if listener.hand_present else "no hand"
            print(f"service={'ok' if listener.connected else '...'} "
                  f"device={listener.device_serial or '?'} fps={fps:3d}  {hand}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
