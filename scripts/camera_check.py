"""Save one frame from each camera index so you can tell which is which.

  .venv/bin/python scripts/camera_check.py            # writes data/camera_check/cam<N>.jpg

Run it from Terminal.app the first time: macOS asks for Camera access for Terminal (click Allow).
Then use the matching indices for --front-cam (overhead) and --wrist-cam (on the gun).
"""

import time
from pathlib import Path

import cv2

OUT = Path(__file__).resolve().parents[1] / "data" / "camera_check"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"camera {i}: not available")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame, t0 = None, time.time()
        while time.time() - t0 < 3:  # cameras need a moment to expose
            ok, f = cap.read()
            if ok and f is not None and f.mean() > 3:
                frame = f
                break
        cap.release()
        if frame is None:
            print(f"camera {i}: opened but no image")
            continue
        path = OUT / f"cam{i}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"camera {i}: {frame.shape[1]}x{frame.shape[0]} → {path}")


if __name__ == "__main__":
    main()
