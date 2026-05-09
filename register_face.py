import cv2
import os
import argparse
import time
from utils.detector import FaceDetector
from utils.recognizer import FaceRecognizer
from utils.drawing import draw_face_box

DATASET_DIR = "dataset"


def parse_args():
    parser = argparse.ArgumentParser(description="Register a new face")
    parser.add_argument("--name", required=True, help="Person's name (used as folder name)")
    parser.add_argument("--samples", type=int, default=30,
                        help="Number of face samples to capture (default: 30)")
    parser.add_argument("--source", default="0",
                        help="Camera source (default: 0 = first webcam)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between auto-captures (default: 0.3)")
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.name.strip()
    save_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    # Count existing samples
    existing = [f for f in os.listdir(save_dir)
                if f.lower().endswith((".jpg", ".png"))]
    start_idx = len(existing)
    print(f"[INFO] Existing samples for '{name}': {start_idx}")
    print(f"[INFO] Will capture {args.samples} more samples → {save_dir}")
    print("[INFO] Press SPACE to manually capture | 'q' to quit")

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {args.source}")
        return

    detector = FaceDetector(use_dnn=False)
    captured = 0
    last_capture = 0

    while captured < args.samples:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        faces = detector.detect(frame)

        # Overlay
        progress = f"Captured: {captured}/{args.samples}"
        cv2.putText(display, progress, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

        for (x, y, w, h) in faces:
            draw_face_box(display, x, y, w, h, label=name)

        cv2.imshow(f"Registering: {name}  [SPACE=capture | q=quit]", display)

        now = time.time()
        key = cv2.waitKey(1) & 0xFF

        # Auto-capture if face found and delay elapsed
        should_capture = (key == ord(' ')) or (now - last_capture >= args.delay and faces)

        if should_capture and faces:
            # Save the largest face crop
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            margin = int(min(w, h) * 0.1)
            fx = max(0, x - margin)
            fy = max(0, y - margin)
            fw = min(frame.shape[1] - fx, w + 2 * margin)
            fh = min(frame.shape[0] - fy, h + 2 * margin)
            face_img = cv2.cvtColor(frame[fy:fy+fh, fx:fx+fw], cv2.COLOR_BGR2GRAY)

            fname = os.path.join(save_dir, f"{name}_{start_idx + captured:04d}.jpg")
            cv2.imwrite(fname, face_img)
            captured += 1
            last_capture = now
            print(f"  [{captured:02d}/{args.samples}] Saved {fname}")

        if key == ord('q'):
            print("[INFO] Quit early.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] Captured {captured} samples for '{name}'.")
    print("[INFO] Now retrain the model with:  python train.py")

    # Offer to retrain immediately
    if captured > 0:
        answer = input("Retrain model now? [y/N]: ").strip().lower()
        if answer == 'y':
            recognizer = FaceRecognizer()
            n = recognizer.train(DATASET_DIR)
            print(f"[INFO] Training complete — {n} sample(s), "
                  f"{len(recognizer.labels)} person(s).")


if __name__ == "__main__":
    main()
