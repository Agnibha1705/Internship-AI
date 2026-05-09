import cv2
import numpy as np
import os
import argparse
import pickle
import time
from utils.detector import FaceDetector
from utils.recognizer import FaceRecognizer
from utils.drawing import draw_face_box

def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection & Recognition")
    parser.add_argument("--source", default="0",
                        help="Source: '0' for webcam, path to image/video file")
    parser.add_argument("--dnn", action="store_true",
                        help="Use DNN-based detector instead of Haar cascade")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Minimum confidence for DNN detection (default: 0.5)")
    parser.add_argument("--threshold", type=float, default=80.0,
                        help="LBPH recognition threshold — lower = stricter (default: 80)")
    parser.add_argument("--no-recognize", action="store_true",
                        help="Detection only, skip recognition")
    return parser.parse_args()


def process_frame(frame, detector, recognizer, threshold, detect_only):
    """Detect faces in a frame and optionally recognize them."""
    faces = detector.detect(frame)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        label = "Unknown"
        confidence = None

        if not detect_only and recognizer.is_trained():
            label, confidence = recognizer.predict(face_roi, threshold)

        draw_face_box(frame, x, y, w, h, label, confidence)

    return frame, len(faces)


def run_on_image(path, detector, recognizer, threshold, detect_only):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Could not load image: {path}")
        return

    result, n = process_frame(frame, detector, recognizer, threshold, detect_only)
    print(f"[INFO] Detected {n} face(s) in {path}")

    out_path = "output_" + os.path.basename(path)
    cv2.imwrite(out_path, result)
    print(f"[INFO] Result saved to: {out_path}")

    cv2.imshow("Face Detection & Recognition", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_video(source, detector, recognizer, threshold, detect_only):
    # Try webcam index or file path
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    fps_start = time.time()
    frame_count = 0

    print("[INFO] Press 'q' to quit | 's' to save screenshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, n_faces = process_frame(frame, detector, recognizer, threshold, detect_only)

        # FPS overlay
        frame_count += 1
        elapsed = time.time() - fps_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(result, f"FPS: {fps:.1f}  Faces: {n_faces}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Detection & Recognition  [q=quit | s=screenshot]", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"screenshot_{ts}.jpg"
            cv2.imwrite(fname, result)
            print(f"[INFO] Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    print("[INFO] Initialising face detector...")
    detector = FaceDetector(use_dnn=args.dnn, min_confidence=args.confidence)

    print("[INFO] Loading face recognizer...")
    recognizer = FaceRecognizer()
    if not args.no_recognize:
        if recognizer.is_trained():
            print(f"[INFO] Recognizer loaded — {len(recognizer.labels)} known person(s): "
                  f"{', '.join(recognizer.labels.values())}")
        else:
            print("[WARN] No trained model found. Run train.py first, or use --no-recognize.")

    source = args.source
    # Detect if source is an image file
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(source)[1].lower()

    if ext in image_exts:
        run_on_image(source, detector, recognizer, args.threshold, args.no_recognize)
    else:
        run_on_video(source, detector, recognizer, args.threshold, args.no_recognize)


if __name__ == "__main__":
    main()
