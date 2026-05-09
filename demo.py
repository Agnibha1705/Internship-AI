import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils.detector import FaceDetector
from utils.recognizer import FaceRecognizer
from utils.drawing import draw_face_box


DATASET_DIR = "dataset"
OUTPUT_PATH = "demo_output.jpg"


def make_synthetic_face(label: int, size=(100, 100), seed=0) -> np.ndarray:
    """
    Generate a very rough synthetic grayscale 'face' image for demo purposes.
    Real usage needs actual photos from register_face.py.
    """
    rng = np.random.default_rng(seed)
    img = np.full(size, 200, dtype=np.uint8)

   
    cx, cy = size[1] // 2, size[0] // 2
    cv2.ellipse(img, (cx, cy), (cx - 10, cy - 5), 0, 0, 360,
                int(180 + label * 15), cv2.FILLED)


    eye_y = cy - 10 + label * 3
    for ex in [cx - 18, cx + 18]:
        cv2.circle(img, (ex, eye_y), 6, 40, cv2.FILLED)
        cv2.circle(img, (ex, eye_y), 3, 10, cv2.FILLED)

    
    cv2.circle(img, (cx, cy + 5), 4, int(150 - label * 10), cv2.FILLED)

    
    cv2.ellipse(img, (cx, cy + 20 + label * 2), (12, 5), 0, 0, 180, 60, 2)

    img = cv2.GaussianBlur(img, (3, 3), 1)
    return img


def create_synthetic_dataset(n_people=3, samples_per_person=20):
    names = ["Alice", "Bob", "Carol"][:n_people]
    for idx, name in enumerate(names):
        person_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        for i in range(samples_per_person):
            face = make_synthetic_face(idx, seed=idx * 1000 + i)
            cv2.imwrite(os.path.join(person_dir, f"{name}_{i:04d}.jpg"), face)
    print(f"[INFO] Created synthetic dataset: {n_people} people × {samples_per_person} samples")
    return names


def create_test_image(names):
   
    h, w = 200, 640
    img = np.full((h, w, 3), 230, dtype=np.uint8)

    
    for x in range(w):
        shade = int(200 + 30 * x / w)
        img[:, x] = [shade - 30, shade - 10, shade]

    positions = [(80, 50), (270, 50), (460, 50)]
    for i, (px, py) in enumerate(positions):
        face = make_synthetic_face(i, size=(100, 100), seed=i * 1000 + 999)
        face_bgr = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        img[py:py+100, px:px+100] = face_bgr

    return img, [(px, py, 100, 100) for (px, py) in positions]


def main():
    print("=" * 50)
    print("  Face Detection & Recognition — Demo")
    print("=" * 50)

    
    names = create_synthetic_dataset(n_people=3, samples_per_person=25)

    
    recognizer = FaceRecognizer()
    recognizer.recognizer.train(
        [make_synthetic_face(i, seed=i * 1000 + j)
         for i in range(3) for j in range(25)],
        np.array([i for i in range(3) for _ in range(25)])
    )
    recognizer.labels = {i: name for i, name in enumerate(names)}
    recognizer._trained = True
    recognizer.save()
    print(f"[INFO] Model trained on {3 * 25} samples for: {', '.join(names)}")

   
    test_img, face_rects = create_test_image(names)

    for (x, y, fw, fh), name in zip(face_rects, names):
        face_roi = test_img[y:y+fh, x:x+fw]
        pred_name, confidence = recognizer.predict(face_roi, threshold=80)
        draw_face_box(test_img, x, y, fw, fh, pred_name, confidence)

    
    cv2.putText(test_img, "Demo — Synthetic Faces", (10, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (30, 30, 100), 1, cv2.LINE_AA)

    cv2.imwrite(OUTPUT_PATH, test_img)
    print(f"\n[DONE] Output saved → {OUTPUT_PATH}")
    print("\nTo use with real faces:")
    print("  1. python register_face.py --name YourName")
    print("  2. python train.py")
    print("  3. python recognize.py --source 0")

    
    try:
        cv2.imshow("Demo Result", test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("  (No display available — see demo_output.jpg)")


if __name__ == "__main__":
    main()
