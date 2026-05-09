import cv2
import numpy as np
import os
import argparse
from utils.recognizer import FaceRecognizer, preprocess

DATASET_DIR = "dataset"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the LBPH face recognizer (v2)")
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip augmentation (faster but less accurate)")
    return parser.parse_args()


def augment_face(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape
    cx, cy = w // 2, h // 2
    variants = [img]

   
    for gamma in [0.5, 0.7, 1.4, 1.8]:
        table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                          for i in range(256)], dtype=np.uint8)
        variants.append(cv2.LUT(img, table))

    
    for alpha, beta in [(0.75, 20), (0.9, -15), (1.1, 15), (1.25, -20)]:
        variants.append(np.clip(img.astype(np.float32) * alpha + beta,
                                0, 255).astype(np.uint8))

    
    shadow = img.copy().astype(np.float32)
    shadow[:h//2, :] *= 0.55          
    variants.append(shadow.clip(0, 255).astype(np.uint8))

    shadow2 = img.copy().astype(np.float32)
    shadow2[:, :w//2] *= 0.55         
    variants.append(shadow2.clip(0, 255).astype(np.uint8))

    
    noise = np.random.normal(0, 12, img.shape).astype(np.float32)
    variants.append(np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8))

    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    variants.append(cv2.filter2D(img, -1, kernel))

    
    variants.append(cv2.GaussianBlur(img, (5, 5), 1.5))

    

    
    variants.append(cv2.flip(img, 1))

    
    for angle in [-15, -10, -5, 5, 10, 15]:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REPLICATE))

   
    for dx, dy in [(-5, 0), (5, 0), (0, -5), (0, 5), (-4, -4), (4, 4)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        variants.append(cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REPLICATE))

   
    for scale in [0.92, 1.08]:
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        variants.append(cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REPLICATE))

    return variants



def load_dataset(dataset_dir: str, augment: bool):
    faces, label_ids = [], []
    label_map: dict[str, int] = {}
    labels: dict[int, str] = {}
    current_id = 0

    print(f"\n[INFO] Scanning: {dataset_dir}")

    for person_name in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        if person_name not in label_map:
            label_map[person_name] = current_id
            labels[current_id] = person_name
            current_id += 1

        pid = label_map[person_name]
        raw_images = []

        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue
           
            raw_images.append(preprocess(img))

        if not raw_images:
            print(f"  [SKIP] {person_name} — no valid images found")
            continue

        person_faces = []
        if augment:
            for img in raw_images:
                person_faces.extend(augment_face(img))
        else:
            person_faces = raw_images

        faces.extend(person_faces)
        label_ids.extend([pid] * len(person_faces))

        raw_count = len(raw_images)
        aug_count = len(person_faces)
        print(f"  • {person_name:20s}  {raw_count:3d} raw → {aug_count:4d} after augment  (id={pid})")

    return faces, np.array(label_ids), labels



def suggest_threshold(n_people: int, samples_per_person: float) -> float:
    """Heuristic threshold recommendation."""
    if samples_per_person < 20:
        return 75.0
    elif samples_per_person < 50:
        return 85.0
    else:
        return 95.0


def main():
    args = parse_args()
    augment = not args.no_augment

    if not os.path.exists(args.dataset):
        print(f"[ERROR] Dataset not found: {args.dataset}")
        print("        Run: python register_face.py --name YourName")
        return

    faces, label_ids, labels = load_dataset(args.dataset, augment)

    if not faces:
        print("[ERROR] No training images found.")
        return

    avg_samples = len(faces) / max(len(labels), 1)
    print(f"\n[INFO] Total samples : {len(faces)}")
    print(f"[INFO] Persons       : {len(labels)}  ({', '.join(labels.values())})")
    print(f"[INFO] Avg per person: {avg_samples:.0f}")
    print(f"[INFO] Training LBPH model ...")

    recognizer = FaceRecognizer()
    recognizer.recognizer.train(faces, label_ids)
    recognizer.labels = labels
    recognizer._trained = True
    recognizer.save()

    threshold = suggest_threshold(len(labels), avg_samples)
    print(f"\n[DONE] Model saved.")
    print(f"[TIP]  Recommended threshold for your dataset size: --threshold {threshold:.0f}")
    print(f"\n       python recognize.py --source 0 --dnn --threshold {threshold:.0f}")


if __name__ == "__main__":
    main()
