import urllib.request
import os
import sys

MODELS_DIR = "models"
FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}


def download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  [SKIP] Already exists: {dest}")
        return

    print(f"  Downloading {os.path.basename(dest)} ...")
    try:
        def _progress(count, block, total):
            pct = count * block / total * 100 if total > 0 else 0
            print(f"\r    {min(pct, 100):.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\r    Done — {os.path.getsize(dest) / 1e6:.1f} MB")
    except Exception as e:
        print(f"\n  [ERROR] Failed to download: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        sys.exit(1)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Downloading DNN model files to ./{MODELS_DIR}/\n")

    for fname, url in FILES.items():
        dest = os.path.join(MODELS_DIR, fname)
        download(url, dest)

    print("\n[DONE] Models ready.")
    print("       Run  python recognize.py --source 0 --dnn  to use the DNN detector.")


if __name__ == "__main__":
    main()
