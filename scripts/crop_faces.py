"""Cut square head crops out of a folder of photos and write a training manifest.

    python scripts/crop_faces.py photos/ -o data/

Faces are found with a YOLO face detector. Each face box is expanded to cover hair
and chin (margins are fractions of the box), made square, and clamped to the frame by
shrinking rather than padding. Heads smaller than --min-face-px or mostly outside the
frame are skipped. Crops mirror the input tree under OUT_DIR, and OUT_DIR/train.jsonl
lists them as {"image": ..., "caption": ""}, the format scripts/train_*.py read. The
VAE needs no captions; fill them in before training the DiT.

The release crops were made with these defaults. Detector weights (AGPL-3.0):
    curl -fsSL -o models/yolov10l-face.pt \\
        https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov10l-face.pt
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def head_box(face: np.ndarray, width: int, height: int, args) -> np.ndarray | None:
    """Detector box -> square head box inside the frame, or None if it fails the filters."""
    x1, y1, x2, y2 = face
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - args.margin_side * w, x2 + args.margin_side * w
    y1, y2 = y1 - args.margin_top * h, y2 + args.margin_bottom * h
    cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) / 2
    box = np.array([cx - half, cy - half, cx + half, cy + half])
    if 2 * half < args.min_face_px:
        return None
    clamped = np.array([max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])])
    if 1 - area(clamped) / area(box) > args.max_clamp_loss:
        return None
    # Growing the square back would only be clamped away again, so shrink to fit.
    side = min(clamped[2] - clamped[0], clamped[3] - clamped[1])
    cx, cy = (clamped[0] + clamped[2]) / 2, (clamped[1] + clamped[3]) / 2
    if side < args.min_face_px:
        return None
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2])


def area(box: np.ndarray) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = area(np.array([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]))
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def detect(model, images: list[Image.Image], args) -> list[np.ndarray]:
    """Boxes [N, 5] (xyxy + confidence) in original-image pixels. Large images are
    downscaled first: the detector runs at --imgsz anyway, and a full-size copy in its
    preprocessing costs memory for nothing. The detector gets BGR arrays: batches of
    PIL images of mixed sizes come back with different confidences."""
    small, scales = [], []
    for image in images:
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        s = min(1.0, args.max_det_side / max(image.size))
        if s < 1:
            bgr = cv2.resize(bgr, (round(image.width * s), round(image.height * s)), interpolation=cv2.INTER_AREA)
        small.append(bgr)
        scales.append(s)
    results = model.predict(small, conf=args.conf, iou=args.nms_iou, imgsz=args.imgsz, verbose=False)
    out = []
    for r, s in zip(results, scales, strict=True):
        xyxy = r.boxes.xyxy.cpu().numpy() / s
        out.append(np.hstack([xyxy, r.boxes.conf.cpu().numpy()[:, None]]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="folder of images, searched recursively")
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("--weights", default="models/yolov10l-face.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="detector threshold (fed to NMS)")
    parser.add_argument("--min-conf", type=float, default=0.7, help="keep faces above this; drops false positives")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640, help="detector input size")
    parser.add_argument("--max-det-side", type=int, default=1024, help="downscale larger images for detection")
    parser.add_argument("--batch", type=int, default=16, help="images per detector pass")
    parser.add_argument("--margin-top", type=float, default=0.45, help="hair and forehead, x box height")
    parser.add_argument("--margin-bottom", type=float, default=0.20, help="chin and jaw, x box height")
    parser.add_argument("--margin-side", type=float, default=0.12, help="x box width")
    parser.add_argument("--min-face-px", type=int, default=256, help="minimum side of the head crop")
    parser.add_argument("--max-clamp-loss", type=float, default=0.7, help="skip if the frame cuts off more area")
    parser.add_argument("--max-size", type=int, default=0, help="downscale larger crops to this side (0: native)")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    paths = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in range(0, len(paths), args.batch):
        batch, images = [], []
        for path in paths[start : start + args.batch]:
            try:
                images.append(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
                batch.append(path)
            except OSError as error:
                print(f"skip {path}: {error}")
        for path, image, faces in zip(batch, images, detect(model, images, args), strict=True):
            kept = []
            for k, face in enumerate(faces):
                if face[4] < args.min_conf:
                    continue
                box = head_box(face[:4], image.width, image.height, args)
                # Two detections of one face can expand to the same head box.
                if box is None or any(iou(box, other) >= 0.5 for other in kept):
                    continue
                kept.append(box)
                x1, y1 = round(box[0]), round(box[1])
                side = min(round(box[2]) - x1, round(box[3]) - y1)
                crop = image.crop((x1, y1, x1 + side, y1 + side))
                if args.max_size and side > args.max_size:
                    crop = crop.resize((args.max_size, args.max_size), Image.BOX)
                rel = path.relative_to(args.input).with_suffix("")
                name = Path("images") / rel.parent / f"{rel.name}_face{k}.jpg"
                (args.out_dir / name).parent.mkdir(parents=True, exist_ok=True)
                crop.save(args.out_dir / name, quality=95)
                rows.append({"image": name.as_posix(), "caption": ""})
        print(f"{min(start + args.batch, len(paths))}/{len(paths)} images, {len(rows)} crops", flush=True)

    manifest = args.out_dir / "train.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
