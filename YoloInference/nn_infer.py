# nn_infer.py

from __future__ import annotations

from pathlib import Path
import json

import fitz  # pymupdf
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------
# CONFIG
# -----------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"
OUTPUT_IMG_ROOT = BASE_DIR / "output_images"
OUTPUT_JSON_ROOT = BASE_DIR / "output_json"

TARGET_WIDTH = 1190
TARGET_HEIGHT = 1684

ID_TO_CLASS_NAME = {
    0: "signature",
    1: "stamp",
    2: "qr",
}

BATCH_SIZE = 8  # pages per YOLO call (for efficiency, without changing outputs)


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

# single model instance reused for all calls
MODEL = YOLO(str(MODEL_PATH))


def box_iou(b1, b2) -> float:
    x1 = max(b1["x1"], b2["x1"])
    y1 = max(b1["y1"], b2["y1"])
    x2 = min(b1["x2"], b2["x2"])
    y2 = min(b1["y2"], b2["y2"])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    a1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    a2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0

    return inter / union


def filter_overlapping_signatures(dets, iou_thresh: float = 0.45):
    sig = [d for d in dets if d["cls_name"] == "signature"]
    others = [d for d in dets if d["cls_name"] != "signature"]

    sig = sorted(sig, key=lambda d: d["conf"], reverse=True)

    kept = []
    for d in sig:
        if all(box_iou(d, k) < iou_thresh for k in kept):
            kept.append(d)

    return kept + others


def _prepare_page_data(doc):
    """
    Pre-render all pages to images and precompute scale factors.
    Returns list of dicts with all per-page data.
    """
    pages_data = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1

        pix = page.get_pixmap(dpi=200)
        w_img, h_img = pix.width, pix.height

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h_img, w_img, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # portrait vs landscape for JSON page size
        if h_img >= w_img:
            page_width_json = TARGET_WIDTH
            page_height_json = TARGET_HEIGHT
        else:
            page_width_json = TARGET_HEIGHT
            page_height_json = TARGET_WIDTH

        # scaling to JSON coordinates
        scale_x_json = page_width_json / w_img
        scale_y_json = page_height_json / h_img

        # scaling to real PDF coordinates
        scale_x_pdf = page.rect.width / w_img
        scale_y_pdf = page.rect.height / h_img

        pages_data.append(
            {
                "page": page,
                "page_idx": page_idx,
                "page_num": page_num,
                "img": img,
                "page_width_json": page_width_json,
                "page_height_json": page_height_json,
                "scale_x_json": scale_x_json,
                "scale_y_json": scale_y_json,
                "scale_x_pdf": scale_x_pdf,
                "scale_y_pdf": scale_y_pdf,
            }
        )

    return pages_data


def run_inference_on_pdf(pdf_path: str | Path, batch_size: int = BATCH_SIZE) -> Path:
    """
    Runs YOLO inference on a PDF and produces:
      - annotated PDF in OUTPUT_IMG_ROOT (same filename)
      - JSON with annotations in OUTPUT_JSON_ROOT (same format as original version)

    Returns path to JSON file.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    OUTPUT_IMG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_ROOT.mkdir(parents=True, exist_ok=True)

    result = {pdf_path.name: {}}

    with fitz.open(str(pdf_path)) as doc:
        pages_data = _prepare_page_data(doc)
        if not pages_data:
            # empty PDF
            out_pdf = OUTPUT_IMG_ROOT / pdf_path.name
            doc.save(str(out_pdf))
        else:
            # process pages in batches to reduce per-call overhead
            for start in range(0, len(pages_data), max(1, batch_size)):
                chunk = pages_data[start : start + batch_size]
                imgs = [p["img"] for p in chunk]

                # YOLO inference on batch of images
                yolo_results = MODEL(imgs, verbose=False)

                # ultralytics returns a list-like for batched input
                for p_info, yolo_res in zip(chunk, yolo_results):
                    raw = []
                    for box in yolo_res.boxes:
                        cls_id = int(box.cls.item())
                        cls_name = ID_TO_CLASS_NAME.get(cls_id, "unknown")

                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        raw.append(
                            {
                                "cls_name": cls_name,
                                "conf": float(box.conf.item()),
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                            }
                        )

                    dets = filter_overlapping_signatures(raw)
                    if not dets:
                        continue

                    page_num = p_info["page_num"]
                    page_key = f"page_{page_num}"
                    page_ann = []

                    # keep annotation numbering per page starting from 1 (same as before)
                    counter = 1

                    scale_x_json = p_info["scale_x_json"]
                    scale_y_json = p_info["scale_y_json"]
                    scale_x_pdf = p_info["scale_x_pdf"]
                    scale_y_pdf = p_info["scale_y_pdf"]

                    page_width_json = p_info["page_width_json"]
                    page_height_json = p_info["page_height_json"]

                    page_obj = p_info["page"]

                    for det in dets:
                        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                        cls_name = det["cls_name"]
                        conf = det["conf"]

                        w = x2 - x1
                        h = y2 - y1

                        # JSON coords (orientation-aware, same as original)
                        x_json = round(x1 * scale_x_json, 2)
                        y_json = round(y1 * scale_y_json, 2)
                        w_json = round(w * scale_x_json, 2)
                        h_json = round(h * scale_y_json, 2)

                        ann_name = f"annotation_{counter}"
                        counter += 1

                        page_ann.append(
                            {
                                ann_name: {
                                    "category": cls_name,
                                    "bbox": {
                                        "x": x_json,
                                        "y": y_json,
                                        "width": w_json,
                                        "height": h_json,
                                    },
                                    "area": round(w_json * h_json, 2),
                                }
                            }
                        )

                        # Draw bounding box on PDF (same coordinates logic as before)
                        page_obj.draw_rect(
                            fitz.Rect(
                                x1 * scale_x_pdf,
                                y1 * scale_y_pdf,
                                x2 * scale_x_pdf,
                                y2 * scale_y_pdf,
                            ),
                            color=(0, 1, 0),
                            width=0.7,
                        )

                        page_obj.insert_textbox(
                            fitz.Rect(
                                x1 * scale_x_pdf,
                                y1 * scale_y_pdf - 12,
                                x2 * scale_x_pdf,
                                y1 * scale_y_pdf + 12,
                            ),
                            f"{cls_name} {conf:.2f}",
                            fontsize=8,
                            color=(0, 1, 0),
                        )

                    # store annotations & page size in result dict (identical schema)
                    result[pdf_path.name][page_key] = {
                        "annotations": page_ann,
                        "page_size": {
                            "width": page_width_json,
                            "height": page_height_json,
                        },
                    }

            out_pdf = OUTPUT_IMG_ROOT / pdf_path.name
            doc.save(str(out_pdf))

    json_path = OUTPUT_JSON_ROOT / f"{pdf_path.stem}_predictions.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"JSON → {json_path}")
    print(f"PDF  → {out_pdf}")

    return json_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_file = Path(sys.argv[1])
    else:
        uploads = BASE_DIR / "uploads"
        pdfs = list(uploads.glob("*.pdf"))
        if not pdfs:
            print("Нет PDF в uploads/")
            raise SystemExit(1)
        pdf_file = pdfs[0]

    run_inference_on_pdf(pdf_file)
