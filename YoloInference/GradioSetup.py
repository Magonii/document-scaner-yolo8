# GradioSetup.py

import os
import shutil
import uuid
import zipfile
import json
from pathlib import Path

import gradio as gr

from nn_infer import run_inference_on_pdf  # direct import instead of subprocess

# === БАЗОВЫЕ ПУТИ (все относительно текущего файла) ===
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output_images"
OUTPUT_JSON_DIR = BASE_DIR / "output_json"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_JSON_DIR.mkdir(exist_ok=True)


def clear_uploads():
    """Полностью очищает uploads/, создаёт заново."""
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def clear_outputs():
    """Очищает output_images и output_json полностью."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if OUTPUT_JSON_DIR.exists():
        shutil.rmtree(OUTPUT_JSON_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# ОДИНОЧНЫЙ PDF
# -----------------------------
def process_single_pdf(temp_pdf_path: str):
    # uploads уже очищен в process_file
    original_name = Path(temp_pdf_path).name
    input_path = UPLOAD_DIR / original_name
    shutil.copy2(temp_pdf_path, input_path)

    try:
        # прямой вызов Python-функции без subprocess (значительно быстрее)
        run_inference_on_pdf(input_path)
    except Exception as e:
        clear_uploads()
        return None, None, f"❌ Error during NN inference:\n{e}"

    output_pdf = OUTPUT_DIR / original_name
    json_file = OUTPUT_JSON_DIR / f"{Path(original_name).stem}_predictions.json"

    clear_uploads()

    if not output_pdf.exists():
        return None, None, "❌ NN finished but annotated PDF not found."

    if not json_file.exists():
        extra = " (JSON file not found)"
        json_path_str = None
    else:
        extra = f" JSON saved: {json_file.name}"
        json_path_str = str(json_file)

    return str(output_pdf), json_path_str, f"✔ Done!{extra}"


# -----------------------------
# ZIP С НЕСКОЛЬКИМИ PDF
# -----------------------------
def process_zip(zip_path: str):
    # uploads уже очищен в process_file
    job_id = uuid.uuid4().hex
    zip_stem = Path(zip_path).stem
    temp_dir = UPLOAD_DIR / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    shutil.unpack_archive(zip_path, temp_dir)

    pdfs = [p for p in temp_dir.rglob("*.pdf")]
    if not pdfs:
        clear_uploads()
        return None, None, "❌ No PDFs inside ZIP."

    merged = {}
    annotated_pdfs = []

    for pdf in pdfs:
        orig_name = pdf.name
        temp_name = f"{job_id}__{orig_name}"
        dst = UPLOAD_DIR / temp_name
        shutil.copy2(pdf, dst)

        try:
            # вызываем YOLO один раз на процесс, на каждый PDF — без запуска отдельного python-процесса
            run_inference_on_pdf(dst)
        except Exception as e:
            clear_uploads()
            return None, None, f"❌ Error during inference of {orig_name}:\n{e}"

        out_pdf = OUTPUT_DIR / temp_name
        out_json = OUTPUT_JSON_DIR / f"{Path(temp_name).stem}_predictions.json"

        if out_json.exists():
            content = json.loads(out_json.read_text(encoding="utf-8"))
            key = list(content.keys())[0]
            # ВОССТАНАВЛИВАЕМ ИМЯ БЕЗ job_id (как и раньше)
            clean_name = key.split("__", 1)[1] if "__" in key else key
            merged[clean_name] = content[key]

        if out_pdf.exists():
            annotated_pdfs.append((out_pdf, orig_name))

        # временный json этого PDF больше не нужен — результат сливается в merged
        if out_json.exists():
            out_json.unlink()

    # ZIP с аннотированными PDF (имена файлов внутри — оригинальные)
    result_zip = OUTPUT_DIR / f"{zip_stem}_annotated.zip"
    with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for temp_pdf_path, clean_name in annotated_pdfs:
            z.write(temp_pdf_path, arcname=clean_name)

    # merged JSON
    merged_json_path = OUTPUT_JSON_DIR / f"{zip_stem}_predictions.json"
    merged_json_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    clear_uploads()

    return (
        str(result_zip),
        str(merged_json_path),
        f"✔ Done! JSON saved: {merged_json_path.name}",
    )


# -----------------------------
# MAIN DISPATCHER
# -----------------------------
def process_file(file_path: str):
    if file_path is None:
        return None, None, "Upload PDF or ZIP first."

    # Очистка перед каждым прогоном
    clear_outputs()
    clear_uploads()

    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return process_single_pdf(file_path)

    if ext == ".zip":
        return process_zip(file_path)

    clear_uploads()
    return None, None, "❌ Unsupported file type. Please upload .pdf or .zip."


# -----------------------------
# GRADIO UI (BRIGHT CYAN THEME)
# -----------------------------

custom_css = """
.gradio-container {
    background: linear-gradient(180deg, #f0faff 0%, #e9f5ff 40%, #ffffff 100%);
    color: #0f172a;
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Navigation bar */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 32px;
    margin-bottom: 18px;
    border-bottom: 1px solid rgba(59, 130, 246, 0.2);
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
}

.nav-left {
    display: flex;
    align-items: center;
    gap: 18px;
}

.nav-logo {
    font-weight: 700;
    font-size: 20px;
    color: #0c4a6e;
}

.nav-tagline {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
}

.nav-links {
    display: flex;
    gap: 16px;
    margin-left: 24px;
    font-size: 14px;
    color: #0369a1;
}

.nav-link:hover {
    color: #0284c7;
    cursor: default;
}

.nav-right {
    display: flex;
    gap: 12px;
}

.nav-btn {
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 13px;
    border: 1px solid rgba(3, 105, 161, 0.3);
    background: white;
    color: #075985;
    cursor: default;
}

.nav-btn.primary {
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    color: white;
    border: none;
}

/* Main wrapper center */
.main-wrapper {
    max-width: 900px;
    margin: 0 auto;
    padding: 30px 20px 50px;
}

/* Card */
.tool-card {
    background: white;
    border-radius: 20px;
    padding: 28px;
    box-shadow:
        0 10px 25px rgba(0, 0, 0, 0.06),
        0 0 0 1px rgba(3, 105, 161, 0.15);
}

.tool-title {
    font-size: 24px;
    font-weight: 700;
    color: #0c4a6e;
}

.tool-subtitle {
    font-size: 14px;
    color: #475569;
    margin-bottom: 18px;
}

/* Input and output panels */
.input-panel, .output-panel {
    background: #f8fcff;
    border-radius: 14px;
    padding: 14px 16px;
    border: 1px solid rgba(0, 119, 182, 0.25);
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.04);
}

/* Button */
.primary-btn button {
    width: 100%;
    background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
    border: none !important;
    padding: 10px 0 !important;
    box-shadow: 0 8px 20px rgba(3, 105, 161, 0.25);
}

.primary-btn button:hover {
    transform: translateY(-1px);
}

/* Status box */
.status-box textarea {
    background: #f0f9ff !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0, 119, 182, 0.35) !important;
}

/* Output file previews */
.output-panel .file-preview {
    border-radius: 12px;
    border: 1px dashed rgba(3, 105, 161, 0.4);
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML(
        """
        <div class="nav-bar">
            <div class="nav-left">
                <div>
                    <div class="nav-logo">ARMETA • VISION</div>
                    <div class="nav-tagline">Bright & Vibrant UI</div>
                </div>
                <div class="nav-links">
                    <span class="nav-link">Home</span>
                    <span class="nav-link">Docs</span>
                    <span class="nav-link">Pricing</span>
                </div>
            </div>
            <div class="nav-right">
                <button class="nav-btn">Sign in</button>
                <button class="nav-btn primary">Register</button>
            </div>
        </div>
        """
    )

    with gr.Column(elem_classes=["main-wrapper"]):
        with gr.Group(elem_classes=["tool-card"]):
            gr.HTML(
                """
                <div class="tool-title">📄 PDF Object Detection Tool</div>
                <div class="tool-subtitle">
                    Upload a PDF or ZIP and receive annotated results + a JSON with detections.
                </div>
                """
            )

            with gr.Row():
                with gr.Column(scale=1, elem_classes=["input-panel"]):
                    file_in = gr.File(label="Upload PDF or ZIP", type="filepath")
                    run_btn = gr.Button(
                        "Run detection",
                        elem_classes=["primary-btn"],
                    )

                with gr.Column(scale=1, elem_classes=["output-panel"]):
                    out_file = gr.File(label="Download annotated PDF / ZIP")
                    out_json = gr.File(label="Download JSON predictions")
                    logs = gr.Textbox(
                        label="Status / Errors",
                        lines=6,
                        elem_classes=["status-box"],
                    )

            run_btn.click(
                process_file,
                inputs=file_in,
                outputs=[out_file, out_json, logs],
            )

if __name__ == "__main__":
    demo.launch()
