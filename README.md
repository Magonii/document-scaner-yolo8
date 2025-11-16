# Armeta * Vision

Armeta Vision is a tool for automatically detecting stamps, signatures, and QR codes inside PDF documents.
It provides a simple and intuitive web interface where users can upload: a single PDF, or a ZIP archive containing multiple PDFs.
The system processes the documents using a pretrained YOLOv8n neural network, performs object detection on every page, and generates downloadable outputs:
  a marked PDF or ZIP with all detected objects highlighted,
  a JSON file containing:
    object classifications,
    coordinates (x, y),
    width and height,
    page references.

## Installation

1. Install YoloInference Folder from repo

2. Install Python 3.11.9

For this project was used python 3.11.9:
👉 https://www.python.org/downloads/release/python-3119/

3. Install CUDA-enabled PyTorch

Go to the official PyTorch installer page and pick your CUDA version:
👉 https://pytorch.org/get-started/locally/

4. Install Python libraries

```bash
pip install gradio
pip install ultralytics
pip install pymupdf
pip install opencv-python
pip install numpy
```
## Usage

1. Open file GradioSetup.py from folder and run
2. It give output with  "Running on local URL:  http://127.0.0.1:XXXX"
3. Copy that http and paste it to browser
4. Choose or drop file into the site and press run
5. Donwload the file or json
