import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import uvicorn
# from pdf2image import convert_from_bytes
from PIL import Image
import openai
import base64
import fitz  # PyMuPDF
from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

# ---------- Configuration ----------
openai.api_key = os.getenv("OPENAI_API_KEY")   # Set your key in the environment
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # GPT model with vision capability
DPI = int(os.getenv("CONVERT_DPI", "200"))     # Raster DPI for pdf2image
# -----------------------------------

app = FastAPI(
    title="Drawing Inspector Backend",
    description="Splits a PDF drawing into pages and inspects each page for ISO/ASME drafting errors.",
    version="1.0.0"
)

# ---------- NEW health-check endpoint ----------
@app.get("/health")
async def health():
    """Render uptime pinger—no OpenAI cost."""
    return {"status": "ok"}
# ----------------------------------------------

PROMPT = (
    "You are a technical drawing reviewer with ISO/ASME drafting expertise.\n\n"
    "Analyze the provided drawing page image and list any of the following issues you detect. "
    "Return a concise bullet-point list. For each issue, mention its approximate location "
    "(e.g., 'top-left corner').\n\n"
    "Issues to detect:\n"
    "• Missing dimensions\n"
    "• Missing tags\n"
    "• Non-aligned views\n"
    "• Non-aligned view tags\n"
    "• Non-aligned dimensions\n"
    "• Overlapping dimensions\n"
    "• Non-consecutive marks of elements (in schedule)\n"
    "• Non-consecutive numbers of views on sheet\n"
    "• Missing or unreadable dimension lines\n"
    "• Unlabeled components\n"
    "• Misaligned annotations\n"
    "• Repeated or overlapping elements\n"
    "• Scale inconsistencies\n"
    "• Missing title block fields (author, revision date, scale)\n"
    "• Incorrect use of standard symbols"
)

def _pil_to_png_bytes(img: Image.Image) -> bytes:
    """Convert PIL Image to PNG byte stream."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _inspect_page(img: Image.Image) -> str:
    # 1) Convert image → PNG bytes → base64 string
    png_bytes = _pil_to_png_bytes(img)
    b64_data  = base64.b64encode(png_bytes).decode("ascii")
    data_url  = f"data:image/png;base64,{b64_data}"

    # 2) Build the multimodal message in the format the API expects
    messages: Iterable[dict] = cast(Iterable[dict], [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                }
            ],
        }
    ])

    resp = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def split_pdf_to_images(pdf_bytes: bytes):
    """Render each PDF page to a PIL Image using PyMuPDF (pure Python)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    try:
        for page in doc:
            # Render at desired DPI → PyMuPDF wants zoom
            zoom = DPI / 72  # 72 dpi is PDF default
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    finally:
        doc.close()
    return images

@app.post("/inspectDrawing")
async def inspect_drawing(file: UploadFile = File(None), request: Request = None):
    """
    Accepts a PDF via multipart/form-data, splits it into pages,
    and runs each page through GPT-4o-vision for drafting issue detection.
    """
    print("---- incoming request ----")
    print("Content-Type:", request.headers.get("content-type"))
    # NOTE: FastAPI reads the stream only once, so don't print body here.
    # ----------------------------------------

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    try:
        pages = split_pdf_to_images(pdf_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {exc}")

    results = []
    for idx, img in enumerate(pages, start=1):
        page_report = _inspect_page(img)
        results.append({"page": idx, "issues": page_report})

    return {
        "filename": file.filename,
        "page_count": len(pages),
        "results": results
    }

if __name__ == "__main__":
    # Local dev: uvicorn drawing_inspector_backend:app --reload
    uvicorn.run(
        "drawing_inspector_backend:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
