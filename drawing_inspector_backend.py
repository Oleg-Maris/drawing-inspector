import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from pdf2image import convert_from_bytes
from PIL import Image
import openai
import base64

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

PROMPT = (
    "You are a technical drawing reviewer with ISO/ASME drafting expertise. "
    "Analyze the provided drawing page image and list any of the following issues you detect. "
    "Return a concise bullet-point list. For each issue, mention its approximate location "
    "(e.g., 'top-left corner')."
    "Issues to detect:"
    "• Missing dimensions"
    "• Missing tags"
    "• Non-aligned views"
    "• Non-aligned view tags"
    "• Non-aligned dimensions"
    "• Overlapping dimensions"
    "• Non-consecutive marks of elements (in schedule)"
    "• Non-consecutive numbers of views on sheet"
    "• Missing or unreadable dimension lines"
    "• Unlabeled components"
    "• Misaligned annotations"
    "• Repeated or overlapping elements"
    "• Scale inconsistencies"
    "• Missing title block fields (author, revision date, scale)"
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
    resp = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": PROMPT },
                    {
                        "type": "image_url",
                        "image_url": { "url": data_url }
                    }
                ],
            }
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

@app.post("/inspectDrawing")
async def inspect_drawing(file: UploadFile = File(...)):
    """
    Accepts a PDF via multipart/form-data, splits it into pages,
    and runs each page through GPT-4o-vision for drafting issue detection.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=DPI)
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
