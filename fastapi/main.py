from fastapi import FastAPI, File, UploadFile, HTTPException
import json
from contextlib import asynccontextmanager
import fitz  # PyMuPDF
from utils import (
    load_vgg16_model,correct_skew, deskew_image, detect_orientation, remove_white_margins,
    process_pdf_file, extract_images, compute_vgg16_similarity, resize_pdf
)

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vgg16_model()
    yield

app = FastAPI(lifespan=lifespan)


@app.post("/SignatureDetection/", operation_id="Signature_Detection")
async def upload_pdfs(Template: UploadFile = File(...), Scanned: UploadFile = File(...), threshold: float = 0.8, json_file: UploadFile = File(...)):
    if not Template.filename.endswith('.pdf') or not Scanned.filename.endswith('.pdf') or not json_file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload PDF and JSON files.")
    if not (0 <= threshold <= 1):
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")

    try:
        template_bytes = await Template.read()
        scanned_bytes = await Scanned.read()
        json_bytes = await json_file.read()
        annotations_info = json.loads(json_bytes)

        processed_scanned_buffer = process_pdf_file(scanned_bytes)
        resized_scanned_buffer = resize_pdf(processed_scanned_buffer.getvalue(), template_bytes)
        
        doc_template = fitz.open(stream=template_bytes, filetype="pdf")
        doc_scanned = fitz.open(stream=resized_scanned_buffer.getvalue(), filetype="pdf")
        
        output_images_template = extract_images(doc_template, annotations_info)
        output_images_scanned = extract_images(doc_scanned, annotations_info)
        
        page_results = {}
        for img_template, img_scanned, annotation in zip(output_images_template, output_images_scanned, annotations_info):
            score = compute_vgg16_similarity(img_template, img_scanned)
            is_signed = score < threshold
            page_number = annotation["page_number"]
            if page_number not in page_results:
                page_results[page_number] = {"signed": 0, "unsigned": 0}
            if is_signed:
                page_results[page_number]["signed"] += 1
            else:
                page_results[page_number]["unsigned"] += 1
        result = [
            f"page no. {page} has {info['signed']} signed boxes and {info['unsigned']} unsigned boxes"
            for page, info in page_results.items()
        ]
        
        return {"Template": Template.filename, "Scanned_original": Scanned.filename, "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
