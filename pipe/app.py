from fastapi import FastAPI, HTTPException, File, UploadFile
from contextlib import asynccontextmanager
import fitz  # PyMuPDF
import os
from pymongo import MongoClient
from utils import (
    load_vgg16_model, process_pdf_file, extract_images, compute_vgg16_similarity, resize_pdf, get_filenames_and_annotations
)

app = FastAPI()

MONGO_CONNECTION_STRING = "mongodb+srv://admin:admin@cluster0.rykip0e.mongodb.net/"
client = MongoClient(MONGO_CONNECTION_STRING)
db = client["pdf_annotations"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vgg16_model()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/list_templates/", operation_id="List_Templates")
async def list_templates():
    templates = db.annotations.find({}, {"pdf_name": 1, "_id": 0})
    return {"templates": list(templates)}

@app.post("/SignatureDetection/", operation_id="Signature_Detection")
async def upload_pdfs(Scanned: UploadFile = File(...), threshold: float = 0.8, pdf_name: str = ''):
    if not Scanned.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")
    if not (0 <= threshold <= 1):
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")
    if not pdf_name:
        raise HTTPException(status_code=400, detail="PDF name must be provided.")

    try:
        pdfs_info = get_filenames_and_annotations(MONGO_CONNECTION_STRING)
        pdf_info = next((item for item in pdfs_info if item['filename'] == pdf_name), None)
        
        if not pdf_info:
            raise HTTPException(status_code=404, detail="PDF not found in the database.")
        
        template_path = os.path.join(os.getcwd(), 'pdfs', pdf_info['filename'])
        if not os.path.exists(template_path):
            raise HTTPException(status_code=404, detail="Template PDF not found locally.")
        
        annotations_info = pdf_info["annotations"]

        with open(template_path, "rb") as f:
            template_bytes = f.read()

        scanned_bytes = await Scanned.read()
        processed_scanned_buffer = process_pdf_file(scanned_bytes)
        resized_scanned_buffer = resize_pdf(processed_scanned_buffer.getvalue(), template_bytes)
        
        doc_template = fitz.open(stream=template_bytes, filetype="pdf")
        doc_scanned = fitz.open(stream=resized_scanned_buffer.getvalue(), filetype="pdf")

        if len(doc_template) != len(doc_scanned):
            raise HTTPException(status_code=400, detail="Template and scanned PDF lengths do not match.")

        results = []
        for annotation in annotations_info:
            page_number = annotation["page_number"] - 1
            if page_number >= len(doc_template) or page_number >= len(doc_scanned):
                raise HTTPException(status_code=400, detail=f"Page number {annotation['page_number']} is out of range.")

            try:
                img_template = extract_images(doc_template, [annotation])[0]
                img_scanned = extract_images(doc_scanned, [annotation])[0]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error extracting images: {str(e)}")

            score = compute_vgg16_similarity(img_template, img_scanned)
            is_present = bool(score < threshold)
            results.append({
                "pageNumber": annotation["page_number"],
                "tagId": annotation["label"],
                "isPresent": is_present,
                "coordinates": {
                    "start_x": annotation["start_x"],
                    "start_y": annotation["start_y"],
                    "end_x": annotation["end_x"],
                    "end_y": annotation["end_y"]
                }
            })
        
        return {"data": results}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
