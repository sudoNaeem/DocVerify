from fastapi import FastAPI, HTTPException, File, UploadFile
from contextlib import asynccontextmanager
import fitz  # PyMuPDF
import os
from pymongo import MongoClient
from utils import (
    load_vgg16_model, process_pdf_file, extract_images, compute_vgg16_similarity, resize_pdf, get_filenames_and_annotations
)
import json

app = FastAPI()

MONGO_CONNECTION_STRING = "mongodb+srv://admin:admin@cluster0.rykip0e.mongodb.net/"
client = MongoClient(MONGO_CONNECTION_STRING)
db = client["pdf_annotations"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vgg16_model()
    yield

app = FastAPI(lifespan=lifespan)

def get_filenames_and_annotations():
    connection_string='mongodb+srv://admin:admin@cluster0.rykip0e.mongodb.net/'
    client = MongoClient(connection_string)
    db = client.pdf_annotations
    collection = db.annotations
    documents = collection.find()
    results = {}
    for document in documents:
        filename = document.get('pdf_name')
        annotations = document.get('annotations')
        if filename:
            results[filename] = annotations
    return results

@app.get("/list_templates/", operation_id="List_Templates")
async def list_templates():
    templates = db.annotations.find({}, {"pdf_name": 1, "_id": 0})
    return {"templates": list(templates)}

@app.post("/SignatureDetection/")
async def upload_pdfs(filename: str, Scanned: UploadFile = File(...), threshold: float = 0.8):
    if not Scanned.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")
    if not (0 <= threshold <= 1):
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")

    try:
        scanned_bytes = await Scanned.read()
        annotations = get_filenames_and_annotations()
        if filename not in annotations:
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

        annotations_info = annotations[filename]

        template_path = f'pdfs/{filename}'
        with open(template_path, 'rb') as f:
            template_bytes = f.read()

        processed_scanned_buffer = process_pdf_file(scanned_bytes)
        resized_scanned_buffer = resize_pdf(processed_scanned_buffer.getvalue(), template_bytes)
        doc_template = fitz.open(stream=template_bytes, filetype="pdf")
        doc_scanned = fitz.open(stream=resized_scanned_buffer.getvalue(), filetype="pdf")
        output_images_template = extract_images(doc_template, annotations_info)
        output_images_scanned = extract_images(doc_scanned, annotations_info)
        
        results = []
        for img_template, img_scanned, annotation in zip(output_images_template, output_images_scanned, annotations_info):
            score = compute_vgg16_similarity(img_template, img_scanned)
            is_present = bool(score < threshold)
            results.append({
                "pageNumber": annotation["page_number"],
                "tagId": annotation["label"],
                "isPresent": is_present
            })
        
        return {"data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
