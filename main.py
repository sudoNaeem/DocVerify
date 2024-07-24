import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
from contextlib import asynccontextmanager
import fitz  # PyMuPDF
import io
import os
from dotenv import load_dotenv
import cv2
from utils import (
    load_vgg16_model, process_pdf_file, extract_images, compute_vgg16_similarity, resize_pdf, get_filenames_and_annotations, detect_document_words)
import psycopg2
import boto3

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

POSTGRESQL_CONNECTION_STRING = os.getenv("POSTGRESQL_CONNECTION_STRING")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vgg16_model()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/list_templates/", operation_id="List_Templates")
async def list_templates():
    try:
        conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT pdf_name FROM annotations")
        templates = cursor.fetchall()
        cursor.close()
        conn.close()
        template_list = [template[0] for template in templates]
        return {"templates": template_list}
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/SignatureDetection/")
async def upload_pdfs(filename: str, Scanned: UploadFile = File(...), threshold: float = 0.5):
    if not Scanned.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")
    if not (0 <= threshold <= 1):
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")

    try:
        scanned_bytes = await Scanned.read()
        annotations = get_filenames_and_annotations()
        if filename not in annotations:
            logger.warning(f"File '{filename}' not found in annotations.")
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

        annotations_info = annotations[filename]

        # Retrieve the template PDF from S3
        s3_key = f"pdfs/{filename}"
        logger.info(f"Retrieving template from S3 with key: {s3_key}")
        try:
            s3_response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
            template_bytes = s3_response['Body'].read()
            logger.info(f"Successfully retrieved template '{filename}' from S3")
        except s3_client.exceptions.NoSuchKey:
            logger.error(f"Template file '{filename}' not found in S3 bucket.")
            raise HTTPException(status_code=404, detail=f"Template file '{filename}' not found in S3 bucket.")
        except Exception as e:
            logger.error(f"Error retrieving template file '{filename}' from S3: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving template file '{filename}' from S3: {str(e)}")

        processed_scanned_buffer = process_pdf_file(scanned_bytes)
        resized_scanned_buffer = resize_pdf(processed_scanned_buffer.getvalue(), template_bytes)
        doc_template = fitz.open(stream=template_bytes, filetype="pdf")
        doc_scanned = fitz.open(stream=resized_scanned_buffer.getvalue(), filetype="pdf")
        output_images_template = extract_images(doc_template, annotations_info)
        output_images_scanned = extract_images(doc_scanned, annotations_info)
        
        results = []   
        for img_template, img_scanned, annotation in zip(output_images_template, output_images_scanned, annotations_info):
            score = float(compute_vgg16_similarity(img_template, img_scanned))
            is_present = bool(score < threshold)
            ocr_text_scanned = detect_document_words(cv2.imencode('.png', img_scanned)[1].tobytes())
            results.append({
                "pageNumber": annotation["page_number"],
                "tagId": annotation["label"],
                "isPresent": is_present,
                "data": ocr_text_scanned.splitlines(),
            })
        
        return {"data": results}
    except Exception as e:
        logger.error(f"Error during signature detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
