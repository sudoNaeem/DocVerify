import logging
from fastapi import FastAPI, HTTPException, File, UploadFile,Query
from contextlib import asynccontextmanager
import fitz  # PyMuPDF
import io
from PIL import Image
import os
from datetime import datetime
import cv2
from utils import (
     process_pdf_file, extract_images, resize_pdf, get_filenames_and_annotations,extract_text,process_pdf_extract_images_and_save_high_res)
import psycopg2
import boto3


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

POSTGRESQL_CONNECTION_STRING='postgresql://postgres.tjnvqtfpfcarwaqcpugt:c3jmkacJGhKD4e@aws-0-us-west-1.pooler.supabase.com:6543/postgres'
AWS_ACCESS_KEY_ID='AKIAWTYXWIPAGEVLW2RZ'
AWS_SECRET_ACCESS_KEY='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'
S3_BUCKET_NAME='pdfsignaturedetection'


# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)



app = FastAPI()

@app.get("/list_templates/", operation_id="List_Templates")
async def list_templates():
    start_time = datetime.now()
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
    finally:
        end_time = datetime.now()
        logger.info(f"Time taken to list templates: {end_time - start_time}")

@app.post("/SignatureDetection/")
async def upload_pdfs(filename: str,
                    Scanned: UploadFile = File(...),
                    Deskewing: bool = Query(False, description="To Deskew Scanned PDFS, it will increase Computing Time")):
    start_time = datetime.now()

    if not Scanned.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")
    # if not (0 <= Threshold <= 1):
    #     raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")

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
        
        retrieval_end_time = datetime.now()
        logger.info(f"Time taken to retrieve template: {retrieval_end_time - start_time}")

        # if Deskewing:
        #     processed_scanned_buffer = process_pdf_file(scanned_bytes)
        #     #resized_scanned_buffer = resize_pdf(processed_scanned_buffer.getvalue(), template_bytes)
        #     new_pdf = process_pdf_extract_images_and_save_high_res(processed_scanned_buffer)
        # else: 
        #     new_pdf = process_pdf_extract_images_and_save_high_res(scanned_bytes)

        if Deskewing:
            # Process the scanned bytes with deskewing
            processed_scanned_buffer = process_pdf_file(scanned_bytes)
            
            # Extract images and save high resolution to create a new PDF
            new_pdf = process_pdf_extract_images_and_save_high_res(processed_scanned_buffer)
            
            # Now, resize the new_pdf to match the template PDF size
            resized_new_pdf = resize_pdf(new_pdf.getvalue(), template_bytes)
            
        else: 
            # Extract images and save high resolution to create a new PDF without deskewing
            new_pdf = process_pdf_extract_images_and_save_high_res(scanned_bytes)
            
            # Now, resize the new_pdf to match the template PDF size
            resized_new_pdf = resize_pdf(new_pdf.getvalue(), template_bytes)

        resize_end_time = datetime.now()
        logger.info(f"Time taken to resize PDFs + Deskewing: {resize_end_time - retrieval_end_time}")



        #doc_template = fitz.open(stream=template_bytes, filetype="pdf")
        doc_scanned = fitz.open(stream=resized_new_pdf, filetype="pdf")
        #output_images_template = extract_images(doc_template, annotations_info)
        output_images_scanned = extract_images(doc_scanned, annotations_info)
        
        extraction_end_time = datetime.now()
        logger.info(f"Time taken to extract images: {extraction_end_time - resize_end_time}")

        results = []   
        for img_scanned, annotation in zip(output_images_scanned, annotations_info):
            param_type = annotation.get("label_type", "Text")
            ocr, is_present = extract_text(param_type, cv2.imencode('.png', img_scanned)[1].tobytes())
            logger.info(f"Ocr extracted text is:{ocr}")
            if is_present == False:
                ocr = " "

            results.append({
                "pageNumber": annotation["page_number"],
                "tagId": annotation["label"],
                "tagName":annotation["label_type"],
                "isPresent": is_present,
                "data": ocr.splitlines() if isinstance(ocr, str) else ocr,
            })
        end_time = datetime.now()
        logger.info(f"Total time taken for signature detection: {end_time - start_time}")

        return {"data": results}
    except Exception as e:
        logger.error(f"Error during signature detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
