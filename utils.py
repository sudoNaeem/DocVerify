import logging
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from scipy.ndimage import rotate
from PIL import Image
import pytesseract
import base64
import requests
import fitz  # PyMuPDF
from io import BytesIO
import json
from decimal import Decimal
from PyPDF2 import PdfReader, PdfWriter
import os
import psycopg2
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


POSTGRESQL_CONNECTION_STRING='postgresql://postgres.tjnvqtfpfcarwaqcpugt:c3jmkacJGhKD4e@aws-0-us-west-1.pooler.supabase.com:6543/postgres'
AWS_ACCESS_KEY_ID='AKIAWTYXWIPAGEVLW2RZ'
AWS_SECRET_ACCESS_KEY='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'
S3_BUCKET_NAME='pdfsignaturedetection'


def correct_skew(image, delta=1, limit=12):
    def determine_score(arr, angle):
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]
    return best_angle

def deskew_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def detect_orientation(image):
    try:
        rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        osd = pytesseract.image_to_osd(rgb_image)
        rotate_angle = 0
        if "Rotate: 180" in osd:
            rotate_angle = 180
        return rotate_angle
    except Exception as e:
        return 0

def process_pdf_file(file_bytes):
    images = convert_from_bytes(file_bytes)
    processed_images = []

    for image in images:
        rotate_angle = detect_orientation(image)
        if rotate_angle == 180:
            image = image.rotate(180, expand=True)
        angle = correct_skew(image)
        deskewed_image = deskew_image(np.array(image), angle)
        final_image = Image.fromarray(deskewed_image)
        #margin_removed_image = remove_white_margins(final_image)
        processed_images.append(final_image.convert('RGB'))
        #processed_images.append(margin_removed_image.convert('RGB'))

    output_buffer = BytesIO()
    processed_images[0].save(output_buffer, format='PDF', save_all=True, append_images=processed_images[1:])
    output_buffer.seek(0)
    return output_buffer


# import os

# def extract_images(doc, annotations_info, output_folder="output_images"):
#     output_images = []
    
#     # Ensure the output directory exists
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     for idx, annotation in enumerate(annotations_info):
#         page_number = annotation["page_number"] - 1  # Adjust for 0-indexing
#         x0, y0, x1, y1 = annotation["start_x"], annotation["start_y"], annotation["end_x"], annotation["end_y"]
        
#         # Load the page
#         page = doc.load_page(page_number)
        
#         # Get page dimensions
#         page_width, page_height = page.rect.width, page.rect.height
        
#         # Ensure the coordinates are within the page bounds
#         x0 = max(0, min(page_width, x0))
#         y0 = max(0, min(page_height, y0))
#         x1 = max(0, min(page_width, x1))
#         y1 = max(0, min(page_height, y1))
        
#         # Create a clipping rectangle with validated coordinates
#         clip = fitz.Rect(x0, y0, x1, y1)
        
#         try:
#             # Get the pixmap using the validated clip
#             pix = page.get_pixmap(clip=clip)
            
#             # Convert pixmap to image
#             img = cv2.imdecode(np.frombuffer(pix.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
#             output_images.append(img)
            
#             # Save the image locally
#             output_image_path = os.path.join(output_folder, f"image_{page_number + 1}_{idx + 1}.png")
#             cv2.imwrite(output_image_path, img)
            
#         except Exception as e:
#             logging.error(f"Error processing page {page_number + 1}: {str(e)}")
#             raise HTTPException(status_code=500, detail=f"Error processing page {page_number + 1}: {str(e)}")
    
#     return output_images

import fitz  # PyMuPDF
import cv2
import numpy as np
import logging
from fastapi import HTTPException

def extract_images(doc, annotations_info,filename,client):
    
    output_images = []
    
    for annotation in annotations_info:
        page_number = annotation["page_number"] - 1  # Adjust for 0-indexing
        tag_id = annotation["label"]  # Use the tag ID for naming
        
        x0, y0, x1, y1 = annotation["start_x"], annotation["start_y"], annotation["end_x"], annotation["end_y"]
        
        # Load the page
        page = doc.load_page(page_number)
        
        # Get page dimensions
        page_width, page_height = page.rect.width, page.rect.height
        
        # Ensure the coordinates are within the page bounds
        x0 = max(0, min(page_width, x0))
        y0 = max(0, min(page_height, y0))
        x1 = max(0, min(page_width, x1))
        y1 = max(0, min(page_height, y1))
        
        # Create a clipping rectangle with validated coordinates
        clip = fitz.Rect(x0, y0, x1, y1)
        
        try:
            # Get the pixmap using the validated clip
            pix = page.get_pixmap(clip=clip)
            
            # Convert pixmap to image
            img = cv2.imdecode(np.frombuffer(pix.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            output_images.append(img)
            
            # Save the image to S3
            s3_key = f"images/{filename}/page#{page_number + 1}_{tag_id}.png"
            _, img_encoded = cv2.imencode('.png', img)
            client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=img_encoded.tobytes(), ContentType='image/png')
        
        except Exception as e:
            logger.error(f"Error processing page {page_number + 1}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing page {page_number + 1}: {str(e)}")
    
    return output_images



def to_float(value):
    if isinstance(value, Decimal):
        return float(value)
    return value

def resize_pdf(scan_pdf_bytes, template_pdf_bytes):
    scan_reader = PdfReader(BytesIO(scan_pdf_bytes))
    template_reader = PdfReader(BytesIO(template_pdf_bytes))
    scan_writer = PdfWriter()

    template_page = template_reader.pages[0]
    template_page_width = to_float(template_page.mediabox.width)
    template_page_height = to_float(template_page.mediabox.height)

    for page_num in range(len(scan_reader.pages)):
        scan_page = scan_reader.pages[page_num]
        scan_page.scale_to(template_page_width, template_page_height)
        scan_writer.add_page(scan_page)

    output_buffer = BytesIO()
    scan_writer.write(output_buffer)
    output_buffer.seek(0)
    return output_buffer.getvalue()



def get_filenames_and_annotations():
    #connection_string = os.getenv("POSTGRESQL_CONNECTION_STRING")
    conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
    cursor = conn.cursor()
    cursor.execute("SELECT pdf_name, annotations FROM annotations")
    documents = cursor.fetchall()
    results = {}
    for document in documents:
        filename, annotations = document
        if filename:
            results[filename] = json.loads(annotations) if isinstance(annotations, str) else annotations
    cursor.close()
    conn.close()
    return results


def extract_text(param_type, image_bytes, temperature=0.2):
        api_key = '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'
        def encode_image(image_bytes):
            return base64.b64encode(image_bytes).decode('utf-8')
        
        base64_image = encode_image(image_bytes)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        prompts = {
                "Name": """You are an OCR for handwritten Names. Extract all handwritten names from the image.\n
                            Only return handwritten text.\n
                            Not all names are english so dont guess names.\n
                            Example: For Ishan Madhan Return Ishan Madhan.\n
                            detect character by character and join them. Don't guess what the word is.\n 
                            If no name is detected, return 'no name detected'.\n
                            DONT ADD PUNCTUATION IN OUTPUT and Don't write 'the name is: ' .\n
                            Give the exact name that you detect, do not try to guess. Give only handwritten names.""",
                
                "Date": """You are an OCR for handwritten Dates. Extract all handwritten dates from the image.\n
                            Example: If the image contains '12/05/2023', return '12/05/2023'.\n
                            If no date is detected, return 'no date detected'.\n
                            and Don't write 'the date is: '\n
                            Give the exact date that you detect, do not try to guess. Give only handwritten dates.""",
                
                "Signature": """You are an OCR for signatures. Detect the signatures from the image.\n
                                If no signature is detected, return 'no signature detected'.\n
                                DONT ADD PUNCTUATION IN OUTPUT and Don't write 'the signature is: '.\n
                                If you detect a signature return 'signature'.""",
                
                "Checkbox": """You are an OCR for detecting checkboxes. Determine if the checkbox is 'marked' or 'not marked'.\n
                                Example: If the checkbox is checked, return 'marked'. If not checked, return 'not marked'.\n
                                DONT ADD PUNCTUATION IN OUTPUT.\n
                                If no checkbox is detected, return 'no checkbox detected'.\n
                                Do not say anything else.""",
                
                "Text": """You are an OCR for text and numbers. Extract all text from the image, Extract Handwritten text too.\n
                            Don't guess what the word is.\n
                            Detect numbers too.\n
                            If no text is detected, return 'no text detected'.\n
                            Don't write 'the Text is: '"""
            }

        if param_type not in prompts:
            raise ValueError(f"Invalid param_type '{param_type}'. Must be one of {list(prompts.keys())}.")
        selected_prompt = prompts[param_type]

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": selected_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": temperature
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_content = response.json()
        ocr_content = response_content['choices'][0]['message']['content'].strip()
        
        non_meaningful_outputs = {
            "Name": ['empty', 'unknown', '',' ', 'no name detected', 'name not found'],
            "Date": ['empty', 'unknown', '',' ', 'no date detected', 'date not found'],
            "Signature": ['empty', 'unknown', '',' ', 'no signature detected', 'signature not found'],
            "Checkbox": ['empty', 'unknown', '',' ', 'not marked', 'no checkbox detected','no checkbox'],
            "Text": ['empty', 'unknown', '',' ', 'no text detected', 'text not found','empty.', 'unknown.', '', 'no text detected.', 'text not found.']
        }

        is_meaningful = ocr_content.lower() not in non_meaningful_outputs[param_type]

        is_present = is_meaningful and len(ocr_content) > 0
        
        return ocr_content, is_present




def process_pdf_extract_images_and_save_high_res(pdf_buffer,output_path="output.pdf", padding=20, tolerance=160, dpi=400):
    # Ensure pdf_buffer is a BytesIO object
    if isinstance(pdf_buffer, bytes):
        pdf_buffer = BytesIO(pdf_buffer)
    
    # Open the PDF document using PyMuPDF
    doc = fitz.open(stream=pdf_buffer.getvalue(), filetype="pdf")
    images = []
    
    # Get the bounding box from the first page
    first_page = doc.load_page(0)
    first_pix = first_page.get_pixmap(dpi=dpi)
    first_img = Image.frombytes("RGB", [first_pix.width, first_pix.height], first_pix.samples)
    img_array = np.array(first_img)
    
    # Convert to grayscale for easier processing
    img_gray = np.mean(img_array, axis=2) if img_array.ndim == 3 else img_array
    
    # Create a mask to identify non-white pixels
    mask = img_gray < tolerance
    
    # Find the coordinates of the non-white pixels
    coords = np.argwhere(mask)
    
    # If no non-white pixels are found, raise an exception
    if coords.size == 0:
        raise ValueError("No non-white content found.")
    
    # Get the bounding box of the non-white pixels
    (y0, x0), (y1, x1) = coords.min(axis=0), coords.max(axis=0) + 1
    
    # Process each page using the bounding box calculated from the first page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_array = np.array(img)
        
        # Apply padding to the bounding box
        x0_padded = max(x0 - padding, 0)
        y0_padded = max(y0 - padding, 0)
        x1_padded = min(x1 + padding, img_array.shape[1])
        y1_padded = min(y1 + padding, img_array.shape[0])
        
        # Crop the image (left, upper, right, lower)
        cropped_img = img.crop((x0_padded, y0_padded, x1_padded, y1_padded))
        
        # Append the cropped image to the list
        images.append(cropped_img)
    if images:
        images[0].save(output_path, format="PDF", save_all=True, append_images=images[1:], resolution=dpi)

    # Save the processed images as a high-resolution PDF to a BytesIO buffer
    output_buffer = BytesIO()
    if images:
        images[0].save(output_buffer, format="PDF", save_all=True, append_images=images[1:], resolution=dpi)
    
    output_buffer.seek(0)
    return output_buffer

