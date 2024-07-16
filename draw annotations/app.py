from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from scipy.ndimage import rotate
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import json
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO

app = FastAPI()

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

def remove_white_margins(image):
    img = np.array(image)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    mask = img < 255 - 5
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return image.crop((y0, x0, y1, x1))

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
        margin_removed_image = remove_white_margins(final_image)
        processed_images.append(margin_removed_image.convert('RGB'))

    output_buffer = BytesIO()
    processed_images[0].save(output_buffer, format='PDF', save_all=True, append_images=processed_images[1:])
    output_buffer.seek(0)
    return output_buffer

def extract_images(doc, annotations_info):
    output_images = []
    for annotation in annotations_info:
        page_number = annotation["page_number"]
        x0, y0, x1, y1 = annotation["start_x"], annotation["start_y"], annotation["end_x"], annotation["end_y"]
        page = doc.load_page(page_number)
        clip = fitz.Rect(x0, y0, x1, y1)
        pix = page.get_pixmap(clip=clip)
        img = cv2.imdecode(np.frombuffer(pix.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        output_images.append(img)
    return output_images

def compute_vgg16_similarity(img1, img2):
    model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    def get_features(img):
        img = cv2.resize(img, (224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return model.predict(img).flatten()

    f1, f2 = get_features(img1), get_features(img2)
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))

def resize_pdf(scan_pdf_bytes, template_pdf_bytes):
    scan_reader = PdfReader(BytesIO(scan_pdf_bytes))
    template_reader = PdfReader(BytesIO(template_pdf_bytes))
    scan_writer = PdfWriter()

    template_page = template_reader.pages[0]
    template_page_width = template_page.mediabox.width
    template_page_height = template_page.mediabox.height

    for page_num in range(len(scan_reader.pages)):
        scan_page = scan_reader.pages[page_num]
        scan_page.scale_to(template_page_width, template_page_height)
        scan_writer.add_page(scan_page)

    output_buffer = BytesIO()
    scan_writer.write(output_buffer)
    output_buffer.seek(0)
    return output_buffer

@app.post("/Signature Detection/", operation_id="Signature_Detection")
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
