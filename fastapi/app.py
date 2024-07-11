from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import shutil
import cv2
import numpy as np
from pdf2image import convert_from_path
from scipy.ndimage import rotate
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import imagehash

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
    rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    osd = pytesseract.image_to_osd(rgb_image)
    rotate_angle = 0
    if "Rotate: 180" in osd:
        rotate_angle = 180
    return rotate_angle

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

def process_pdf_file(file_path):
    images = convert_from_path(file_path)
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

    output_path = os.path.splitext(file_path)[0] + "-processed.pdf"
    processed_images[0].save(output_path, save_all=True, append_images=processed_images[1:])
    return output_path

def extract_annotations(doc, color='red'):
    annotations_info = []
    color_map = {'red': ([0, 0, 250], [10, 10, 255])}

    for page_number in range(len(doc)):
        page = doc[page_number]
        pix = page.get_pixmap()

        if pix.samples:
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            lower_color, upper_color = color_map[color] if color in color_map else ([0, 0, 0], [255, 255, 255])
            lower_red = np.array(lower_color)
            upper_red = np.array(upper_color)

            mask = cv2.inRange(image, lower_red, upper_red)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for rect in [cv2.boundingRect(cnt) for cnt in contours]:
                x0, y0, w, h = rect
                x1, y1 = x0 + w, y0 + h
                annotations_info.append((page_number + 1, x0, y0, x1, y1))

    return annotations_info

def remove_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        img = img[y:y+h, x:x+w]
    return img, (x, y, w, h)

def extract_images(doc, annotations_info, output_folder, remove_border_flag=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_images = []
    new_annotations_info = []
    for idx, (page_number, x0, y0, x1, y1) in enumerate(annotations_info):
        page = doc.load_page(page_number - 1)
        clip = fitz.Rect(x0, y0, x1, y1)
        pix = page.get_pixmap(clip=clip)
        img = cv2.imdecode(np.frombuffer(pix.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if remove_border_flag:
            img, (border_x, border_y, border_w, border_h) = remove_border(img)
            new_x0 = x0 + border_x
            new_y0 = y0 + border_y
            new_x1 = x0 + border_x + border_w
            new_y1 = y0 + border_y + border_h
            new_annotations_info.append((page_number, new_x0, new_y0, new_x1, new_y1))
        else:
            new_annotations_info.append((page_number, x0, y0, x1, y1))
        output_images.append(img)
        img_path = os.path.join(output_folder, f"extracted_image_{idx + 1}.png")
        cv2.imwrite(img_path, img)
    return output_images, new_annotations_info

def adjust_annotations_for_pdf2(original_annotations, source_rect, target_rect):
    adjusted_annotations = []
    x_scale = target_rect.width / source_rect.width
    y_scale = target_rect.height / source_rect.height
    for (page_number, x0, y0, x1, y1) in original_annotations:
        new_x0 = x0 * x_scale
        new_y0 = y0 * y_scale
        new_x1 = x1 * x_scale
        new_y1 = y1 * y_scale
        adjusted_annotations.append((page_number, new_x0, new_y0, new_x1, new_y1))
    return adjusted_annotations

def compute_hash_similarity(img1, img2):
    img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    hash1 = imagehash.phash(img1_pil)
    hash2 = imagehash.phash(img2_pil)
    similarity = 1 - (hash1 - hash2) / len(hash1.hash) ** 2
    return similarity

@app.post("/Signature Detection/",operation_id="Signature_Detection")
async def upload_pdfs(Template: UploadFile = File(...), Scanned: UploadFile = File(...), threshold: float = 0.61):
    if not Template.filename.endswith('.pdf') or not Scanned.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload PDF files.")
    if not (0 <= threshold <= 1):
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")

    try:
        temp_template_path = f"temp_{Template.filename}"
        temp_scanned_path = f"temp_{Scanned.filename}"
        with open(temp_template_path, 'wb') as out_file:
            shutil.copyfileobj(Template.file, out_file)
        with open(temp_scanned_path, 'wb') as out_file:
            shutil.copyfileobj(Scanned.file, out_file)

        processed_scanned_path = process_pdf_file(temp_scanned_path)
        doc_template = fitz.open(temp_template_path)
        doc_scanned = fitz.open(processed_scanned_path)
        annotations_info_template = extract_annotations(doc_template)
        output_folder_template = "Template_images"
        output_folder_scanned = "Scanned_images"
        output_images_template, new_annotations_info_template = extract_images(doc_template, annotations_info_template, output_folder_template, remove_border_flag=True)
        source_rect = doc_template[0].rect
        target_rect = doc_scanned[0].rect
        adjusted_annotations_scanned = adjust_annotations_for_pdf2(annotations_info_template, source_rect, target_rect)
        output_images_scanned, _ = extract_images(doc_scanned, adjusted_annotations_scanned, output_folder_scanned, remove_border_flag=True)
        page_results = {}
        for idx, (img_template, img_scanned, annotation) in enumerate(zip(output_images_template, output_images_scanned, annotations_info_template)):
            score = compute_hash_similarity(img_template, img_scanned)
            is_signed = score < threshold
            page_number = annotation[0]
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
        os.remove(temp_template_path)
        os.remove(temp_scanned_path)
        return {"Template": Template.filename, "Scanned_original": Scanned.filename, "Scanned_processed": processed_scanned_path, "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
