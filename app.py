import fitz  # PyMuPDF
import cv2
import numpy as np
import os
from PIL import Image
import imagehash
import gradio as gr

# Function to extract annotations
def extract_annotations(doc):
    annotations_info = []
    for page_number, page in enumerate(doc):
        if page.annots():
            for annot in page.annots(types=fitz.PDF_ANNOT_SQUARE):
                rect = annot.rect  # Get the rectangle bounding box
                annotations_info.append((page_number + 1, rect.x0, rect.y0, rect.x1, rect.y1))  # Store page and coordinates
    return annotations_info

# Function to extract images from a PDF using coordinates
def extract_images(doc, annotations_info, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_images = []
    for idx, (page_number, x0, y0, x1, y1) in enumerate(annotations_info):
        page = doc.load_page(page_number - 1)  # Page numbers are 0-based in PyMuPDF
        clip = fitz.Rect(x0, y0, x1, y1)  # Define the clipping rectangle
        pix = page.get_pixmap(clip=clip)  # Get the pixmap for the clipped region
        img = cv2.imdecode(np.frombuffer(pix.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        output_images.append(img_rgb)
        img_path = os.path.join(output_folder, f"extracted_image_{idx + 1}.png")
        cv2.imwrite(img_path, img_rgb)
    return output_images

# Function to compute hash similarity between two images
def compute_hash_similarity(img1, img2):
    # Convert images to PIL format
    img1_pil = Image.fromarray(img1)
    img2_pil = Image.fromarray(img2)
    # Compute pHash
    hash1 = imagehash.phash(img1_pil)
    hash2 = imagehash.phash(img2_pil)
    # Compute similarity
    similarity = 1 - (hash1 - hash2) / len(hash1.hash) ** 2
    return similarity

# Function to process PDFs and compute similarities
def process_pdfs(pdf1_path, pdf2_path):
    # Load the PDF files
    doc1 = fitz.open(pdf1_path)
    doc2 = fitz.open(pdf2_path)

    # Extract annotations from the first PDF
    annotations_info1 = extract_annotations(doc1)

    # Extract images from both PDFs using the coordinates and save them in separate folders
    output_folder1 = "extracted_images_pdf1"
    output_folder2 = "extracted_images_pdf2"
    output_images1 = extract_images(doc1, annotations_info1, output_folder1)
    output_images2 = extract_images(doc2, annotations_info1, output_folder2)

    # Calculate hash similarity for each pair of images
    similarities = []
    extracted_images = []
    for img1, img2 in zip(output_images1, output_images2):
        score = compute_hash_similarity(img1, img2)
        similarities.append(score)
        extracted_images.append(img2)

    # Return similarities and images from the second PDF
    return similarities, extracted_images

# Gradio interface
def gradio_interface(pdf1, pdf2):
    similarities, images = process_pdfs(pdf1.name, pdf2.name)
    return similarities, [Image.fromarray(img) for img in images]

# Create Gradio interface
gr.Interface(
    fn=gradio_interface,
    inputs=[gr.File(label="Upload PDF 1"), gr.File(label="Upload PDF 2")],
    outputs=[gr.Textbox(label="Similarities"), gr.Gallery(label="Extracted Images from PDF 2")]
).launch()


