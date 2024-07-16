import fitz  # PyMuPDF
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pymongo

class PDFAnnotator:
    def __init__(self):
        self.pdf_document = None
        self.boxes = []
        self.client = pymongo.MongoClient("mongodb+srv://:<password>@cluster0.rykip0e.mongodb.net/")
        self.db = self.client["pdf_annotations"]

    def open_pdf(self, file):
        self.pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        return self.get_all_pages_images()

    def annotate_pdf(self, page_number, annotations):
        page = self.pdf_document.load_page(page_number)
        for annotation in annotations:
            if annotation["type"] == "rect":
                start_x = annotation["left"]
                start_y = annotation["top"]
                end_x = annotation["left"] + annotation["width"]
                end_y = annotation["top"] + annotation["height"]
                rect = fitz.Rect(start_x, start_y, end_x, end_y)
                label = annotation.get("label", "")
                page.add_rect_annot(rect)
                if label:
                    text_rect = fitz.Rect(end_x, start_y, end_x + 100, start_y + 30)
                    page.insert_textbox(text_rect, label, fontsize=12, color=(1, 0, 0))
                self.boxes.append({
                    "page_number": page_number,
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": end_x,
                    "end_y": end_y,
                    "label": label
                })

    def save_annotations(self):
        # Save annotations to MongoDB
        self.db.annotations.insert_one({
            "annotations": self.boxes
        })
        return "Annotations saved."

    def get_all_pages_images(self):
        images = []
        for page_num in range(len(self.pdf_document)):
            page = self.pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append((page_num, np.array(img)))
        return images

pdf_annotator = PDFAnnotator()

st.title("PDF Annotator")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file is not None:
    pages_images = pdf_annotator.open_pdf(uploaded_file)

    st.write("Annotate PDF:")
    
    for page_num, pdf_image in pages_images:
        st.write(f"Page {page_num + 1}")

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=3,
            stroke_color="red",
            background_image=Image.fromarray(pdf_image),
            update_streamlit=True,
            height=pdf_image.shape[0],
            width=pdf_image.shape[1],
            drawing_mode="rect",
            key=f"canvas{page_num}",
        )

        if canvas_result.json_data is not None:
            for obj in canvas_result.json_data["objects"]:
                obj["label"] = st.text_input(f"Label for annotation on Page {page_num + 1}", key=f"label{page_num}_{obj['left']}_{obj['top']}")

            pdf_annotator.annotate_pdf(page_num, canvas_result.json_data["objects"])

    if st.button("Save Annotations"):
        result = pdf_annotator.save_annotations()
        st.write(result)
