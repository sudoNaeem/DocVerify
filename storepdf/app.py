import fitz  # PyMuPDF
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pymongo
import gridfs
import io

class PDFAnnotator:
    def __init__(self):
        self.pdf_document = None
        self.boxes = []
        self.client = pymongo.MongoClient("mongodb+srv://admin:admin@cluster0.rykip0e.mongodb.net/")
        self.db = self.client["pdf_annotations"]
        self.fs = gridfs.GridFS(self.db)
        self.pdf_name = ""
        self.pdf_id = None

    def open_pdf(self, file, name):
        self.pdf_name = name
        file_data = io.BytesIO(file.getbuffer())
        self.pdf_id = self.fs.put(file_data, filename=self.pdf_name)
        pdf_data = self.fs.get(self.pdf_id).read()
        self.pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
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
                    "page_number": page_number + 1,
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": end_x,
                    "end_y": end_y,
                    "label": label
                })

    def save_annotations(self):
        annotation_data = {
            "pdf_name": self.pdf_name,
            "pdf_id": self.pdf_id,
            "annotations": self.boxes
        }
        self.db.annotations.insert_one(annotation_data)
        return annotation_data

    def get_all_pages_images(self):
        images = []
        for page_num in range(len(self.pdf_document)):
            page = self.pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append((page_num, np.array(img)))
        return images

    def retrieve_pdf(self, pdf_name):
        annotation = self.db.annotations.find_one({"pdf_name": pdf_name})
        if annotation:
            pdf_id = annotation["pdf_id"]
            pdf_data = self.fs.get(pdf_id).read()
            self.pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            return self.get_all_pages_images()
        return None

pdf_annotator = PDFAnnotator()

st.title("PDF Annotator")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file is not None:
    pdf_name = uploaded_file.name
    existing_pdf = pdf_annotator.db.annotations.find_one({"pdf_name": pdf_name})
    
    if existing_pdf:
        st.error(f"The name '{pdf_name}' already exists. Please upload a PDF with a different name.")
    else:
        pages_images = pdf_annotator.open_pdf(uploaded_file, pdf_name)
        st.success(f"Annotations can now be added to '{pdf_name}'.")

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
            st.success("Annotations saved.")
            st.write("Annotations saved to database:")
            st.json(result)

st.write("Retrieve and Display PDF:")

pdf_to_retrieve = st.text_input("Enter the name of the PDF to retrieve")
if st.button("Retrieve PDF"):
    retrieved_pages_images = pdf_annotator.retrieve_pdf(pdf_to_retrieve)
    if retrieved_pages_images:
        st.success(f"Retrieved and displaying '{pdf_to_retrieve}'")
        for page_num, pdf_image in retrieved_pages_images:
            st.write(f"Page {page_num + 1}")
            st.image(pdf_image)
    else:
        st.error(f"No PDF found with the name '{pdf_to_retrieve}'")
