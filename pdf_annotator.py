import fitz  # PyMuPDF
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pymongo
import gridfs
import io
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")

class PDFAnnotator:
    def __init__(self):
        self.pdf_document = None
        self.client = pymongo.MongoClient(MONGO_CONNECTION_STRING)
        self.db = self.client["signature_detection"]
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
        boxes = []
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
                boxes.append({
                    "page_number": page_number + 1,
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": end_x,
                    "end_y": end_y,
                    "label": label
                })
        return boxes

    def save_annotations(self, boxes):
        unique_boxes = self.deduplicate_annotations(boxes)
        annotation_data = {
            "pdf_name": self.pdf_name,
            "pdf_id": self.pdf_id,
            "annotations": unique_boxes
        }
        self.db.annotations.insert_one(annotation_data)
        return annotation_data

    def deduplicate_annotations(self, boxes):
        seen = set()
        unique_boxes = []
        for box in boxes:
            box_tuple = (box["page_number"], box["start_x"], box["start_y"], box["end_x"], box["end_y"])
            if box_tuple not in seen:
                seen.add(box_tuple)
                unique_boxes.append(box)
        return unique_boxes

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

if "annotations" not in st.session_state:
    st.session_state.annotations = {}

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file is not None:
    pdf_name = uploaded_file.name
    existing_pdf = pdf_annotator.db.annotations.find_one({"pdf_name": pdf_name})

    if existing_pdf:
        st.error(f"The name '{pdf_name}' already exists. Please upload a PDF with a different name.")
    else:
        pages_images = pdf_annotator.open_pdf(uploaded_file, pdf_name)
        st.success(f"Annotations can now be added to '{pdf_name}'.")

        total_pages = len(pages_images)
        current_page = st.session_state.current_page

        st.write(f"Annotate PDF - Page {current_page + 1} of {total_pages}")

        page_num, pdf_image = pages_images[current_page]
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
            if f"page_{page_num}" not in st.session_state.annotations:
                st.session_state.annotations[f"page_{page_num}"] = []

            new_annotations = []
            for obj in canvas_result.json_data["objects"]:
                label = st.text_input(f"Label for annotation on Page {page_num + 1}", key=f"label{page_num}_{obj['left']}_{obj['top']}")
                if label.strip() == "":
                    st.error("Label cannot be empty. Please provide a label for all annotations.")
                else:
                    obj["label"] = label
                    new_annotations.append(obj)

            # Deduplicate annotations for the current page before saving to session state
            seen = set()
            unique_annotations = []
            for annotation in new_annotations:
                annotation_tuple = (annotation["left"], annotation["top"], annotation["width"], annotation["height"])
                if annotation_tuple not in seen:
                    seen.add(annotation_tuple)
                    unique_annotations.append(annotation)

            st.session_state.annotations[f"page_{page_num}"].extend(unique_annotations)

        col1, col2, col3, col4 = st.columns(4)
        
        if col1.button("Previous Page") and current_page > 0:
            st.session_state.current_page -= 1

        page_number_input = col2.text_input("Go to Page", value=str(current_page + 1))
        if col2.button("Go"):
            try:
                page_num_input = int(page_number_input) - 1
                if 0 <= page_num_input < total_pages:
                    st.session_state.current_page = page_num_input
            except ValueError:
                st.error("Invalid page number.")

        if col3.button("Next Page") and current_page < total_pages - 1:
            st.session_state.current_page += 1

        if col4.button("Save Annotations"):
            all_boxes = []
            for page, annotations in st.session_state.annotations.items():
                page_number = int(page.split("_")[1])
                boxes = pdf_annotator.annotate_pdf(page_number, annotations)
                all_boxes.extend(boxes)

            result = pdf_annotator.save_annotations(all_boxes)
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
