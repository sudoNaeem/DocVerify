import fitz
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import boto3
import os
#from dotenv import load_dotenv
import psycopg2
import json

#load_dotenv()



st.set_page_config(layout="wide")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)



POSTGRESQL_CONNECTION_STRING = os.getenv("POSTGRESQL_CONNECTION_STRING")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

def get_pg_connection():
    pg_conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
    return pg_conn

@st.cache_resource
def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

s3_client = get_s3_client()

class PDFManager:
    def __init__(self):
        self.s3_client = s3_client
        self.bucket_name = S3_BUCKET_NAME

    def upload_pdf(self, file, pdf_name):
        file_data = file.read()
        s3_key = f"pdfs/{pdf_name}"
        self.s3_client.put_object(Bucket=self.bucket_name, Key=s3_key, Body=file_data)
        return s3_key

    def retrieve_pdf(self, pdf_name):
        s3_key = f"pdfs/{pdf_name}"
        pdf_data = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)['Body'].read()
        return pdf_data

class AnnotationManager:
    def __init__(self, pg_conn):
        self.pg_conn = pg_conn

    def save_annotations(self, pdf_name, annotations):
        unique_annotations = self.deduplicate_annotations(annotations)
        annotation_data = {
            "pdf_name": pdf_name,
            "annotations": unique_annotations
        }
        with self.pg_conn.cursor() as pg_cursor:
            pg_cursor.execute(
                "INSERT INTO annotations (pdf_name, annotations) VALUES (%s, %s)",
                (pdf_name, json.dumps(unique_annotations))
            )
            self.pg_conn.commit()
        return annotation_data

    def retrieve_annotations(self, pdf_name):
        with self.pg_conn.cursor() as pg_cursor:
            pg_cursor.execute("SELECT annotations FROM annotations WHERE pdf_name = %s", (pdf_name,))
            result = pg_cursor.fetchone()
        if result:
            return result[0] if isinstance(result[0], list) else json.loads(result[0])
        return None

    def deduplicate_annotations(self, annotations):
        seen = set()
        unique_annotations = []
        for annotation in annotations:
            annotation_tuple = (annotation["page_number"], annotation["start_x"], annotation["start_y"], annotation["end_x"], annotation["end_y"], annotation["label"])
            if annotation_tuple not in seen:
                seen.add(annotation_tuple)
                unique_annotations.append(annotation)
        return unique_annotations

pdf_manager = PDFManager()

st.title("PDF Annotator")

if "annotations" not in st.session_state:
    st.session_state.annotations = {}

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file is not None:
    pdf_name = uploaded_file.name

    pg_conn = get_pg_connection()
    annotation_manager = AnnotationManager(pg_conn)

    existing_annotations = annotation_manager.retrieve_annotations(pdf_name)
    if existing_annotations:
        st.error(f"The name '{pdf_name}' already exists. Please upload a PDF with a different name.")
        pg_conn.close()
    else:
        s3_key = pdf_manager.upload_pdf(uploaded_file, pdf_name)
        st.success(f"PDF uploaded to S3 with key '{s3_key}'. Annotations can now be added to '{pdf_name}'.")

        pdf_data = pdf_manager.retrieve_pdf(pdf_name)
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        pages_images = [(page_num, np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples)))
                        for page_num, page in enumerate(pdf_document)
                        for pix in [page.get_pixmap()]]
        total_pages = len(pages_images)
        current_page = st.session_state.current_page

        st.write(f"Annotate PDF - Page {current_page + 1} of {total_pages}")

        page_num, pdf_image = pages_images[current_page]
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=1,
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
                start_x = obj["left"]
                start_y = obj["top"]
                end_x = obj["left"] + obj["width"]
                end_y = obj["top"] + obj["height"]
                label = st.text_input(f"Label for annotation on Page {page_num + 1}", key=f"label{page_num}_{obj['left']}_{obj['top']}")
                if label.strip() == "":
                    st.error("Label cannot be empty. Please provide a label for all annotations.")
                else:
                    new_annotations.append({
                        "page_number": page_num + 1,
                        "start_x": start_x,
                        "start_y": start_y,
                        "end_x": end_x,
                        "end_y": end_y,
                        "label": label
                    })

            seen = set()
            unique_annotations = []
            for annotation in new_annotations:
                annotation_tuple = (annotation["page_number"], annotation["start_x"], annotation["start_y"], annotation["end_x"], annotation["end_y"], annotation["label"])
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
                all_boxes.extend(annotations)

            result = annotation_manager.save_annotations(pdf_name, all_boxes)
            st.success("Annotations saved.")
            st.write("Annotations saved to database:")
            st.json(result)
            pg_conn.close()

st.write("Retrieve and Display PDF:")

pdf_to_retrieve = st.text_input("Enter the name of the PDF to retrieve")
if st.button("Retrieve PDF"):
    pdf_data = pdf_manager.retrieve_pdf(pdf_to_retrieve)
    if pdf_data:
        st.success(f"Retrieved and displaying '{pdf_to_retrieve}'")
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        pages_images = [(page_num, np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples)))
                        for page_num, page in enumerate(pdf_document)
                        for pix in [page.get_pixmap()]]
        for page_num, pdf_image in pages_images:
            st.write(f"Page {page_num + 1}")
            st.image(pdf_image)
    else:
        st.error(f"No PDF found with the name '{pdf_to_retrieve}'")

annotations_to_retrieve = st.text_input("Enter the name of the PDF to retrieve annotations for")
if st.button("Retrieve Annotations"):
    pg_conn = get_pg_connection()
    annotation_manager = AnnotationManager(pg_conn)
    
    annotations = annotation_manager.retrieve_annotations(annotations_to_retrieve)
    if annotations:
        st.success(f"Annotations for '{annotations_to_retrieve}'")
        st.json(annotations)
    else:
        st.error(f"No annotations found for '{annotations_to_retrieve}'")
    pg_conn.close()
