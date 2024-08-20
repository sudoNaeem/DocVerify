import fitz
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import numpy as np
import boto3
from io import BytesIO
import psycopg2
import json
import logging
from utils import process_pdf_extract_images_and_save_high_res

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(layout="wide")

# Hide Streamlit menu and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

POSTGRESQL_CONNECTION_STRING='postgresql://postgres.tjnvqtfpfcarwaqcpugt:c3jmkacJGhKD4e@aws-0-us-west-1.pooler.supabase.com:6543/postgres'
AWS_ACCESS_KEY_ID='AKIAWTYXWIPAGEVLW2RZ'
AWS_SECRET_ACCESS_KEY='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'
S3_BUCKET_NAME='pdfsignaturedetection'

def get_pg_connection():
    try:
        return psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
    except Exception as e:
        logging.error(f"Error connecting to PostgreSQL: {e}")
        st.error("Error connecting to database.")
        raise

@st.cache_resource
def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

s3_client = get_s3_client()

# Main PDF Manager Class
class PDFManager:
    def __init__(self):
        self.s3_client = s3_client
        self.bucket_name = S3_BUCKET_NAME

    def retrieve_pdf(self, pdf_name):
        try:
            s3_key = f"pdfs/{pdf_name}"
            pdf_data = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)['Body'].read()
            logging.info(f"Retrieved PDF '{pdf_name}' from S3.")
            return pdf_data
        except Exception as e:
            logging.error(f"Error retrieving PDF '{pdf_name}': {e}")
            st.error(f"Error retrieving PDF '{pdf_name}'.")

    def upload_pdf_to_s3(self, file, pdf_name):
        try:
            # Process the uploaded PDF
            pdf_buffer = BytesIO(file.read())
            processed_pdf = process_pdf_extract_images_and_save_high_res(pdf_buffer)

            # Check if file exists in S3
            s3_key = f"pdfs/{pdf_name}"
            if self.file_exists_in_s3(s3_key):
                logging.warning(f"File '{pdf_name}' already exists in S3.")
                return None  # File already exists

            # Upload the processed PDF to S3
            self.s3_client.put_object(Bucket=self.bucket_name, Key=s3_key, Body=processed_pdf.getvalue())
            logging.info(f"Uploaded PDF '{pdf_name}' to S3.")
            return s3_key
        except Exception as e:
            logging.error(f"Error uploading PDF '{pdf_name}' to S3: {e}")
            st.error(f"Error uploading PDF '{pdf_name}'.")

    def file_exists_in_s3(self, s3_key):
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except self.s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logging.error(f"Error checking if file exists in S3: {e}")
                raise

# Annotation Manager Class
class AnnotationManager:
    def __init__(self, pg_conn):
        self.pg_conn = pg_conn

    def save_annotations(self, pdf_name, annotations):
        try:
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
            logging.info(f"Saved annotations for PDF '{pdf_name}'.")
            return annotation_data
        except Exception as e:
            logging.error(f"Error saving annotations for PDF '{pdf_name}': {e}")
            st.error(f"Error saving annotations.")

    def retrieve_annotations(self, pdf_name):
        try:
            with self.pg_conn.cursor() as pg_cursor:
                pg_cursor.execute("SELECT annotations FROM annotations WHERE pdf_name = %s", (pdf_name,))
                result = pg_cursor.fetchone()
            if result:
                return json.loads(result[0]) if isinstance(result[0], str) else result[0]
            return None
        except Exception as e:
            logging.error(f"Error retrieving annotations for PDF '{pdf_name}': {e}")
            st.error(f"Error retrieving annotations.")

    def deduplicate_annotations(self, annotations):
        seen = {}
        for annotation in annotations:
            annotation_key = (annotation["page_number"], annotation["start_x"], annotation["start_y"], annotation["end_x"], annotation["end_y"])
            seen[annotation_key] = annotation
        return list(seen.values())

# Instantiate PDFManager and AnnotationManager
pdf_manager = PDFManager()

st.title("PDF Annotator")
st.sidebar.title("PDF Tools")
choice = st.sidebar.radio("Select an option:", ( "Upload PDF","Annotate PDF", "PDF Annotated"))

if choice == "Annotate PDF":
    st.header("Annotate PDF from S3")

    pdf_name = st.text_input("Enter the name of the PDF to retrieve from S3")
    if pdf_name:
        pdf_data = pdf_manager.retrieve_pdf(pdf_name)
        if pdf_data:
            st.success(f"Retrieved '{pdf_name}' for annotation.")

            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            pages_images = [(page_num, np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples)))
                            for page_num, page in enumerate(pdf_document)
                            for pix in [page.get_pixmap()]]
            total_pages = len(pages_images)
            
            if "annotations" not in st.session_state:
                st.session_state.annotations = {}
            if "current_page" not in st.session_state:
                st.session_state.current_page = 0
            
            current_page = st.session_state.current_page

            def prev_page():
                if st.session_state.current_page > 0:
                    st.session_state.current_page -= 1

            def next_page(total_pages):
                if st.session_state.current_page < total_pages - 1:
                    st.session_state.current_page += 1

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
                    label_options = ["Name", "Date", "Signature", "Checkbox", "Text"]
                    label_type = st.selectbox(f"Select label type for annotation on Page {page_num + 1}", label_options, key=f"label_type{page_num}_{obj['left']}_{obj['top']}")

                    label = st.text_input(f"Enter ID or name for '{label_type}' on Page {page_num + 1}", key=f"label_id{page_num}_{obj['left']}_{obj['top']}")

                    if label_type and label.strip():
                        new_annotations.append({
                            "page_number": page_num + 1,
                            "start_x": start_x,
                            "start_y": start_y,
                            "end_x": end_x,
                            "end_y": end_y,
                            "label": label,
                            "label_type": label_type
                        })

                st.session_state.annotations[f"page_{page_num}"].extend(new_annotations)

            col1, col2, col3 = st.columns(3)
            col1.button("Previous Page", on_click=prev_page)
            col2.button("Next Page", on_click=next_page, args=(total_pages,))

            if col3.button("Save Annotations"):
                pg_conn = get_pg_connection()
                annotation_manager = AnnotationManager(pg_conn)

                all_boxes = []
                for page, annotations in st.session_state.annotations.items():
                    page_number = int(page.split("_")[1])
                    all_boxes.extend(annotations)

                result = annotation_manager.save_annotations(pdf_name, all_boxes)
                st.success("Annotations saved.")
                st.write("Annotations saved to database:")
                st.json(result)
                pg_conn.close()
        else:
            st.error(f"No PDF found with the name '{pdf_name}'")

elif choice == "Upload PDF":
    st.header("Upload and Process PDF to S3")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        pdf_name = uploaded_file.name

        s3_key = pdf_manager.upload_pdf_to_s3(uploaded_file, pdf_name)
        if s3_key:
            st.success(f"PDF processed and uploaded to S3 with key '{s3_key}'")
        else:
            st.error(f"A file with the name '{pdf_name}' already exists in S3. Please rename your file and try again.")



elif choice == "PDF Annotated":
    pdf_to_annotate = st.text_input("Enter the name of the PDF to annotate and display")
    if st.button("Retrieve and Annotate PDF"):
        try:
            pdf_data = pdf_manager.retrieve_pdf(pdf_to_annotate)
            if pdf_data:
                st.success(f"Retrieved and displaying annotated '{pdf_to_annotate}'")
                
                try:
                    pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
                    pg_conn = get_pg_connection()
                    annotation_manager = AnnotationManager(pg_conn)
                    annotations = annotation_manager.retrieve_annotations(pdf_to_annotate)
                    
                    pages_images = [(page_num, np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples)))
                                    for page_num, page in enumerate(pdf_document)
                                    for pix in [page.get_pixmap()]]
                    
                    for page_num, pdf_image in pages_images:
                        st.write(f"Page {page_num + 1}")
                        
                        # Convert image to an editable format
                        annotated_image = Image.fromarray(pdf_image)
                        draw = ImageDraw.Draw(annotated_image)
                        
                        # Draw annotations
                        if annotations:
                            for annotation in annotations:
                                if annotation['page_number'] == page_num + 1:
                                    start = (annotation['start_x'], annotation['start_y'])
                                    end = (annotation['end_x'], annotation['end_y'])
                                    draw.rectangle([start, end], outline="red", width=2)
                                    label_position = (start[0], start[1] - 15)
                                    draw.text(label_position, annotation['label'], fill="red")
                        
                        # Display the annotated image
                        st.image(annotated_image)
                    
                    pg_conn.close()
                except Exception as e:
                    st.error("Failed to process or annotate PDF data.")
                    st.error(str(e))
            else:
                st.error(f"No PDF found with the name '{pdf_to_annotate}'")
        except Exception as e:
            st.error(str(e))
