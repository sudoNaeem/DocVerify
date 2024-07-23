# Project Overview

This project consists of a FastAPI application for PDF Comparison and a Streamlit application for PDF annotation. The following files are included:

## Files

### `downloadmodel.py`
This script allows you to download the VGG16 model with ImageNet weights.

### `main.py`
This is the main FastAPI application. It provides the backend services for the project. It has two endpoints:
- **GET** `/list_templates/`: Lists the template PDFs in the database.
- **POST** `/SignatureDetection/`: Uploads PDFs and performs the signature detection function.
![FastAPI Interface](https://github.com/ab-ark/Signature-Detection/blob/main/img/fastapi.png)


### `pdf_annotator.py`
This Streamlit application allows users to upload PDF files, annotate them, and store the annotations in the database.
![Streamlit application](https://github.com/ab-ark/Signature-Detection/blob/main/img/streamlit.png)

### `requirements.txt`
Contains all the dependencies required for the project.

### `utils.py`
This file includes utility functions used across the project, including:
- **PDF Processing**: Functions to correct skew, deskew, detect orientation, and remove white margins from PDF images.
- **Image Extraction**: Functions to extract images from PDF annotations.
- **Similarity Computation**: Functions to compute VGG16 model similarity between images.
- **Database Interaction**: Functions to interact with MongoDB to retrieve filenames and annotations.

## Setup Instructions

1. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

2. **Download VGG16 Model**
   Run the downloadmodel.py script to download the VGG16 model with ImageNet weights.
   ```sh
   python3 downloadmodel.py
   ```

3. **Run Streamlit Application:**
   ```sh
   streamlit run pdf_annotator.py
   ```

4. **Run FastAPI Application:**
   ```sh
   fastapi dev main.py
   ```

