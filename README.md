# Project Overview

This project consists of a FastAPI application for backend services and a Streamlit application for PDF annotation. The following files are included:

## Files

### `downloadmodel.py`
This script allows you to download the VGG16 model with ImageNet weights.

### `main.py`
This is the main FastAPI application. It provides the backend services for the project.

### `pdf_annotator.py`
This Streamlit application allows users to upload PDF files, annotate them, and store the annotations in the database.

### `requirements.txt`
Contains all the dependencies required for the project.

### `utils.py`
Utility functions used across the project.

## Setup Instructions

1. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
