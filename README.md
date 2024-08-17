# Multimodal-Image-Search-and-Processing

## Overview
**Image Search and Processing** is an advanced image management and processing application. It allows users to upload images, perform text extraction (OCR), generate descriptive captions, detect objects, recognize faces, and tag those faces. The application integrates with MongoDB to store image metadata and processing results, and provides a user-friendly interface built with Streamlit.

## Features
- **Image Upload:** Users can upload images via ZIP files or individually. The app processes and stores these images in MongoDB.
- **OCR (Optical Character Recognition):** Extracts text from images using the Tesseract OCR engine. Useful for documents, signs, or any image with textual content.
- **Image Captioning:** Generates captions for images using the BLIP model from Hugging Face, providing a descriptive text about the content of the image.
- **Object Detection:** Identifies objects in images using YOLO (You Only Look Once) models, which helps in classifying and labeling objects present in the images.
- **Face Detection and Tagging:** Detects and processes faces within images using the `face_recognition` library. Users can tag faces for easier identification and management.
- **Search Functionality:** Allows searching images by keywords, text from OCR, or similar faces. Supports various search queries to retrieve relevant images based on different criteria.

## Installation

### Prerequisites
- **Python 3.7 or Higher:** Ensure you have Python 3.7 or a newer version installed. You can check your Python version with `python --version`.
- **MongoDB:** You need MongoDB installed and running. You can use a local installation or a MongoDB Atlas cluster.

### Dependencies
Create a virtual environment and install the required dependencies. Here's how you can do it:

1. **Create and activate a virtual environment:**

    ```bash
    python -m venv myenv  # Replace 'myenv' with your preferred name for the virtual environment.
    myenv\Scripts\activate  # To activate the virtual environment on Windows.
    ```

2. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Requirements.txt
```plaintext
transformers==4.42.3
pytesseract==0.3.10
face_recognition==1.3.0
sentence-transformers==2.2.2
streamlit==1.22.0
pymongo==4.3.0
torch==2.0.0
Pillow==10.0.0
nltk==3.8.1


##Setup
  1.Clone the Repository:
     git clone https://github.com/your-username/image-search-and-processing.git
     cd image-search-and-processing
  2.Install Dependencies
  3.Configure MongoDB:
    ->Local MongoDB: Ensure MongoDB is running locally. The default connection string is mongodb://localhost:27017/.
    ->MongoDB Atlas: If using MongoDB Atlas, update the connection.py file with your Atlas connection string.
##Usage
  1.Start the Streamlit Application:
      streamlit run app.py
   2.Open the Application:
      Navigate to http://localhost:8501 in your web browser to interact with the application interface.
      
##How to Use
Upload Images: Use the upload button in the Streamlit interface to upload images or ZIP files containing images.
Process Images: After uploading, the application will process the images, perform OCR, generate captions, detect object, recognize faces,tagging of faces in images with custom labels, automatically applying the same label to similar faces across the database. It supports case-insensitive and partial match searches to retrieve and display tagged images. 
Search Images: Enter keywords in the search box to find images based on labels, extracted text, or detected faces.

##File Descriptions
app.py: Main Streamlit application script. Handles the user interface, image uploads, and interactions.
caption.py: Contains functions for generating captions from images using the BLIP model.
connection.py: Manages the connection to MongoDB, including functions for database operations.
embeddings.py: Provides functions for generating and comparing text embeddings.
face_detect.py: Implements face detection algorithms using the face_recognition library.
face_tagging.py: Manages face tagging, including assigning and retrieving face labels.
obj_det.py: Implements object detection using YOLO models.
ocr.py: Handles Optical Character Recognition tasks, extracting text from images.
wordnet.py: Contains setup for NLTK WordNet, which can be used for lexical processing. This file may include functions for accessing and utilizing the WordNet lexical database, useful for enhancing text search capabilities by understanding the relationships between words and their synonyms.

##Contributing
 Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes or improvements.




























