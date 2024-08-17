#ocr.py
from PIL import Image
import io
import pytesseract

def extract_text_from_image(image_bytes):
    try:
        # Check if image_bytes is empty
        if not image_bytes:
            raise ValueError("No image data provided.")

        # Attempt to open the image
        image = Image.open(io.BytesIO(image_bytes))
        
        image.verify()  # Verify that the image is valid

        # Reopen the image after verification
        image = Image.open(io.BytesIO(image_bytes))
        image.load()  # Load the image data

        # Perform OCR processing
        text = pytesseract.image_to_string(image)
        return text

    except Exception as e:
        print(f"Error in extract_text_from_image: {e}")
        raise
