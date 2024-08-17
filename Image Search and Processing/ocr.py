#ocr.py
from PIL import Image
import io
import pytesseract

def extract_text_from_image(image_bytes):
    try:
        if not image_bytes:
            raise ValueError("No image data provided.")
        image = Image.open(io.BytesIO(image_bytes))
        
        image.verify() 

        image = Image.open(io.BytesIO(image_bytes))
        image.load() 

        text = pytesseract.image_to_string(image)
        return text

    except Exception as e:
        print(f"Error in extract_text_from_image: {e}")
        raise
