#caption.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate captions
def generate_caption(image: Image.Image) -> str:
    inputs = caption_processor(images=image, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    caption = caption_processor.decode(outputs[0], skip_special_tokens=True)
    return caption