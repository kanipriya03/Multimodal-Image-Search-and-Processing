#face_detect.py
import zipfile
import os
import uuid 
import hashlib
from PIL import Image
from io import BytesIO
import numpy as np
import io
import base64
from bson.binary import Binary 
from collections import Counter
import streamlit as st
import face_recognition
from connection import get_db, get_collections

db = get_db()
(images_collection, ocr_collection, caption_collection, object_collection, faces_collection, face_tags_collection, faces_with_boxes_collection) = get_collections(db)

def encode_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()

def extract_images_from_zip(zip_file):
    images = []
    try:
        with zipfile.ZipFile(zip_file, 'r') as z:
            for name in z.namelist():
                if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_data = z.read(name)
                    image = Image.open(BytesIO(image_data))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    images.append((name, image, image_data))
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP file.")
    return images

def generate_unique_id():
    return str(faces_collection.count_documents({}) + 1)

def upload_images(images):
    for filename, image, image_data in images:
        face_image = np.array(image)
        face_locations = face_recognition.face_locations(face_image)
        face_encodings = face_recognition.face_encodings(face_image, face_locations)

        img_bytes = encode_image(image)
        image_hash = hashlib.md5(img_bytes).hexdigest()

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            face = face_image[top:bottom, left:right]
            face_pil_image = Image.fromarray(face)
            buffered = io.BytesIO()
            face_pil_image.save(buffered, format="JPEG")
            face_img_bytes = buffered.getvalue()

            face_hash = hashlib.md5(face_img_bytes).hexdigest()

            existing_face = faces_with_boxes_collection.find_one({"hash": face_hash})

            if existing_face:
                face_id = existing_face["unique_id"]
            else:
                similar_face_id = None
                for existing_face in faces_with_boxes_collection.find():
                    existing_encoding = np.frombuffer(existing_face["encoding"], dtype=np.float64)
                    if face_recognition.compare_faces([existing_encoding], face_encoding, tolerance=0.6)[0]:
                        similar_face_id = existing_face["unique_id"]
                        break

                if similar_face_id is None:
                    face_id = generate_unique_id()
                else:
                    face_id = similar_face_id

            faces_with_boxes_collection.insert_one({
                "filename": filename,  
                "hash": face_hash, 
                "face_image": Binary(face_img_bytes),
                "face_location": {"top": top, "right": right, "bottom": bottom, "left": left},
                "unique_id": face_id,
                "encoding": face_encoding.tobytes()
            })

            faces_collection.update_one(
                {"hash": face_hash},
                {"$set": {
                    "filename": filename,
                    "image": Binary(img_bytes),
                    "face_encodings": [face_encoding.tolist()],
                    "unique_id": face_id
                }},
                upsert=True 
            )

def group_similar_faces():
    face_groups = {}
    for doc in faces_collection.find():
        encodings = [np.array(enc) for enc in doc['face_encodings']]
        if encodings:
            for encoding in encodings:
                match_found = False
                for face_id in face_groups.keys():
                    matches = face_recognition.compare_faces(face_groups[face_id], encoding, tolerance=0.6)
                    if True in matches:
                        face_groups[face_id].append(encoding)
                        faces_collection.update_one(
                            {"hash": doc['hash']},
                            {"$set": {"unique_id": face_id}}
                        )
                        match_found = True
                        break
                if not match_found:
                    new_face_id = generate_unique_id()
                    face_groups[new_face_id] = [encoding]
                    faces_collection.update_one(
                        {"hash": doc['hash']},
                        {"$set": {"unique_id": new_face_id}}
                    )
    return face_groups

def get_frequent_faces():
    face_ids = [doc['unique_id'] for doc in faces_collection.find({"unique_id": {"$exists": True}})]
    face_counts = Counter(face_ids) 
    most_frequent_faces = face_counts.most_common(3) 
    
    frequent_faces = []
    for face_id, count in most_frequent_faces:
        face_doc = faces_collection.find_one({"unique_id": face_id})
        if face_doc:
            # Ensure image data is in bytes
            image_data = face_doc.get('image')
            if isinstance(image_data, str):  # Check if the image is stored as base64 string
                image_data = base64.b64decode(image_data)
            
            frequent_faces.append((face_id, count, image_data))
    
    return frequent_faces

