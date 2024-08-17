#face_tagging.py
import numpy as np
from face_recognition import face_encodings, compare_faces
from connection import get_db, get_collections
import streamlit as st

db = get_db()
images_collection, ocr_collection, caption_collection, object_collection, faces_collection, face_tags_collection, faces_with_boxes_collection = get_collections(db)

def retrieve_by_label(label):
    # Retrieve faces with the given label from faces_collection
    matching_faces = faces_collection.find({"label": {"$regex": label, "$options": "i"}})
    faces_with_label = list(faces_collection.find({"label": {"$regex": label, "$options": "i"}}))
    # Extract filenames from the matching faces
    filenames = {face.get("filename") for face in matching_faces}
    # Extract unique hashes from the faces
    image_hashes = {face.get("hash") for face in faces_with_label}

    
    # Retrieve and return images with those filenames from images_collection
    images = []
    for filename in filenames:
        image_doc = images_collection.find_one({"filename": filename})
        if image_doc:
            images.append(image_doc.get("image"))
    
    return images

def get_unique_faces():
    unique_faces = []
    for doc in faces_with_boxes_collection.find():
        face_id = doc.get('unique_id')
        if face_id:
            face = faces_with_boxes_collection.find_one({"unique_id": face_id})
            if face and face not in unique_faces:
                unique_faces.append(face)
    return unique_faces

def get_existing_tags():
    # Retrieve existing tags
    return list(face_tags_collection.distinct("label"))

def tag_face(label, hash, unique_id):
    # Find faces by hash and unique_id in faces_collection
    matching_faces = faces_collection.find({"hash": hash, "unique_id": unique_id,"label":label})
    
    # Tag each matching face with the provided label
    for face in matching_faces:
        # Add to face_tags_collection for historical record
        face_tags_collection.insert_one({
            "label": label,
            "hash": face["hash"],
            "unique_id": face["unique_id"]
        })
        
        # Update faces_collection with the new label
        faces_collection.update_one(
            {"hash": face["hash"], "unique_id": face["unique_id"]},
            {"$set": {"label": label}},
            upsert=True  # Ensure the document is created if it does not exist
        )

    face_record = faces_collection.find_one({'unique_id': unique_id})
    if not face_record:
        st.error("Selected face is not found in the database.")
        return
    
    if 'face_encodings' not in face_record or not face_record['face_encodings']:
        st.error("Selected face does not have face encodings.")
        return
    
    face_encoding = np.array(face_record['face_encodings'][0])
    similar_faces = faces_collection.find()
    
    tagged_count = 0
    for record in similar_faces:
        if 'face_encodings' in record and record['face_encodings']:
            db_face_encoding = np.array(record['face_encodings'][0])
            matches = compare_faces([face_encoding], db_face_encoding)
            if matches[0]:
                faces_collection.update_one({'_id': record['_id']}, {'$set': {'label': label}})
                face_tags_collection.update_one({'_id': record['_id']}, {'$set': {'label': label}}, upsert=True)
                tagged_count += 1
    
    if tagged_count > 0:
        st.success(f"Face and {tagged_count-1} similar faces tagged as '{label}'.")
    else:
        st.warning("No similar faces found to tag.")

