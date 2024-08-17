#connection.py
from pymongo import MongoClient

def get_db():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['Final_image_data']
    return db

def get_collections(db):
    images_collection = db['images']
    ocr_collection = db['ocr']
    caption_collection = db['captions']
    object_collection = db['objects']
    faces_collection = db['faces']
    face_tags_collection = db['face_tags']
    faces_with_boxes_collection = db['faces_with_boxes']  
    return images_collection, ocr_collection, caption_collection, object_collection, faces_collection, face_tags_collection, faces_with_boxes_collection
