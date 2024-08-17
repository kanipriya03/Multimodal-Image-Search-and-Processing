#app.py
import streamlit as st
import io
import hashlib
from PIL import Image, UnidentifiedImageError
from connection import get_db, get_collections
import ocr
from caption import generate_caption
from obj_det import detect_objects
from face_detect import upload_images, extract_images_from_zip, get_frequent_faces
from face_tagging import tag_face, retrieve_by_label, get_existing_tags,get_unique_faces
from embeddings import compute_embedding, cosine_similarity
import nltk
import zipfile
from io import BytesIO
import numpy as np
try:
    from nltk.corpus import wordnet
except LookupError:
    nltk.download('wordnet')
    from nltk.corpus import wordnet
    
db = get_db()
images_collection, ocr_collection, caption_collection, object_collection, faces_collection, face_tags_collection, faces_with_boxes_collection = get_collections(db)
def resize_image(image, max_size=(800, 800)):
    image.thumbnail(max_size)
    return image
def extract_images_from_zip(zip_file):
    image_list = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_data = zip_ref.read(file)
                image = Image.open(BytesIO(image_data))
                image_list.append((file, image, image_data))
    return image_list

def calculate_image_hash(image_data):
    return hashlib.md5(image_data).hexdigest()

def expand_query(query):
    synonyms = set()
    for syn in wordnet.synsets(query):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)
# Function to encode image to store in MongoDB
def encode_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()

def load_image_from_db(image_data):
    try:
        return Image.open(io.BytesIO(image_data))
    except UnidentifiedImageError:
        st.error("Image cannot be identified or is corrupted.")
        raise

# Streamlit app
st.title('Image Processing and Face Tagging')

if 'images_uploaded' not in st.session_state:
    st.session_state.images_uploaded = False
if 'show_face_tagging' not in st.session_state:
    st.session_state.show_face_tagging = False

uploaded_file = st.file_uploader("Choose a ZIP file or an image", type=["zip", "jpg", "jpeg", "png"])
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            img_bytes = uploaded_file.read()
            try:
                image = Image.open(io.BytesIO(img_bytes))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                images = [(uploaded_file.name, image, img_bytes)]
            except UnidentifiedImageError as e:
                st.error(f"Cannot identify image file: {e}")
                st.stop()
        except Exception as e:
            st.error(f"Error opening image: {e}")
            st.stop()
    else:
        try:
            images = extract_images_from_zip(uploaded_file)
        except Exception as e:
            st.error(f"Error extracting images from ZIP file: {e}")
            st.stop()
    
    if images:
        for filename, image, image_data in images:
            try:
                img_bytes = image_data
                image_hash = calculate_image_hash(img_bytes)
                if not images_collection.find_one({"hash": image_hash}):
                    text = ocr.extract_text_from_image(img_bytes)
                    embedding = compute_embedding(text)
                    ocr_collection.insert_one({"filename": filename, "hash": image_hash, "text": text, "embedding": embedding.tolist(), "image": img_bytes})
                    
                    caption = generate_caption(image)
                    embedding = compute_embedding(caption)
                    caption_collection.insert_one({"filename": filename, "hash": image_hash, "caption": caption, "embedding": embedding.tolist(), "image": img_bytes})

                     
                    objects = detect_objects(image)
                    object_text = " ".join(obj['label'] for obj in objects)
                    embedding = compute_embedding(object_text)
                    object_collection.insert_one({"filename": filename, "hash": image_hash, "objects": objects, "embedding": embedding.tolist(), "image": img_bytes})
                     
                    images_collection.insert_one({
                        "filename": filename,
                        "hash": image_hash,
                        "image": img_bytes
                    })
                    
                    upload_images([(filename, image, image_data)])
                
            except Exception as e:
                st.error(f"Error processing file {filename}: {e}")

        st.session_state.images_uploaded = True
        st.success("Images uploaded and processed successfully. Please enter a keyword to search.")
       
        st.session_state.show_face_tagging = True

if st.session_state.images_uploaded:
    query = st.text_input("Enter keyword to search")
    if st.button("Search Keyword"):
        if query:
            query_embeddings = [compute_embedding(query)]
            expanded_queries = expand_query(query)
            for eq in expanded_queries:
                query_embeddings.append(compute_embedding(eq))

            def find_similar_texts(collection, query_embeddings):
                results = []
                for document in collection.find():
                    document_embedding = np.array(document['embedding'])
                    for query_embedding in query_embeddings:
                        similarity = cosine_similarity(query_embedding, document['embedding'])
                        if similarity > 0.5: 
                            results.append((similarity, document))
                            break  
                return sorted(results, key=lambda x: x[0], reverse=True)

            ocr_results = find_similar_texts(ocr_collection, query_embeddings)
            caption_results = find_similar_texts(caption_collection, query_embeddings)
            object_results = find_similar_texts(object_collection, query_embeddings)

            ocr_keyword_results = ocr_collection.find({"text": {"$regex": query, "$options": "i"}})
            caption_keyword_results = caption_collection.find({"caption": {"$regex": query, "$options": "i"}})
            object_keyword_results = object_collection.find({"objects.label": {"$regex": query, "$options": "i"}})
            
            displayed_hashes = set()

            for _, result in ocr_results:
                image_hash = result["hash"]
                if image_hash not in displayed_hashes:
                    st.image(result["image"])
                    displayed_hashes.add(image_hash)

            for result in ocr_keyword_results:
                image_hash = result["hash"]
                if image_hash not in displayed_hashes:
                    st.image(result["image"])
                    displayed_hashes.add(image_hash)

            for _, result in caption_results:
                image_hash = result["hash"]
                if image_hash not in displayed_hashes:
                    st.image(result["image"])
                    displayed_hashes.add(image_hash)

            for result in caption_keyword_results:
                image_hash = result["hash"]
                if image_hash not in displayed_hashes:
                    st.image(result["image"])
                    displayed_hashes.add(image_hash)

            for _, result in object_results:
                image_hash = result["hash"]
                if image_hash not in displayed_hashes:
                    st.image(result["image"])
                    displayed_hashes.add(image_hash)

            for result in object_keyword_results:
                image_hash = result["hash"]
                if image_hash not in displayed_hashes:
                    st.image(result["image"])
                    displayed_hashes.add(image_hash)
            

    if st.button("Show Frequent Faces"):
        try:
            most_frequent_faces = get_frequent_faces()
            if most_frequent_faces:
                for face_id, count, image_data in most_frequent_faces:
                    st.write(f"Face ID: {face_id}, Count: {count}")
                    try:
                        img = load_image_from_db(image_data)
                        st.image(img, caption=f"Face ID: {face_id}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying face image: {e}")
            else:
                st.write("No faces found.")
        except Exception as e:
            st.error(f"Error retrieving frequent faces: {e}")

    if st.session_state.show_face_tagging:
        option = st.selectbox("Choose an option", ["Search by Label", "Tag Face"])

        def display_faces_grid(faces, enable_tagging=True):
            unique_face_ids = set()
            cols = st.columns(4) 
            displayed_faces = 0  

            if 'loaded_faces' not in st.session_state:
                st.session_state.loaded_faces = faces
                st.session_state.face_tagging_state = {}
            # Use the faces from session state
            for face in st.session_state.loaded_faces:
                face_id = face.get('unique_id')
                if face_id not in unique_face_ids:
                    unique_face_ids.add(face_id)
                    image_data = face.get('face_image') 
                    if image_data:
                        try:
                            img = Image.open(io.BytesIO(image_data))
                            col = cols[displayed_faces % 4]  
                            with col:
                                # Display the image
                                st.image(img, caption=f"Face ID: {face_id}", use_column_width=True)
                                
                                if enable_tagging:
                                    if face_id not in st.session_state.face_tagging_state:
                                        st.session_state.face_tagging_state[face_id] = False
                                    if st.button(f"Tag Face {face_id}", key=f"tag_button_{face_id}"):
                                        st.session_state[f"show_tagging_{face_id}"] = True
                                        
                                    if st.session_state.get(f"show_tagging_{face_id}", False):
                                        existing_tags = get_existing_tags()
                                        selected_label = st.selectbox("Select label", options=existing_tags, key=f"selectbox_{face_id}")
                                     
                                        add_new_label = st.checkbox("Add new label", key=f"add_new_{face_id}")
                                        if add_new_label:
                                            new_label = st.text_input("Enter new label", key=f"new_label_{face_id}")
                                        else:
                                            new_label = ""

                                        if st.button("Confirm Tag", key=f"confirm_tag_{face_id}"):
                                            label_to_use = new_label if new_label else selected_label

                                            if label_to_use:
                                                similar_faces = [f for f in st.session_state.loaded_faces if f.get('unique_id') == face_id]
                                                for sim_face in similar_faces:
                                                    tag_face(label_to_use, sim_face.get('hash'), sim_face.get('unique_id'))
                                                
                                                num_tagged_faces = len(similar_faces)
                                                st.session_state[f"show_tagging_{face_id}"] = False
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                    displayed_faces += 1

        if option == "Tag Face":
            unique_faces=get_unique_faces()
            if unique_faces:
                display_faces_grid(unique_faces)
            else:
                st.write("No unique faces found.")

        
        elif option == "Search by Label":
            label_search = st.text_input("Enter label to search")
            if st.button("Search by Label"):
                if label_search:
                    try:
                        matching_images = retrieve_by_label(label_search)
                        if matching_images:
                            for img_data in matching_images:
                                st.image(load_image_from_db(img_data), caption="Similar Image", use_column_width=True)
                        else:
                            st.write("No images found for the label.")
                    except Exception as e:
                        st.error(f"Error searching by label: {e}")

