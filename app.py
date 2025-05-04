import streamlit as st
import json
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from uuid import uuid4
from fuzzywuzzy import fuzz
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("Hugging Face token not found. Set HF_TOKEN in .env or environment variables.")
    st.stop()

# Function to call Hugging Face Inference API
def call_hf_llm(query, context):
    """
    Call the Hugging Face Inference API to generate a response using Mistral-7B-Instruct-v0.3.
    """
    try:
        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        prompt = (
            "You are a crop disease expert. Use the following data to answer queries about crop diseases:\n\n"
            f"{context}\n\n"
            f"User Query: {query}\n\n"
            "Provide a concise, accurate, and conversational response about the crop disease or related information."
        )
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()[0]["generated_text"].strip()
    #except Exception as e:
        #st.error(f"Error calling Hugging Face API: {str(e)}")
        #return None
    except requests.exceptions.HTTPError:
        return None
    except Exception:
        return None
        

# Function to create context from crops_data
def create_llm_context(crops_data, crop_name=None, disease_name=None):
    """
    Create a context string from crops_data for the LLM, optionally filtering by crop or disease.
    """
    context = []
    for crop in crops_data:
        if crop_name and fuzz.partial_ratio(crop['crop'].lower(), crop_name.lower()) < 90:
            continue
        crop_info = f"Crop: {crop['crop']}\nAliases: {', '.join(crop.get('aliases', []))}\nDiseases:\n"
        for disease in crop.get('diseases', []):
            if disease_name and fuzz.partial_ratio(disease['name'].lower(), disease_name.lower()) < 90:
                continue
            disease_info = (
                f"- {disease['name']}\n"
                f"  Causal Organism: {disease['causal_organism']}\n"
                f"  Symptoms: {disease['symptoms']}\n"
                f"  Favorable Conditions: {disease.get('favorable_conditions', 'Not specified')}\n"
                f"  Prevention: {disease['prevention']}\n"
            )
            crop_info += disease_info
        context.append(crop_info)
    return "\n".join(context) if context else "No relevant crop or disease data found."

# Flatten nested crop data
def flatten_crops_data(data):
    crops = []
    for item in data:
        if 'crop' in item:
            crops.append(item)
        elif 'crops' in item:
            crops.extend(item['crops'])
    return crops

# Load disease data with error handling
try:
    with st.spinner("Loading crop disease data..."):
        with open('crop_diseases.json') as f:
            raw_data = json.load(f)
        crops_data = flatten_crops_data(raw_data)
except FileNotFoundError:
    st.error("Error: crop_diseases.json file not found.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error: Invalid JSON format in crop_diseases.json.")
    st.stop()

# Load pre-trained models with error handling
try:
    pre_finetune_model = tf.keras.models.load_model('pre_finetune_model.h5')
    finetuned_model = tf.keras.models.load_model('finetuned_model.h5')
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Class names for disease prediction
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', ' thrustingPotato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Format class names for display
def format_class_name(class_name):
    return class_name.replace("___", " - ").replace("_", " ")

# Image preprocessing
def preprocess_image(image):
    try:
        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        if img_array.shape[-1] != 3:  # Handle grayscale images
            img_array = np.stack([img_array] * 3, axis=-1)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

# Predict disease
def predict_disease(image, model):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    return np.argmax(prediction, axis=1)[0], np.max(prediction)

# Token-based symptom matching with scoring
def match_symptoms(user_input, disease_symptoms):
    common_words = {'how', 'to', 'prevent', 'in', 'on', 'and', 'the', 'of', 'for', 'with', 'at', 'is', 'are'}
    input_words = [word for word in user_input.lower().split() if word not in common_words]
    symptom_words = [word for word in disease_symptoms.lower().split() if word not in common_words]
    
    matches = sum(1 for word in input_words if word in symptom_words)
    score = matches / max(len(input_words), 1)
    return score, matches

# Chatbot response with Hugging Face LLM
def chatbot_response(user_input):
    user_input = user_input.lower().strip()
    if not user_input:
        return "Please provide a valid query about crop diseases."

    # Try Hugging Face LLM first
    best_crop = None
    best_disease = None
    best_crop_score = 0
    
    # Find the best matching crop and disease for context
    for crop in crops_data:
        crop_names = [crop['crop'].lower()] + [alias.lower() for alias in crop.get('aliases', [])]
        crop_score = max(fuzz.partial_ratio(user_input, name) for name in crop_names)
        if crop_score > best_crop_score:
            best_crop_score = crop_score
            best_crop = crop
            for disease in crop.get('diseases', []):
                if disease['name'].lower() in user_input:
                    best_disease = disease
                    break

    # Create context for LLM
    context = create_llm_context(crops_data, 
                              crop_name=best_crop['crop'] if best_crop else None, 
                              disease_name=best_disease['name'] if best_disease else None)
    
    # Call Hugging Face LLM
    llm_response = call_hf_llm(user_input, context)
    if llm_response:
        return llm_response

    # Fallback to original logic if LLM fails
    best_response = None
    best_score = 0
    best_matches = 0

    for crop in crops_data:
        crop_names = [crop['crop'].lower()] + [alias.lower() for alias in crop.get('aliases', [])]
        crop_match_score = max(fuzz.partial_ratio(user_input, name) for name in crop_names)
        
        if crop_match_score > 80:
            for disease in crop.get('diseases', []):
                if disease['name'].lower() in user_input:
                    return (
                        f"**Crop**: {crop['crop']}\n"
                        f"**Disease**: {disease['name']}\n"
                        f"**Causal Organism**: {disease['causal_organism']}\n"
                        f"**Symptoms**: {disease['symptoms']}\n"
                        f"**Favorable Conditions**: {disease.get('favorable_conditions', 'Not specified')}\n"
                        f"**Prevention**: {disease['prevention']}"
                    )
                
                score, matches = match_symptoms(user_input, disease['symptoms'])
                if score > best_score or (score == best_score and matches > best_matches):
                    best_score = score
                    best_matches = matches
                    best_response = (
                        f"**Crop**: {crop['crop']}\n"
                        f"**Disease**: {disease['name']}\n"
                        f"**Causal Organism**: {disease['causal_organism']}\n"
                        f"**Symptoms**: {disease['symptoms']}\n"
                        f"**Favorable Conditions**: {disease.get('favorable_conditions', 'Not specified')}\n"
                        f"**Prevention**: {disease['prevention']}"
                    )
            
            if best_score < 0.3:
                diseases = crop.get('diseases', [])
                if diseases:
                    return f"**{crop['crop']} Diseases**: {', '.join([d['name'] for d in diseases])}"
                else:
                    return f"No disease information available for {crop['crop']}."

    return best_response or "Sorry, I couldn't find a match. Try specifying a crop or disease, e.g., 'Rice Blast' or 'Tomato leaf spots'."

# Disease information retrieval
def get_disease_info(crop_name, disease_name):
    for crop in crops_data:
        if fuzz.partial_ratio(crop['crop'].lower(), crop_name.lower()) > 90:
            for disease in crop.get('diseases', []):
                if fuzz.partial_ratio(disease['name'].lower(), disease_name.lower()) > 90:
                    return disease
    return None

# Streamlit UI
st.title("Crop Disease Assistant 🌱")

tab1, tab2 = st.tabs(["Chatbot", "Disease Recognition"])

# Chatbot Tab
with tab1:
    st.header("Crop Disease Chatbot")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    user_input = st.chat_input("Ask about crop diseases...")
    
    if user_input:
        response = chatbot_response(user_input)
        st.session_state.chat_history.insert(0, {
            "id": str(uuid4()),
            "user": user_input,
            "bot": response
        })
    
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message["user"])
        with st.chat_message("assistant"):
            st.markdown(message["bot"])

# Disease Recognition Tab (Unchanged)
with tab2:
    st.header("📸 Disease Detection")
    upload_option = st.radio("Choose input method:", ("Upload Image", "Use Camera"))

    image = None
    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Upload a leaf image", type=None)
        if uploaded_file:
            filename = uploaded_file.name.lower()
            _, ext = os.path.splitext(filename)
            if ext not in ['.jpg', '.jpeg', '.png']:
                st.error(f"Invalid file type: {ext}. Please upload JPG/JPEG/PNG.")
                st.stop()
            try:
                image = Image.open(uploaded_file).convert('RGB')
            except Exception as e:
                st.error(f"Invalid image file: {str(e)}")
                st.stop()
    else:
        camera_image = st.camera_input("Take a photo")
        if camera_image:
            try:
                image = Image.open(camera_image).convert('RGB')
            except Exception as e:
                st.error(f"Invalid image file: {str(e)}")
                st.stop()

    if image:
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            try:
                pre_class_idx, pre_conf = predict_disease(image, pre_finetune_model)
                fine_class_idx, fine_conf = predict_disease(image, finetuned_model)

                st.write("### Prediction Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Before Fine-Tuning**")
                    st.write(f"- Disease: {format_class_name(CLASS_NAMES[pre_class_idx])}")
                    st.write(f"- Confidence: {pre_conf:.2%}")

                with col2:
                    st.write("**After Fine-Tuning**")
                    st.write(f"- Disease: {format_class_name(CLASS_NAMES[fine_class_idx])}")
                    st.write(f"- Confidence: {fine_conf:.2%}")

                if 'healthy' not in CLASS_NAMES[fine_class_idx].lower():
                    st.warning("⚠️ Disease detected! Consult an expert.")
                    predicted_disease = format_class_name(CLASS_NAMES[fine_class_idx])
                    crop_name = predicted_disease.split(" - ")[0]
                    disease_name = predicted_disease.split(" - ")[1]
                    disease_info = get_disease_info(crop_name, disease_name)
                    if disease_info:
                        st.markdown("#### Detailed Disease Information")
                        st.markdown(
                            f"**Crop**: {crop_name}\n"
                            f"**Disease**: {disease_info['name']}\n"
                            f"**Causal Organism**: {disease_info['causal_organism']}\n"
                            f"**Symptoms**: {disease_info['symptoms']}\n"
                            f"**Favorable Conditions**: {disease_info.get('favorable_conditions', 'Not specified')}\n"
                            f"**Prevention**: {disease_info['prevention']}"
                        )
                    else:
                        st.info(f"No detailed information found for {disease_name} in {crop_name}.")
                else:
                    st.success("✅ Leaf appears healthy!")

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")





hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stActionButton {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

