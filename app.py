import streamlit as st
from utils import load_model, prepare_image, predict_disease

# Load the model
model = load_model()

st.title("Plant Disease Prediction")
st.write("Upload an image of a plant leaf to predict its health status.")

uploaded_file = st.file_uploader("Choose a leaf image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file to the current directory
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Prepare and predict
    image = prepare_image("temp.jpg")
    prediction = predict_disease(model, image)
    
    st.image("temp.jpg", caption=f"Uploaded Image - Predicted as {prediction}")
    st.write(f"Prediction: **{prediction}**")

    # Cleanup
    import os
    os.remove("temp.jpg")
