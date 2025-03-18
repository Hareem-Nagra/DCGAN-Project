import streamlit as st
import os
import subprocess
from PIL import Image

# Set the path for the generated image
image_path = "generated_sample.png"

# Run model.py to generate the image (if it hasn't been generated already)
if not os.path.exists(image_path):
    st.info("Running the model to generate an image...")
    subprocess.run(["python", "model.py"])

# Streamlit app
st.title("Generated Image Viewer")

if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, caption="Generated Image", use_column_width=True)
else:
    st.error("Generated image not found. Please check model.py execution.")