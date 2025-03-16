import streamlit as st
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define latent dimension
latent_dim = 100

# Load the trained Generator
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define your generator architecture here
        pass

    def forward(self, x):
        # Define forward pass
        pass

# Load model
generator = Generator()
generator.load_state_dict(torch.load("models/generator.pth", map_location=device))
generator.to(device).eval()

# Streamlit UI
st.title("🎨 GAN Image Generator")
st.write("Click the button to generate a new image!")

if st.button("Generate Image ⚡"):
    with torch.no_grad():
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        generated_image = generator(noise).cpu()

    # Convert to PIL Image
    img = generated_image.squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 255).astype("uint8")
    img_pil = Image.fromarray(img)

    # Save to BytesIO
    img_buffer = BytesIO()
    img_pil.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    # Display Image
    st.image(img_pil, caption="Generated Image", use_column_width=True)

    # Add download button
    st.download_button(label="Download Image", data=img_buffer, file_name="generated_image.png", mime="image/png")
